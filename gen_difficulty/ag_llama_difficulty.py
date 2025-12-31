import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from transformers.optimization import Adafactor
import evaluate
import time
import json
import shutil
from huggingface_hub import whoami, snapshot_download
import packaging.version
import inspect

# --- Environment Setup ---
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Hugging Face Token ---
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]
    print("Removed HF_TOKEN environment variable to use cached token or CLI login.")
try:
    user_info = whoami()
    print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
except Exception as e:
    print(f"Hugging Face login check error: {e}. Ensure CLI login for gated models.")

# --- Reproducibility ---
RANDOM_SEED = 63
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Configuration for LLAMA ---
models_to_run_from_mapping = [
    "meta-llama/Meta-Llama-3.1-8B",
    # Add other Llama models here if needed
]

model_name_mapping = {
    "meta-llama/Meta-Llama-3.1-8B": "llama-3.1-8B",
    # Contextual mappings (can be trimmed if only Llama is ever run by this script)
    "Qwen/Qwen2.5-7B": "qwen2.5_7b_base",
    "microsoft/deberta-base": "deberta_base",
    "bert-base-uncased": "bert_base_uncased",
    "roberta-base": "roberta_base",
    "albert-base-v2": "albert_base_v2",
    "distilbert-base-uncased": "distilbert_base_uncased",
    "gpt2": "gpt2",
    "google/electra-base-discriminator": "electra_base_discriminator",
    "facebook/bart-base": "bart_base",
    "t5-base": "t5_base",
    "t5-3b": "t5_3b",
    "xlnet-base-cased": "xlnet_base_cased",
}

DATASET_ID = "contemmcm/ag_news"
RESULTS_OUTPUT_DIR = "ag_news_results"
TASK_NAME_FOR_RESULTS_FILE = "ag_news_llama_20ft_80eval" # Specific for Llama

MAX_LENGTH = 128
# NUM_LABELS_FROM_DATASET will be determined dynamically

# User-specified batch sizes for Llama
LLAMA_TRAIN_BATCH_SIZE_PER_DEVICE = 32
LLAMA_EVAL_BATCH_SIZE_PER_DEVICE = 64

model_train_batch_sizes_per_device = {
    "default": LLAMA_TRAIN_BATCH_SIZE_PER_DEVICE, # Default for this script
    "meta-llama/Meta-Llama-3.1-8B": LLAMA_TRAIN_BATCH_SIZE_PER_DEVICE,
}
LLM_TARGET_EFFECTIVE_BATCH_SIZE = 32 # If physical BS is 32, grad accum will be 1

model_eval_batch_sizes_per_device = {
    "default": LLAMA_EVAL_BATCH_SIZE_PER_DEVICE, # Default for this script
    "meta-llama/Meta-Llama-3.1-8B": LLAMA_EVAL_BATCH_SIZE_PER_DEVICE,
}

epochs_list = [0, 1, 3, 5, 10]
LEARNING_RATE_ADAFACTOR = 5e-5 # For Adafactor with Llama
LEARNING_RATE_DEFAULT = 2e-5   # Fallback (not primary for Llama with Adafactor)

accuracy_metric_evaluator = evaluate.load("accuracy")

# --- Dataset Loading and Initial Preparation ---
print(f"Loading dataset: {DATASET_ID}")
raw_dataset_full = load_dataset(DATASET_ID, cache_dir=os.environ["HF_DATASETS_CACHE"])

if 'complete' not in raw_dataset_full:
    raise ValueError(f"Dataset {DATASET_ID} does not have a 'complete' split.")
complete_dataset_view = raw_dataset_full['complete']

label_feature = complete_dataset_view.features['label']
if hasattr(label_feature, 'num_classes'):
    NUM_LABELS_FROM_DATASET = label_feature.num_classes
    label_names_temp = label_feature.names if hasattr(label_feature, 'names') else [str(i) for i in range(NUM_LABELS_FROM_DATASET)]
    print(f"Dynamically determined number of labels (from num_classes): {NUM_LABELS_FROM_DATASET} (Labels: {label_names_temp})")
elif hasattr(label_feature, 'names'):
    NUM_LABELS_FROM_DATASET = len(label_feature.names)
    print(f"Dynamically determined number of labels (from names): {NUM_LABELS_FROM_DATASET} (Labels: {label_feature.names})")
else:
    unique_labels = sorted(list(set(complete_dataset_view.unique("label")['label'])))
    if unique_labels and all(isinstance(l, int) for l in unique_labels):
        NUM_LABELS_FROM_DATASET = len(unique_labels)
        print(f"Dynamically determined number of labels (from unique values): {NUM_LABELS_FROM_DATASET} (Labels: {unique_labels})")
    else:
        raise ValueError("Label names or distinct integer labels not found/inconsistent in the dataset's 'label' feature.")

print(f"Filtering 'complete' split for labels 0 to {NUM_LABELS_FROM_DATASET-1}...")
complete_filtered = complete_dataset_view.filter(
    lambda example: 0 <= example['label'] < NUM_LABELS_FROM_DATASET,
    num_proc=min(4, os.cpu_count() if os.cpu_count() else 1)
)
print(f"Number of examples after filtering: {len(complete_filtered)}")
if len(complete_filtered) == 0: raise ValueError("Dataset is empty after filtering.")
complete_filtered = complete_filtered.shuffle(seed=RANDOM_SEED)

train_temp_split = complete_filtered.train_test_split(test_size=0.2, seed=RANDOM_SEED)
dataset_for_finetuning = train_temp_split['test']
dataset_for_evaluation = train_temp_split['train']

print("Dataset splits assigned:")
print(f"  For Fine-tuning (20%): {len(dataset_for_finetuning)} examples")
print(f"  For Evaluation (80%): {len(dataset_for_evaluation)} examples")

# --- Tokenization Function ---
def baseline_tokenize_for_batch(examples_batch, tokenizer_to_use, max_len_param):
    list_of_texts = examples_batch.get("text", [])
    if not list_of_texts and "text" not in examples_batch: # Should be "text" for AG News
        try:
            any_other_column_key = next(iter(examples_batch))
            batch_size = len(examples_batch[any_other_column_key])
            list_of_texts = [""] * batch_size
            print(f"Warning: 'text' key not found; creating {batch_size} empty strings.")
        except StopIteration:
            print("Warning: Empty batch passed to tokenization.")
            list_of_texts = []
    processed_text_list = [str(text_item) if text_item is not None else "" for text_item in list_of_texts]
    return tokenizer_to_use(processed_text_list, padding="max_length", truncation=True, max_length=max_len_param)

# --- Main Execution Loop ---
print(f"\n--- Starting {TASK_NAME_FOR_RESULTS_FILE.upper()} Experiment ---")

potential_columns_to_remove = ["text", "html", "title", "content", "description", "id"]
num_proc_tokenizer = min(4, os.cpu_count() if os.cpu_count() else 1)

os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved in: {RESULTS_OUTPUT_DIR}")

for model_checkpoint_id in models_to_run_from_mapping:
    if model_checkpoint_id not in model_name_mapping:
        print(f"Warning: Model '{model_checkpoint_id}' not in model_name_mapping. Using fallback for short name.")
    if not (model_checkpoint_id in model_train_batch_sizes_per_device or "default" in model_train_batch_sizes_per_device):
        print(f"Warning: Model {model_checkpoint_id} has no train batch size. Skipping.")
        continue
    if not (model_checkpoint_id in model_eval_batch_sizes_per_device or "default" in model_eval_batch_sizes_per_device):
        print(f"Warning: Model {model_checkpoint_id} has no eval batch size. Skipping.")
        continue

    print(f"\n===== Processing Model ID: {model_checkpoint_id} =====")

    model_snapshot_path = None
    tokenizer_instance = None
    tokenized_data = {}
    added_new_pad_token = False

    try:
        print(f"Downloading/verifying model snapshot for {model_checkpoint_id}...")
        model_snapshot_path = snapshot_download(
            repo_id=model_checkpoint_id, cache_dir=os.environ["TRANSFORMERS_CACHE"],
            resume_download=True, local_files_only=False,
        )
        print(f"Model snapshot for {model_checkpoint_id} is at: {model_snapshot_path}")

        # For Llama, is_qwen_model will be False
        is_qwen_model = "qwen" in model_checkpoint_id.lower() # Should be False here
        tokenizer_instance = AutoTokenizer.from_pretrained(
            model_snapshot_path,
            use_fast=True, # Llama tokenizers are typically fast
            trust_remote_code=False # Llama 3.1 doesn't need remote code for tokenizer
        )
        print(f"Tokenizer loaded for: {model_checkpoint_id} (Fast: True, TrustRemoteCode: False)")

        if tokenizer_instance.pad_token is None:
            if tokenizer_instance.eos_token is not None:
                tokenizer_instance.pad_token = tokenizer_instance.eos_token
                if tokenizer_instance.pad_token_id is None:
                    tokenizer_instance.pad_token_id = tokenizer_instance.eos_token_id
                print(f"Tokenizer: pad_token set to eos_token ('{tokenizer_instance.eos_token}'). ID: {tokenizer_instance.pad_token_id}")
            else:
                new_pad_token_str = '[PAD]'
                tokenizer_instance.add_special_tokens({'pad_token': new_pad_token_str})
                added_new_pad_token = True
                print(f"Tokenizer: Added new '{new_pad_token_str}' token. ID: {tokenizer_instance.pad_token_id}")

        if tokenizer_instance.pad_token is not None and tokenizer_instance.pad_token_id is None:
            tokenizer_instance.pad_token_id = tokenizer_instance.convert_tokens_to_ids(tokenizer_instance.pad_token)
            print(f"Tokenizer: pad_token_id explicitly set for '{tokenizer_instance.pad_token}': {tokenizer_instance.pad_token_id}")

        print(f"Final Tokenizer for {model_checkpoint_id}: Pad Token: '{tokenizer_instance.pad_token}', ID: {tokenizer_instance.pad_token_id}, Side: '{tokenizer_instance.padding_side}'")

        print(f"Tokenizing datasets using num_proc={num_proc_tokenizer}...")
        actual_cols_to_remove_finetune = [col for col in potential_columns_to_remove if col in dataset_for_finetuning.column_names]
        tokenized_data['data_for_finetune'] = dataset_for_finetuning.map(
            lambda exs: baseline_tokenize_for_batch(exs, tokenizer_instance, MAX_LENGTH), batched=True,
            num_proc=num_proc_tokenizer, remove_columns=actual_cols_to_remove_finetune
        ).rename_column("label", "labels")

        actual_cols_to_remove_eval = [col for col in potential_columns_to_remove if col in dataset_for_evaluation.column_names]
        tokenized_data['data_for_eval'] = dataset_for_evaluation.map(
            lambda exs: baseline_tokenize_for_batch(exs, tokenizer_instance, MAX_LENGTH), batched=True,
            num_proc=num_proc_tokenizer, remove_columns=actual_cols_to_remove_eval
        ).rename_column("label", "labels")
        print("Tokenization complete.")

    except Exception as e:
        print(f"Error during setup for model {model_checkpoint_id}: {e}")
        import traceback; traceback.print_exc()
        if 'tokenizer_instance' in locals() and tokenizer_instance is not None: del tokenizer_instance
        continue

    per_device_train_batch_size = model_train_batch_sizes_per_device.get(model_checkpoint_id, model_train_batch_sizes_per_device["default"])
    per_device_eval_batch_size = model_eval_batch_sizes_per_device.get(model_checkpoint_id, model_eval_batch_sizes_per_device["default"])
    # is_llm_family will be true for Llama
    is_llm_family = "llama" in model_checkpoint_id.lower() or "qwen" in model_checkpoint_id.lower() or "t5" in model_checkpoint_id.lower()

    for num_train_epochs_current in epochs_list:
        print(f"\n--- Model: {model_checkpoint_id}, Fine-tuning Epochs: {num_train_epochs_current} ---")

        model_instance = None
        trainer = None
        prediction_trainer = None

        training_start_time = time.time()

        short_model_name_temp = model_name_mapping.get(model_checkpoint_id, model_checkpoint_id.replace('/', '_'))
        output_dir_current_run = f"./trainer_output_temp/{short_model_name_temp}_epochs_{num_train_epochs_current}"
        predict_output_dir_epoch0_specific = f"./predict_output_temp_epoch0/{short_model_name_temp}"

        try:
            # For Llama, is_qwen_model will be False
            is_qwen_model_check_local = "qwen" in model_checkpoint_id.lower() # Redundant check, will be false
            config = AutoConfig.from_pretrained(model_snapshot_path, num_labels=NUM_LABELS_FROM_DATASET,
                                                trust_remote_code=False) # Llama 3.1 doesn't need remote code for config

            if tokenizer_instance.pad_token_id is not None and config.pad_token_id != tokenizer_instance.pad_token_id:
                config.pad_token_id = tokenizer_instance.pad_token_id
                print(f"Model config for {model_checkpoint_id} updated with pad_token_id: {tokenizer_instance.pad_token_id}")
            elif tokenizer_instance.pad_token_id is None:
                print(f"CRITICAL WARNING: tokenizer.pad_token_id is None for {model_checkpoint_id} before model config sync.")

            model_load_kwargs = {"config": config,
                                 "trust_remote_code": False} # Llama 3.1 no remote code for model
            if is_llm_family: # True for Llama
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): model_load_kwargs["torch_dtype"] = torch.bfloat16; print(f"Loading {model_checkpoint_id} with torch.bfloat16")
                elif torch.cuda.is_available(): model_load_kwargs["torch_dtype"] = torch.float16; print(f"Loading {model_checkpoint_id} with torch.float16")

            model_instance = AutoModelForSequenceClassification.from_pretrained(model_snapshot_path, **model_load_kwargs)

            if added_new_pad_token and hasattr(model_instance, 'resize_token_embeddings'):
                if len(tokenizer_instance) > model_instance.config.vocab_size:
                    print(f"Resizing model token embeddings for {model_checkpoint_id} from {model_instance.config.vocab_size} to {len(tokenizer_instance)}.")
                    model_instance.resize_token_embeddings(len(tokenizer_instance))
                    if model_instance.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
                        model_instance.config.pad_token_id = tokenizer_instance.pad_token_id
                        print(f"Model config pad_token_id re-affirmed to {tokenizer_instance.pad_token_id} after resize.")

            if model_instance.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
                model_instance.config.pad_token_id = tokenizer_instance.pad_token_id
                print(f"Final model.config.pad_token_id sync for {model_checkpoint_id} to: {tokenizer_instance.pad_token_id}")

            if is_llm_family and hasattr(model_instance, 'gradient_checkpointing_enable'): # True for Llama
                print(f"Explicitly enabling gradient checkpointing for {model_checkpoint_id}")
                gc_enable_kwargs = {}
                if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
                    try:
                        sig = inspect.signature(model_instance.gradient_checkpointing_enable)
                        if 'gradient_checkpointing_kwargs' in sig.parameters: gc_enable_kwargs['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
                    except: pass
                model_instance.gradient_checkpointing_enable(**gc_enable_kwargs)
                if hasattr(model_instance.config, "use_cache") and model_instance.config.use_cache:
                    model_instance.config.use_cache = False; print("Set model.config.use_cache = False for gradient checkpointing.")

            optimizer_for_trainer_tuple = (None, None) # Will be replaced by Adafactor

            if num_train_epochs_current > 0:
                use_gradient_checkpointing_arg_trainer = is_llm_family # True
                use_bf16_training_arg, use_fp16_training_arg = False, False
                gradient_accumulation_steps_arg = 1

                # Adafactor for Llama
                current_learning_rate = LEARNING_RATE_ADAFACTOR
                print(f"Using Adafactor optimizer for {model_checkpoint_id} with LR: {current_learning_rate}")
                adafactor_optimizer = Adafactor(model_instance.parameters(), lr=current_learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
                optimizer_for_trainer_tuple = (adafactor_optimizer, None)
                lr_scheduler_type_arg = "constant_with_warmup"

                if is_llm_family: # True
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): use_bf16_training_arg = True
                    elif torch.cuda.is_available(): use_fp16_training_arg = True

                    if per_device_train_batch_size > 0:
                        gradient_accumulation_steps_arg = max(1, LLM_TARGET_EFFECTIVE_BATCH_SIZE // per_device_train_batch_size)
                    # If per_device_train_batch_size is 32 and LLM_TARGET_EFFECTIVE_BATCH_SIZE is 32, grad_accum will be 1
                    print(f"LLM Family ({model_checkpoint_id}): Physical BS: {per_device_train_batch_size}, Grad Accum: {gradient_accumulation_steps_arg}, Effective BS: {per_device_train_batch_size * gradient_accumulation_steps_arg}")

                training_args_dict = {
                    "output_dir": output_dir_current_run, "num_train_epochs": num_train_epochs_current,
                    "per_device_train_batch_size": per_device_train_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps_arg,
                    "save_strategy": "no", "logging_dir": f"{output_dir_current_run}/logs",
                    "logging_steps": max(1, len(tokenized_data['data_for_finetune']) // (per_device_train_batch_size * gradient_accumulation_steps_arg * (torch.cuda.device_count() if torch.cuda.is_available() else 1) * 5) + 1),
                    "report_to": "none", "learning_rate": current_learning_rate,
                    "weight_decay": 0.0, # No weight decay with Adafactor
                    "warmup_ratio": 0.05, "lr_scheduler_type": lr_scheduler_type_arg,
                    "gradient_checkpointing": use_gradient_checkpointing_arg_trainer,
                    "bf16": use_bf16_training_arg, "fp16": use_fp16_training_arg,
                    "dataloader_num_workers": min(2, os.cpu_count() if os.cpu_count() else 1), "eval_strategy": "no",
                }
                if use_gradient_checkpointing_arg_trainer and packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
                    if hasattr(torch.utils.checkpoint, 'checkpoint') and 'use_reentrant' in inspect.signature(torch.utils.checkpoint.checkpoint).parameters:
                        training_args_dict["gradient_checkpointing_kwargs"] = {'use_reentrant': False}

                training_args = TrainingArguments(**training_args_dict)
                trainer = Trainer(model=model_instance, args=training_args, train_dataset=tokenized_data['data_for_finetune'],
                                  tokenizer=tokenizer_instance, optimizers=optimizer_for_trainer_tuple)
                print(f"Starting fine-tuning for {num_train_epochs_current} epochs...")
                trainer.train()
                print("Fine-tuning complete.")
            else:
                print("Skipping fine-tuning (0 epochs specified).")

            fine_tuning_duration = time.time() - training_start_time

            print(f"Predicting on 'data_for_eval' dataset...")
            predict_args_output_dir_current = predict_output_dir_epoch0_specific if num_train_epochs_current == 0 else f"{output_dir_current_run}_prediction_temp"

            predict_bf16_flag, predict_fp16_flag = False, False
            if trainer and hasattr(trainer.args, 'bf16'):
                predict_bf16_flag = trainer.args.bf16; predict_fp16_flag = trainer.args.fp16
            elif is_llm_family: # True for Llama
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported(): predict_bf16_flag = True
                elif torch.cuda.is_available(): predict_fp16_flag = True

            predict_args = TrainingArguments(output_dir=predict_args_output_dir_current,
                                               per_device_eval_batch_size=per_device_eval_batch_size,
                                               report_to="none", bf16=predict_bf16_flag, fp16=predict_fp16_flag,
                                               dataloader_num_workers=min(2, os.cpu_count() if os.cpu_count() else 1))
            prediction_trainer = Trainer(model=model_instance, args=predict_args, tokenizer=tokenizer_instance)

            _ = prediction_trainer.model.to(prediction_trainer.args.device)
            predictions_output = prediction_trainer.predict(test_dataset=tokenized_data['data_for_eval'])
            logits_output, labels_output = predictions_output.predictions, predictions_output.label_ids

            if logits_output is None or labels_output is None: raise ValueError("Predictions output or labels are None.")

            current_logits_to_process = logits_output
            if isinstance(logits_output, tuple):
                print(f"INFO: For model '{model_checkpoint_id}', predictions is a tuple. Length: {len(logits_output)}")
                if len(logits_output) > 0 and isinstance(logits_output[0], np.ndarray):
                    current_logits_to_process = logits_output[0]
                    print(f"  Using first element (shape: {current_logits_to_process.shape}) as logits.")
                else: raise TypeError(f"Model '{model_checkpoint_id}' output tuple error: first element not np.ndarray.")
            elif not isinstance(logits_output, np.ndarray):
                raise TypeError(f"Logits for '{model_checkpoint_id}' not np.ndarray or handled tuple. Type: {type(logits_output)}")

            try:
                softmax_logits = torch.nn.functional.softmax(torch.from_numpy(current_logits_to_process), dim=-1).numpy()
            except TypeError as e:
                print(f"ERROR converting logits for '{model_checkpoint_id}'. Type: {type(current_logits_to_process)}, Shape: {getattr(current_logits_to_process, 'shape', 'N/A')}")
                raise e

            preds_numerical = np.argmax(softmax_logits, axis=1)
            responses_correctness = np.equal(preds_numerical, labels_output).astype(int)
            accuracy_on_eval = responses_correctness.mean() if responses_correctness.size > 0 else 0.0

            print(f"\nAccuracy on 'data_for_eval': {accuracy_on_eval:.4f}")
            print(f"Fine-tuning time (if any): {fine_tuning_duration:.2f} seconds")

            filename = os.path.join(RESULTS_OUTPUT_DIR,
                                    f"{short_model_name_temp}_{TASK_NAME_FOR_RESULTS_FILE}_{fine_tuning_duration:.2f}_Acc_{accuracy_on_eval:.4f}_epochs_{num_train_epochs_current}.json")
            data_to_save = {"logits": softmax_logits.tolist(), "responses": responses_correctness.tolist()}
            with open(filename, "w") as f: json.dump(data_to_save, f)
            print(f"Saved prediction results to {filename}")

        except Exception as e_epoch:
            print(f"Error during epoch run {num_train_epochs_current} for model {model_checkpoint_id}: {e_epoch}")
            import traceback; traceback.print_exc()
        finally:
            del model_instance
            del trainer
            del prediction_trainer
            model_instance, trainer, prediction_trainer = None, None, None

            if torch.cuda.is_available(): torch.cuda.empty_cache()

            if os.path.exists(output_dir_current_run):
                try: shutil.rmtree(output_dir_current_run); print(f"Cleaned up: {output_dir_current_run}")
                except Exception as e_clean: print(f"Error cleaning {output_dir_current_run}: {e_clean}")

            predict_args_output_dir_to_clean = predict_output_dir_epoch0_specific if num_train_epochs_current == 0 else f"{output_dir_current_run}_prediction_temp"
            if os.path.exists(predict_args_output_dir_to_clean):
                 try: shutil.rmtree(predict_args_output_dir_to_clean); print(f"Cleaned up: {predict_args_output_dir_to_clean}")
                 except Exception as e_clean: print(f"Error cleaning {predict_args_output_dir_to_clean}: {e_clean}")

    if 'tokenizer_instance' in locals() and tokenizer_instance is not None:
        del tokenizer_instance; tokenizer_instance = None
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print(f"\n===== {TASK_NAME_FOR_RESULTS_FILE.upper()} Experiment (Llama Models) Completed =====")
