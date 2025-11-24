import os
import datetime
import random
import numpy as np
import torch
import json
from tqdm import tqdm
import shutil
import traceback
import sys
import time
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,  # Key change for these models
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
    AutoConfig,
)
from huggingface_hub import whoami  # From your AG News script

# ----- Environment Setup -----
HF_HOME_SPECIFIED = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(HF_HOME_SPECIFIED):
    HF_HOME = HF_HOME_SPECIFIED
else:
    print(f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' does not exist. Using default.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ----- Hugging Face Token Check -----
if "HF_TOKEN" not in os.environ:
    print("HF_TOKEN environment variable not set. Some private models may not be accessible.")
try:
    user_info = whoami()
    print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
except Exception as e:
    print(f"Hugging Face login check error: {e}. Ensure CLI login or set HF_TOKEN for private models.")

# ----- Reproducibility -----
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
print(f"Using random seed: {random_seed}")

# ----- Configuration -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

models_to_run = [
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "distilbert-base-uncased",
    "xlnet-base-cased",
    "google/electra-base-discriminator",
    "facebook/bart-base",
]

model_name_mapping = {
    "bert-base-uncased": "bert_base_uncased",
    "roberta-base": "roberta_base",
    "albert-base-v2": "albert_base_v2",
    "distilbert-base-uncased": "distilbert_base_uncased",
    "xlnet-base-cased": "xlnet_base_cased",
    "google/electra-base-discriminator": "electra_base_discriminator",
    "facebook/bart-base": "bart_base",
}

DATASET_ID_MC = "GBaker/MedQA-USMLE-4-options"
TASK_NAME_FOR_RESULTS_FILE_MC = "MedQA_MC_Encoders"  # Task name for this script run

MAX_LENGTH_MC = 512
# Batch sizes (can be fine-tuned per model if needed)
# Using general settings from your AG News script as a base, may need adjustment for MC
model_train_batch_sizes_mc = {
    "default": 8,  # Encoder models for MC can take more memory than seq class
    "xlnet-base-cased": 4,  # XLNet can be larger
    "facebook/bart-base": 4,  # BART can also be larger
}
model_eval_batch_sizes_mc_custom = {
    "default": 16,
}
GENERAL_TARGET_EFFECTIVE_BATCH_SIZE_MC = 32

num_epochs_total_mc = 10
epochs_to_evaluate_mc = [0, 1, 3, 5, 10]
LEARNING_RATE_MC = 2e-5

ANSWER_MAP_MC = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
OPTION_KEYS_MC = ['A', 'B', 'C', 'D']
NUM_CHOICES_MC = len(OPTION_KEYS_MC)

# --- Output Directories ---
current_datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
base_output_dir_script = f"./MedQA_MC_Encoders_run_{current_datetime_str}"  # Main dir for this script's runs
# RESULTS_OUTPUT_DIR will be inside model_base_output_dir to group by model
# training_checkpoints_dir will also be inside model_base_output_dir

os.makedirs(base_output_dir_script, exist_ok=True)
print(f"Base directory for this script run: {base_output_dir_script}")

# --- Load MedQA Dataset ---
print(f"Loading MedQA dataset: {DATASET_ID_MC}")
try:
    raw_dataset_medqa_full = load_dataset(DATASET_ID_MC, cache_dir=os.environ["HF_DATASETS_CACHE"])
except Exception as e:
    print(f"Error loading MedQA dataset: {e}");
    traceback.print_exc();
    sys.exit(1)


def filter_medqa(example):
    return example["answer_idx"] is not None and example["answer_idx"].strip().upper() in ANSWER_MAP_MC


raw_dataset_medqa_full = raw_dataset_medqa_full.filter(filter_medqa)

if 'train' not in raw_dataset_medqa_full or 'test' not in raw_dataset_medqa_full or \
        len(raw_dataset_medqa_full["train"]) == 0 or len(raw_dataset_medqa_full["test"]) == 0:
    print(f"Error: MedQA Dataset {DATASET_ID_MC} must contain non-empty 'train' and 'test' splits after filtering.")
    sys.exit(1)

medqa_dataset_for_training_raw = raw_dataset_medqa_full["test"]
medqa_dataset_for_evaluation_raw = raw_dataset_medqa_full["train"]
print(f"Using {len(medqa_dataset_for_training_raw)} MedQA 'test' examples for fine-tuning.")
print(f"Using {len(medqa_dataset_for_evaluation_raw)} MedQA 'train' examples for evaluation.")


# --- Preprocessing Function for Encoder-style Multiple Choice ---
def preprocess_encoder_mc_style(examples, tokenizer_instance, max_len):
    num_examples_in_batch = len(examples["question"])
    first_sentences = [[examples["question"][i]] * NUM_CHOICES_MC for i in range(num_examples_in_batch)]
    second_sentences = []
    labels = []
    for i in range(num_examples_in_batch):
        options_dict = examples["options"][i]
        answer_idx_key = examples["answer_idx"][i].strip().upper()
        current_option_texts = [options_dict.get(key, "[Option unavailable]") for key in OPTION_KEYS_MC] \
            if isinstance(options_dict, dict) else ["[Invalid options format]"] * NUM_CHOICES_MC
        second_sentences.append(current_option_texts)
        labels.append(ANSWER_MAP_MC.get(answer_idx_key, -100))

    first_sentences_flat = [s for sublist in first_sentences for s in sublist]
    second_sentences_flat = [s for sublist in second_sentences for s in sublist]

    tokenized_output = tokenizer_instance(
        first_sentences_flat,
        second_sentences_flat,
        max_length=max_len,
        truncation=True,
        padding=False
    )
    unflattened = {k: [v[i:i + NUM_CHOICES_MC] for i in range(0, len(v), NUM_CHOICES_MC)] for k, v in
                   tokenized_output.items()}
    unflattened["labels"] = labels
    return unflattened


# --- Evaluation Function for Encoder-style Multiple Choice models ---
def evaluate_encoder_mc_model(model_to_eval, current_tokenizer, eval_dataset_raw, current_data_collator,
                              epoch, fine_tuning_duration, model_checkpoint_name_str, task_name_str,
                              model_specific_results_dir):
    print(f"\nCustom evaluation for {model_checkpoint_name_str}, Epoch {epoch}...")
    model_to_eval.eval()
    model_to_eval.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    num_proc_tok_eval = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    tokenized_eval_data = eval_dataset_raw.map(
        lambda exs: preprocess_encoder_mc_style(exs, current_tokenizer, MAX_LENGTH_MC),
        batched=True, num_proc=num_proc_tok_eval, remove_columns=eval_dataset_raw.column_names,  # Remove original cols
        desc=f"Tokenizing eval data for {model_checkpoint_name_str} epoch {epoch}"
    )

    all_logits_probs = []
    all_responses_correctness = []
    all_true_labels_int = []

    batch_size_eval = model_eval_batch_sizes_mc_custom.get(model_checkpoint_name_str,
                                                           model_eval_batch_sizes_mc_custom["default"])

    for i in tqdm(range(0, len(tokenized_eval_data), batch_size_eval),
                  desc=f"Predicting {model_checkpoint_name_str} Epoch {epoch}"):
        batch_examples = [tokenized_eval_data[j] for j in range(i, min(i + batch_size_eval, len(tokenized_eval_data)))]
        collated_batch = current_data_collator(batch_examples)

        input_ids = collated_batch["input_ids"].to(model_to_eval.device)
        attention_mask = collated_batch["attention_mask"].to(model_to_eval.device)
        token_type_ids = collated_batch.get("token_type_ids")  # Not all models use this
        if token_type_ids is not None: token_type_ids = token_type_ids.to(model_to_eval.device)

        batch_true_labels_int = collated_batch["labels"].numpy()
        all_true_labels_int.extend(batch_true_labels_int)

        with torch.no_grad():
            outputs = model_to_eval(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids  # Pass if exists, else None
            )
            batch_logits = outputs.logits

        batch_probs = torch.softmax(batch_logits, dim=-1).cpu().numpy()
        all_logits_probs.extend(batch_probs.tolist())

        batch_pred_indices = np.argmax(batch_logits.cpu().numpy(), axis=1)
        for pred_idx, true_idx in zip(batch_pred_indices, batch_true_labels_int):
            if true_idx != -100:
                all_responses_correctness.append(int(pred_idx == true_idx))

    valid_indices_for_accuracy = [idx for idx, label in enumerate(all_true_labels_int) if label != -100]
    if not valid_indices_for_accuracy:  # No valid labels to compute accuracy
        accuracy = 0.0
    else:
        # Ensure all_responses_correctness only contains scores for valid labels
        # The current logic already does this by checking true_idx != -100 before appending
        accuracy = np.mean(all_responses_correctness) if all_responses_correctness else 0.0

    model_name_part = model_name_mapping.get(model_checkpoint_name_str, model_checkpoint_name_str.replace('/', '_'))
    output_filename = f"{model_name_part}_{task_name_str}_{fine_tuning_duration:.2f}_Acc_{accuracy:.4f}_epochs_{epoch}.json"
    output_filepath = os.path.join(model_specific_results_dir, output_filename)  # Save in model-specific results dir
    data_to_save = {"logits": all_logits_probs, "responses": all_responses_correctness}
    with open(output_filepath, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved Encoder-MC results for epoch {epoch} to {output_filepath} (Acc: {accuracy:.4f})")
    return accuracy


# --- Main Execution Loop ---
print(f"\n--- Starting {TASK_NAME_FOR_RESULTS_FILE_MC.upper()} Experiment ---")

num_proc_tokenizer_map = min(4, os.cpu_count() if os.cpu_count() else 1)

for model_checkpoint_id in models_to_run:
    model_short_name = model_name_mapping.get(model_checkpoint_id, model_checkpoint_id.replace('/', '_'))
    print(f"\n===== Processing Model: {model_checkpoint_id} (Short Name: {model_short_name}) =====")

    # Model-specific output directories within the main script run directory
    model_base_output_dir = os.path.join(base_output_dir_script, model_short_name)
    model_results_dir = os.path.join(model_base_output_dir, "MedQA_Results_JSON")
    model_checkpoints_dir = os.path.join(model_base_output_dir, "Training_Checkpoints")
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(model_checkpoints_dir, exist_ok=True)

    tokenizer_instance = None
    added_new_pad_token_flag = False
    total_training_time_for_run = 0.0  # Initialize for each model

    try:
        print(f"Loading tokenizer for {model_checkpoint_id}...")
        use_fast_tokenizer = True
        # Some models prefer/require the slow tokenizer
        if "xlnet" in model_checkpoint_id.lower() or "deberta-v2" in model_checkpoint_id.lower() or "deberta-v3" in model_checkpoint_id.lower():
            use_fast_tokenizer = False

        tokenizer_instance = AutoTokenizer.from_pretrained(model_checkpoint_id, use_fast=use_fast_tokenizer,
                                                           cache_dir=os.environ["TRANSFORMERS_CACHE"])
        print(f"Tokenizer for {model_checkpoint_id} loaded (Fast: {use_fast_tokenizer}).")

        # Pad token handling (from AG News script)
        if tokenizer_instance.pad_token is None:
            if tokenizer_instance.eos_token is not None:
                tokenizer_instance.pad_token = tokenizer_instance.eos_token
                print(f"Set pad_token to eos_token ('{tokenizer_instance.eos_token}') for {model_checkpoint_id}")
            else:
                new_pad_token_val = "[PAD]"
                # Check if [PAD] already exists as a special token but not set as pad_token
                if new_pad_token_val in tokenizer_instance.vocab and \
                        tokenizer_instance.convert_tokens_to_ids(
                            new_pad_token_val) == tokenizer_instance.unk_token_id:  # Avoid using UNK as PAD
                    new_pad_token_val = "<|pad|>"  # Use a more unique pad token

                if new_pad_token_val not in tokenizer_instance.get_vocab():  # get_vocab() might be slow, use direct check if possible
                    tokenizer_instance.add_special_tokens({'pad_token': new_pad_token_val})
                    added_new_pad_token_flag = True
                    print(f"Added new pad_token '{new_pad_token_val}' for {model_checkpoint_id}")
                else:  # Token exists, just set it
                    tokenizer_instance.pad_token = new_pad_token_val
                    print(f"Set existing token '{new_pad_token_val}' as pad_token for {model_checkpoint_id}")

        if tokenizer_instance.pad_token is not None and tokenizer_instance.pad_token_id is None:
            tokenizer_instance.pad_token_id = tokenizer_instance.convert_tokens_to_ids(tokenizer_instance.pad_token)
        elif tokenizer_instance.pad_token_id is not None and tokenizer_instance.pad_token is None:  # Ensure consistency
            tokenizer_instance.pad_token = tokenizer_instance.convert_ids_to_tokens(tokenizer_instance.pad_token_id)

        print(
            f"Final Tokenizer Check: Pad Token: '{tokenizer_instance.pad_token}', ID: {tokenizer_instance.pad_token_id}, Side: {tokenizer_instance.padding_side}")

        # Data Collator for Multiple Choice (used in eval and training)
        data_collator_mc = DataCollatorForMultipleChoice(tokenizer=tokenizer_instance, padding="longest")

        # --- EPOCH 0 EVALUATION ---
        print(f"\n--- Epoch 0 Evaluation for {model_checkpoint_id} ---")
        eval_epoch0_start_time = time.time()

        config_epoch0 = AutoConfig.from_pretrained(model_checkpoint_id, num_labels=NUM_CHOICES_MC,
                                                   cache_dir=os.environ["TRANSFORMERS_CACHE"])
        if tokenizer_instance.pad_token_id is not None and config_epoch0.pad_token_id != tokenizer_instance.pad_token_id:
            config_epoch0.pad_token_id = tokenizer_instance.pad_token_id
            print(f"Updated Epoch 0 model config pad_token_id to {tokenizer_instance.pad_token_id}")

        model_epoch0 = AutoModelForMultipleChoice.from_pretrained(
            model_checkpoint_id,
            config=config_epoch0,
            cache_dir=os.environ["TRANSFORMERS_CACHE"]
        )
        if added_new_pad_token_flag and hasattr(model_epoch0, 'resize_token_embeddings'):
            if len(tokenizer_instance) > model_epoch0.config.vocab_size:
                print(
                    f"Resizing Epoch 0 model embeddings: {model_epoch0.config.vocab_size} -> {len(tokenizer_instance)}")
                model_epoch0.resize_token_embeddings(len(tokenizer_instance))
        if model_epoch0.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
            model_epoch0.config.pad_token_id = tokenizer_instance.pad_token_id

        evaluate_encoder_mc_model(model_epoch0, tokenizer_instance, medqa_dataset_for_evaluation_raw, data_collator_mc,
                                  0, 0.0,
                                  model_checkpoint_id, TASK_NAME_FOR_RESULTS_FILE_MC, model_results_dir)
        del model_epoch0
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"Epoch 0 evaluation took {time.time() - eval_epoch0_start_time:.2f}s")

        # --- TOKENIZE MEDQA "TEST" SPLIT FOR TRAINING ---
        print(f"\nTokenizing MedQA 'test' split for SFT with {model_checkpoint_id}...")
        tokenized_medqa_for_sft = medqa_dataset_for_training_raw.map(
            lambda exs: preprocess_encoder_mc_style(exs, tokenizer_instance, MAX_LENGTH_MC),
            batched=True, num_proc=num_proc_tokenizer_map, remove_columns=medqa_dataset_for_training_raw.column_names,
            desc=f"Tokenizing MedQA for {model_short_name} SFT"
        )

        # --- TRAINING (10 Epochs) ---
        print(f"\n--- Training {model_checkpoint_id} for {num_epochs_total_mc} epochs ---")
        config_train = AutoConfig.from_pretrained(model_checkpoint_id, num_labels=NUM_CHOICES_MC,
                                                  cache_dir=os.environ["TRANSFORMERS_CACHE"])
        if tokenizer_instance.pad_token_id is not None and config_train.pad_token_id != tokenizer_instance.pad_token_id:
            config_train.pad_token_id = tokenizer_instance.pad_token_id
            print(f"Updated training model config pad_token_id to {tokenizer_instance.pad_token_id}")

        model_for_training = AutoModelForMultipleChoice.from_pretrained(
            model_checkpoint_id,
            config=config_train,
            cache_dir=os.environ["TRANSFORMERS_CACHE"]
        )
        if added_new_pad_token_flag and hasattr(model_for_training, 'resize_token_embeddings'):
            if len(tokenizer_instance) > model_for_training.config.vocab_size:
                print(
                    f"Resizing Training model embeddings: {model_for_training.config.vocab_size} -> {len(tokenizer_instance)}")
                model_for_training.resize_token_embeddings(len(tokenizer_instance))
        if model_for_training.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
            model_for_training.config.pad_token_id = tokenizer_instance.pad_token_id

        current_train_bs = model_train_batch_sizes_mc.get(model_checkpoint_id, model_train_batch_sizes_mc["default"])
        grad_accum = max(1, GENERAL_TARGET_EFFECTIVE_BATCH_SIZE_MC // current_train_bs) if current_train_bs > 0 else 1

        training_args = TrainingArguments(
            output_dir=model_checkpoints_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs_total_mc,
            per_device_train_batch_size=current_train_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=LEARNING_RATE_MC,
            weight_decay=0.01,
            save_strategy="epoch",
            eval_strategy="no",
            logging_strategy="steps",
            logging_steps=max(1, len(tokenized_medqa_for_sft) // (current_train_bs * grad_accum * (
                torch.cuda.device_count() if torch.cuda.is_available() else 1) * 10) + 1),  # Log ~10 times per epoch
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=True,  # Handled by .map remove_columns but good safety.
            seed=random_seed,
            save_total_limit=None,  # Keep all for eval
            dataloader_num_workers=min(2, os.cpu_count() if os.cpu_count() else 1),
        )
        trainer = Trainer(
            model=model_for_training,
            args=training_args,
            train_dataset=tokenized_medqa_for_sft,
            tokenizer=tokenizer_instance,
            data_collator=data_collator_mc,
        )

        print(f"Starting actual training for {model_checkpoint_id}...")
        train_start_time = time.time()
        train_result_metrics = trainer.train()
        total_training_time_for_run = time.time() - train_start_time  # More reliable direct timing
        print(f"Training for {model_checkpoint_id} completed in {total_training_time_for_run:.2f}s.")

        del model_for_training, trainer
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- CHECKPOINT EVALUATION ---
        print(f"\n--- Checkpoint Evaluation for {model_checkpoint_id} ---")
        for desired_epoch in epochs_to_evaluate_mc:
            if desired_epoch == 0: continue

            found_ckpt, ckpt_path_to_load = False, None
            if os.path.exists(model_checkpoints_dir):
                ckpt_folders = sorted([
                    os.path.join(model_checkpoints_dir, d) for d in os.listdir(model_checkpoints_dir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_checkpoints_dir, d))
                ])
                for ckpt_candidate in ckpt_folders:
                    state_file = os.path.join(ckpt_candidate, "trainer_state.json")
                    if os.path.exists(state_file):
                        with open(state_file, "r") as f:
                            state = json.load(f)
                        if round(state.get("epoch", 0.0)) == desired_epoch:
                            ckpt_path_to_load = ckpt_candidate
                            found_ckpt = True;
                            break

            if found_ckpt and ckpt_path_to_load:
                print(f"Loading checkpoint for epoch {desired_epoch} from {ckpt_path_to_load}")
                # Config will be loaded from checkpoint; num_labels should be correct.
                # Sync pad_token_id just in case it differs from global tokenizer due to old config in ckpt
                config_ckpt = AutoConfig.from_pretrained(ckpt_path_to_load)
                if tokenizer_instance.pad_token_id is not None and hasattr(config_ckpt, "pad_token_id") and \
                        config_ckpt.pad_token_id != tokenizer_instance.pad_token_id:
                    config_ckpt.pad_token_id = tokenizer_instance.pad_token_id
                    print(f"Synced checkpoint config pad_token_id for epoch {desired_epoch}")

                model_checkpoint = AutoModelForMultipleChoice.from_pretrained(ckpt_path_to_load, config=config_ckpt)

                if model_checkpoint.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
                    model_checkpoint.config.pad_token_id = tokenizer_instance.pad_token_id

                evaluate_encoder_mc_model(model_checkpoint, tokenizer_instance, medqa_dataset_for_evaluation_raw,
                                          data_collator_mc,
                                          desired_epoch, total_training_time_for_run, model_checkpoint_id,
                                          TASK_NAME_FOR_RESULTS_FILE_MC, model_results_dir)
                del model_checkpoint
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            else:
                print(
                    f"Warning: Checkpoint for epoch {desired_epoch} for model {model_checkpoint_id} not found in {model_checkpoints_dir}")

    except Exception as e_model_loop:
        print(f"!!!!! MAJOR ERROR processing model {model_checkpoint_id}: {e_model_loop} !!!!!")
        traceback.print_exc()
    finally:
        if 'tokenizer_instance' in locals() and tokenizer_instance is not None: del tokenizer_instance
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Optional: Clean up model_checkpoints_dir after a model is fully processed if space is critical
        # print(f"To save space, consider manually cleaning: {model_checkpoints_dir}")

print(f"\n===== {TASK_NAME_FOR_RESULTS_FILE_MC.upper()} Experiment Completed =====")