import os
import datetime
import random
import numpy as np
import torch
import json
import time
from tqdm import tqdm
import shutil
import traceback
import sys

from datasets import load_dataset, DatasetDict
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    AutoConfig,
)
from evaluate import load as load_metric
from huggingface_hub import whoami

# ----- Environment Setup -----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_HOME_SPECIFIED = HF_HOME
if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(HF_HOME_SPECIFIED):
    HF_HOME = HF_HOME_SPECIFIED
else:
    print(f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' does not exist. Using default.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ----- Random Seed -----
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
print(f"Using random seed: {random_seed}")

# ----- Config -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_name = "gpt2"
dataset_id = "GBaker/MedQA-USMLE-4-options"

MAX_SFT_SEQ_LENGTH = 512
MAX_EVAL_PROMPT_LENGTH = 512

PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE_TARGET = 32
VALIDATION_SUBSET_SIZE = 200

NUM_TRAIN_EPOCHS = 20  # CHANGED BACK TO 20
LEARNING_RATE = 5e-5

current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# Updated run_name to reflect 20 epochs potentially
run_name = f"gpt2_medqa_sft_baseline_{current_time_str}_trainbs{PER_DEVICE_TRAIN_BATCH_SIZE}_evalbs{PER_DEVICE_EVAL_BATCH_SIZE}_epochs{NUM_TRAIN_EPOCHS}"
output_dir_base = f"./{run_name}_output_3"
best_model_dir = os.path.join(output_dir_base, "best_model_final")

ANSWER_MAP_KEYS = ['A', 'B', 'C', 'D']
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS)

# ----- Load Dataset -----
print(f"Loading dataset: {dataset_id}")
try:
    dataset_full = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])
except Exception as e:
    print(f"Error loading dataset: {e}");
    traceback.print_exc();
    sys.exit(1)


def filter_valid_answers(example):
    return example["answer_idx"] is not None and example["answer_idx"].strip().upper() in ANSWER_MAP_KEYS


dataset_full = dataset_full.filter(filter_valid_answers)

if 'train' not in dataset_full or 'test' not in dataset_full or \
        len(dataset_full["train"]) == 0 or len(dataset_full["test"]) == 0:
    print("Error: Dataset must contain non-empty 'train' and 'test' splits after filtering.")
    sys.exit(1)

if 'validation' not in dataset_full:
    print("Validation split not found. Splitting 'train' (80% train / 20% validation)...")
    original_test_split = dataset_full.get('test')
    train_val_split = dataset_full['train'].train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
    processed_dataset_dict_args = {
        'train': train_val_split['train'],
        'validation': train_val_split['test']
    }
    if original_test_split:
        processed_dataset_dict_args['test'] = original_test_split
    else:
        processed_dataset_dict_args['test'] = dataset_full['test']
    processed_dataset = DatasetDict(processed_dataset_dict_args)
else:
    processed_dataset = DatasetDict(dataset_full)
print("Final dataset splits:", processed_dataset)

# ----- Load Tokenizer -----
print(f"Loading {model_name} tokenizer...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
except Exception as e:
    print(f"Error loading tokenizer: {e}");
    traceback.print_exc();
    sys.exit(1)

added_pad_token = False
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    else:
        new_pad_token_val = '[PAD]'
        tokenizer.add_special_tokens({'pad_token': new_pad_token_val})
        added_pad_token = True
        print(f"Added new pad_token '{new_pad_token_val}' (ID: {tokenizer.pad_token_id})")
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
elif tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"
print(f"Tokenizer padding side: {tokenizer.padding_side}, Pad token ID: {tokenizer.pad_token_id}")


# ----- Preprocessing Functions for SFT and Evaluation -----
def create_sft_text_gpt2(example_dict):
    question_text = example_dict["question"]
    options_dict = example_dict["options"]
    answer_key = example_dict["answer_idx"].strip().upper()
    prompt_parts = [f"Question: {question_text}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option unavailable]')}")
    else:
        for key_char in ANSWER_MAP_KEYS: prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    prompt_text = "\n".join(prompt_parts)
    return prompt_text + " " + answer_key


def preprocess_sft_function_gpt2(examples_batch):
    texts_for_sft = []
    batch_size = len(examples_batch[next(iter(examples_batch))])
    for i in range(batch_size):
        single_example_dict = {key: examples_batch[key][i] for key in examples_batch.keys()}
        texts_for_sft.append(create_sft_text_gpt2(single_example_dict))
    tokenized_output = tokenizer(
        texts_for_sft, max_length=MAX_SFT_SEQ_LENGTH, padding=False, truncation=True, add_special_tokens=True
    )
    return tokenized_output


def create_gpt2_eval_prompt(example_dict):
    question_text = example_dict["question"]
    options_dict = example_dict["options"]
    prompt_parts = [f"Question: {question_text}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option unavailable]')}")
    else:
        for key_char in ANSWER_MAP_KEYS: prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


print("Tokenizing datasets for SFT...")
try:
    num_proc_map = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    tokenized_datasets = processed_dataset.map(
        preprocess_sft_function_gpt2, batched=True, num_proc=num_proc_map,
        remove_columns=processed_dataset["train"].column_names
    )
    print("Tokenization complete.")
except Exception as e:
    print(f"Error during tokenization: {e}");
    traceback.print_exc();
    exit()

# ----- Data Collator -----
print("Initializing DataCollatorForLanguageModeling...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ----- Metric for Trainer's internal eval (based on loss) -----
def compute_metrics_for_trainer(eval_pred):
    eval_loss = eval_pred.metrics.get("eval_loss", None)
    metrics_to_return = {}
    if eval_loss is not None:
        metrics_to_return["eval_loss"] = eval_loss
        try:
            perplexity = np.exp(eval_loss)
            metrics_to_return["perplexity"] = perplexity
        except OverflowError:
            metrics_to_return["perplexity"] = float('inf')
    else:
        metrics_to_return["eval_loss"] = -1.0
        metrics_to_return["perplexity"] = -1.0
    return metrics_to_return


# ----- Load Model Function -----
def load_model_for_training_or_eval(model_checkpoint_name_or_path, tokenizer_instance, new_pad_token_added_flag,
                                    is_training=True):
    purpose = "training" if is_training else "evaluation"
    print(f"Loading model {model_checkpoint_name_or_path} for {purpose}...")
    config = AutoConfig.from_pretrained(model_checkpoint_name_or_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])

    if tokenizer_instance.pad_token_id is not None:
        config.pad_token_id = tokenizer_instance.pad_token_id
    if hasattr(config, "eos_token_id") and tokenizer_instance.eos_token_id is not None:
        config.eos_token_id = tokenizer_instance.eos_token_id
    if hasattr(config, "bos_token_id") and tokenizer_instance.bos_token_id is not None:
        config.bos_token_id = tokenizer_instance.bos_token_id

    model_instance = GPT2LMHeadModel.from_pretrained(model_checkpoint_name_or_path, config=config,
                                                     cache_dir=os.environ["TRANSFORMERS_CACHE"])

    if new_pad_token_added_flag:
        if hasattr(model_instance.config, "vocab_size") and len(tokenizer_instance) > model_instance.config.vocab_size:
            print(
                f"Resizing model token embeddings for {purpose}: {model_instance.config.vocab_size} -> {len(tokenizer_instance)}")
            model_instance.resize_token_embeddings(len(tokenizer_instance))
            if model_instance.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
                model_instance.config.pad_token_id = tokenizer_instance.pad_token_id

    if model_instance.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
        model_instance.config.pad_token_id = tokenizer_instance.pad_token_id
    return model_instance


# ----- Training Arguments -----
gradient_accumulation_steps = max(1, EFFECTIVE_BATCH_SIZE_TARGET // PER_DEVICE_TRAIN_BATCH_SIZE)
os.makedirs(output_dir_base, exist_ok=True)
training_output_dir = os.path.join(output_dir_base, "training_output")

training_args = TrainingArguments(
    output_dir=training_output_dir,
    overwrite_output_dir=True, num_train_epochs=NUM_TRAIN_EPOCHS,  # Uses 20 now
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
    logging_steps=max(1, (len(tokenized_datasets["train"]) // (
                PER_DEVICE_TRAIN_BATCH_SIZE * gradient_accumulation_steps * (
            torch.cuda.device_count() if torch.cuda.is_available() else 1))) // 20 + 1),
    learning_rate=LEARNING_RATE, weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type="linear",
    save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="loss", greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none", seed=random_seed,
    dataloader_num_workers=min(2, os.cpu_count() if os.cpu_count() else 1),
    prediction_loss_only=True,
)

# --- Prepare Model and Validation Subset for Trainer ---
model = load_model_for_training_or_eval(model_name, tokenizer, added_pad_token, is_training=True)

validation_dataset_for_trainer_eval = tokenized_datasets["validation"]
if len(validation_dataset_for_trainer_eval) > VALIDATION_SUBSET_SIZE:
    print(f"Using a subset of {VALIDATION_SUBSET_SIZE} samples from validation set for Trainer's internal evaluation.")
    validation_dataset_for_trainer_eval = validation_dataset_for_trainer_eval.select(range(VALIDATION_SUBSET_SIZE))

# ----- Trainer -----
print("Initializing Trainer...")
trainer = Trainer(
    model=model, args=training_args, train_dataset=tokenized_datasets["train"],
    eval_dataset=validation_dataset_for_trainer_eval,
    tokenizer=tokenizer, data_collator=data_collator,
    compute_metrics=compute_metrics_for_trainer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3 if NUM_TRAIN_EPOCHS > 1 else 100)]  # Patience is 3
)

# ----- Train -----
print(f"Starting training (max {NUM_TRAIN_EPOCHS} epochs)...")
training_successful = False
if torch.cuda.is_available(): torch.cuda.empty_cache()
try:
    train_result = trainer.train()
    training_successful = True
    print("Training finished.")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_state()
    if training_successful:
        trainer.save_model(best_model_dir)
        print(f"Best model (based on val_loss) saved to {best_model_dir}")
except Exception as e:
    print(f"Error during training: {e}");
    traceback.print_exc()
finally:
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ----- Evaluation on Full Test Set with Custom Logic for Accuracy -----
print("\nEvaluating on the full test set with the best model (custom accuracy calculation)...")
if torch.cuda.is_available(): torch.cuda.empty_cache()

if training_successful and os.path.exists(best_model_dir):
    try:
        print(f"Loading best model from: {best_model_dir} for final test evaluation...")
        model_for_test = load_model_for_training_or_eval(best_model_dir, tokenizer, added_pad_token, is_training=False)

        print("Using custom evaluation logic for test set accuracy...")
        test_eval_start_time = time.time()
        model_for_test.eval()
        model_for_test.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        test_set_raw = processed_dataset["test"]
        test_prompts = [create_gpt2_eval_prompt(test_set_raw[i]) for i in
                        tqdm(range(len(test_set_raw)), desc="Creating Test Prompts")]
        test_true_letters = [test_set_raw[i]["answer_idx"].strip().upper() for i in range(len(test_set_raw))]

        letter_token_ids = {}
        for letter_char in ANSWER_MAP_KEYS:
            tokens = tokenizer.encode(letter_char, add_special_tokens=False)
            if len(tokens) == 1:
                letter_token_ids[letter_char] = tokens[0]
            else:
                print(f"Warning: Test Eval - Letter '{letter_char}' tokenized to {tokens}.")
                letter_token_ids[letter_char] = -1

        test_all_choice_probs = []
        test_all_correctness = []

        for i in tqdm(range(0, len(test_prompts), PER_DEVICE_EVAL_BATCH_SIZE), desc="Predicting on Test Set"):
            batch_p_text = test_prompts[i: i + PER_DEVICE_EVAL_BATCH_SIZE]
            inputs = tokenizer(
                batch_p_text, return_tensors="pt", padding="longest", truncation=True,
                max_length=MAX_EVAL_PROMPT_LENGTH, add_special_tokens=True
            ).to(model_for_test.device)
            with torch.no_grad():
                outputs = model_for_test(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
            next_token_probs_softmax = torch.softmax(next_token_logits, dim=-1).cpu()

            for j_batch, s_probs in enumerate(next_token_probs_softmax):
                choice_p = np.zeros(NUM_CHOICES_MC, dtype=float)
                for choice_i, key_l in enumerate(ANSWER_MAP_KEYS):
                    tid = letter_token_ids.get(key_l, -1)
                    if tid != -1: choice_p[choice_i] = s_probs[tid].item()

                sum_p_test = np.sum(choice_p)
                if sum_p_test > 1e-6:
                    choice_p /= sum_p_test
                else:
                    choice_p[:] = 1.0 / NUM_CHOICES_MC
                test_all_choice_probs.append(choice_p.tolist())

                pred_l_idx = np.argmax(choice_p)
                pred_l_char = ANSWER_MAP_KEYS[pred_l_idx]
                true_l_char = test_true_letters[i + j_batch]
                test_all_correctness.append(int(pred_l_char == true_l_char))

        tokenizer.padding_side = original_padding_side
        test_accuracy_custom = np.mean(test_all_correctness) if test_all_correctness else 0.0
        test_eval_duration = time.time() - test_eval_start_time

        print(f"Custom Test Set Evaluation Accuracy: {test_accuracy_custom:.4f}")
        print(f"Custom Test Set Evaluation Duration: {test_eval_duration:.2f}s")

        test_results_summary = {
            "test_accuracy_custom": test_accuracy_custom,
            "test_custom_eval_duration_seconds": round(test_eval_duration, 2),
            "num_test_samples": len(test_prompts),
            "best_model_source": best_model_dir
        }
        test_metrics_path = os.path.join(output_dir_base, "test_results_summary_custom_accuracy.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_results_summary, f, indent=4)
        print(f"Custom test metrics saved to {test_metrics_path}")

        detailed_test_output_path = os.path.join(output_dir_base, "test_set_detailed_predictions.json")
        with open(detailed_test_output_path, "w") as f:
            json.dump({"logits_probs": test_all_choice_probs, "responses_correctness": test_all_correctness}, f,
                      indent=2)
        print(f"Detailed test predictions saved to {detailed_test_output_path}")

        del model_for_test
    except Exception as e:
        print(f"Error during evaluation on test set: {e}");
        traceback.print_exc()
else:
    print("Skipping final test evaluation as training was not successful or best model not found.")
    if not training_successful: print("  Reason: Training did not complete.")
    if not os.path.exists(best_model_dir): print(f"  Reason: Best model directory not found at {best_model_dir}")

if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"\nScript finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output and best model (if training successful) in: {output_dir_base}")