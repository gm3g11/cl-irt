import sys
import os
import datetime
import random
import traceback
import json
from tqdm import tqdm
import re
import time

import torch
import numpy as np

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler as TorchAmpGradScaler, autocast as torch_amp_autocast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
# import evaluate # Not directly used with a metric object, can be commented if not needed elsewhere

# --- IRT Scoring (from your irt_scoring.py) ---
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import expit


def theta_fn_irt(difficulties, student_prior, response_pattern):
    def fn(theta_val):
        theta_val = theta_val[0]
        probabilities = expit(theta_val - difficulties)
        log_likelihood = student_prior.logpdf(theta_val)
        probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
        for i, rp_val in enumerate(response_pattern):
            if rp_val == 1:
                log_likelihood += np.log(probabilities[i])
            elif rp_val == -1:
                log_likelihood += np.log(1 - probabilities[i])
        return -log_likelihood

    return fn


def calculate_theta_irt(difficulties, response_pattern, num_obs=-1, initial_theta_val=0.0):
    start_time_irt_calc = time.time()
    difficulties_np = np.array(difficulties, dtype=float)
    response_pattern_np = np.array(response_pattern, dtype=float)

    if len(difficulties_np) == 0 or len(difficulties_np) != len(response_pattern_np):
        return initial_theta_val, time.time() - start_time_irt_calc

    valid_indices = ~np.isnan(difficulties_np) & ~np.isnan(response_pattern_np)  # Ensure responses are also valid
    difficulties_filt = difficulties_np[valid_indices]
    response_pattern_filt = response_pattern_np[valid_indices]

    if len(difficulties_filt) == 0:
        print(
            "  calculate_theta_irt: No valid (non-NaN) difficulties/responses to estimate theta after filtering. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt_calc

    student_prior = norm(loc=0., scale=1.)
    if num_obs > 0 and len(difficulties_filt) > num_obs:
        idx = np.random.choice(len(difficulties_filt), num_obs, replace=False)
        difficulties_sample, response_pattern_sample = difficulties_filt[idx], response_pattern_filt[idx]
    else:
        difficulties_sample, response_pattern_sample = difficulties_filt, response_pattern_filt

    if len(difficulties_sample) == 0:
        print(
            "  calculate_theta_irt: No samples to estimate theta after potential sampling. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt_calc

    fn_min = theta_fn_irt(difficulties_sample, student_prior, response_pattern_sample)
    res = minimize(fn_min, [initial_theta_val], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-4})
    est_theta = res['x'][0]

    if np.isnan(est_theta) or np.isinf(est_theta):
        print(
            f"  calculate_theta_irt: Estimated theta is NaN/Inf. Optimizer success: {res.success}, Message: {res.message}. Returning initial_theta_val.")
        est_theta = initial_theta_val
    return est_theta, time.time() - start_time_irt_calc


# --- End IRT Scoring ---

# --- PUDF Data Selection Logic ---
def select_data_for_pudf_epoch(
        full_hf_train_dataset, capacity_theta, difficulty_col='difficulty',
        pudf_ordering='easiest', lower_offset=-float('inf'), upper_offset=0.0,
        min_samples_per_epoch=100):
    print(f"  Selecting data: capacity_theta={capacity_theta:.4f}, "
          f"window=[{capacity_theta + lower_offset:.4f}, {capacity_theta + upper_offset:.4f}), "
          f"min_samples={min_samples_per_epoch}")
    if difficulty_col not in full_hf_train_dataset.column_names:
        print(f"  Error: Difficulty column '{difficulty_col}' not found. Returning empty selection.")
        return full_hf_train_dataset.select([])  # Return empty selection of the same type

    # Filter out None or NaN difficulties before selection
    filtered_for_selection = full_hf_train_dataset.filter(
        lambda x: x[difficulty_col] is not None and not np.isnan(x[difficulty_col]), load_from_cache_file=False
    )
    if len(filtered_for_selection) == 0:
        print(f"  Warning: No valid difficulties in dataset for selection. Returning empty selection.")
        return full_hf_train_dataset.select([])

    min_diff_target = capacity_theta + lower_offset;
    max_diff_target = capacity_theta + upper_offset

    selected_hf_dataset = filtered_for_selection.filter(
        lambda x: min_diff_target <= x[difficulty_col] < max_diff_target, load_from_cache_file=False)

    print(f"  Initially selected {len(selected_hf_dataset)} samples by difficulty window.")

    if len(selected_hf_dataset) < min_samples_per_epoch:
        print(f"  Selected data ({len(selected_hf_dataset)}) < min_samples ({min_samples_per_epoch}).")
        if len(
            filtered_for_selection) == 0: return selected_hf_dataset  # Should be empty if filtered_for_selection is empty

        num_to_take = min(min_samples_per_epoch, len(filtered_for_selection))
        print(f"  Taking {num_to_take} samples via '{pudf_ordering}' ordering from dataset with valid difficulties.")

        # dataset_to_sort was already filtered for non-None difficulties (filtered_for_selection)
        # No need to filter again unless original full_hf_train_dataset is intended
        dataset_to_sort = filtered_for_selection

        if len(dataset_to_sort) == 0:  # Should not happen if filtered_for_selection was not empty
            print(f"  Warning: All difficulties were None for sorting. Cannot fulfill min_samples with sorting.")
            return selected_hf_dataset  # Which would be the initially window-selected one (possibly empty)

        reverse_sort = True if pudf_ordering == 'hardest' else False
        try:
            # Ensure indices are reset if any previous selections happened on this object if it's not a fresh copy
            sorted_ds = dataset_to_sort.sort(difficulty_col, reverse=reverse_sort, load_from_cache_file=False)
            actual_take = min(num_to_take, len(sorted_ds))
            if actual_take > 0:
                selected_hf_dataset = sorted_ds.select(range(actual_take))
            else:  # If nothing to take, return an empty selection
                selected_hf_dataset = dataset_to_sort.select([])
        except Exception as e:
            print(f"  Error sorting for min_samples: {e}. Returning current selection.")
            # Fallback to current selected_hf_dataset which might be smaller than min_samples_per_epoch

    print(f"  Final samples for epoch: {len(selected_hf_dataset)}")
    return selected_hf_dataset


# --- End PUDF Data Selection ---

# ----- Configuration -----
# !!! IMPORTANT: Verify if Qwen/Qwen2.5-7B is a BASE model or an INSTRUCT/CHAT model.
# !!! If it's an INSTRUCT/CHAT model, the prompt format in `create_prompt_and_target_letter_pudf`
# !!! and `preprocess_sft_format_pudf` MUST be changed to use the Qwen chat template.
# !!! Failure to do so will likely result in very poor performance (like 0% accuracy).
model_id_hf = "Qwen/Qwen2.5-7B"
dataset_id = "GBaker/MedQA-USMLE-4-options"

max_seq_length_sft = 512 + 10  # Max length for combined prompt + target for SFT
max_prompt_len_sft_preprocess = 512  # Max length for the prompt part during preprocessing
max_target_len_sft_preprocess = 10  # Max length for the target part (e.g., "A" + EOS) during preprocessing

per_device_train_bs_pudf = 2
per_device_eval_bs_pudf = 4
grad_accum_steps_pudf = 16
num_pudf_epochs = 10
early_stopping_patience_pudf = 3  # Increased slightly
learning_rate_pudf = 1e-4  # Might be too high for full SFT, but okay for LoRA
weight_decay_pudf = 0.01

lora_r_pudf = 16
lora_alpha_pudf = 32  # Often 2*r
lora_dropout_pudf = 0.05
# For Qwen2 models, check recommended target_modules. Common ones include:
# "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
# Some models might use "self_attn.q_proj", "self_attn.k_proj", etc., or "mlp.gate_proj"
# Verify with the specific Qwen2 model architecture if issues persist.
lora_target_modules_pudf = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

random_seed = 63
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bf16_pudf_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

ANSWER_MAP_KEYS = ["A", "B", "C", "D"]
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS)
max_new_tokens_for_choice_pred = 3  # Increased slightly to allow for space + letter + EOS if needed by tokenizer

DIFFICULTY_FILE_PATH = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/MeD_QA/merged_jsonlines_output/test-1pl/best_parameters.json"
DIFFICULTY_JSON_KEY = "diff"
INITIAL_CAPACITY_THETA = 0.0
NUM_OBS_THETA_ESTIMATION = -1  # Use all valid observations
PUDF_LOWER_OFFSET = -float('inf')
PUDF_UPPER_OFFSET = 0.0
PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH = 100
PUDF_ORDERING = 'easiest'  # or 'closest_to_theta' could be an alternative strategy

current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
run_name_pudf = f"qwen2.5_7b_medqa_PUDF_qlora_{current_time_str}_ep{num_pudf_epochs}"
output_dir_pudf_main = f"./{run_name_pudf}"
best_adapter_pudf_path = os.path.join(output_dir_pudf_main, "best_pudf_qlora_adapter")
os.makedirs(output_dir_pudf_main, exist_ok=True)
os.makedirs(best_adapter_pudf_path, exist_ok=True)

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

print(f"Running with model: {model_id_hf}")
print(f"BF16 enabled for PUDF: {bf16_pudf_enabled}")
print(f"Output directory: {output_dir_pudf_main}")
print(f"Using device: {DEVICE}, Random seed: {random_seed}")


def load_difficulties_from_file(difficulty_file_path, difficulty_key):
    print(f"Loading difficulty scores from: {difficulty_file_path}")
    try:
        with open(difficulty_file_path, 'r') as f:
            data = json.load(f)
        scores = data.get(difficulty_key) if isinstance(data, dict) and difficulty_key in data else data if isinstance(
            data, list) else None
        if scores is None: raise KeyError(f"Key '{difficulty_key}' not found or data not a list of scores.")
        return np.array(scores, dtype=float)
    except Exception as e:
        print(f"Error loading difficulty scores: {e}");
        traceback.print_exc();
        raise


print(f"Loading dataset: {dataset_id}")
raw_dataset_loaded = load_dataset(dataset_id)
dataset_filtered = raw_dataset_loaded.filter(
    lambda ex: ex["answer_idx"] and ex["answer_idx"].strip().upper() in ANSWER_MAP_KEYS,
    load_from_cache_file=False
)

if 'train' not in dataset_filtered or len(dataset_filtered["train"]) == 0:
    print("Error: Dataset must contain non-empty 'train' split after filtering.");
    sys.exit(1)

original_train_for_diff = dataset_filtered['train']
difficulty_scores = load_difficulties_from_file(DIFFICULTY_FILE_PATH, DIFFICULTY_JSON_KEY)
if len(difficulty_scores) != len(original_train_for_diff):
    raise ValueError(
        f"Mismatch! Diff scores ({len(difficulty_scores)}) != original train ({len(original_train_for_diff)}).")

train_full_with_diff = original_train_for_diff.add_column("difficulty",
                                                          difficulty_scores.tolist())  # Ensure it's a list for add_column
print(f"Added 'difficulty' to original 'train'. Columns: {train_full_with_diff.column_names}")

pudf_test_pre_sft = dataset_filtered['test']
if "validation" not in dataset_filtered or len(dataset_filtered["validation"]) == 0:
    print("Validation split not found/empty. Splitting 'train' (with diff) into 80/20 for PUDF...")
    train_val_split = train_full_with_diff.train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
    pudf_train_raw, pudf_validation_raw = train_val_split['train'], train_val_split['test']
else:
    print("Using predefined 'validation' split.")
    pudf_train_raw = train_full_with_diff  # Use the full train set with difficulties
    pudf_validation_raw = dataset_filtered['validation']
    # If predefined validation doesn't have difficulty, it's an issue for IRT.
    # This script doesn't add difficulties to a predefined val set. This should be handled upstream if needed.
    if 'difficulty' not in pudf_validation_raw.column_names:
        print(
            f"CRITICAL WARNING: Predefined 'validation' lacks 'difficulty'. IRT theta on it will be impaired (use initial theta).")

pudf_dataset_pre_sft = DatasetDict(
    {'train': pudf_train_raw, 'validation': pudf_validation_raw, 'test': pudf_test_pre_sft})
print("\nPUDF dataset splits (pre-SFT formatting):")
for n, ds in pudf_dataset_pre_sft.items():
    print(f"  {n}: {len(ds)} examples, cols: {ds.column_names}")
    if 'difficulty' in ds.column_names and n in ['train', 'validation']:
        # Check for NaNs in difficulty column if it exists
        # Ensure 'difficulty' is treated as a numeric type for np.isnan
        difficulties_in_split = np.array(ds['difficulty'], dtype=float)
        num_nans = np.sum(np.isnan(difficulties_in_split))
        if num_nans > 0: print(f"    WARNING: '{n}' has {num_nans} NaN in 'difficulty'. These will be filtered by IRT.")

print(f"Loading tokenizer for {model_id_hf}...")
tokenizer = AutoTokenizer.from_pretrained(model_id_hf, padding_side="left", trust_remote_code=True)
original_vocab_size = len(tokenizer)

if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    # Common practice for Qwen and other models is to use eos_token as pad_token if no specific pad_token is set.
    # However, for supervised fine-tuning where we might append EOS, having a distinct pad_token can be clearer.
    # Let's try setting it to EOS if pad is None, as Qwen often does.
    # If issues arise, consider adding a new special token like "␂".
    if tokenizer.eos_token:
        print(f"Tokenizer pad_token is None or same as EOS. Setting pad_token to eos_token ('{tokenizer.eos_token}').")
        tokenizer.pad_token = tokenizer.eos_token
    else:  # If no EOS token either, this is problematic. Add a custom one.
        new_pad_token_str = "␂"  # Or "<|pad|>"
        print(f"Tokenizer pad_token and eos_token are None. Adding NEW special pad_token: '{new_pad_token_str}'")
        tokenizer.add_special_tokens({"pad_token": new_pad_token_str})

# Ensure pad_token_id is correctly set as an integer
if tokenizer.pad_token and not isinstance(tokenizer.pad_token_id, int):
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

print(
    f"Tokenizer: Vocab size: {len(tokenizer)}. Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}. EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")


def create_prompt_and_target_letter_pudf(example):
    # !!! IMPORTANT: If using an INSTRUCT/CHAT Qwen model, this function needs to change
    # !!! to create a list of messages for tokenizer.apply_chat_template.
    # Example for INSTRUCT:
    # messages = [
    #    {"role": "user", "content": "Question: ... Options: ... Answer:"}
    # ]
    # return {"messages": messages, "target_letter": target_letter, "difficulty": ...}

    question = example["question"].strip()
    options_dict = example["options"]
    answer_idx_key = example["answer_idx"]  # This is "A", "B", "C", or "D"

    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for k_opt in ANSWER_MAP_KEYS:  # Ensure order A, B, C, D
            prompt_parts.append(f"{k_opt}) {options_dict.get(k_opt, '[Invalid Option]')}")
    else:  # Fallback if options are not a dict (should not happen with MedQA)
        for i, k_opt in enumerate(ANSWER_MAP_KEYS):
            prompt_parts.append(f"{k_opt}) [Option {i + 1} Malformed]")

    prompt_parts.append("\nAnswer:")  # Prompt ends with "Answer:", model should predict the letter.
    prompt_text = "\n".join(prompt_parts)

    target_letter = answer_idx_key.strip().upper()
    # Ensure target_letter is valid, otherwise, this example might be problematic for training/eval
    if target_letter not in ANSWER_MAP_KEYS:
        # Handle invalid target_letter, e.g., by choosing a default or logging an error
        # For now, let's pass it through; downstream checks on letter_tokens_for_eval might catch it.
        print(f"Warning: Invalid target_letter '{target_letter}' in example. Question: {question[:50]}...")

    return {"prompt": prompt_text, "target_letter": target_letter, "difficulty": example.get("difficulty", np.nan)}


def preprocess_sft_format_pudf(examples_batch):
    # REVISED SFT PREPROCESSING - Tokenizes prompt and target separately
    inputs_tokenized_batch, labels_tokenized_batch, difficulties_batch = [], [], []
    # Check if the first item in examples_batch (which is a dict of lists) has 'question'
    if not examples_batch or "question" not in examples_batch or not examples_batch["question"]:
        return {"input_ids": [], "labels": [], "difficulty": []}  # Return empty if batch is malformed

    num_examples_in_batch = len(examples_batch["question"])

    for i in range(num_examples_in_batch):
        current_example_raw = {k: examples_batch[k][i] for k in examples_batch}
        processed = create_prompt_and_target_letter_pudf(current_example_raw)

        # !!! IMPORTANT: If using an INSTRUCT/CHAT Qwen model, 'processed' would contain 'messages'.
        # !!! The logic here would use tokenizer.apply_chat_template(processed['messages'], ...)
        # !!! to get the prompt_text, and target_letter would be the assistant's response.
        prompt_text, target_letter = processed["prompt"], processed["target_letter"]

        tokenized_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_prompt_len_sft_preprocess,
            padding=False,
            add_special_tokens=True  # Typically adds BOS
        )
        prompt_input_ids = tokenized_prompt.input_ids

        tokenized_target = tokenizer(
            target_letter,  # e.g., "A"
            truncation=True,  # Target should be short, so no truncation ideally
            max_length=max_target_len_sft_preprocess,
            padding=False,
            add_special_tokens=False  # No BOS/EOS for the target part itself
        )
        target_input_ids = tokenized_target.input_ids

        input_ids = prompt_input_ids + target_input_ids
        labels = ([-100] * len(prompt_input_ids)) + target_input_ids

        if tokenizer.eos_token_id is not None:
            if not input_ids or input_ids[-1] != tokenizer.eos_token_id:  # Avoid double EOS
                if len(input_ids) < max_seq_length_sft:  # Check space before appending
                    input_ids.append(tokenizer.eos_token_id)
                    labels.append(tokenizer.eos_token_id)  # EOS should be predicted

        final_input_ids = input_ids[:max_seq_length_sft]
        final_labels = labels[:max_seq_length_sft]

        if len(final_labels) < len(final_input_ids):
            final_labels.extend([-100] * (len(final_input_ids) - len(final_labels)))
        elif len(final_input_ids) < len(final_labels):
            final_labels = final_labels[:len(final_input_ids)]

        # Debug print for the first example of the first batch
        if i == 0 and not hasattr(preprocess_sft_format_pudf, 'printed_once'):
            print("--- Debug SFT Preprocessing (First example of first batch) ---")
            print(f"Prompt Text: {prompt_text}")
            print(f"Target Letter: {target_letter}")
            print(f"Final Input IDs ({len(final_input_ids)}): {final_input_ids}")
            print(f"Tokens: {tokenizer.decode(final_input_ids)}")  # More readable than individual tokens
            # print(f"Tokens (indiv): {tokenizer.convert_ids_to_tokens(final_input_ids)}")
            print(f"Final Labels ({len(final_labels)}): {final_labels}")
            label_tokens_debug = []
            for lbl_idx, label_id in enumerate(final_labels):
                if label_id != -100:
                    # Check if label_id is a valid token ID before converting
                    if 0 <= label_id < tokenizer.vocab_size:
                        label_tokens_debug.append(tokenizer.convert_ids_to_tokens([label_id])[0])
                    else:  # Handle potential out-of-range token IDs (e.g. if EOS was from a different model)
                        label_tokens_debug.append(f"[ID:{label_id}]")
                else:
                    label_tokens_debug.append("[-100]")
            print(f"Label Tokens: {' '.join(label_tokens_debug)}")
            preprocess_sft_format_pudf.printed_once = True
            print("-----------------------------------------------------------------")

        inputs_tokenized_batch.append(final_input_ids)
        labels_tokenized_batch.append(final_labels)
        difficulties_batch.append(processed["difficulty"])

    output_dict = {"input_ids": inputs_tokenized_batch, "labels": labels_tokenized_batch}
    if difficulties_batch:
        output_dict["difficulty"] = difficulties_batch
    return output_dict


print(f"\nTokenizing dataset for SFT. Max SFT sequence length: {max_seq_length_sft}")
# Ensure 'difficulty' column is preserved during tokenization if it exists.
# The `remove_columns` will keep 'difficulty' if it's not in the list of columns to remove.
cols_to_remove = [c for c in pudf_dataset_pre_sft["train"].column_names if c not in ['difficulty']]

tokenized_sft_datasets = pudf_dataset_pre_sft.map(
    preprocess_sft_format_pudf, batched=True,
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    remove_columns=cols_to_remove,
    load_from_cache_file=False  # Disable caching for easier debugging of preprocessing
)
print("SFT Tokenization complete.")
for split_n, ds_s in tokenized_sft_datasets.items():
    print(f"Cols SFT tokenized '{split_n}': {ds_s.column_names}")
    if 'difficulty' not in ds_s.column_names and split_n != 'test':  # Test set might not have difficulty
        print(f"  CRITICAL WARNING: SFT '{split_n}' DOES NOT have 'difficulty' column for PUDF.")

data_collator_sft = DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100, padding="longest")

print("\nConfiguring BitsAndBytes for QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if bf16_pudf_enabled else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
print(f"Loading base model {model_id_hf} for QLoRA PUDF training...")
# Try with explicit torch_dtype for from_pretrained, matching compute_dtype
model_dtype = torch.bfloat16 if bf16_pudf_enabled else torch.float16
base_model_for_pudf = AutoModelForCausalLM.from_pretrained(
    model_id_hf,
    quantization_config=bnb_config,
    device_map="auto",  # Handles multi-GPU or CPU if no CUDA
    trust_remote_code=True,
    torch_dtype=model_dtype
)

model_vocab_size_before_resize = base_model_for_pudf.get_input_embeddings().weight.size(0)
if len(tokenizer) > model_vocab_size_before_resize:
    print(f"Resizing base model token embeddings: {model_vocab_size_before_resize} -> {len(tokenizer)}")
    base_model_for_pudf.resize_token_embeddings(len(tokenizer))

# Sync pad_token_id in model config
if hasattr(base_model_for_pudf.config, 'pad_token_id') and tokenizer.pad_token_id is not None:
    if base_model_for_pudf.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Syncing base model pad_token_id to {tokenizer.pad_token_id}")
        base_model_for_pudf.config.pad_token_id = tokenizer.pad_token_id
elif tokenizer.pad_token_id is not None:  # If model.config doesn't have pad_token_id but tokenizer does
    print(f"Setting base model pad_token_id to {tokenizer.pad_token_id} (was not initially present in model config)")
    base_model_for_pudf.config.pad_token_id = tokenizer.pad_token_id

base_model_for_pudf = prepare_model_for_kbit_training(base_model_for_pudf, use_gradient_checkpointing=True)
peft_config_pudf = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=lora_r_pudf,
    lora_alpha=lora_alpha_pudf,
    lora_dropout=lora_dropout_pudf,
    target_modules=lora_target_modules_pudf,
    bias="none"
)
pudf_peft_model = get_peft_model(base_model_for_pudf, peft_config_pudf)
print("QLoRA Causal LM PEFT model prepared for PUDF training.")
pudf_peft_model.print_trainable_parameters()

optimizer_pudf = AdamW(pudf_peft_model.parameters(), lr=learning_rate_pudf, weight_decay=weight_decay_pudf)

# Calculate total training steps more carefully
if 'train' in tokenized_sft_datasets and len(tokenized_sft_datasets['train']) > 0:
    # This is an approximation as data selection per PUDF epoch varies
    # Let's assume on average 75% of the data (or min_samples) is used per PUDF inner epoch.
    # This approximation is mainly for the LR scheduler.
    avg_samples_per_pudf_epoch = max(PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH,
                                     int(len(tokenized_sft_datasets['train']) * 0.5))  # Heuristic
    steps_per_pudf_outer_epoch_approx = avg_samples_per_pudf_epoch // (per_device_train_bs_pudf * grad_accum_steps_pudf)
    total_pudf_training_steps_approx = max(1, int(steps_per_pudf_outer_epoch_approx * num_pudf_epochs))
else:
    total_pudf_training_steps_approx = num_pudf_epochs  # Fallback, should not happen if train set exists

print(f"Approximate total PUDF training steps (for LR scheduler): {total_pudf_training_steps_approx}")

lr_scheduler_pudf = get_linear_schedule_with_warmup(
    optimizer_pudf,
    num_warmup_steps=max(1, int(0.1 * total_pudf_training_steps_approx)),  # Ensure num_warmup_steps >= 1
    num_training_steps=total_pudf_training_steps_approx
)
scaler_pudf = TorchAmpGradScaler(enabled=bf16_pudf_enabled)  # scaler is fine even if bf16 is false

# Ensure letter tokens are valid and single tokens
letter_tokens_for_eval = {}
for L in ANSWER_MAP_KEYS:
    encoded_L = tokenizer.encode(L, add_special_tokens=False)
    if encoded_L and len(encoded_L) == 1:
        letter_tokens_for_eval[L] = encoded_L[0]

print(f"Letter token IDs for eval: {letter_tokens_for_eval}")
if len(letter_tokens_for_eval) != NUM_CHOICES_MC:
    print(
        f"CRITICAL WARNING! Could not get single token IDs for all {NUM_CHOICES_MC} answer letters: {letter_tokens_for_eval}. This will break evaluation.")
    # Consider exiting if this happens, as eval will be meaningless:
    # sys.exit("Exiting due to incomplete letter token mapping for evaluation.")


def create_custom_evaluation_prompt(example):
    # Same logic as create_prompt_and_target_letter_pudf for the prompt part
    # !!! IMPORTANT: If using an INSTRUCT/CHAT Qwen model, this function needs to change
    # !!! to create the appropriate prompt format (e.g., list of messages or templated string).
    question = example["question"].strip()
    options_dict = example["options"]
    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for k_opt in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{k_opt}) {options_dict.get(k_opt, '[Invalid Option]')}")
    else:
        for i, k_opt in enumerate(ANSWER_MAP_KEYS):
            prompt_parts.append(f"{k_opt}) [Option {i + 1} Malformed]")
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


def evaluate_qwen_sft_and_estimate_theta(
        peft_model_eval, validation_sft_tokenized_ds, original_raw_validation_ds,
        tokenizer_eval, device_eval, epoch_curr, theta_init_calc,
        eval_prompt_max_len, letter_token_ids_map, max_new_toks_for_gen_choice):
    peft_model_eval.eval()
    print(f"  E{epoch_curr} Val: Calculating SFT LM loss...")
    sft_val_cols = ['input_ids', 'attention_mask', 'labels']
    current_sft_val_ds_cols = list(validation_sft_tokenized_ds.column_names)
    cols_to_set_format = [col for col in sft_val_cols if col in current_sft_val_ds_cols]

    if not all(c in cols_to_set_format for c in ['input_ids', 'labels']):
        print(
            f"  Eval Fn Error E{epoch_curr}: 'input_ids' or 'labels' missing from tokenized validation set. Cols: {cols_to_set_format}")
        return float('nan'), float('inf'), theta_init_calc, 0.0

    try:
        validation_sft_tokenized_ds.set_format(type='torch', columns=cols_to_set_format)
    except Exception as e:
        print(f"  Eval Fn Error E{epoch_curr}: Failed to set format for validation_sft_tokenized_ds. Error: {e}")
        return float('nan'), float('inf'), theta_init_calc, 0.0

    sft_val_dataloader = DataLoader(validation_sft_tokenized_ds, batch_size=per_device_eval_bs_pudf,
                                    collate_fn=data_collator_sft, shuffle=False)
    total_sft_loss, num_sft_batches = 0, 0
    for sft_batch in tqdm(sft_val_dataloader, desc=f"  E{epoch_curr} SFT Val Loss", leave=False):
        sft_batch = {k: v.to(device_eval) for k, v in sft_batch.items() if k in cols_to_set_format and hasattr(v, 'to')}
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=bf16_pudf_enabled):
            outputs = peft_model_eval(**sft_batch)
            total_sft_loss += outputs.loss.item()
            num_sft_batches += 1

    avg_sft_loss = total_sft_loss / num_sft_batches if num_sft_batches > 0 else float('nan')
    perplexity = np.exp(avg_sft_loss) if not np.isnan(avg_sft_loss) else float('inf')
    print(f"  E{epoch_curr} Val: Avg SFT Loss = {avg_sft_loss:.4f}, Perplexity = {perplexity:.4f}")

    print(f"  E{epoch_curr} Val: Predicting choices for Theta estimation...")
    item_diffs_theta, resp_patt_theta = [], []

    # Check if original_raw_validation_ds has 'difficulty' and is not empty
    has_diff_irt_val = "difficulty" in original_raw_validation_ds.column_names and len(original_raw_validation_ds) > 0

    if not has_diff_irt_val:
        print(
            f"  E{epoch_curr} Val: 'difficulty' missing or empty original_raw_validation_ds for IRT. Theta defaults to initial.")
        return avg_sft_loss, perplexity, theta_init_calc, 0.0

    if not letter_token_ids_map:  # If no valid letter tokens were found
        print(f"  E{epoch_curr} Val: letter_token_ids_map is empty. Cannot perform IRT choice prediction.")
        return avg_sft_loss, perplexity, theta_init_calc, 0.0

    prompts_for_gen = [create_custom_evaluation_prompt(ex) for ex in original_raw_validation_ds]
    true_letters_for_gen = [ex["answer_idx"].strip().upper() for ex in original_raw_validation_ds]
    difficulties_for_gen_raw = original_raw_validation_ds["difficulty"]

    time_theta_start_local = time.time()
    original_padding_side = tokenizer_eval.padding_side
    tokenizer_eval.padding_side = "left"  # Important for generation

    for i in tqdm(range(0, len(prompts_for_gen), per_device_eval_bs_pudf), desc=f"  E{epoch_curr} Theta Pred",
                  leave=False):
        batch_prompts = prompts_for_gen[i:i + per_device_eval_bs_pudf]
        batch_true_letters = true_letters_for_gen[i:i + per_device_eval_bs_pudf]
        batch_difficulties_raw = difficulties_for_gen_raw[i:i + per_device_eval_bs_pudf]

        # Convert difficulties to float and handle potential NaNs for this batch
        batch_difficulties = np.array(batch_difficulties_raw, dtype=float)

        inputs = tokenizer_eval(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=eval_prompt_max_len
        ).to(device_eval)

        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=bf16_pudf_enabled):
            generated_ids = peft_model_eval.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_toks_for_gen_choice,
                pad_token_id=tokenizer_eval.pad_token_id,
                eos_token_id=tokenizer_eval.eos_token_id,  # Good to provide EOS for generate
                do_sample=False  # Greedy decoding
            )

        for j_idx, gen_ids_sample in enumerate(generated_ids):
            current_difficulty = batch_difficulties[j_idx]
            if np.isnan(current_difficulty):  # Skip if difficulty is NaN for this item
                continue

            # input_len is the length of the prompt tokens for this specific sample in the batch
            # For left-padded sequences, inputs.input_ids[j_idx] might have padding.
            # We need the length of the actual prompt part.
            # A common way is to find where padding starts or use the length of the non-padded sequence if available.
            # Simpler for generate: generated tokens are appended after the input_ids length.
            prompt_part_len = inputs.input_ids[j_idx].shape[0]

            predicted_letter = None
            if len(gen_ids_sample) > prompt_part_len:
                # First generated token ID
                predicted_token_id = gen_ids_sample[prompt_part_len].item()
                # Find which letter (A,B,C,D) this token ID corresponds to
                predicted_letter = next((L for L, tid in letter_token_ids_map.items() if tid == predicted_token_id),
                                        None)

            is_correct = 1 if predicted_letter is not None and predicted_letter == batch_true_letters[j_idx] else -1
            resp_patt_theta.append(is_correct)
            item_diffs_theta.append(current_difficulty)

    tokenizer_eval.padding_side = original_padding_side  # Restore padding side
    theta_est_time_local = time.time() - time_theta_start_local

    final_theta_est = theta_init_calc
    if item_diffs_theta and resp_patt_theta:
        final_theta_est, _ = calculate_theta_irt(item_diffs_theta, resp_patt_theta, NUM_OBS_THETA_ESTIMATION,
                                                 theta_init_calc)
    else:
        print(
            f"  E{epoch_curr} Val: No valid data (difficulties/responses) for IRT theta estimation. Using initial value.")

    return avg_sft_loss, perplexity, final_theta_est, theta_est_time_local


# ----- PUDF Training Loop for Qwen QLoRA SFT -----
print("\nStarting Qwen SFT QLoRA PUDF training loop...")
current_cap_theta = INITIAL_CAPACITY_THETA
best_val_sft_loss_pudf = float('inf')
early_stop_counter = 0
all_pudf_stats_qwen = []

# Ensure train set is not empty before proceeding
if 'train' not in tokenized_sft_datasets or len(tokenized_sft_datasets['train']) == 0:
    print("CRITICAL ERROR: Tokenized training set is empty. Cannot start PUDF loop.")
    sys.exit(1)
full_tokenized_sft_train_hf = tokenized_sft_datasets['train']

for pudf_epoch_idx in range(num_pudf_epochs):
    epoch_time_start = time.time()
    print(f"\n===== PUDF Epoch {pudf_epoch_idx + 1}/{num_pudf_epochs} =====")

    guiding_theta_for_current_selection = current_cap_theta

    print(
        f"  Estimating capacity & SFT val loss (theta_start_epoch_for_irt_opt={guiding_theta_for_current_selection:.4f})...")
    # Ensure validation sets are not empty
    if len(tokenized_sft_datasets['validation']) == 0 or len(pudf_dataset_pre_sft['validation']) == 0:
        print(
            f" PUDF Epoch {pudf_epoch_idx + 1}: Validation data is empty. Skipping evaluation and training for this epoch.")
        # Decide how to handle current_cap_theta: maybe keep it, or slightly increase. For now, keep.
        # Or, stop if validation is critical.
        all_pudf_stats_qwen.append({
            "pudf_epoch": pudf_epoch_idx + 1, "cap_theta_select": guiding_theta_for_current_selection,
            "num_sel_sft_train": 0, "avg_inner_sft_loss": float('nan'),
            "val_sft_loss": float('nan'), "val_sft_ppl": float('inf'),
            "theta_val_post_train": guiding_theta_for_current_selection, "duration_s": time.time() - epoch_time_start,
            "theta_est_time_s": 0.0, "notes": "Skipped due to empty validation set"
        })
        early_stop_counter += 1  # Count as a non-improvement
        if early_stop_counter >= early_stopping_patience_pudf:
            print(f"  Early stopping at E{pudf_epoch_idx + 1} due to persistent validation issues / no improvement.");
            break
        continue

    sft_val_loss, sft_val_ppl, estimated_theta_from_eval, theta_est_time_current_epoch = evaluate_qwen_sft_and_estimate_theta(
        pudf_peft_model, tokenized_sft_datasets['validation'], pudf_dataset_pre_sft['validation'],
        tokenizer, DEVICE, pudf_epoch_idx + 1, guiding_theta_for_current_selection,
        max_prompt_len_sft_preprocess, letter_tokens_for_eval, max_new_tokens_for_choice_pred
    )

    theta_for_selection = estimated_theta_from_eval
    if np.isnan(theta_for_selection):
        print(
            f"  Theta estimation for selection was NaN. Using guiding theta from start of epoch: {guiding_theta_for_current_selection:.4f}")
        theta_for_selection = guiding_theta_for_current_selection

    # User's rule for first epoch (adjust if needed)
    if pudf_epoch_idx == 0 and theta_for_selection < 1.0:  # Applied only to derived theta_for_selection
        print(
            f"  PUDF Epoch 1: Effective theta for data selection {theta_for_selection:.4f} is < 1.0. Adjusting to 1.0.")
        theta_for_selection = 1.0

    print(
        f"  E{pudf_epoch_idx + 1}: Capacity(Theta) for selection this epoch = {theta_for_selection:.4f} (Derived from estimate: {estimated_theta_from_eval:.4f}, Est_time={theta_est_time_current_epoch:.2f}s)")
    print(
        f"  E{pudf_epoch_idx + 1}: Val SFT Loss (pre-inner-train) = {sft_val_loss:.4f}, PPL (pre-inner-train) = {sft_val_ppl:.4f}")

    # Update current_cap_theta (to be used as IRT init for NEXT epoch's eval) using nudge logic
    if not np.isnan(
            estimated_theta_from_eval) and estimated_theta_from_eval <= guiding_theta_for_current_selection and pudf_epoch_idx > 0:
        current_cap_theta = guiding_theta_for_current_selection + 0.05
        print(
            f"  Theta for next IRT init nudged. Est: {estimated_theta_from_eval:.4f} <= Prev IRT Init: {guiding_theta_for_current_selection:.4f}. Next IRT Init: {current_cap_theta:.4f}")
    elif not np.isnan(estimated_theta_from_eval):
        current_cap_theta = estimated_theta_from_eval
    else:
        print(
            f"  Theta estimation was NaN. Next IRT init will use previous value: {guiding_theta_for_current_selection:.4f}")
        current_cap_theta = guiding_theta_for_current_selection

    epoch_sft_train_data_hf = select_data_for_pudf_epoch(
        full_tokenized_sft_train_hf, theta_for_selection, 'difficulty',
        PUDF_ORDERING, PUDF_LOWER_OFFSET, PUDF_UPPER_OFFSET, PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH
    )
    num_selected_sft_samples = len(epoch_sft_train_data_hf)
    avg_inner_sft_loss = float('nan')

    if num_selected_sft_samples > 0:
        cols_for_sft_loader_train = ['input_ids', 'attention_mask', 'labels']
        current_epoch_train_cols = list(epoch_sft_train_data_hf.column_names)
        cols_to_set_format_train = [col for col in cols_for_sft_loader_train if col in current_epoch_train_cols]

        if not all(c in cols_to_set_format_train for c in ['input_ids', 'labels']):
            print(
                f"  Warning E{pudf_epoch_idx + 1}: Selected training data missing SFT columns ('input_ids' or 'labels'). Skipping inner train.")
        else:
            epoch_sft_train_data_hf.set_format(type='torch', columns=cols_to_set_format_train)
            inner_sft_dataloader = DataLoader(
                epoch_sft_train_data_hf, batch_size=per_device_train_bs_pudf,
                collate_fn=data_collator_sft, shuffle=True,
                num_workers=min(2, os.cpu_count() if os.cpu_count() else 1)  # Max 2 workers or available CPUs
            )
            pudf_peft_model.train()
            total_inner_loss, grad_steps_inner = 0, 0
            for step, sft_batch_train in enumerate(
                    tqdm(inner_sft_dataloader, desc=f"  InnerTrain SFT E{pudf_epoch_idx + 1}", leave=False)):
                optimizer_pudf.zero_grad()
                sft_batch_train = {k: v.to(DEVICE) for k, v in sft_batch_train.items() if
                                   k in cols_to_set_format_train and hasattr(v, 'to')}

                with torch_amp_autocast(device_type=DEVICE.type, enabled=bf16_pudf_enabled):
                    outputs_train = pudf_peft_model(**sft_batch_train)
                    loss_train = outputs_train.loss
                    if grad_accum_steps_pudf > 1:  # Only divide if accumulating
                        loss_train = loss_train / grad_accum_steps_pudf

                scaler_pudf.scale(loss_train).backward()
                total_inner_loss += outputs_train.loss.item()  # Log the non-accumulated loss item

                if (step + 1) % grad_accum_steps_pudf == 0 or (step + 1) == len(inner_sft_dataloader):
                    scaler_pudf.unscale_(optimizer_pudf)
                    torch.nn.utils.clip_grad_norm_(pudf_peft_model.parameters(), 1.0)
                    scaler_pudf.step(optimizer_pudf)
                    scaler_pudf.update()
                    lr_scheduler_pudf.step()
                    optimizer_pudf.zero_grad()
                    grad_steps_inner += 1

            avg_inner_sft_loss = total_inner_loss / (step + 1) if (step + 1) > 0 else float(
                'nan')  # Avg loss over all items
            print(
                f"  E{pudf_epoch_idx + 1}: Inner SFT train avg loss = {avg_inner_sft_loss:.4f} over {grad_steps_inner} grad steps.")
    else:
        print(
            f"  E{pudf_epoch_idx + 1}: No SFT data selected for training (samples = {num_selected_sft_samples}). Skipping inner train.")

    # Re-evaluate SFT loss and theta on validation after inner training
    # Pass current_cap_theta (which was updated after pre-train eval) as IRT init for this post-train eval
    sft_val_loss_post, sft_val_ppl_post, theta_val_post, _ = evaluate_qwen_sft_and_estimate_theta(
        pudf_peft_model, tokenized_sft_datasets['validation'], pudf_dataset_pre_sft['validation'],
        tokenizer, DEVICE, pudf_epoch_idx + 1, current_cap_theta,  # Use updated current_cap_theta for IRT init
        max_prompt_len_sft_preprocess, letter_tokens_for_eval, max_new_tokens_for_choice_pred
    )
    print(
        f"  E{pudf_epoch_idx + 1}: Val SFT Loss (post-inner-train) = {sft_val_loss_post:.4f}, PPL (post-inner-train) = {sft_val_ppl_post:.4f}")
    epoch_duration = time.time() - epoch_time_start

    all_pudf_stats_qwen.append({
        "pudf_epoch": pudf_epoch_idx + 1, "cap_theta_select": theta_for_selection,
        "num_sel_sft_train": num_selected_sft_samples, "avg_inner_sft_loss": avg_inner_sft_loss,
        "val_sft_loss_pre_train": sft_val_loss, "val_sft_ppl_pre_train": sft_val_ppl,  # Log pre-train val loss
        "val_sft_loss_post_train": sft_val_loss_post, "val_sft_ppl_post_train": sft_val_ppl_post,
        "theta_val_post_train": theta_val_post, "duration_s": epoch_duration,
        "theta_est_time_s": theta_est_time_current_epoch
    })

    if not np.isnan(sft_val_loss_post) and sft_val_loss_post < best_val_sft_loss_pudf:
        print(f"  New best val_sft_loss: {sft_val_loss_post:.4f} (prev {best_val_sft_loss_pudf:.4f}). Saving adapter.")
        best_val_sft_loss_pudf = sft_val_loss_post
        early_stop_counter = 0
        pudf_peft_model.save_pretrained(best_adapter_pudf_path)
        tokenizer.save_pretrained(best_adapter_pudf_path)
    elif not np.isnan(sft_val_loss_post):  # Loss is valid but not an improvement
        early_stop_counter += 1
        print(f"  Val SFT loss not improved. EarlyStop: {early_stop_counter}/{early_stopping_patience_pudf}")
    else:  # Loss is NaN
        early_stop_counter += 1
        print(f"  Val SFT loss is NaN. EarlyStop: {early_stop_counter}/{early_stopping_patience_pudf}")

    if early_stop_counter >= early_stopping_patience_pudf:
        print(f"  Early stopping at E{pudf_epoch_idx + 1}.")
        break
    print(f"  PUDF E{pudf_epoch_idx + 1} ended. Duration: {epoch_duration:.2f}s")

# --- End PUDF Loop ---
print(f"\nPUDF Qwen SFT Training finished. Best val_sft_loss: {best_val_sft_loss_pudf:.4f}")
stats_file_pudf_qwen = os.path.join(output_dir_pudf_main, "pudf_qwen_sft_training_stats.json")
with open(stats_file_pudf_qwen, 'w') as f:
    json.dump(all_pudf_stats_qwen, f, indent=4,
              default=lambda x: float(x) if isinstance(x, (np.float32, np.float64, np.float_)) else "NaN" if isinstance(
                  x, float) and np.isnan(x) else x)
print(f"PUDF Qwen SFT stats saved: {stats_file_pudf_qwen}")

# ----- Final Evaluation on Test Set -----
if os.path.exists(best_adapter_pudf_path) and \
        (os.path.exists(os.path.join(best_adapter_pudf_path, "adapter_model.safetensors")) or \
         os.path.exists(os.path.join(best_adapter_pudf_path, "adapter_model.bin"))):
    print(f"\nLoading base model with best PUDF QLoRA adapter from {best_adapter_pudf_path} for final test eval...")

    # Clear CUDA cache before loading new model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Reload base model for final evaluation
    base_model_final_eval = AutoModelForCausalLM.from_pretrained(
        model_id_hf,
        quantization_config=bnb_config,  # Use the same quantization
        torch_dtype=model_dtype,  # Use same dtype as training
        device_map="auto",
        trust_remote_code=True
    )

    model_vocab_size_final = base_model_final_eval.get_input_embeddings().weight.size(0)
    if len(tokenizer) > model_vocab_size_final:
        print(f"Resizing final eval model token embeddings: {model_vocab_size_final} -> {len(tokenizer)}")
        base_model_final_eval.resize_token_embeddings(len(tokenizer))

    if hasattr(base_model_final_eval.config, 'pad_token_id') and tokenizer.pad_token_id is not None:
        if base_model_final_eval.config.pad_token_id != tokenizer.pad_token_id:
            base_model_final_eval.config.pad_token_id = tokenizer.pad_token_id
    elif tokenizer.pad_token_id is not None:
        base_model_final_eval.config.pad_token_id = tokenizer.pad_token_id

    final_peft_model_eval = PeftModel.from_pretrained(base_model_final_eval, best_adapter_pudf_path)
    final_peft_model_eval.eval()
    print("PEFT model for final evaluation loaded and set to eval mode.")

    test_set_raw_final_eval = pudf_dataset_pre_sft["test"]
    if len(test_set_raw_final_eval) == 0:
        print("Test set is empty. Skipping final evaluation.")
    elif not letter_tokens_for_eval:
        print("Letter tokens for evaluation are not properly set. Skipping final evaluation.")
    else:
        all_prompts_final_eval = [create_custom_evaluation_prompt(ex) for ex in test_set_raw_final_eval]
        # Get true letters using the same robust function
        all_true_letters_final_eval = [create_prompt_and_target_letter_pudf(ex)["target_letter"] for ex in
                                       test_set_raw_final_eval]
        all_predicted_letters_final_eval = []

        original_padding_side_final_eval = tokenizer.padding_side
        tokenizer.padding_side = "left"  # Ensure left padding for generation

        with torch.inference_mode():
            for i in tqdm(range(0, len(all_prompts_final_eval), per_device_eval_bs_pudf), desc="Final Test Custom Eval",
                          leave=False):
                batch_prompts_final = all_prompts_final_eval[i: i + per_device_eval_bs_pudf]
                inputs_final_test = tokenizer(
                    batch_prompts_final, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_prompt_len_sft_preprocess  # Use prompt max length
                ).to(DEVICE)

                generated_ids_final = final_peft_model_eval.generate(
                    input_ids=inputs_final_test['input_ids'],
                    attention_mask=inputs_final_test['attention_mask'],
                    max_new_tokens=max_new_tokens_for_choice_pred,  # Should be small (e.g., 1-3)
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False  # MODIFIED: Removed temperature and top_p
                )

                for j_idx_final, gen_ids_sample_final in enumerate(generated_ids_final):
                    prompt_part_len_final = inputs_final_test['input_ids'][j_idx_final].shape[0]
                    predicted_letter_final = None
                    if len(gen_ids_sample_final) > prompt_part_len_final:
                        predicted_token_id_final = gen_ids_sample_final[prompt_part_len_final].item()
                        predicted_letter_final = next(
                            (L for L, tid in letter_tokens_for_eval.items() if tid == predicted_token_id_final), None)
                    all_predicted_letters_final_eval.append(predicted_letter_final)

        tokenizer.padding_side = original_padding_side_final_eval  # Restore

        correct_final = sum(1 for pred, true in zip(all_predicted_letters_final_eval, all_true_letters_final_eval) if
                            pred is not None and pred == true)
        total_final = len(all_true_letters_final_eval)
        accuracy_custom_final = correct_final / total_final if total_final > 0 else 0.0

        print(
            f"Final Custom Test Set Accuracy (PUDF QLoRA Qwen model): {accuracy_custom_final:.4f} ({correct_final}/{total_final})")
        summary_test_final_qwen = {
            "best_val_sft_loss_pudf": best_val_sft_loss_pudf,
            "final_test_acc_custom_pudf": accuracy_custom_final,
            "correct_final": correct_final,
            "total_final": total_final
        }
        with open(os.path.join(output_dir_pudf_main, "final_pudf_qwen_sft_test_summary.json"), 'w') as f:
            json.dump(summary_test_final_qwen, f, indent=4)
else:
    print(f"Best PUDF QLoRA adapter not found at {best_adapter_pudf_path} or test set issues. Skipping final test.")

if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"\nScript finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Outputs saved in: {output_dir_pudf_main}")