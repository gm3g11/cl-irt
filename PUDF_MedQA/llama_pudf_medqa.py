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
import evaluate

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
    start_time_irt = time.time()
    difficulties_np = np.array(difficulties, dtype=float)
    response_pattern_np = np.array(response_pattern, dtype=float)
    if len(difficulties_np) == 0 or len(difficulties_np) != len(response_pattern_np):
        return initial_theta_val, time.time() - start_time_irt
    valid_indices = ~np.isnan(difficulties_np)
    difficulties_filt, response_pattern_filt = difficulties_np[valid_indices], response_pattern_np[valid_indices]

    # Corrected block from user feedback
    if len(difficulties_filt) == 0:
        print(
            "  calculate_theta_irt: No valid (non-NaN) difficulties to estimate theta after filtering. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt

    student_prior = norm(loc=0., scale=1.)
    if num_obs > 0 and len(difficulties_filt) > num_obs:
        idx = np.random.choice(len(difficulties_filt), num_obs, replace=False)
        difficulties_sample, response_pattern_sample = difficulties_filt[idx], response_pattern_filt[idx]
    else:
        difficulties_sample, response_pattern_sample = difficulties_filt, response_pattern_filt
    if len(difficulties_sample) == 0:
        print(
            "  calculate_theta_irt: No samples to estimate theta after potential sampling. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt
    fn_min = theta_fn_irt(difficulties_sample, student_prior, response_pattern_sample)
    res = minimize(fn_min, [initial_theta_val], method='Nelder-Mead', options={'xtol': 1e-4, 'ftol': 1e-4})
    est_theta = res['x'][0]
    if np.isnan(est_theta) or np.isinf(est_theta):
        print(
            f"  calculate_theta_irt: Estimated theta is NaN/Inf. Optimizer success: {res.success}, Message: {res.message}. Returning initial_theta_val.")
        est_theta = initial_theta_val
    return est_theta, time.time() - start_time_irt


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
        return full_hf_train_dataset.select([])
    min_diff_target = capacity_theta + lower_offset;
    max_diff_target = capacity_theta + upper_offset
    selected_hf_dataset = full_hf_train_dataset.filter(
        lambda x: x[difficulty_col] is not None and \
                  min_diff_target <= x[difficulty_col] < max_diff_target, load_from_cache_file=False)
    print(f"  Initially selected {len(selected_hf_dataset)} samples by difficulty window.")
    if len(selected_hf_dataset) < min_samples_per_epoch:
        print(f"  Selected data ({len(selected_hf_dataset)}) < min_samples ({min_samples_per_epoch}).")
        if len(full_hf_train_dataset) == 0: return selected_hf_dataset
        num_to_take = min(min_samples_per_epoch, len(full_hf_train_dataset))
        print(f"  Taking {num_to_take} samples via '{pudf_ordering}' ordering.")
        dataset_to_sort = full_hf_train_dataset.filter(lambda x: x[difficulty_col] is not None,
                                                       load_from_cache_file=False)
        if len(dataset_to_sort) == 0:
            print(f"  Warning: All difficulties were None for sorting. Cannot fulfill min_samples with sorting.")
            return selected_hf_dataset
        reverse_sort = True if pudf_ordering == 'hardest' else False
        try:
            sorted_ds = dataset_to_sort.sort(difficulty_col, reverse=reverse_sort, load_from_cache_file=False)
            actual_take = min(num_to_take, len(sorted_ds))
            selected_hf_dataset = sorted_ds.select(range(actual_take))
        except Exception as e:
            print(f"  Error sorting for min_samples: {e}.")
    print(f"  Final samples for epoch: {len(selected_hf_dataset)}")
    return selected_hf_dataset


# --- End PUDF Data Selection ---

# ----- Configuration -----
model_id = "meta-llama/Meta-Llama-3.1-8B"
dataset_id = "GBaker/MedQA-USMLE-4-options"

max_seq_length_sft = 512 + 10
max_prompt_len_for_sft_preprocess = 512
max_target_len_for_sft_preprocess = 5

per_device_train_bs = 8
grad_accum_steps = 2
num_pudf_epochs = 10
early_stopping_patience_pudf = 3
learning_rate = 2e-4
weight_decay_train = 0.01

lora_r = 16;
lora_alpha = 16;
lora_dropout = 0.05
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

random_seed = 63
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"BF16 enabled: {bf16_enabled}")

per_device_eval_bs_pudf = 16
max_new_tokens_for_choice_pred = 1

ANSWER_MAP_KEYS = ["A", "B", "C", "D"]
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS)  # <<< *** THE FIX IS HERE ***

DIFFICULTY_FILE_PATH = MEDQA_DIFFICULTY_FILE
DIFFICULTY_JSON_KEY = "diff";
INITIAL_CAPACITY_THETA = 0.0;
NUM_OBS_THETA_ESTIMATION = -1
PUDF_LOWER_OFFSET = -float('inf');
PUDF_UPPER_OFFSET = 0.0
PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH = 100;
PUDF_ORDERING = 'easiest'

current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
run_name = f"llama3_medqa_sft_PUDF_qlora_{current_time_str}_ep{num_pudf_epochs}"
output_dir_pudf = f"./{run_name}"
best_adapter_pudf_path = os.path.join(output_dir_pudf, "best_pudf_qlora_adapter")
os.makedirs(output_dir_pudf, exist_ok=True);
os.makedirs(best_adapter_pudf_path, exist_ok=True)

torch.manual_seed(random_seed);
np.random.seed(random_seed);
random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
print(f"Output directory: {output_dir_pudf}")
print(f"Using device: {DEVICE}, Random seed: {random_seed}")


def load_difficulties_from_file(difficulty_file_path, difficulty_key):
    print(f"Loading difficulty scores from: {difficulty_file_path}")
    try:
        with open(difficulty_file_path, 'r') as f:
            data = json.load(f)
        scores = data[difficulty_key] if difficulty_key in data else data if isinstance(data, list) else None
        if scores is None: raise KeyError(f"Key '{difficulty_key}' not found or data not a list.")
        return np.array(scores, dtype=float)
    except Exception as e:
        print(f"Error loading difficulty scores: {e}"); traceback.print_exc(); raise


print(f"Loading dataset: {dataset_id}")
dataset_raw = load_dataset(dataset_id)
dataset_filtered = dataset_raw.filter(
    lambda ex: ex["answer_idx"] and ex["answer_idx"].strip().upper() in ANSWER_MAP_KEYS)

if 'train' not in dataset_filtered or len(dataset_filtered["train"]) == 0:
    print("Error: Dataset must contain non-empty 'train' split after filtering.");
    sys.exit(1)

original_train_for_diff = dataset_filtered['train']
difficulty_scores = load_difficulties_from_file(DIFFICULTY_FILE_PATH, DIFFICULTY_JSON_KEY)
if len(difficulty_scores) != len(original_train_for_diff):
    raise ValueError(
        f"Mismatch! Diff scores ({len(difficulty_scores)}) != original train ({len(original_train_for_diff)}).")
train_full_with_diff = original_train_for_diff.add_column("difficulty", difficulty_scores)
print(f"Added 'difficulty' to original 'train'. Columns: {train_full_with_diff.column_names}")

pudf_test_pre_sft = dataset_filtered['test']
if "validation" not in dataset_filtered or len(dataset_filtered["validation"]) == 0:
    print("Validation split not found/empty. Splitting 'train' (with diff) into 80/20...")
    train_val_split = train_full_with_diff.train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
    pudf_train_raw = train_val_split['train']
    pudf_validation_raw = train_val_split['test']
else:
    print("Using predefined 'validation' split.")
    pudf_train_raw = train_full_with_diff
    pudf_validation_raw = dataset_filtered['validation']
    if 'difficulty' not in pudf_validation_raw.column_names:
        print(f"CRITICAL WARNING: Predefined 'validation' lacks 'difficulty'. IRT theta on it will be impaired.")

pudf_dataset_pre_sft = DatasetDict({
    'train': pudf_train_raw, 'validation': pudf_validation_raw, 'test': pudf_test_pre_sft
})
print("\nPUDF dataset splits (pre-SFT formatting):");
for n, ds in pudf_dataset_pre_sft.items():
    print(f"  {n}: {len(ds)} examples, cols: {ds.column_names}")
    if 'difficulty' in ds.column_names and n in ['train', 'validation']:
        num_nans = np.sum(np.isnan(np.array(ds['difficulty'])))
        if num_nans > 0: print(f"    WARNING: '{n}' has {num_nans} NaN in 'difficulty'.")

print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
original_vocab_size = len(tokenizer)
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    new_pad_token_str = "␂"
    if new_pad_token_str not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": new_pad_token_str})
        print(f"Added NEW special pad_token: '{new_pad_token_str}', ID: {tokenizer.pad_token_id}")
    else:
        tokenizer.pad_token = new_pad_token_str
        print(f"Set pad_token to EXISTING token in vocab: '{new_pad_token_str}', ID: {tokenizer.pad_token_id}")
elif tokenizer.pad_token != "␂":
    print(f"Tokenizer already has a pad_token: '{tokenizer.pad_token}'. Using it.")
if tokenizer.pad_token and tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
print(
    f"Tokenizer: Current vocab: {len(tokenizer)}. Pad: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}, EOS: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")


def preprocess_sft_letter_target_pudf(examples_batch, tokenizer_ref):
    inputs_tokenized_batch, labels_tokenized_batch, difficulties_batch = [], [], []
    num_examples_in_batch = len(examples_batch["question"])
    for i in range(num_examples_in_batch):
        question, options_dict, answer_idx_key = examples_batch["question"][i].strip(), examples_batch["options"][i], \
        examples_batch["answer_idx"][i]
        prompt_parts = [f"Question: {question}\n\nOptions:"]
        if isinstance(options_dict, dict):
            [prompt_parts.append(f"{k}) {options_dict.get(k, '')}") for k in ANSWER_MAP_KEYS]
        else:
            [prompt_parts.append(f"{k}) [Invalid Opt]") for k in ANSWER_MAP_KEYS]
        prompt_parts.append("\nAnswer:")
        prompt_text = "\n".join(prompt_parts)
        target_letter = answer_idx_key.strip().upper() if answer_idx_key.strip().upper() in ANSWER_MAP_KEYS else ""
        tokenized_prompt = tokenizer_ref(prompt_text, truncation=True, max_length=max_prompt_len_for_sft_preprocess,
                                         padding=False, add_special_tokens=True)
        tokenized_target = tokenizer_ref(target_letter, truncation=True, max_length=max_target_len_for_sft_preprocess,
                                         padding=False, add_special_tokens=False)
        prompt_ids, target_ids = tokenized_prompt.input_ids, tokenized_target.input_ids
        input_ids, labels = prompt_ids + target_ids, ([-100] * len(prompt_ids)) + target_ids
        if tokenizer_ref.eos_token_id is not None: input_ids.append(tokenizer_ref.eos_token_id); labels.append(
            tokenizer_ref.eos_token_id)
        input_ids, labels = input_ids[:max_seq_length_sft], labels[:max_seq_length_sft]
        inputs_tokenized_batch.append(input_ids);
        labels_tokenized_batch.append(labels)
        if "difficulty" in examples_batch: difficulties_batch.append(examples_batch["difficulty"][i])
    output_dict = {"input_ids": inputs_tokenized_batch, "labels": labels_tokenized_batch}
    if difficulties_batch and "difficulty" in examples_batch: output_dict[
        "difficulty"] = difficulties_batch  # Ensure key matches input
    return output_dict


print(f"\nTokenizing dataset for SFT (target: letter). Max SFT seq_len: {max_seq_length_sft}")
tokenized_sft_datasets = pudf_dataset_pre_sft.map(
    lambda ex: preprocess_sft_letter_target_pudf(ex, tokenizer), batched=True,
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    remove_columns=[c for c in pudf_dataset_pre_sft["train"].column_names if c not in ['difficulty']]
)
print("SFT Tokenization complete.")
for split_n, ds_s in tokenized_sft_datasets.items():
    print(f"Cols SFT tokenized '{split_n}': {ds_s.column_names}")
    if 'difficulty' not in ds_s.column_names and split_n != 'test':
        print(f"  CRITICAL WARNING: SFT '{split_n}' DOES NOT have 'difficulty'.")

data_collator_sft = DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100, padding="longest")

print("\nConfiguring BitsAndBytes for QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
print(f"Loading base model {model_id} for QLoRA PUDF training...")
base_model_for_pudf = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16 if bf16_enabled else torch.float16
)
current_model_vocab_size = base_model_for_pudf.get_input_embeddings().weight.size(0)
if len(tokenizer) > current_model_vocab_size:
    print(f"Resizing base model token embeddings: {current_model_vocab_size} -> {len(tokenizer)}")
    base_model_for_pudf.resize_token_embeddings(len(tokenizer))
if base_model_for_pudf.config.pad_token_id != tokenizer.pad_token_id and tokenizer.pad_token_id is not None:
    print(f"Syncing base model pad_token_id to {tokenizer.pad_token_id}")
    base_model_for_pudf.config.pad_token_id = tokenizer.pad_token_id

base_model_for_pudf = prepare_model_for_kbit_training(base_model_for_pudf, use_gradient_checkpointing=True)
peft_config_pudf = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
                              lora_dropout=lora_dropout, target_modules=lora_target_modules, bias="none")
pudf_peft_model = get_peft_model(base_model_for_pudf, peft_config_pudf)
print("QLoRA Causal LM PEFT model prepared for PUDF training.")
pudf_peft_model.print_trainable_parameters()

optimizer_pudf = AdamW(pudf_peft_model.parameters(), lr=learning_rate, weight_decay=weight_decay_train)
num_total_train_items_pudf = len(tokenized_sft_datasets['train'])
steps_per_pudf_outer_epoch_approx = (num_total_train_items_pudf * 0.75) // (per_device_train_bs * grad_accum_steps)
total_pudf_training_steps_approx = int(steps_per_pudf_outer_epoch_approx * num_pudf_epochs)
if total_pudf_training_steps_approx == 0: total_pudf_training_steps_approx = num_pudf_epochs
lr_scheduler_pudf = get_linear_schedule_with_warmup(optimizer_pudf,
                                                    num_warmup_steps=int(0.1 * total_pudf_training_steps_approx),
                                                    num_training_steps=max(1, total_pudf_training_steps_approx))
scaler_pudf = TorchAmpGradScaler(enabled=bf16_enabled)

letter_tokens_for_eval = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in ANSWER_MAP_KEYS
                          if len(tokenizer.encode(L, add_special_tokens=False)) == 1}
print(f"Letter token IDs for eval: {letter_tokens_for_eval}")
# This is where the error occurred. NUM_CHOICES_MC is now defined.
if len(letter_tokens_for_eval) != NUM_CHOICES_MC:
    print(f"Warning! Could not get single token IDs for all {NUM_CHOICES_MC} answer letters: {letter_tokens_for_eval}")


def create_custom_evaluation_prompt(example):  # From baseline
    question = example["question"].strip();
    options_dict = example["options"]
    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        [prompt_parts.append(f"{k}) {options_dict.get(k, '')}") for k in ANSWER_MAP_KEYS]
    else:
        [prompt_parts.append(f"{k}) [Invalid Opt]") for k in ANSWER_MAP_KEYS]
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


def evaluate_llama_sft_and_estimate_theta(
        peft_model_eval, validation_sft_tokenized_ds, original_raw_validation_ds,
        tokenizer_eval, device_eval, epoch_curr, theta_init_calc,
        eval_prompt_max_len, letter_token_ids, max_new_toks_choice):
    peft_model_eval.eval();
    print(f"  E{epoch_curr} Val: Calculating SFT LM loss...")
    sft_val_cols = ['input_ids', 'attention_mask', 'labels']
    current_sft_val_ds_cols = list(validation_sft_tokenized_ds.column_names)
    cols_to_set_format = [col for col in sft_val_cols if col in current_sft_val_ds_cols]
    validation_sft_tokenized_ds.set_format(type='torch', columns=cols_to_set_format)  # Ensure only existing cols
    sft_val_dataloader = DataLoader(validation_sft_tokenized_ds, batch_size=per_device_eval_bs_pudf,
                                    collate_fn=data_collator_sft, shuffle=False)
    total_sft_loss, num_sft_batches = 0, 0
    for sft_batch in tqdm(sft_val_dataloader, desc=f"  E{epoch_curr} SFT Val Loss", leave=False):
        sft_batch = {k: v.to(device_eval) for k, v in sft_batch.items() if k in cols_to_set_format}
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=bf16_enabled):
            outputs = peft_model_eval(**sft_batch)
            total_sft_loss += outputs.loss.item();
            num_sft_batches += 1
    avg_sft_loss = total_sft_loss / num_sft_batches if num_sft_batches > 0 else float('nan')
    perplexity = np.exp(avg_sft_loss) if not np.isnan(avg_sft_loss) else float('inf')
    print(f"  E{epoch_curr} Val: Avg SFT Loss = {avg_sft_loss:.4f}, Perplexity = {perplexity:.4f}")

    print(f"  E{epoch_curr} Val: Predicting choices for Theta estimation...")
    item_diffs_theta, resp_patt_theta = [], []
    has_diff_irt_val = "difficulty" in original_raw_validation_ds.column_names
    if not has_diff_irt_val:
        print(
            f"  E{epoch_curr} Val: 'difficulty' missing in original_raw_validation_ds for IRT. Theta defaults to initial.")
        return avg_sft_loss, perplexity, theta_init_calc, 0.0
    prompts_for_gen = [create_custom_evaluation_prompt(ex) for ex in original_raw_validation_ds]
    true_letters_for_gen = [ex["answer_idx"].strip().upper() for ex in original_raw_validation_ds]
    difficulties_for_gen = original_raw_validation_ds["difficulty"] if has_diff_irt_val else [np.nan] * len(
        original_raw_validation_ds)
    time_theta_start = time.time()
    original_padding_side = tokenizer_eval.padding_side;
    tokenizer_eval.padding_side = "left"
    for i in tqdm(range(0, len(prompts_for_gen), per_device_eval_bs_pudf), desc=f"  E{epoch_curr} Theta Pred",
                  leave=False):
        batch_prompts, batch_true_letters, batch_difficulties = prompts_for_gen[
                                                                i:i + per_device_eval_bs_pudf], true_letters_for_gen[
                                                                                                i:i + per_device_eval_bs_pudf], difficulties_for_gen[
                                                                                                                                i:i + per_device_eval_bs_pudf]
        inputs = tokenizer_eval(batch_prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=eval_prompt_max_len).to(device_eval)
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=bf16_enabled):
            outputs_gen = peft_model_eval(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            next_token_logits = outputs_gen.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).cpu()
        for j_idx, single_probs in enumerate(next_token_probs):
            if np.isnan(batch_difficulties[j_idx]): continue
            choice_scores = np.zeros(NUM_CHOICES_MC, dtype=float)  # NUM_CHOICES_MC is now defined
            for choice_k_idx, key_char in enumerate(ANSWER_MAP_KEYS):
                token_id = letter_token_ids.get(key_char, -1)
                if token_id != -1: choice_scores[choice_k_idx] = single_probs[token_id].item()
            predicted_choice_idx = np.argmax(choice_scores)
            predicted_letter = ANSWER_MAP_KEYS[predicted_choice_idx]
            is_correct = 1 if predicted_letter == batch_true_letters[j_idx] else -1
            resp_patt_theta.append(is_correct);
            item_diffs_theta.append(batch_difficulties[j_idx])
    tokenizer_eval.padding_side = original_padding_side
    theta_est_time = time.time() - time_theta_start
    final_theta_est = theta_init_calc
    if item_diffs_theta and resp_patt_theta:
        final_theta_est, _ = calculate_theta_irt(item_diffs_theta, resp_patt_theta, NUM_OBS_THETA_ESTIMATION,
                                                 theta_init_calc)
    else:
        print(f"  E{epoch_curr} Val: No valid data for IRT theta. Using initial.")
    return avg_sft_loss, perplexity, final_theta_est, theta_est_time


# ----- PUDF Training Loop for Llama QLoRA SFT -----
print("\nStarting Llama SFT QLoRA PUDF training loop...")
current_cap_theta = INITIAL_CAPACITY_THETA;
best_val_sft_loss_pudf = float('inf');
early_stop_counter = 0
all_pudf_stats_llama = []
full_tokenized_sft_train_hf = tokenized_sft_datasets['train']

for pudf_epoch_idx in range(num_pudf_epochs):
    epoch_time_start = time.time()
    print(f"\n===== PUDF Epoch {pudf_epoch_idx + 1}/{num_pudf_epochs} =====")
    print(f"  Estimating capacity & SFT val loss (theta_start_epoch={current_cap_theta:.4f})...")
    sft_val_loss, sft_val_ppl, new_theta, theta_est_time = evaluate_llama_sft_and_estimate_theta(
        pudf_peft_model, tokenized_sft_datasets['validation'], pudf_dataset_pre_sft['validation'],
        tokenizer, DEVICE, pudf_epoch_idx + 1, current_cap_theta,
        max_prompt_len_for_sft_preprocess, letter_tokens_for_eval, max_new_tokens_for_choice_pred
    )
    if not np.isnan(new_theta) and new_theta <= current_cap_theta and pudf_epoch_idx > 0 and not np.isnan(
            current_cap_theta):
        current_cap_theta += 0.05;
        print(f"  Theta nudge. New cap: {current_cap_theta:.4f}")
    elif not np.isnan(new_theta):
        current_cap_theta = new_theta
    else:
        print(f"  Theta est NaN. Keeping prev: {current_cap_theta:.4f}")
    print(
        f"  E{pudf_epoch_idx + 1}: Capacity(Theta) for selection = {current_cap_theta:.4f} (est_time={theta_est_time:.2f}s)")
    print(f"  E{pudf_epoch_idx + 1}: Val SFT Loss (pre-train) = {sft_val_loss:.4f}, PPL = {sft_val_ppl:.4f}")

    epoch_sft_train_data_hf = select_data_for_pudf_epoch(full_tokenized_sft_train_hf, current_cap_theta,
                                                         'difficulty', PUDF_ORDERING, PUDF_LOWER_OFFSET,
                                                         PUDF_UPPER_OFFSET, PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH)
    num_selected_sft_samples = len(epoch_sft_train_data_hf);
    avg_inner_sft_loss = float('nan')

    if num_selected_sft_samples > 0:
        cols_for_sft_loader_train = ['input_ids', 'attention_mask', 'labels']
        current_epoch_train_cols = list(epoch_sft_train_data_hf.column_names)
        cols_to_set_format_train = [col for col in cols_for_sft_loader_train if col in current_epoch_train_cols]
        epoch_sft_train_data_hf.set_format(type='torch', columns=cols_to_set_format_train)
        inner_sft_dataloader = DataLoader(epoch_sft_train_data_hf, batch_size=per_device_train_bs,
                                          collate_fn=data_collator_sft, shuffle=True,
                                          num_workers=min(2, os.cpu_count() if os.cpu_count() else 1))
        pudf_peft_model.train();
        total_inner_loss, grad_steps_inner = 0, 0
        for step, sft_batch_train in enumerate(
                tqdm(inner_sft_dataloader, desc=f"  InnerTrain SFT E{pudf_epoch_idx + 1}", leave=False)):
            optimizer_pudf.zero_grad()
            sft_batch_train = {k: v.to(DEVICE) for k, v in sft_batch_train.items() if k in cols_to_set_format_train}
            with torch_amp_autocast(device_type=DEVICE.type, enabled=bf16_enabled):
                outputs_train = pudf_peft_model(**sft_batch_train);
                loss_train = outputs_train.loss / grad_accum_steps
            scaler_pudf.scale(loss_train).backward()
            total_inner_loss += loss_train.item() * grad_accum_steps
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(inner_sft_dataloader):
                scaler_pudf.unscale_(optimizer_pudf);
                torch.nn.utils.clip_grad_norm_(pudf_peft_model.parameters(), 1.0)
                scaler_pudf.step(optimizer_pudf);
                scaler_pudf.update();
                lr_scheduler_pudf.step();
                optimizer_pudf.zero_grad()
                grad_steps_inner += 1
        avg_inner_sft_loss = total_inner_loss / grad_steps_inner if grad_steps_inner > 0 else float('nan')
        print(f"  E{pudf_epoch_idx + 1}: Inner SFT train avg loss = {avg_inner_sft_loss:.4f}")
    else:
        print(f"  E{pudf_epoch_idx + 1}: No SFT data selected. Skipping inner train.")

    sft_val_loss_post, sft_val_ppl_post, theta_val_post, _ = evaluate_llama_sft_and_estimate_theta(
        pudf_peft_model, tokenized_sft_datasets['validation'], pudf_dataset_pre_sft['validation'],
        tokenizer, DEVICE, pudf_epoch_idx + 1, current_cap_theta,
        max_prompt_len_for_sft_preprocess, letter_tokens_for_eval, max_new_tokens_for_choice_pred)
    print(f"  E{pudf_epoch_idx + 1}: Val SFT Loss (post-train) = {sft_val_loss_post:.4f}, PPL = {sft_val_ppl_post:.4f}")
    epoch_time_end = time.time() - epoch_time_start
    all_pudf_stats_llama.append({
        "pudf_epoch": pudf_epoch_idx + 1, "cap_theta_select": current_cap_theta,
        "num_sel_sft_train": num_selected_sft_samples, "avg_inner_sft_loss": avg_inner_sft_loss,
        "val_sft_loss": sft_val_loss_post, "val_sft_ppl": sft_val_ppl_post,
        "theta_val_post_train": theta_val_post, "duration_s": epoch_time_end, "theta_est_time_s": theta_est_time})
    if not np.isnan(sft_val_loss_post) and sft_val_loss_post < best_val_sft_loss_pudf:
        print(f"  New best val_sft_loss: {sft_val_loss_post:.4f} (prev {best_val_sft_loss_pudf:.4f}). Saving adapter.")
        best_val_sft_loss_pudf = sft_val_loss_post;
        early_stop_counter = 0
        pudf_peft_model.save_pretrained(best_adapter_pudf_path);
        tokenizer.save_pretrained(best_adapter_pudf_path)
    elif not np.isnan(sft_val_loss_post):  # Only increment if val_loss is valid
        early_stop_counter += 1;
        print(f"  Val SFT loss not improved. EarlyStop: {early_stop_counter}/{early_stopping_patience_pudf}")
    else:  # val_loss is NaN, don't update best, but count as non-improvement for early stopping
        early_stop_counter += 1;
        print(f"  Val SFT loss is NaN. EarlyStop: {early_stop_counter}/{early_stopping_patience_pudf}")

    if early_stop_counter >= early_stopping_patience_pudf: print(f"  Early stopping at E{pudf_epoch_idx + 1}."); break
    print(f"  PUDF E{pudf_epoch_idx + 1} ended. Duration: {epoch_time_end:.2f}s")

print(f"\nPUDF Llama SFT Training finished. Best val_sft_loss: {best_val_sft_loss_pudf:.4f}")
stats_file_pudf_llama = os.path.join(output_dir_pudf, "pudf_llama_sft_training_stats.json")
with open(stats_file_pudf_llama, 'w') as f: json.dump(all_pudf_stats_llama, f, indent=4,
                                                      default=lambda x: float(x) if isinstance(x, (
                                                      np.float32, np.float64, np.float_)) else "NaN" if isinstance(x,
                                                                                                                   float) and np.isnan(
                                                          x) else x)
print(f"PUDF Llama SFT stats saved: {stats_file_pudf_llama}")

if os.path.exists(best_adapter_pudf_path) and \
        (os.path.exists(os.path.join(best_adapter_pudf_path, "adapter_model.safetensors")) or \
         os.path.exists(os.path.join(best_adapter_pudf_path, "adapter_model.bin"))):
    print(f"\nLoading base model with best PUDF QLoRA adapter from {best_adapter_pudf_path} for final test eval...")
    base_model_final_eval = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                                                 torch_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
                                                                 device_map="auto", trust_remote_code=True)
    current_final_eval_vocab = base_model_final_eval.get_input_embeddings().weight.size(0)
    if len(tokenizer) > current_final_eval_vocab: base_model_final_eval.resize_token_embeddings(len(tokenizer))
    if base_model_final_eval.config.pad_token_id != tokenizer.pad_token_id and tokenizer.pad_token_id is not None:
        base_model_final_eval.config.pad_token_id = tokenizer.pad_token_id
    final_peft_model_eval = PeftModel.from_pretrained(base_model_final_eval, best_adapter_pudf_path)
    final_peft_model_eval.eval();
    print("PEFT model for final evaluation loaded and set to eval mode.")
    test_set_raw_final_eval = pudf_dataset_pre_sft["test"]
    all_prompts_final_eval = [create_custom_evaluation_prompt(ex) for ex in test_set_raw_final_eval]
    all_true_letters_final_eval = [ex["answer_idx"].strip().upper() for ex in test_set_raw_final_eval]
    all_predicted_letters_final_eval = []
    original_padding_side_final_eval = tokenizer.padding_side;
    tokenizer.padding_side = "left"
    with torch.inference_mode():
        for i in tqdm(range(0, len(all_prompts_final_eval), per_device_eval_bs_pudf), desc="Final Test Custom Eval",
                      leave=False):
            batch_prompts_final = all_prompts_final_eval[i: i + per_device_eval_bs_pudf]
            inputs_final_test = tokenizer(batch_prompts_final, return_tensors="pt", padding=True, truncation=True,
                                          max_length=max_prompt_len_for_sft_preprocess).to(DEVICE)
            generated_ids_final = final_peft_model_eval.generate(
                input_ids=inputs_final_test['input_ids'], attention_mask=inputs_final_test['attention_mask'],
                max_new_tokens=max_new_tokens_for_choice_pred, pad_token_id=tokenizer.pad_token_id,
                temperature=0.1, top_p=0.9, do_sample=False)
            for j_idx_final, gen_ids_sample_final in enumerate(generated_ids_final):
                input_len_final = inputs_final_test['input_ids'].shape[1]
                # Handle potential empty generation or EOS before target token
                if len(gen_ids_sample_final) > input_len_final:
                    predicted_token_id_final = gen_ids_sample_final[input_len_final].item()
                    predicted_letter_final = next(
                        (L for L, tid in letter_tokens_for_eval.items() if tid == predicted_token_id_final), None)
                else:  # Model generated nothing new or only EOS right away.
                    predicted_letter_final = None
                all_predicted_letters_final_eval.append(predicted_letter_final)
    tokenizer.padding_side = original_padding_side_final_eval
    correct_final = sum(
        1 for pred, true in zip(all_predicted_letters_final_eval, all_true_letters_final_eval) if pred == true)
    total_final = len(all_true_letters_final_eval)
    accuracy_custom_final = correct_final / total_final if total_final > 0 else 0.0
    print(f"Final Custom Test Set Accuracy (PUDF QLoRA Llama model): {accuracy_custom_final:.4f}")
    summary_test_final_llama = {"best_val_sft_loss_pudf": best_val_sft_loss_pudf,
                                "final_test_acc_custom_pudf": accuracy_custom_final}
    with open(os.path.join(output_dir_pudf, "final_pudf_llama_sft_test_summary.json"), 'w') as f:
        json.dump(summary_test_final_llama, f, indent=4)
else:
    print(f"Best PUDF QLoRA adapter not found at {best_adapter_pudf_path}. Skipping final test.")

if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"\nScript finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Outputs saved in: {output_dir_pudf}")