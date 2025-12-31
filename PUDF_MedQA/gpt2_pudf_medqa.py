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

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from transformers.data.data_collator import DataCollatorForMultipleChoice  # Base for custom
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler as TorchAmpGradScaler, autocast as torch_amp_autocast
from evaluate import load as load_metric
from huggingface_hub import whoami

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
        print(
            f"  calculate_theta_irt: Invalid inputs (empty or mismatched lengths: diffs={len(difficulties_np)}, resps={len(response_pattern_np)}). Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt

    valid_indices = ~np.isnan(difficulties_np)
    difficulties_filt = difficulties_np[valid_indices]
    response_pattern_filt = response_pattern_np[valid_indices]

    # Corrected block:
    if len(difficulties_filt) == 0:
        print(
            "  calculate_theta_irt: No valid (non-NaN) difficulties to estimate theta after filtering. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt

    student_prior = norm(loc=0., scale=1.)
    if num_obs > 0 and len(difficulties_filt) > num_obs:
        idx = np.random.choice(len(difficulties_filt), num_obs, replace=False)
        difficulties_sample = difficulties_filt[idx]
        response_pattern_sample = response_pattern_filt[idx]
    else:
        difficulties_sample = difficulties_filt
        response_pattern_sample = response_pattern_filt

    if len(difficulties_sample) == 0:  # Should be redundant if above check is fine, but good safeguard
        print(
            "  calculate_theta_irt: No samples to estimate theta after potential sampling (e.g. num_obs=0). Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt

    fn_min = theta_fn_irt(difficulties_sample, student_prior, response_pattern_sample)
    res = minimize(fn_min, [initial_theta_val], method='Nelder-Mead')

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
    print(f"  Selecting data for PUDF epoch: capacity_theta={capacity_theta:.4f}, "
          f"difficulty window=[{capacity_theta + lower_offset:.4f}, {capacity_theta + upper_offset:.4f}), "
          f"min_samples={min_samples_per_epoch}")
    if difficulty_col not in full_hf_train_dataset.column_names:
        print(f"  Error: Difficulty column '{difficulty_col}' not found. Returning empty selection.")
        return full_hf_train_dataset.select([])
    min_diff_target = capacity_theta + lower_offset;
    max_diff_target = capacity_theta + upper_offset
    selected_hf_dataset = full_hf_train_dataset.filter(
        lambda x: x[difficulty_col] is not None and \
                  min_diff_target <= x[difficulty_col] < max_diff_target, load_from_cache_file=False)
    print(f"  Initially selected {len(selected_hf_dataset)} samples based on difficulty window.")
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

# --- Custom Data Collator (Only needed for DeBERTa-like MC, not GPT2 LM SFT) ---
# For GPT2 SFT, DataCollatorForLanguageModeling is used and doesn't need 'difficulty'.
# If 'difficulty' is passed in a batch to it, it should ideally ignore it.
# The DeBERTa multiple choice needed a custom one due to its specific flattening logic.

# ----- Environment Setup -----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_HOME_SPECIFIED = HF_HOME
HF_HOME = HF_HOME_SPECIFIED if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(
    HF_HOME_SPECIFIED) else os.path.expanduser("~/.cache/huggingface")
if HF_HOME != HF_HOME_SPECIFIED: print(
    f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' not found. Using default.")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ----- Random Seed -----
random_seed = 63
torch.manual_seed(random_seed);
np.random.seed(random_seed);
random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
print(f"Using random seed: {random_seed}")

# ----- Config -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_name_hf = "gpt2"
dataset_id = "GBaker/MedQA-USMLE-4-options"
MAX_SFT_SEQ_LENGTH = 512;
MAX_EVAL_PROMPT_LENGTH = 512
PER_DEVICE_TRAIN_BATCH_SIZE_SFT = 4;
PER_DEVICE_EVAL_BATCH_SIZE_SFT = 8
GRAD_ACCUM_STEPS_SFT = 8
NUM_PUDF_EPOCHS = 20
LEARNING_RATE_SFT = 5e-5;
WEIGHT_DECAY_SFT = 0.01
EARLY_STOPPING_PATIENCE_SFT = 3

current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
run_name = f"gpt2_medqa_sft_PUDF_{current_time_str}_trainbs{PER_DEVICE_TRAIN_BATCH_SIZE_SFT}_epochs{NUM_PUDF_EPOCHS}"
output_dir_base = f"./{run_name}_output"
best_model_pudf_path = os.path.join(output_dir_base, "best_pudf_model")
os.makedirs(output_dir_base, exist_ok=True);
os.makedirs(best_model_pudf_path, exist_ok=True)

ANSWER_MAP_KEYS = ['A', 'B', 'C', 'D'];
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DIFFICULTY_FILE_PATH = MEDQA_DIFFICULTY_FILE
DIFFICULTY_JSON_KEY = "diff";
INITIAL_CAPACITY_THETA = 0.0;
NUM_OBS_THETA_ESTIMATION = -1
PUDF_LOWER_OFFSET = -float('inf');
PUDF_UPPER_OFFSET = 0.0
PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH = 100;
PUDF_ORDERING = 'easiest'

fp16_enabled = torch.cuda.is_available()  # Define for scaler and autocast


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
dataset_full_raw = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])
dataset_full_filtered = dataset_full_raw.filter(
    lambda ex: ex["answer_idx"] is not None and ex["answer_idx"].strip().upper() in ANSWER_MAP_KEYS)

if 'train' not in dataset_full_filtered or len(dataset_full_filtered["train"]) == 0:
    print("Error: Dataset must contain non-empty 'train' split after filtering.");
    sys.exit(1)

original_train_hf_for_diff = dataset_full_filtered['train']
difficulty_scores = load_difficulties_from_file(DIFFICULTY_FILE_PATH, DIFFICULTY_JSON_KEY)
if len(difficulty_scores) != len(original_train_hf_for_diff):
    raise ValueError(
        f"Mismatch! Diff scores ({len(difficulty_scores)}) != original train ({len(original_train_hf_for_diff)}).")
train_with_difficulties_added = original_train_hf_for_diff.add_column("difficulty", difficulty_scores)
print(f"Added 'difficulty' to original 'train'. Columns: {train_with_difficulties_added.column_names}")

pudf_test_dataset_pre_sft = dataset_full_filtered['test']
if 'validation' not in dataset_full_filtered:
    print("Validation split not found. Splitting 'train' (with diff) into 80/20...")
    train_val_split_pudf = train_with_difficulties_added.train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
    pudf_train_raw = train_val_split_pudf['train']
    pudf_validation_raw = train_val_split_pudf['test']
else:
    print("Using predefined 'validation' split. 'train' has difficulties.")
    pudf_train_raw = train_with_difficulties_added
    pudf_validation_raw = dataset_full_filtered['validation']
    if 'difficulty' not in pudf_validation_raw.column_names:
        print(f"CRITICAL WARNING: Predefined 'validation' lacks 'difficulty'. IRT theta on it will be impaired.")

pudf_dataset_pre_sft = DatasetDict({
    'train': pudf_train_raw, 'validation': pudf_validation_raw,
    'test': pudf_test_dataset_pre_sft
})
print("\nPUDF dataset splits (pre-SFT formatting):");
for n, ds in pudf_dataset_pre_sft.items():
    print(f"  {n}: {len(ds)} examples, cols: {ds.column_names}")
    if 'difficulty' in ds.column_names and n in ['train', 'validation']:
        num_nans = np.isnan(np.array(ds['difficulty'])).sum()
        if num_nans > 0: print(f"    WARNING: '{n}' has {num_nans} NaN in 'difficulty'.")

print(f"Loading {model_name_hf} tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name_hf, cache_dir=os.environ["TRANSFORMERS_CACHE"])
added_pad_token = False
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    if tokenizer.pad_token == '[PAD]' and tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': '[PAD]'});
        added_pad_token = True
    print(f"Set tokenizer.pad_token to: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
tokenizer.padding_side = "right"


def create_sft_text_gpt2(example_dict):
    q, opts, ans_key = example_dict["question"], example_dict["options"], example_dict["answer_idx"].strip().upper()
    prompt = [f"Question: {q}\n\nOptions:"]
    if isinstance(opts, dict):
        [prompt.append(f"{k}) {opts.get(k, '')}") for k in ANSWER_MAP_KEYS]
    else:
        [prompt.append(f"{k}) [Invalid Opt]") for k in ANSWER_MAP_KEYS]
    prompt.append("\nAnswer:")
    return "\n".join(prompt) + " " + ans_key


def preprocess_sft_map_function(examples_batch):
    texts_for_sft, difficulties_batch = [], []
    batch_len = len(examples_batch[next(iter(examples_batch))])
    for i in range(batch_len):
        single_example = {k: examples_batch[k][i] for k in examples_batch}
        texts_for_sft.append(create_sft_text_gpt2(single_example))
        if 'difficulty' in single_example: difficulties_batch.append(single_example['difficulty'])
    tokenized = tokenizer(texts_for_sft, max_length=MAX_SFT_SEQ_LENGTH, padding=False, truncation=True,
                          add_special_tokens=True)  # Ensure BOS/EOS if needed
    if difficulties_batch and 'difficulty' in examples_batch: tokenized['difficulty'] = difficulties_batch
    return tokenized


print("\nTokenizing dataset for SFT (PUDF)...");
num_proc_map = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
tokenized_sft_datasets = DatasetDict()
for split_n, ds_s in pudf_dataset_pre_sft.items():
    print(f"  Tokenizing {split_n}...");
    original_cols = list(ds_s.column_names)
    # Columns to remove are those not used by preprocess_sft_map_function (q, opts, ans_idx)
    # and not 'difficulty' which is handled inside preprocess_sft_map_function
    cols_to_remove_for_map = [c for c in original_cols if c not in ['question', 'options', 'answer_idx', 'difficulty']]
    temp_ds_map = ds_s.remove_columns(cols_to_remove_for_map) if cols_to_remove_for_map else ds_s

    # After map, only columns returned by preprocess_sft_map_function will exist (+ original columns not in remove_columns)
    # So, 'difficulty' will be there if returned by preprocess_sft_map_function.
    # 'input_ids', 'attention_mask' are returned by tokenizer.
    # `remove_columns` in `.map` refers to columns of `temp_ds_map` to drop *after* mapping if they are not in output of `preprocess_sft_map_function`.
    # It's safer to let `preprocess_sft_map_function` return all it needs, and `remove_columns` be empty or specific.
    # Here, we want to keep 'difficulty' from `preprocess_sft_map_function`'s output.
    tokenized_sft_datasets[split_n] = temp_ds_map.map(
        preprocess_sft_map_function, batched=True, num_proc=num_proc_map,
        remove_columns=[c for c in temp_ds_map.column_names if c not in ['difficulty']]
        # Keep only 'difficulty' from original columns if it wasn't explicitly returned by map_fn
        # and let map_fn define other output columns (input_ids etc)
    )
print("SFT Tokenization complete.")
for split_n, ds_s in tokenized_sft_datasets.items():
    print(f"Cols SFT tokenized '{split_n}': {ds_s.column_names}")
    if 'difficulty' not in ds_s.column_names and split_n != 'test':
        print(f"  CRITICAL WARNING: SFT '{split_n}' DOES NOT have 'difficulty'.")

data_collator_sft = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def load_model_gpt2(model_path_or_name, tokenizer_inst, new_pad_added):
    print(f"Loading GPT-2 model from {model_path_or_name}...")
    config = AutoConfig.from_pretrained(model_path_or_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if tokenizer_inst.pad_token_id is not None: config.pad_token_id = tokenizer_inst.pad_token_id
    # Ensure eos_token_id is also set if tokenizer has it, some GPT2 configs might miss it
    if tokenizer_inst.eos_token_id is not None and hasattr(config, "eos_token_id"):
        config.eos_token_id = tokenizer_inst.eos_token_id

    model_inst = GPT2LMHeadModel.from_pretrained(model_path_or_name, config=config,
                                                 cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if new_pad_added:  # Resize if new token was added
        current_embeds = model_inst.get_input_embeddings().weight.size(0)
        if len(tokenizer_inst) > current_embeds:
            print(f"Resizing model token embeddings: {current_embeds} -> {len(tokenizer_inst)}")
            model_inst.resize_token_embeddings(len(tokenizer_inst))
            # Ensure pad_token_id in model config is updated if it was part of resize implicitly
            if model_inst.config.pad_token_id != tokenizer_inst.pad_token_id and tokenizer_inst.pad_token_id is not None:
                model_inst.config.pad_token_id = tokenizer_inst.pad_token_id
    return model_inst


def evaluate_gpt2_sft_and_estimate_theta(
        model_eval, validation_sft_dataset_tokenized,
        original_validation_raw_dataset,
        tokenizer_eval, device_eval, epoch_curr, theta_init_calc,
        eval_prompt_max_len, letter_tok_ids):
    model_eval.eval();
    model_eval.to(device_eval)
    print(f"  E{epoch_curr} Val: Calculating SFT LM loss...")
    sft_val_dl_config = {'input_ids', 'attention_mask', 'labels'}  # What DataCollatorForLM expects
    validation_sft_dataset_tokenized.set_format(type='torch', columns=list(
        sft_val_dl_config.intersection(validation_sft_dataset_tokenized.column_names)))
    validation_sft_dataloader = DataLoader(validation_sft_dataset_tokenized,
                                           batch_size=PER_DEVICE_EVAL_BATCH_SIZE_SFT, collate_fn=data_collator_sft,
                                           shuffle=False)

    total_sft_eval_loss = 0;
    num_sft_eval_batches = 0
    for sft_batch in tqdm(validation_sft_dataloader, desc=f"  E{epoch_curr} SFT Val Loss", leave=False):
        sft_batch = {k: v.to(device_eval) for k, v in sft_batch.items()}
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=fp16_enabled):
            outputs = model_eval(**sft_batch)
            total_sft_eval_loss += outputs.loss.item();
            num_sft_eval_batches += 1
    avg_sft_eval_loss = total_sft_eval_loss / num_sft_eval_batches if num_sft_eval_batches > 0 else float('nan')
    perplexity = np.exp(avg_sft_eval_loss) if not np.isnan(avg_sft_eval_loss) else float('inf')
    print(f"  E{epoch_curr} Val: Avg SFT Loss = {avg_sft_eval_loss:.4f}, Perplexity = {perplexity:.4f}")

    print(f"  E{epoch_curr} Val: Predicting choices for Theta estimation...")
    item_diffs_theta, resp_patt_theta = [], []
    has_diff_irt = "difficulty" in validation_sft_dataset_tokenized.column_names and \
                   "difficulty" in original_validation_raw_dataset.column_names  # Redundant but safe
    if not has_diff_irt:
        print(f"  E{epoch_curr} Val: 'difficulty' missing for IRT. Theta defaults to initial.")
        return avg_sft_eval_loss, perplexity, theta_init_calc, 0.0

    prompts_gen, true_letters_gen, difficulties_gen = [], [], []
    # Ensure original_validation_raw_dataset has difficulty for alignment, or use validation_sft_dataset_tokenized['difficulty']
    # Assuming validation_sft_dataset_tokenized[i]['difficulty'] is aligned with original_validation_raw_dataset[i]
    # This was ensured if preprocess_sft_map_function correctly carries 'difficulty' for each example.

    for i in range(len(original_validation_raw_dataset)):
        raw_ex = original_validation_raw_dataset[i]
        q_text, opts_dict = raw_ex["question"], raw_ex["options"]
        prompt_parts = [f"Question: {q_text}\n\nOptions:"]
        if isinstance(opts_dict, dict):
            [prompt_parts.append(f"{k}) {opts_dict.get(k, '')}") for k in ANSWER_MAP_KEYS]
        else:
            [prompt_parts.append(f"{k}) [Invalid Opt]") for k in ANSWER_MAP_KEYS]
        prompt_parts.append("\nAnswer:")
        prompts_gen.append("\n".join(prompt_parts))
        true_letters_gen.append(raw_ex["answer_idx"].strip().upper())
        try:
            # This assumes pudf_dataset_pre_sft['validation'] was used to create tokenized_sft_datasets['validation']
            # and it had the difficulty column correctly.
            difficulties_gen.append(pudf_dataset_pre_sft['validation'][i]["difficulty"])
        except (KeyError, IndexError) as e:
            print(f"  Error accessing difficulty for validation example {i}: {e}. Skipping for theta.")
            difficulties_gen.append(np.nan)  # Mark as NaN to filter later

    time_theta_start = time.time()
    orig_pad_side = tokenizer_eval.padding_side;
    tokenizer_eval.padding_side = "left"
    for i in tqdm(range(0, len(prompts_gen), PER_DEVICE_EVAL_BATCH_SIZE_SFT), desc=f"  E{epoch_curr} Theta Pred",
                  leave=False):
        batch_prompts = prompts_gen[i: i + PER_DEVICE_EVAL_BATCH_SIZE_SFT]
        batch_true_letters = true_letters_gen[i: i + PER_DEVICE_EVAL_BATCH_SIZE_SFT]
        batch_difficulties = difficulties_gen[i: i + PER_DEVICE_EVAL_BATCH_SIZE_SFT]
        inputs = tokenizer_eval(batch_prompts, return_tensors="pt", padding="longest",
                                truncation=True, max_length=eval_prompt_max_len).to(device_eval)
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=fp16_enabled):
            outputs_gen = model_eval(**inputs);
            next_token_logits = outputs_gen.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1).cpu()
        for j_idx, single_probs in enumerate(next_token_probs):
            if np.isnan(batch_difficulties[j_idx]): continue  # Skip if difficulty was NaN
            choice_scores = np.zeros(NUM_CHOICES_MC, dtype=float)
            for choice_k_idx, key_char in enumerate(ANSWER_MAP_KEYS):
                token_id = letter_tok_ids.get(key_char, -1)
                if token_id != -1: choice_scores[choice_k_idx] = single_probs[token_id].item()
            predicted_choice_idx = np.argmax(choice_scores)
            predicted_letter = ANSWER_MAP_KEYS[predicted_choice_idx]
            is_correct = 1 if predicted_letter == batch_true_letters[j_idx] else -1
            resp_patt_theta.append(is_correct);
            item_diffs_theta.append(batch_difficulties[j_idx])
    tokenizer_eval.padding_side = orig_pad_side
    theta_est_time = time.time() - time_theta_start
    final_theta_est = theta_init_calc
    if item_diffs_theta and resp_patt_theta:
        final_theta_est, _ = calculate_theta_irt(item_diffs_theta, resp_patt_theta, NUM_OBS_THETA_ESTIMATION,
                                                 theta_init_calc)
    else:
        print(f"  E{epoch_curr} Val: No valid data for IRT theta. Using initial.")
    return avg_sft_eval_loss, perplexity, final_theta_est, theta_est_time


# ----- Main PUDF Loop -----
print("\nStarting GPT-2 SFT PUDF training loop...")
sft_model = load_model_gpt2(model_name_hf, tokenizer, added_pad_token)
sft_model.to(DEVICE)
letter_token_ids_for_eval = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in ANSWER_MAP_KEYS if
                             len(tokenizer.encode(L, add_special_tokens=False)) == 1}
print(f"Letter token IDs for eval: {letter_token_ids_for_eval}")
if len(letter_token_ids_for_eval) != NUM_CHOICES_MC: print("Warning: Some answer choice letters not single tokens!")

optimizer_sft = AdamW(sft_model.parameters(), lr=LEARNING_RATE_SFT, weight_decay=WEIGHT_DECAY_SFT)
num_total_train_items = len(tokenized_sft_datasets['train'])
steps_per_outer_epoch_approx = (num_total_train_items * 0.75) // (
            PER_DEVICE_TRAIN_BATCH_SIZE_SFT * GRAD_ACCUM_STEPS_SFT)  # Avg 75% data
total_training_steps_approx = int(steps_per_outer_epoch_approx * NUM_PUDF_EPOCHS)
if total_training_steps_approx == 0: total_training_steps_approx = NUM_PUDF_EPOCHS
lr_scheduler_sft = get_linear_schedule_with_warmup(optimizer_sft,
                                                   num_warmup_steps=int(0.1 * total_training_steps_approx),
                                                   num_training_steps=max(1, total_training_steps_approx))
scaler_sft = TorchAmpGradScaler(enabled=fp16_enabled)

current_cap_theta = INITIAL_CAPACITY_THETA;
best_val_metric_pudf = float('inf');
early_stop_count = 0
all_stats_pudf = []
full_tokenized_sft_train_hf = tokenized_sft_datasets['train']

for epoch_pudf_idx in range(NUM_PUDF_EPOCHS):
    time_epoch_start = time.time()
    print(f"\n===== PUDF Epoch {epoch_pudf_idx + 1}/{NUM_PUDF_EPOCHS} =====")
    print(f"  Estimating capacity & SFT val loss (theta_start_epoch={current_cap_theta:.4f})...")
    sft_val_loss, sft_val_ppl, new_theta, time_theta = evaluate_gpt2_sft_and_estimate_theta(
        sft_model, tokenized_sft_datasets['validation'], pudf_dataset_pre_sft['validation'],
        tokenizer, DEVICE, epoch_pudf_idx + 1, current_cap_theta,
        MAX_EVAL_PROMPT_LENGTH, letter_token_ids_for_eval
    )
    if not np.isnan(new_theta) and new_theta <= current_cap_theta and epoch_pudf_idx > 0 and not np.isnan(
            current_cap_theta):
        current_cap_theta += 0.05;
        print(
            f"  Theta nudge: NewEst {new_theta:.4f} <= Curr {current_cap_theta - 0.05:.4f}. Nudged Cap: {current_cap_theta:.4f}")
    elif not np.isnan(new_theta):
        current_cap_theta = new_theta
    else:
        print(f"  Theta est NaN. Keeping prev theta: {current_cap_theta:.4f}")

    print(
        f"  E{epoch_pudf_idx + 1}: Capacity(Theta) for selection = {current_cap_theta:.4f} (est_time={time_theta:.2f}s)")
    print(f"  E{epoch_pudf_idx + 1}: Val SFT Loss (pre-train) = {sft_val_loss:.4f}, PPL = {sft_val_ppl:.4f}")

    epoch_sft_train_data_hf = select_data_for_pudf_epoch(full_tokenized_sft_train_hf, current_cap_theta,
                                                         'difficulty', PUDF_ORDERING, PUDF_LOWER_OFFSET,
                                                         PUDF_UPPER_OFFSET, PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH)
    num_sel_samples = len(epoch_sft_train_data_hf);
    avg_loss_inner = float('nan')

    if num_sel_samples > 0:
        # SFT data already has input_ids, attention_mask. DataCollator will make labels.
        # No need to set_format again if it's already dicts of lists/tensors.
        # We need 'input_ids', 'attention_mask' primarily.
        cols_for_sft_loader = list(
            set(epoch_sft_train_data_hf.column_names).intersection({'input_ids', 'attention_mask'}))
        epoch_sft_train_data_hf.set_format(type='torch', columns=cols_for_sft_loader)

        inner_sft_dl = DataLoader(epoch_sft_train_data_hf, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE_SFT,
                                  collate_fn=data_collator_sft, shuffle=True,
                                  num_workers=min(2, os.cpu_count() if os.cpu_count() else 1))
        sft_model.train();
        total_loss_inner, grad_steps_inner = 0, 0
        for step, sft_batch_tr in enumerate(
                tqdm(inner_sft_dl, desc=f"  InnerTrain SFT E{epoch_pudf_idx + 1}", leave=False)):
            sft_batch_tr = {k: v.to(DEVICE) for k, v in sft_batch_tr.items()}
            with torch_amp_autocast(device_type=DEVICE.type, enabled=fp16_enabled):
                outputs_tr = sft_model(**sft_batch_tr);
                loss_tr = outputs_tr.loss / GRAD_ACCUM_STEPS_SFT
            scaler_sft.scale(loss_tr).backward()
            total_loss_inner += loss_tr.item() * GRAD_ACCUM_STEPS_SFT
            if (step + 1) % GRAD_ACCUM_STEPS_SFT == 0 or (step + 1) == len(inner_sft_dl):
                scaler_sft.unscale_(optimizer_sft);
                torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
                scaler_sft.step(optimizer_sft);
                scaler_sft.update();
                lr_scheduler_sft.step();
                optimizer_sft.zero_grad()
                grad_steps_inner += 1
        avg_loss_inner = total_loss_inner / grad_steps_inner if grad_steps_inner > 0 else float('nan')
        print(f"  E{epoch_pudf_idx + 1}: Inner SFT train avg loss = {avg_loss_inner:.4f}")
    else:
        print(f"  E{epoch_pudf_idx + 1}: No SFT data selected. Skipping inner train.")

    sft_val_loss_post, sft_val_ppl_post, theta_val_post, _ = evaluate_gpt2_sft_and_estimate_theta(
        sft_model, tokenized_sft_datasets['validation'], pudf_dataset_pre_sft['validation'],
        tokenizer, DEVICE, epoch_pudf_idx + 1, current_cap_theta,
        MAX_EVAL_PROMPT_LENGTH, letter_token_ids_for_eval)
    print(f"  E{epoch_pudf_idx + 1}: Val SFT Loss (post-train) = {sft_val_loss_post:.4f}, PPL = {sft_val_ppl_post:.4f}")

    time_epoch_end = time.time() - time_epoch_start
    all_stats_pudf.append({
        "pudf_epoch": epoch_pudf_idx + 1, "cap_theta_select": current_cap_theta, "num_sel_sft_train": num_sel_samples,
        "avg_inner_sft_loss": avg_loss_inner, "val_sft_loss": sft_val_loss_post, "val_sft_ppl": sft_val_ppl_post,
        "theta_val_post_train": theta_val_post, "duration_s": time_epoch_end, "theta_est_time_s": time_theta})
    if sft_val_loss_post < best_val_metric_pudf:  # Loss, so lower is better
        print(f"  New best val_sft_loss: {sft_val_loss_post:.4f} (prev {best_val_metric_pudf:.4f}). Saving model.")
        best_val_metric_pudf = sft_val_loss_post;
        early_stop_count = 0
        sft_model.save_pretrained(best_model_pudf_path);
        tokenizer.save_pretrained(best_model_pudf_path)
    else:
        early_stop_count += 1; print(
            f"  Val SFT loss not improved. EarlyStop: {early_stop_count}/{EARLY_STOPPING_PATIENCE_SFT}")
    if early_stop_count >= EARLY_STOPPING_PATIENCE_SFT: print(f"  Early stopping at E{epoch_pudf_idx + 1}."); break
    print(f"  PUDF E{epoch_pudf_idx + 1} ended. Duration: {time_epoch_end:.2f}s")

# --- End PUDF Loop ---
print(f"\nPUDF SFT Training finished. Best val_sft_loss: {best_val_metric_pudf:.4f}")
stats_file_pudf_sft = os.path.join(output_dir_base, "pudf_sft_training_stats.json")
with open(stats_file_pudf_sft, 'w') as f: json.dump(all_stats_pudf, f, indent=4,
                                                    default=lambda x: float(x) if isinstance(x, (
                                                    np.float32, np.float64, np.float_)) else "NaN" if isinstance(x,
                                                                                                                 float) and np.isnan(
                                                        x) else x)
print(f"PUDF SFT stats saved: {stats_file_pudf_sft}")

# ----- Final Evaluation on Test Set (Custom Accuracy from baseline) -----
if os.path.exists(os.path.join(best_model_pudf_path, "pytorch_model.bin")) or \
        os.path.exists(os.path.join(best_model_pudf_path, "model.safetensors")):  # GPT2 usually saves pytorch_model.bin
    print(f"\nLoading best PUDF SFT model from {best_model_pudf_path} for final test eval...")
    model_best_test_sft = load_model_gpt2(best_model_pudf_path, tokenizer,
                                          added_pad_token)  # Use the same load function
    model_best_test_sft.to(DEVICE);
    model_best_test_sft.eval()

    test_set_raw_final = pudf_dataset_pre_sft["test"]
    test_prompts_final = []
    for i in range(len(test_set_raw_final)):  # Create prompts for test set
        raw_ex_test = test_set_raw_final[i]
        q_text_test, opts_dict_test = raw_ex_test["question"], raw_ex_test["options"]
        prompt_parts_test = [f"Question: {q_text_test}\n\nOptions:"]
        if isinstance(opts_dict_test, dict):
            [prompt_parts_test.append(f"{k}) {opts_dict_test.get(k, '')}") for k in ANSWER_MAP_KEYS]
        else:
            [prompt_parts_test.append(f"{k}) [Invalid Opt]") for k in ANSWER_MAP_KEYS]
        prompt_parts_test.append("\nAnswer:")
        test_prompts_final.append("\n".join(prompt_parts_test))

    test_true_letters_final = [test_set_raw_final[i]["answer_idx"].strip().upper() for i in
                               range(len(test_set_raw_final))]
    test_correctness_final, test_choice_probs_final = [], []
    original_pad_side_final = tokenizer.padding_side;
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(test_prompts_final), PER_DEVICE_EVAL_BATCH_SIZE_SFT), desc="Final Test Custom Eval",
                  leave=False):
        batch_prompts_test = test_prompts_final[i: i + PER_DEVICE_EVAL_BATCH_SIZE_SFT]
        inputs_test = tokenizer(batch_prompts_test, return_tensors="pt", padding="longest", truncation=True,
                                max_length=MAX_EVAL_PROMPT_LENGTH, add_special_tokens=True).to(
            DEVICE)  # Ensure add_special_tokens consistent with training/SFT
        with torch.no_grad(), torch_amp_autocast(device_type=DEVICE.type, enabled=fp16_enabled):
            outputs_test_gen = model_best_test_sft(**inputs_test)
            next_tok_logits_test = outputs_test_gen.logits[:, -1, :]
        next_tok_probs_test = torch.softmax(next_tok_logits_test, dim=-1).cpu()
        for j_batch_test, s_probs_test in enumerate(next_tok_probs_test):
            choice_p_test = np.zeros(NUM_CHOICES_MC, dtype=float)
            for choice_i_test, key_l_test in enumerate(ANSWER_MAP_KEYS):
                tid_test = letter_token_ids_for_eval.get(key_l_test, -1)
                if tid_test != -1: choice_p_test[choice_i_test] = s_probs_test[tid_test].item()
            sum_p_test = np.sum(choice_p_test);
            if sum_p_test > 1e-9:
                choice_p_test = choice_p_test / sum_p_test
            else:
                choice_p_test = np.full_like(choice_p_test, 1.0 / NUM_CHOICES_MC)  # Uniform if all zero
            test_choice_probs_final.append(choice_p_test.tolist())
            pred_l_idx_test = np.argmax(choice_p_test);
            pred_l_char_test = ANSWER_MAP_KEYS[pred_l_idx_test]
            true_l_char_test = test_true_letters_final[i + j_batch_test]
            test_correctness_final.append(int(pred_l_char_test == true_l_char_test))

    tokenizer.padding_side = original_pad_side_final  # Restore padding side
    acc_test_custom_final = np.mean(test_correctness_final) if test_correctness_final else 0.0
    print(f"Final Custom Test Set Accuracy (PUDF SFT model): {acc_test_custom_final:.4f}")
    summary_test_final = {"best_val_sft_loss_pudf": best_val_metric_pudf,
                          "final_test_acc_custom_pudf": acc_test_custom_final}
    with open(os.path.join(output_dir_base, "final_pudf_sft_test_summary.json"), 'w') as f:
        json.dump(summary_test_final, f, indent=4)
else:
    print(f"Best PUDF SFT model not found at {best_model_pudf_path}. Skipping final test.")

if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"\nScript finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Outputs saved in: {output_dir_base}")