import os
import datetime
import random
import numpy as np
import torch
import json
import time
import shutil
import glob
import traceback

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForMultipleChoice,
    # DataCollatorForMultipleChoice, # We will use our custom one
    get_linear_schedule_with_warmup
)
from transformers.data.data_collator import DataCollatorForMultipleChoice  # Import base for subclassing
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler as TorchAmpGradScaler, autocast as torch_amp_autocast  # Updated
from tqdm.auto import tqdm
from evaluate import load as load_metric

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

    if len(difficulties_filt) == 0:
        print("  calculate_theta_irt: No valid data after filtering NaNs. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt

    student_prior = norm(loc=0., scale=1.)
    if num_obs > 0 and len(difficulties_filt) > num_obs:
        samples_idx = np.random.choice(len(difficulties_filt), num_obs, replace=False)
        difficulties_sample = difficulties_filt[samples_idx]
        response_pattern_sample = response_pattern_filt[samples_idx]
    else:
        difficulties_sample = difficulties_filt
        response_pattern_sample = response_pattern_filt

    if len(difficulties_sample) == 0:
        print(
            "  calculate_theta_irt: No samples to estimate theta after potential sampling. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt

    fn_to_minimize = theta_fn_irt(difficulties_sample, student_prior, response_pattern_sample)
    result = minimize(fn_to_minimize, [initial_theta_val], method='Nelder-Mead')

    estimated_theta = result['x'][0]
    if np.isnan(estimated_theta) or np.isinf(estimated_theta):
        print(
            f"  calculate_theta_irt: Estimated theta is NaN/Inf. Optimizer success: {result.success}, Message: {result.message}. Returning initial_theta_val.")
        estimated_theta = initial_theta_val
    return estimated_theta, time.time() - start_time_irt


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

    min_diff_target = capacity_theta + lower_offset
    max_diff_target = capacity_theta + upper_offset

    selected_hf_dataset = full_hf_train_dataset.filter(
        lambda x: x[difficulty_col] is not None and \
                  min_diff_target <= x[difficulty_col] < max_diff_target,
        load_from_cache_file=False
    )
    print(f"  Initially selected {len(selected_hf_dataset)} samples based on difficulty window.")

    if len(selected_hf_dataset) < min_samples_per_epoch:
        print(
            f"  Selected data ({len(selected_hf_dataset)}) is less than min_samples_per_epoch ({min_samples_per_epoch}).")
        if len(full_hf_train_dataset) == 0:
            return selected_hf_dataset

        num_to_take = min(min_samples_per_epoch, len(full_hf_train_dataset))
        print(f"  Taking {num_to_take} samples based on '{pudf_ordering}' ordering to meet min_samples_per_epoch.")

        dataset_to_sort = full_hf_train_dataset.filter(lambda x: x[difficulty_col] is not None,
                                                       load_from_cache_file=False)
        if len(dataset_to_sort) == 0:
            print(f"  Warning: All difficulties were None for sorting. Cannot fulfill min_samples with sorting.")
            return selected_hf_dataset

        reverse_sort = True if pudf_ordering == 'hardest' else False
        try:
            sorted_full_train_dataset = dataset_to_sort.sort(difficulty_col, reverse=reverse_sort,
                                                             load_from_cache_file=False)
            actual_num_to_take = min(num_to_take, len(sorted_full_train_dataset))
            selected_hf_dataset = sorted_full_train_dataset.select(range(actual_num_to_take))
        except Exception as e:
            print(f"  Error during sorting for min_samples: {e}. Returning initially selected data.")

    print(f"  Final number of samples for this epoch: {len(selected_hf_dataset)}")
    return selected_hf_dataset


# --- End PUDF Data Selection ---

# --- Custom Data Collator ---
class CustomDataCollatorForMultipleChoice(DataCollatorForMultipleChoice):
    def torch_call(self, features: list[dict[str, any]]) -> dict[str, any]:
        # Pop 'difficulty' if it exists, similar to how 'labels' are handled by parent
        difficulty_values = None
        if "difficulty" in features[0]:
            # Ensure all features have it or handle missing (though set_format should ensure consistency)
            if all("difficulty" in feature for feature in features):
                difficulty_values = [feature.pop("difficulty") for feature in features]
            else:
                # Fallback or error if 'difficulty' is inconsistent among features
                print("Warning: 'difficulty' key inconsistent in batch features. Not including in batch.")

        # Call the parent's torch_call method for standard processing
        batch = super().torch_call(features)

        # Add 'difficulty' back to the batch if it was popped
        if difficulty_values is not None:
            # Convert list of scalar tensors (or scalars) to a single tensor for the batch
            if isinstance(difficulty_values[0], torch.Tensor):  # if they are already tensors
                batch["difficulty"] = torch.stack(difficulty_values)
            else:  # if they are Python scalars
                batch["difficulty"] = torch.tensor(difficulty_values, dtype=torch.float32)
        return batch


# --- End Custom Data Collator ---


# ----- Environment Setup & Seed -----
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
if not os.path.exists(os.path.dirname(HF_HOME)):
    print(f"Warning: Path {os.path.dirname(HF_HOME)} does not exist. Using default Hugging Face cache directory.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

random_seed = 63
torch.manual_seed(random_seed);
np.random.seed(random_seed);
random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
print(f"Using random seed: {random_seed}")

# ----- Timestamp and Config -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_name = "microsoft/deberta-v3-base"
dataset_id = "GBaker/MedQA-USMLE-4-options"
max_length = 512;
train_batch_size = 16;
eval_batch_size = train_batch_size * 2
num_pudf_epochs = 20;
learning_rate = 2e-5;
weight_decay = 0.01
early_stopping_patience_val = 2;
fp16_enabled = torch.cuda.is_available()

DIFFICULTY_FILE_PATH = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/MeD_QA/merged_jsonlines_output/test-1pl/best_parameters.json"
DIFFICULTY_JSON_KEY = "diff";
INITIAL_CAPACITY_THETA = 0.0;
NUM_OBS_THETA_ESTIMATION = -1
PUDF_LOWER_OFFSET = -float('inf');
PUDF_UPPER_OFFSET = 0.0
PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH = 100;
PUDF_ORDERING = 'easiest'

run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
base_output_dir = f"./deberta_v3_base_medqa_PUDF_{run_timestamp}"
best_model_path = os.path.join(base_output_dir, "best_pudf_model")
os.makedirs(base_output_dir, exist_ok=True);
os.makedirs(best_model_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
raw_dataset = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])

if 'train' not in raw_dataset: print("FATAL: 'train' split not found!"); exit()
original_train_hf = raw_dataset['train']
difficulty_scores = load_difficulties_from_file(DIFFICULTY_FILE_PATH, DIFFICULTY_JSON_KEY)
if len(difficulty_scores) != len(original_train_hf):
    raise ValueError(f"Mismatch! Diff scores ({len(difficulty_scores)}) != original train ({len(original_train_hf)}).")
train_full_with_diff = original_train_hf.add_column("difficulty", difficulty_scores)
print(f"Added 'difficulty' to original 'train'. Columns: {train_full_with_diff.column_names}")

pudf_test_dataset = raw_dataset['test']
if 'validation' not in raw_dataset:
    print("Validation split not found. Splitting 'train' (with diff) into 80/20...")
    split = train_full_with_diff.train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
    pudf_train_dataset, pudf_validation_dataset = split['train'], split['test']
    print("Train/Validation split for PUDF complete. Both have 'difficulty'.")
else:
    print("Using predefined 'train', 'validation', 'test' splits.")
    pudf_train_dataset = train_full_with_diff
    pudf_validation_dataset = raw_dataset['validation']
    if 'difficulty' not in pudf_validation_dataset.column_names:
        print(
            f"CRITICAL WARNING: Predefined 'validation' lacks 'difficulty'. IRT theta estimation on it will be impaired.")
    else:
        print("Predefined 'validation' has 'difficulty' column.")

final_dataset_splits = DatasetDict(
    {'train': pudf_train_dataset, 'validation': pudf_validation_dataset, 'test': pudf_test_dataset})
print("\nFinal dataset splits for PUDF:");
for n, ds in final_dataset_splits.items(): print(f"  {n}: {len(ds)} examples, cols: {ds.column_names}")

tokenizer = DebertaV2Tokenizer.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
model = DebertaV2ForMultipleChoice.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
model.to(device)
answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3};
option_keys = ['A', 'B', 'C', 'D']


def preprocess_function_pudf(examples):
    num_ex = len(examples["question"])
    first_s = [[examples["question"][i]] * len(option_keys) for i in range(num_ex)]
    second_s = [
        [examples["options"][i].get(k, "") if isinstance(examples["options"][i], dict) else "" for k in option_keys] for
        i in range(num_ex)]
    labels = [answer_map.get(examples["answer_idx"][i], -100) for i in range(num_ex)]
    flat_first = [s for sub in first_s for s in sub];
    flat_second = [s for sub in second_s for s in sub]
    tok_in = tokenizer(flat_first, flat_second, max_length=max_length, truncation=True, padding=False)
    proc_ex = {k: [v[i:i + len(option_keys)] for i in range(0, len(v), len(option_keys))] for k, v in tok_in.items()}
    proc_ex["labels"] = labels
    if "difficulty" in examples: proc_ex["difficulty"] = examples["difficulty"]
    return proc_ex


print("\nTokenizing dataset for PUDF...");
num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
encoded_dataset = DatasetDict()
for split_n, ds_s in final_dataset_splits.items():
    print(f"  Tokenizing {split_n}...");
    orig_cols = list(ds_s.column_names)
    cols_to_rem = [c for c in orig_cols if c not in ['question', 'options', 'answer_idx', 'difficulty']]
    temp_ds_map = ds_s.remove_columns(cols_to_rem) if cols_to_rem else ds_s
    encoded_dataset[split_n] = temp_ds_map.map(preprocess_function_pudf, batched=True, num_proc=num_proc,
                                               remove_columns=[c for c in temp_ds_map.column_names if
                                                               c not in ['difficulty']])
print("Tokenization complete.")
for split_n, ds_s in encoded_dataset.items(): print(f"Cols tokenized '{split_n}': {ds_s.column_names}")

accuracy_metric_hf = load_metric("accuracy")
data_collator = CustomDataCollatorForMultipleChoice(tokenizer=tokenizer, padding="longest")  # Use Custom Collator

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
avg_subset_factor = 0.75
est_batches_epoch = (len(encoded_dataset["train"]) * avg_subset_factor) // train_batch_size
total_est_steps = int(est_batches_epoch * num_pudf_epochs)
if total_est_steps <= 0: total_est_steps = num_pudf_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_est_steps),
                                               num_training_steps=max(1, total_est_steps))
scaler = TorchAmpGradScaler(enabled=fp16_enabled)  # Updated GradScaler


def evaluate_and_estimate_theta_on_val(model_eval, dataloader_eval, device_eval, epoch_curr, theta_init_calc):
    model_eval.eval();
    total_loss_eval, all_logits_eval, all_lbls_eval, all_diffs_theta = 0, [], [], []
    has_diff_val = "difficulty" in dataloader_eval.dataset.column_names
    if not has_diff_val and epoch_curr == 1: print(
        f"  EvalFnWarn: 'difficulty' missing in val data. Theta est default.")

    for batch_eval in tqdm(dataloader_eval, desc=f"E{epoch_curr} Eval/Theta", leave=False):
        # Correctly unpack and move all expected items by data_collator and set_format
        input_ids = batch_eval['input_ids'].to(device_eval)
        attention_mask = batch_eval['attention_mask'].to(device_eval)
        labels_eval = batch_eval['labels'].to(device_eval)
        token_type_ids_eval = batch_eval.get('token_type_ids')
        if token_type_ids_eval is not None: token_type_ids_eval = token_type_ids_eval.to(device_eval)

        batch_model_in = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels_eval}
        if token_type_ids_eval is not None: batch_model_in['token_type_ids'] = token_type_ids_eval

        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type,
                                                 enabled=fp16_enabled):  # Updated autocast
            outputs_eval = model_eval(**batch_model_in)
        total_loss_eval += outputs_eval.loss.item()
        all_logits_eval.append(outputs_eval.logits.cpu().numpy())
        all_lbls_eval.append(labels_eval.cpu().numpy())
        if has_diff_val and "difficulty" in batch_eval:
            all_diffs_theta.append(batch_eval["difficulty"].cpu().numpy())

    avg_loss_eval = total_loss_eval / len(dataloader_eval) if dataloader_eval else float('nan')
    logits_np = np.concatenate(all_logits_eval) if all_logits_eval else np.array([])
    preds_np = np.argmax(logits_np, axis=-1) if logits_np.size > 0 else np.array([])
    lbls_np = np.concatenate(all_lbls_eval) if all_lbls_eval else np.array([])

    acc_eval, theta_est, theta_time = 0.0, theta_init_calc, 0.0
    if preds_np.size > 0 and lbls_np.size > 0:
        valid_idx = (lbls_np != -100)
        valid_preds_eval, valid_lbls_eval = preds_np[valid_idx], lbls_np[valid_idx]
        if len(valid_lbls_eval) > 0: acc_eval = \
        accuracy_metric_hf.compute(predictions=valid_preds_eval, references=valid_lbls_eval)['accuracy']

        if has_diff_val and all_diffs_theta:
            diffs_np_theta = np.concatenate(all_diffs_theta)
            if len(diffs_np_theta) == len(lbls_np):
                diffs_align_theta = diffs_np_theta[valid_idx]
                resp_patt_theta = np.where(valid_preds_eval == valid_lbls_eval, 1, -1)
                if len(diffs_align_theta) > 0 and len(diffs_align_theta) == len(resp_patt_theta):
                    theta_est, theta_time = calculate_theta_irt(diffs_align_theta, resp_patt_theta,
                                                                NUM_OBS_THETA_ESTIMATION, theta_init_calc)
                else:
                    print(f"  Warn E{epoch_curr}: Not enough aligned data for theta est.")
            else:
                print(
                    f"  Warn E{epoch_curr}: Mismatch diffs ({len(diffs_np_theta)}) vs lbls ({len(lbls_np)}) for theta est.")
        elif has_diff_val:
            print(f"  Warn E{epoch_curr}: 'difficulty' in val cols, but none collected. Theta est uses init.")
    else:
        print(f"  Warn E{epoch_curr}: No preds/lbls from val. Metrics/Theta skipped.")
    return acc_eval, avg_loss_eval, theta_est, theta_time


print("\nStarting PUDF training loop...")
current_cap_theta = INITIAL_CAPACITY_THETA;
best_val_acc_pudf = 0.0;
early_stop_count = 0
all_stats_pudf = []

val_cols_tensor = ['input_ids', 'attention_mask', 'labels']
if "difficulty" in encoded_dataset["validation"].column_names: val_cols_tensor.append('difficulty')
if "token_type_ids" in encoded_dataset["validation"].column_names: val_cols_tensor.append('token_type_ids')
encoded_dataset["validation"].set_format(type='torch', columns=val_cols_tensor)
full_val_dl = DataLoader(encoded_dataset["validation"], batch_size=eval_batch_size, collate_fn=data_collator,
                         shuffle=False)
full_tok_train_hf = encoded_dataset["train"]

for epoch_pudf in range(num_pudf_epochs):
    time_epoch_start = time.time()
    print(f"\n===== PUDF Epoch {epoch_pudf + 1}/{num_pudf_epochs} =====")
    print(f"  Estimating capacity (theta_start_epoch={current_cap_theta:.4f})...")
    val_acc_pre, val_loss_pre, new_theta_est, time_theta = evaluate_and_estimate_theta_on_val(model, full_val_dl,
                                                                                              device, epoch_pudf + 1,
                                                                                              current_cap_theta)

    if not np.isnan(new_theta_est) and new_theta_est <= current_cap_theta and epoch_pudf > 0 and not np.isnan(
            current_cap_theta):
        current_cap_theta += 0.05;
        print(f"  Theta nudge: {new_theta_est:.4f} <= {current_cap_theta - 0.05:.4f}. New cap: {current_cap_theta:.4f}")
    elif not np.isnan(new_theta_est):
        current_cap_theta = new_theta_est
    else:
        print(f"  Theta est NaN. Keeping prev theta: {current_cap_theta:.4f}")

    print(f"  E{epoch_pudf + 1}: Capacity(Theta) for selection = {current_cap_theta:.4f} (est_time={time_theta:.2f}s)")
    print(f"  E{epoch_pudf + 1}: Val Acc (pre-train) = {val_acc_pre:.4f}, Val Loss = {val_loss_pre:.4f}")

    epoch_train_hf = select_data_for_pudf_epoch(full_tok_train_hf, current_cap_theta, 'difficulty', PUDF_ORDERING,
                                                PUDF_LOWER_OFFSET, PUDF_UPPER_OFFSET, PUDF_MIN_TRAIN_SAMPLES_PER_EPOCH)
    num_sel_samples = len(epoch_train_hf);
    avg_loss_inner = float('nan')

    if num_sel_samples > 0:
        train_cols_dl = ['input_ids', 'attention_mask', 'labels']
        if "token_type_ids" in epoch_train_hf.column_names: train_cols_dl.append('token_type_ids')
        epoch_train_hf.set_format(type='torch', columns=train_cols_dl)
        inner_train_dl = DataLoader(epoch_train_hf, batch_size=train_batch_size, collate_fn=data_collator, shuffle=True)
        model.train();
        total_loss_inner, num_batches_inner = 0, 0
        for batch_train in tqdm(inner_train_dl, desc=f"  InnerTrain E{epoch_pudf + 1}", leave=False):
            optimizer.zero_grad()
            input_ids_tr = batch_train['input_ids'].to(device);
            attention_mask_tr = batch_train['attention_mask'].to(device)
            labels_tr = batch_train['labels'].to(device)
            token_type_ids_tr = batch_train.get('token_type_ids');
            if token_type_ids_tr is not None: token_type_ids_tr = token_type_ids_tr.to(device)
            batch_model_in_tr = {'input_ids': input_ids_tr, 'attention_mask': attention_mask_tr, 'labels': labels_tr}
            if token_type_ids_tr is not None: batch_model_in_tr['token_type_ids'] = token_type_ids_tr
            with torch_amp_autocast(device_type=device.type, enabled=fp16_enabled):
                outputs_tr = model(**batch_model_in_tr);loss_tr = outputs_tr.loss
            scaler.scale(loss_tr).backward();
            scaler.step(optimizer);
            scaler.update();
            lr_scheduler.step()
            total_loss_inner += loss_tr.item();
            num_batches_inner += 1
        avg_loss_inner = total_loss_inner / num_batches_inner if num_batches_inner > 0 else float('nan')
        print(f"  E{epoch_pudf + 1}: Inner train avg loss = {avg_loss_inner:.4f}")
    else:
        print(f"  E{epoch_pudf + 1}: No training data selected. Skipping inner train.")

    val_acc_post, val_loss_post, theta_val_post, _ = evaluate_and_estimate_theta_on_val(model, full_val_dl, device,
                                                                                        epoch_pudf + 1,
                                                                                        current_cap_theta)
    print(f"  E{epoch_pudf + 1}: Val Acc (post-train) = {val_acc_post:.4f}, Val Loss = {val_loss_post:.4f}")

    time_epoch_end = time.time() - time_epoch_start
    all_stats_pudf.append({
        "pudf_epoch": epoch_pudf + 1, "cap_theta_select": current_cap_theta, "num_sel_train": num_sel_samples,
        "avg_inner_loss": avg_loss_inner, "val_acc": val_acc_post, "val_loss": val_loss_post,
        "theta_val_post": theta_val_post, "duration_s": time_epoch_end, "theta_est_time_s": time_theta})
    if val_acc_post > best_val_acc_pudf:
        print(f"  New best val_acc: {val_acc_post:.4f} (prev {best_val_acc_pudf:.4f}). Saving model.")
        best_val_acc_pudf = val_acc_post;
        early_stop_count = 0
        model.save_pretrained(best_model_path);
        tokenizer.save_pretrained(best_model_path)
    else:
        early_stop_count += 1; print(
            f"  Val acc not improved. EarlyStop: {early_stop_count}/{early_stopping_patience_val}")
    if early_stop_count >= early_stopping_patience_val: print(f"  Early stopping at E{epoch_pudf + 1}."); break
    print(f"  PUDF E{epoch_pudf + 1} ended. Duration: {time_epoch_end:.2f}s")

print(f"\nPUDF Training finished. Best val_acc: {best_val_acc_pudf:.4f}")
stats_file_pudf = os.path.join(base_output_dir, "pudf_training_stats.json")
with open(stats_file_pudf, 'w') as f_stats: json.dump(all_stats_pudf, f_stats, indent=4,
                                                      default=lambda x: float(x) if isinstance(x, (
                                                      np.float32, np.float64, np.float_)) else x if not (
                                                                  isinstance(x, float) and np.isnan(x)) else "NaN")
print(f"PUDF stats saved: {stats_file_pudf}")

if os.path.exists(os.path.join(best_model_path, "pytorch_model.bin")) or \
        os.path.exists(os.path.join(best_model_path, "model.safetensors")):
    print(f"\nLoading best PUDF model from {best_model_path} for final test eval...")
    model_best_test = DebertaV2ForMultipleChoice.from_pretrained(best_model_path);
    model_best_test.to(device);
    model_best_test.eval()
    test_cols_tensor = ['input_ids', 'attention_mask', 'labels']
    if "token_type_ids" in encoded_dataset["test"].column_names: test_cols_tensor.append('token_type_ids')
    encoded_dataset["test"].set_format(type='torch', columns=test_cols_tensor)
    dl_test = DataLoader(encoded_dataset["test"], batch_size=eval_batch_size, collate_fn=data_collator, shuffle=False)
    all_logits_test, all_lbls_test = [], []
    for batch_test in tqdm(dl_test, desc="Final Test Eval", leave=False):
        input_ids_test = batch_test['input_ids'].to(device);
        att_mask_test = batch_test['attention_mask'].to(device)
        lbls_test = batch_test['labels'].to(device)
        tok_type_ids_test = batch_test.get('token_type_ids');
        if tok_type_ids_test is not None: tok_type_ids_test = tok_type_ids_test.to(device)
        batch_model_in_test = {'input_ids': input_ids_test, 'attention_mask': att_mask_test, 'labels': lbls_test}
        if tok_type_ids_test is not None: batch_model_in_test['token_type_ids'] = tok_type_ids_test
        with torch.no_grad(), torch_amp_autocast(device_type=device.type, enabled=fp16_enabled):
            outputs_test = model_best_test(**batch_model_in_test); logits_test = outputs_test.logits
        all_logits_test.append(logits_test.cpu().numpy());
        all_lbls_test.append(lbls_test.cpu().numpy())
    acc_test_final = 0.0
    if all_logits_test and all_lbls_test:
        logits_np_test = np.concatenate(all_logits_test);
        preds_np_test = np.argmax(logits_np_test, axis=-1)
        lbls_np_test = np.concatenate(all_lbls_test);
        valid_idx_test = (lbls_np_test != -100)
        if np.sum(valid_idx_test) > 0: acc_test_final = \
        accuracy_metric_hf.compute(predictions=preds_np_test[valid_idx_test], references=lbls_np_test[valid_idx_test])[
            'accuracy']
    print(f"Final Test Accuracy (PUDF model): {acc_test_final:.4f}")
    summary_test_res = {"best_val_acc_pudf": best_val_acc_pudf, "final_test_acc_pudf": acc_test_final}
    with open(os.path.join(base_output_dir, "final_pudf_test_summary.json"), 'w') as f_sum:
        json.dump(summary_test_res, f_sum, indent=4)
else:
    print(f"Best PUDF model not found at {best_model_path}. Skipping final test.")
print(f"\nScript finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Outputs saved in: {base_output_dir}")