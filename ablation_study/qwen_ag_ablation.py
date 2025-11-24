# qwen_agnews_ablation_scaled_v5.py
import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.optimization import Adafactor
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, Sampler
from typing import Sized, Iterator
import evaluate
import types
import json
import gc
import traceback
from tqdm.auto import tqdm
from huggingface_hub import login, whoami
from math import ceil
import copy
import time
import argparse
import re
import packaging

# Assuming these custom modules are in the same directory or Python path
from build_features_Qwen import get_epoch_training_data
from irt_scoring import calculate_theta

# --- Default Configurations (can be overridden by args or specific logic) ---
MODEL_CHECKPOINT = "Qwen/Qwen2.5-7B"
DATASET_ID = "contemmcm/ag_news"
DEFAULT_PUDF_DIFFICULTY_FILE_PATH = "../gen_difficulty/merged_jsonlines_output/test-1pl/best_parameters.json"
DIFFICULTY_JSON_KEY = "diff"

RANDOM_SEED = 63
MAX_LENGTH = 128

# --- PUDF Scheduler Specific Defaults ---
DEFAULT_THETA_ESTIMATION_SET_SIZE = 1000
DEFAULT_NUM_TRAIN_EPOCHS_PUDF = 15
DEFAULT_LEARNING_RATE_ADAFACTOR_PUDF = 2e-5
DEFAULT_WEIGHT_DECAY_PUDF = 0.01
DEFAULT_PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE_PUDF = 64  # From original PUDF script
DEFAULT_GRADIENT_ACCUMULATION_STEPS_PUDF = 16  # From original PUDF script
DEFAULT_PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE_PUDF = 64
DEFAULT_PUDF_STRATEGY = 'theta'
DEFAULT_PUDF_ORDERING = 'easiest'
DEFAULT_PUDF_LOWER_BOUND = -np.inf
DEFAULT_PUDF_UPPER_BOUND = 0.0
DEFAULT_PUDF_MIN_TRAIN_LENGTH = 500
DEFAULT_PATIENCE_EARLY_STOPPING_PUDF = 3
DEFAULT_USE_GRADIENT_CHECKPOINTING_PUDF = True

# --- Heuristic Scheduler Specific Defaults ---
DEFAULT_LEARNING_RATE_HEURISTIC = 2e-5  # Aligning with PUDF's LR for Adafactor
DEFAULT_ADAFACOR_LR_HEURISTIC = 2e-5  # Explicit for Adafactor
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE_HEURISTIC = 64  # CHANGED to match PUDF script
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE_HEURISTIC = 64  # Matching PUDF script
DEFAULT_GRADIENT_ACCUMULATION_STEPS_HEURISTIC = 16  # CHANGED to match PUDF script
DEFAULT_NUM_TRAIN_EPOCHS_HEURISTIC = 15
DEFAULT_WEIGHT_DECAY_HEURISTIC = 0.01
DEFAULT_LOGGING_STEPS_HEURISTIC = 100  # Can be adjusted based on new step count
DEFAULT_GRADIENT_CHECKPOINTING_HEURISTIC = True
DEFAULT_EARLY_STOPPING_PATIENCE_HEURISTIC = 3
DEFAULT_HEURISTIC_ORDERING = 'easiest'
DEFAULT_COMPETENCY_PARAM_HEURISTIC = 5
DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC = 0.05
DEFAULT_C_INIT_HEURISTIC = 0.01


# ... (rest of the script remains the same as qwen_agnews_ablation_scaled_v4.py)
# The functions setup_environment, set_seed, simple_tokenize_for_rarity,
# get_example_rarities, calculate_heuristic_difficulty_scores, min_max_scale,
# create_qwen_tensor_dataset, evaluate_and_estimate_qwen_theta_pudf,
# evaluate_qwen_main_val_pudf, HeuristicSampler, CustomHeuristicTrainer,
# compute_metrics_for_trainer, run_training_with_heuristic_sampler,
# run_training_with_pudf_scheduler, and main will be identical to the
# qwen_agnews_ablation_scaled_v4.py version I provided in the previous response.
# The only change is the DEFAULT constants above for the Heuristic Scheduler path.

# --- Environment Setup (User Provided) ---
def setup_environment(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
        print("Removed HF_TOKEN environment variable to use cached token.")
    try:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        user_info = whoami(token=hf_token)
        print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
    except Exception as e:
        print(
            f"Hugging Face login check failed: {e}. Public models should still work if already cached or no auth required.")


# --- Seed Setting (User Provided) ---
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed_value}")


def simple_tokenize_for_rarity(sent):
    if not isinstance(sent, str): return []
    sent = re.sub(r'\s+', ' ', sent)
    tokens = [x.strip() for x in re.findall(r"[\w']+|[^\w\s]", sent) if x.strip()]
    return tokens


def get_example_rarities(texts):
    if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts): return [0.0] * len(texts)
    tokenized_corpus = [simple_tokenize_for_rarity(text) for text in texts];
    vocab = set();
    counts = dict();
    N = 0
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t];
        vocab.update(valid_tokens);
        N += len(valid_tokens)
        for tok in valid_tokens: counts.setdefault(tok, 0); counts[tok] += 1
    if N == 0: return [0.0] * len(texts)
    result = [];
    epsilon = 1e-9
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]
        if not valid_tokens:
            p_hat = 0.0
        else:
            log_probs = [np.log(counts.get(tok, 0) / N + epsilon) for tok in valid_tokens]; p_hat = -np.mean(
                log_probs) if log_probs else 0.0
        result.append(p_hat)
    return result


def calculate_heuristic_difficulty_scores(dataset_split_hf, measurer_type, text_column_name='text'):
    print(f"Calculating '{measurer_type}' difficulty scores...")
    texts = dataset_split_hf[text_column_name]
    if measurer_type == 'sentence_length':
        scores = [len(str(text)) if text is not None else 0 for text in texts]
        print("Calculated sentence length difficulty.")
    elif measurer_type == 'word_rarity':
        str_texts = [str(text) if text is not None else "" for text in texts]
        scores = get_example_rarities(str_texts)
        print("Calculated word rarity difficulty.")
    else:
        raise ValueError(f"Unsupported heuristic difficulty measurer: {measurer_type}")
    return scores


def min_max_scale(scores_to_scale, target_min, target_max):
    scores_np = np.array(scores_to_scale, dtype=float)
    if scores_np.size == 0: return np.array([], dtype=float)
    min_raw_score = scores_np.min()
    max_raw_score = scores_np.max()
    if not (isinstance(target_min, (int, float)) and isinstance(target_max, (int, float))):
        print(f"Warning: Invalid target_min ({target_min}) or target_max ({target_max}) for scaling. Using raw scores.")
        return scores_np
    if min_raw_score == max_raw_score:
        return np.full_like(scores_np, (target_min + target_max) / 2.0)
    scaled_scores = ((scores_np - min_raw_score) / (max_raw_score - min_raw_score)) * (
                target_max - target_min) + target_min
    return scaled_scores


def create_qwen_tensor_dataset(hf_tokenized_dataset_split, partition_name="unknown", include_difficulty=True):
    print(f"Creating TensorDataset for Qwen {partition_name} (include_difficulty={include_difficulty})...")
    actual_columns = hf_tokenized_dataset_split.column_names
    final_columns_ordered = ['input_ids', 'attention_mask']
    if 'labels' in actual_columns:
        final_columns_ordered.append('labels')
    elif partition_name in ["actual_train_pool_for_pudf", "theta_estimation_set",
                            "validation_main_eval"] and 'labels' not in actual_columns:
        raise ValueError(f"'labels' column expected for {partition_name} but not found in {actual_columns}.")

    if include_difficulty:
        if 'difficulty' in actual_columns:
            final_columns_ordered.append('difficulty')
        elif partition_name in ["actual_train_pool_for_pudf", "theta_estimation_set"]:
            raise RuntimeError(
                f"Difficulty column required for {partition_name} (include_difficulty=True) but missing from {actual_columns}.")
    try:
        current_format = hf_tokenized_dataset_split.format
        hf_tokenized_dataset_split.set_format(type='torch', columns=final_columns_ordered)
        tensors_to_extract = []
        for col_name in final_columns_ordered:
            tensor_data = hf_tokenized_dataset_split[col_name]
            if col_name == 'labels':
                if tensor_data.ndim > 1 and tensor_data.shape[-1] == 1:
                    tensor_data = tensor_data.squeeze(-1)
                if tensor_data.ndim != 1:
                    raise ValueError(f"Labels tensor for {partition_name} not 1D. Shape: {tensor_data.shape}")
                tensor_data = tensor_data.long()
            elif col_name == 'difficulty':
                if not isinstance(tensor_data, torch.Tensor):
                    tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
                elif tensor_data.dtype != torch.float32:
                    tensor_data = tensor_data.to(torch.float32)
            tensors_to_extract.append(tensor_data)

        if current_format and current_format['type'] != hf_tokenized_dataset_split.format['type']:
            hf_tokenized_dataset_split.set_format(**current_format)
        elif not current_format and hf_tokenized_dataset_split.format['type'] == 'torch':
            hf_tokenized_dataset_split.reset_format()
        return TensorDataset(*tensors_to_extract), final_columns_ordered
    except Exception as e:
        print(f"ERROR creating TensorDataset for {partition_name}: {e}");
        traceback.print_exc();
        raise


def evaluate_and_estimate_qwen_theta_pudf(model, dataloader, device, column_order, num_labels, num_obs_theta=-1,
                                          desc_prefix="QwenThetaEst"):
    print(f"Estimating Qwen Theta ({desc_prefix})...")
    model.eval()
    all_preds_logits_list, all_labels_list, all_difficulties_list = [], [], []
    use_bf16_eval = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype_eval = torch.bfloat16 if use_bf16_eval else torch.float16

    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        labels_idx = column_order.index('labels')
        difficulty_idx = column_order.index('difficulty')
    except ValueError as e:
        print(f"FATAL ERROR ({desc_prefix}): Column missing for theta. Expected in order: {column_order}. Error: {e}");
        return 0.0, 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc_prefix, leave=False)):
            if desc_prefix.startswith("Ep1 ThetaEst") and batch_idx == 0:
                if batch[difficulty_idx].numel() > 0:
                    print(
                        f"  DEBUG ({desc_prefix}) Batch 0, Difficulty sample (min/max): {batch[difficulty_idx].cpu().numpy().min():.4f} / {batch[difficulty_idx].cpu().numpy().max():.4f}")
                else:
                    print(f"  DEBUG ({desc_prefix}) Batch 0, Difficulty tensor is empty.")

            input_ids = batch[input_ids_idx].to(device, non_blocking=True)
            attention_mask = batch[attention_mask_idx].to(device, non_blocking=True)
            labels_batch = batch[labels_idx].to(device, non_blocking=True)
            difficulty_tensor_for_batch = batch[difficulty_idx].cpu().numpy()
            with autocast(device.type, dtype=amp_dtype_eval, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            all_preds_logits_list.append(logits.cpu().numpy())
            all_labels_list.append(labels_batch.cpu().numpy())
            all_difficulties_list.append(difficulty_tensor_for_batch)

    theta_hat, model_capacity_time = 0.0, 0.0
    if all_difficulties_list and all(isinstance(d, np.ndarray) and d.size > 0 for d in all_difficulties_list):
        final_preds_logits, final_labels, concatenated_difficulties_raw = map(np.concatenate,
                                                                              [all_preds_logits_list, all_labels_list,
                                                                               all_difficulties_list])
        if concatenated_difficulties_raw.ndim > 1:
            concatenated_difficulties = concatenated_difficulties_raw.squeeze().astype(float)
        else:
            concatenated_difficulties = concatenated_difficulties_raw.astype(float)

        if final_preds_logits.shape[0] == final_labels.shape[0] == concatenated_difficulties.shape[
            0] and final_labels.size > 0:
            time_s = time.time()
            predictions = np.argmax(final_preds_logits, axis=1)
            response_pattern = (predictions == final_labels).astype(int) * 2 - 1
            if response_pattern.ndim > 1: response_pattern = response_pattern.squeeze()

            num_total = len(concatenated_difficulties)
            indices = np.arange(num_total)
            if num_obs_theta > 0 and num_obs_theta < num_total:
                indices = np.random.choice(num_total, num_obs_theta, replace=False)

            if len(indices) > 0:
                try:
                    theta_hat = calculate_theta(concatenated_difficulties[indices], response_pattern[indices])[0]
                except Exception as e:
                    print(
                        f"ERROR ({desc_prefix}) theta calc: {e}. Trace: {traceback.format_exc()}. Theta=0.0"); theta_hat = 0.0
            else:
                print(
                    f"Warning ({desc_prefix}): No samples selected for theta estimation after indexing (num_total={num_total}, num_obs_theta={num_obs_theta}).")
            model_capacity_time = time.time() - time_s
        else:
            print(
                f"Warning ({desc_prefix}): Mismatch in lengths or empty data for theta. P: {final_preds_logits.shape}, L: {final_labels.shape}, D: {concatenated_difficulties.shape}")
    else:
        print(
            f"Warning ({desc_prefix}): No difficulty scores collected or all_difficulties_list was empty/contained empty arrays for theta estimation.")
    print(f"Theta estimated ({desc_prefix}): {theta_hat:.4f} in {model_capacity_time:.2f}s")
    return theta_hat, model_capacity_time


def evaluate_qwen_main_val_pudf(model, dataloader, device, column_order, num_labels, desc_prefix="QwenMainVal",
                                pudf_config_for_theta_val=None):
    print(f"Evaluating Qwen on Main Validation ({desc_prefix})...")
    model.eval()
    accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
    total_loss, num_batches = 0.0, 0
    use_bf16_eval = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype_eval = torch.bfloat16 if use_bf16_eval else torch.float16
    all_preds_np_list, all_labels_np_list, all_logits_for_theta, all_difficulties_main_val = [], [], [], []

    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        labels_idx = column_order.index('labels')
        has_difficulty_in_val = 'difficulty' in column_order
        difficulty_idx_main_val = column_order.index('difficulty') if has_difficulty_in_val else -1
    except ValueError as e:
        print(f"FATAL ({desc_prefix}): Column missing for main val. Order: {column_order}. Error: {e}");
        return 0.0, float('inf'), 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc_prefix, leave=False):
            num_batches += 1
            input_ids, attention_mask, labels_batch = batch[input_ids_idx].to(device), batch[attention_mask_idx].to(
                device), batch[labels_idx].to(device)
            if has_difficulty_in_val and batch[difficulty_idx_main_val].numel() > 0: all_difficulties_main_val.append(
                batch[difficulty_idx_main_val].cpu().numpy())
            with autocast(device.type, dtype=amp_dtype_eval, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss, logits = outputs.loss, outputs.logits.float()
            if loss is not None: total_loss += loss.item()
            all_preds_np_list.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels_np_list.append(labels_batch.cpu().numpy())
            if has_difficulty_in_val and batch[difficulty_idx_main_val].numel() > 0: all_logits_for_theta.append(
                logits.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    val_accuracy = 0.0
    if all_labels_np_list and all_preds_np_list and np.concatenate(all_labels_np_list).size > 0:
        final_preds_np_cat = np.concatenate(all_preds_np_list)
        final_labels_np_cat = np.concatenate(all_labels_np_list)
        if len(final_preds_np_cat) == len(final_labels_np_cat) and len(final_labels_np_cat) > 0:
            val_accuracy = accuracy_metric.compute(predictions=final_preds_np_cat, references=final_labels_np_cat)[
                'accuracy']

    theta_main, theta_time = 0.0, 0.0
    can_calc_theta = (has_difficulty_in_val and
                      all_difficulties_main_val and all(
                isinstance(d, np.ndarray) and d.size > 0 for d in all_difficulties_main_val) and
                      all_logits_for_theta and all(
                isinstance(l, np.ndarray) and l.size > 0 for l in all_logits_for_theta) and
                      all_labels_np_list and all(
                isinstance(lb, np.ndarray) and lb.size > 0 for lb in all_labels_np_list) and
                      pudf_config_for_theta_val)

    if can_calc_theta:
        s_time = time.time()
        try:
            logits_c, labels_c, diffs_c_raw = map(np.concatenate,
                                                  [all_logits_for_theta, all_labels_np_list, all_difficulties_main_val])
            if diffs_c_raw.ndim > 1:
                diffs_c = diffs_c_raw.squeeze().astype(float)
            else:
                diffs_c = diffs_c_raw.astype(float)

            if len(logits_c) == len(labels_c) == len(diffs_c) and len(labels_c) > 0:
                preds_idx = np.argmax(logits_c, axis=1);
                rps = (preds_idx == labels_c).astype(int) * 2 - 1
                if rps.ndim > 1: rps = rps.squeeze()

                num_obs_theta_val = pudf_config_for_theta_val.num_obs_theta
                num_total_diffs = len(diffs_c)

                if num_total_diffs > 0:
                    if not (num_obs_theta_val > 0 and num_obs_theta_val < num_total_diffs):
                        num_obs_to_use = num_total_diffs
                    else:
                        num_obs_to_use = num_obs_theta_val
                    indices_theta_val = np.random.choice(num_total_diffs, num_obs_to_use, replace=False)
                    theta_main = calculate_theta(diffs_c[indices_theta_val], rps[indices_theta_val])[0]
                else:
                    print(
                        f"Warning ({desc_prefix}): No difficulty scores available for main val theta estimation after processing.")
            else:
                print(
                    f"Warning ({desc_prefix}): Mismatch lengths or empty data for main_val theta. P: {logits_c.shape}, L: {labels_c.shape}, D: {diffs_c.shape}")
        except Exception as e_th:
            print(f"Error calc theta for {desc_prefix}: {e_th}. Trace: {traceback.format_exc()}"); theta_main = 0.0
        theta_time = time.time() - s_time
    elif has_difficulty_in_val and not pudf_config_for_theta_val:
        print(
            f"Info ({desc_prefix}): Difficulty present in val set but no PUDF config provided for theta estimation on val.")
    elif not has_difficulty_in_val:
        print(f"Info ({desc_prefix}): No difficulty scores in validation set, skipping theta estimation on val.")

    print(
        f"Main Validation Results ({desc_prefix}): Acc={val_accuracy:.4f}, Loss={avg_loss:.4f}, Theta={theta_main:.4f} ({theta_time:.2f}s)")
    return val_accuracy, avg_loss, theta_main, theta_time


class HeuristicSampler(Sampler[int]):
    def __init__(self, num_samples_total: int, batch_size: int, sorted_indices: list[int], heuristic_config: dict,
                 num_replicas: int = 1, rank: int = 0, seed: int = RANDOM_SEED):
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas: raise ValueError("Invalid num_replicas or rank.")
        if not isinstance(batch_size, int) or batch_size <= 0: raise ValueError("batch_size should be positive.")
        self.num_replicas = num_replicas;
        self.rank = rank;
        self.epoch = 0;
        self.seed = seed;
        self.batch_size = batch_size
        self._full_data_len = num_samples_total;
        self._sorted_indices = sorted_indices;
        self.heuristic_config = heuristic_config
        min_train_percent_float = float(
            self.heuristic_config.get('min_train_percent', DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC))

        min_len_from_percent = int(min_train_percent_float * self._full_data_len)
        # Ensure batch_size*num_replicas is considered, especially if it's small relative to dataset
        min_len_from_batch_replicas = batch_size * num_replicas if self._full_data_len >= batch_size * num_replicas else self._full_data_len

        self._min_train_length = max(1, min_len_from_percent, min_len_from_batch_replicas)

        if self._min_train_length > self._full_data_len: self._min_train_length = self._full_data_len
        self.indices_for_epoch = [];
        self.num_samples_epoch_replica = 0;
        self.replica_indices_epoch = []
        self.set_epoch(0)

    def _get_num_samples_for_epoch(self, epoch: int) -> int:
        scheduler_type = self.heuristic_config['scheduler_type']
        num_total = self._full_data_len
        competency_epoch_val = float(self.heuristic_config.get('competency_param', DEFAULT_COMPETENCY_PARAM_HEURISTIC))
        competency_epoch = max(1.0, competency_epoch_val)
        c_init = float(self.heuristic_config.get('c_init', DEFAULT_C_INIT_HEURISTIC))
        current_epoch_float = float(epoch)

        if scheduler_type == 'linear':
            progress_ratio = current_epoch_float / competency_epoch
            epoch_competency = c_init + (1.0 - c_init) * progress_ratio if progress_ratio < 1.0 else 1.0
        elif scheduler_type == 'root':
            progress_ratio = current_epoch_float / competency_epoch
            sqrt_arg = max(0.0, progress_ratio)
            epoch_competency = c_init + (1.0 - c_init) * np.sqrt(sqrt_arg) if progress_ratio < 1.0 else 1.0
        else:
            raise NotImplementedError(f"Scheduler '{scheduler_type}' unknown.")
        num_train = int(epoch_competency * num_total)
        num_train = max(self._min_train_length, num_train);
        num_train = min(num_train, num_total)
        return num_train

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        num_samples_for_epoch = self._get_num_samples_for_epoch(epoch)
        new_indices = self._sorted_indices[:num_samples_for_epoch]
        if epoch == 0 or len(new_indices) != len(self.indices_for_epoch) or epoch % 5 == 0:
            print(
                f"[HeuristicSampler] Epoch {epoch}: Selecting {len(new_indices)} samples for training (out of {self._full_data_len}). Min length was {self._min_train_length}.")
        self.indices_for_epoch = new_indices

        num_indices_this_epoch = len(self.indices_for_epoch)
        if self.num_replicas > 1:
            if num_indices_this_epoch == 0:
                self.replica_indices_epoch = []
                self.num_samples_epoch_replica = 0
                return

            samples_per_replica_base = num_indices_this_epoch // self.num_replicas
            num_extra_samples = num_indices_this_epoch % self.num_replicas

            start_idx = self.rank * samples_per_replica_base + min(self.rank, num_extra_samples)
            self.num_samples_epoch_replica = samples_per_replica_base + (1 if self.rank < num_extra_samples else 0)
            end_idx = start_idx + self.num_samples_epoch_replica
            self.replica_indices_epoch = self.indices_for_epoch[start_idx:end_idx]
        else:
            self.replica_indices_epoch = self.indices_for_epoch
            self.num_samples_epoch_replica = num_indices_this_epoch

    def __iter__(self) -> Iterator[int]:
        if not self.replica_indices_epoch: return iter([])
        g = torch.Generator();
        g.manual_seed(self.seed + self.epoch)
        indices_shuffled = [self.replica_indices_epoch[i] for i in
                            torch.randperm(len(self.replica_indices_epoch), generator=g).tolist()]
        return iter(indices_shuffled)

    def __len__(self) -> int:
        return self.num_samples_epoch_replica


class CustomHeuristicTrainer(Trainer):
    def __init__(self, *args, sorted_indices_sampler=None, heuristic_config_sampler=None,
                 num_samples_total_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        if sorted_indices_sampler is None or heuristic_config_sampler is None or num_samples_total_sampler is None:
            raise ValueError(
                "CustomHeuristicTrainer requires sorted_indices_sampler, heuristic_config_sampler, and num_samples_total_sampler.")
        self.sorted_indices_sampler = sorted_indices_sampler
        self.heuristic_config_sampler = heuristic_config_sampler
        self.num_samples_total_sampler = num_samples_total_sampler
        print("CustomHeuristicTrainer initialized.")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None: raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset;
        data_collator = self.data_collator
        world_size = self.args.world_size
        process_index = self.args.process_index

        heuristic_sampler = HeuristicSampler(
            num_samples_total=self.num_samples_total_sampler,
            batch_size=self._train_batch_size,
            sorted_indices=self.sorted_indices_sampler,
            heuristic_config=self.heuristic_config_sampler,
            num_replicas=world_size,
            rank=process_index,
            seed=self.args.seed
        )
        return DataLoader(
            train_dataset, batch_size=self._train_batch_size, sampler=heuristic_sampler,
            collate_fn=data_collator, drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def compute_metrics_for_trainer(p):
    accuracy_metric_ht = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    labels = p.label_ids
    if not isinstance(labels, np.ndarray): labels = np.array(labels)
    if preds.size == 0 or labels.size == 0: return {"accuracy": 0.0}
    if labels.shape != preds.shape: print(f"Warning: Label shape {labels.shape} != Pred shape {preds.shape}"); return {
        "accuracy": 0.0}
    valid_idx = labels != -100
    if not np.any(valid_idx): return {"accuracy": 0.0}
    preds = preds[valid_idx];
    labels = labels[valid_idx]
    if preds.size == 0: return {"accuracy": 0.0}
    return accuracy_metric_ht.compute(predictions=preds, references=labels)


def run_training_with_heuristic_sampler(args, train_full_hf_with_difficulty, val_hf, test_hf, tokenizer_qwen,
                                        run_output_dir):
    print(f"\nStarting Heuristic Sampler Run: Diff='{args.difficulty_measurer}', Sched='{args.training_scheduler}'")
    print(f"Output Dir: {run_output_dir}")
    num_labels = train_full_hf_with_difficulty.features['label'].num_classes
    print(f"Sorting training data by 'difficulty' ({DEFAULT_HEURISTIC_ORDERING} first)...")
    difficulty_values_np = np.array(train_full_hf_with_difficulty['difficulty'])
    if difficulty_values_np.size == 0: raise ValueError("Difficulty column is empty for Heuristic Sampler.")
    if DEFAULT_HEURISTIC_ORDERING == 'easiest':
        sorted_original_indices = np.argsort(difficulty_values_np).tolist()
    elif DEFAULT_HEURISTIC_ORDERING == 'hardest':
        sorted_original_indices = np.argsort(difficulty_values_np)[::-1].tolist()
    else:
        raise NotImplementedError(f"Ordering '{DEFAULT_HEURISTIC_ORDERING}' not supported for heuristic sampler.")
    num_samples_total_for_sampler = len(train_full_hf_with_difficulty)

    def tokenize_fn_heuristic(examples):
        tokenized_output = tokenizer_qwen(
            examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
        )
        tokenized_output["labels"] = examples["label"]
        return tokenized_output

    print("Tokenizing datasets for heuristic trainer...")
    map_num_procs = args.num_workers if args.num_workers > 0 else None

    tokenized_train_all_cols = train_full_hf_with_difficulty.map(tokenize_fn_heuristic, batched=True,
                                                                 num_proc=map_num_procs)
    tokenized_val_all_cols = val_hf.map(tokenize_fn_heuristic, batched=True, num_proc=map_num_procs)
    tokenized_test_all_cols = test_hf.map(tokenize_fn_heuristic, batched=True, num_proc=map_num_procs)

    final_model_columns = ['input_ids', 'attention_mask', 'labels']
    tokenized_train = tokenized_train_all_cols.select_columns(final_model_columns)
    tokenized_val = tokenized_val_all_cols.select_columns(final_model_columns)
    tokenized_test = tokenized_test_all_cols.select_columns(final_model_columns)

    if 'labels' not in tokenized_train.column_names:
        raise ValueError("'labels' column missing from tokenized_train after selection.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_ready = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    model_config = AutoConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels, trust_remote_code=True)
    if tokenizer_qwen.pad_token_id is not None:
        model_config.pad_token_id = tokenizer_qwen.pad_token_id
    else:
        print("Warning: tokenizer_qwen.pad_token_id is None before model config assignment.")

    model_load_kwargs = {"config": model_config, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
    if args.use_bf16 and bf16_ready: model_load_kwargs["torch_dtype"] = torch.bfloat16

    print(
        f"Loading model {MODEL_CHECKPOINT} with torch_dtype: {model_load_kwargs.get('torch_dtype', 'default (float32)')} and attn_implementation: {model_load_kwargs.get('attn_implementation')}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, **model_load_kwargs)

    if hasattr(model, 'resize_token_embeddings') and len(tokenizer_qwen) > model.config.vocab_size:
        print(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer_qwen)}")
        model.resize_token_embeddings(len(tokenizer_qwen))

    if model.config.pad_token_id != tokenizer_qwen.pad_token_id:
        print(f"Syncing model.config.pad_token_id to tokenizer's: {tokenizer_qwen.pad_token_id}")
        model.config.pad_token_id = tokenizer_qwen.pad_token_id
    elif model.config.pad_token_id is None and tokenizer_qwen.pad_token_id is not None:
        print(f"Model config pad_token_id is None, setting from tokenizer: {tokenizer_qwen.pad_token_id}")
        model.config.pad_token_id = tokenizer_qwen.pad_token_id

    if model.config.pad_token_id is None:
        raise ValueError("Model config pad_token_id is None after all sync attempts. This will cause issues.")
    model.to(device)

    try:
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() and torch.distributed.is_available() else 1
    except Exception:
        world_size = 1

    train_batch_size_actual = args.train_batch_size if args.train_batch_size is not None else DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE_HEURISTIC
    grad_accum_actual = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else DEFAULT_GRADIENT_ACCUMULATION_STEPS_HEURISTIC
    num_epochs_actual = args.num_epochs if args.num_epochs is not None else DEFAULT_NUM_TRAIN_EPOCHS_HEURISTIC
    lr_actual = args.learning_rate if args.learning_rate is not None else DEFAULT_ADAFACOR_LR_HEURISTIC
    base_lr_for_args = args.learning_rate if args.learning_rate is not None else DEFAULT_LEARNING_RATE_HEURISTIC
    weight_decay_actual = args.weight_decay if args.weight_decay is not None else DEFAULT_WEIGHT_DECAY_HEURISTIC
    patience_actual = args.patience_early_stopping if args.patience_early_stopping is not None else DEFAULT_EARLY_STOPPING_PATIENCE_HEURISTIC
    grad_checkpoint_actual = args.use_gradient_checkpointing if args.use_gradient_checkpointing is not None else DEFAULT_GRADIENT_CHECKPOINTING_HEURISTIC

    total_train_batch_size_eff = train_batch_size_actual * world_size * grad_accum_actual
    if num_samples_total_for_sampler == 0 and total_train_batch_size_eff > 0:
        steps_per_epoch_full_data = 0
    elif total_train_batch_size_eff == 0 and num_samples_total_for_sampler > 0:
        raise ValueError("Total effective batch size is 0 but there are samples.")
    elif num_samples_total_for_sampler == 0 and total_train_batch_size_eff == 0:
        steps_per_epoch_full_data = 0
    else:
        steps_per_epoch_full_data = ceil(num_samples_total_for_sampler / total_train_batch_size_eff)

    calculated_max_steps = ceil(
        num_epochs_actual * steps_per_epoch_full_data) if steps_per_epoch_full_data > 0 else num_epochs_actual
    print(
        f"Effective Batch Size (Heuristic): {total_train_batch_size_eff}, Steps/Epoch (Full Data): {steps_per_epoch_full_data}")
    print(f"Calculated max_steps for {num_epochs_actual} epochs (Heuristic): {calculated_max_steps}")

    use_bf16_for_trainer = bf16_ready and args.use_bf16
    use_fp16_for_trainer = not use_bf16_for_trainer and torch.cuda.is_available()

    training_args_hf = TrainingArguments(
        output_dir=run_output_dir, eval_strategy="epoch", save_strategy="epoch",
        learning_rate=base_lr_for_args,
        per_device_train_batch_size=train_batch_size_actual,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE_HEURISTIC,
        gradient_accumulation_steps=grad_accum_actual,
        max_steps=max(1, calculated_max_steps),
        weight_decay=weight_decay_actual, logging_dir=os.path.join(run_output_dir, "logs"),
        logging_steps=DEFAULT_LOGGING_STEPS_HEURISTIC, gradient_checkpointing=grad_checkpoint_actual,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_checkpoint_actual and packaging.version.parse(
            torch.__version__) >= packaging.version.parse("2.0.0") else {},
        bf16=use_bf16_for_trainer,
        fp16=use_fp16_for_trainer,
        load_best_model_at_end=True, metric_for_best_model="eval_accuracy", greater_is_better=True,
        save_total_limit=2, dataloader_num_workers=args.num_workers,
        report_to="none", seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    adafactor_optimizer = Adafactor(
        model.parameters(), scale_parameter=False, relative_step=False, lr=lr_actual,
        warmup_init=False, weight_decay=weight_decay_actual
    )
    optimizers_tuple = (adafactor_optimizer, None)

    heuristic_config_dict_for_sampler = {
        "scheduler_type": args.training_scheduler,
        "ordering": DEFAULT_HEURISTIC_ORDERING,
        "competency_param": args.competency_param_heuristic if args.competency_param_heuristic is not None else DEFAULT_COMPETENCY_PARAM_HEURISTIC,
        "min_train_percent": args.min_train_percent_heuristic if args.min_train_percent_heuristic is not None else DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC,
        "c_init": args.c_init_heuristic if args.c_init_heuristic is not None else DEFAULT_C_INIT_HEURISTIC,
    }
    trainer = CustomHeuristicTrainer(
        model=model, args=training_args_hf, train_dataset=tokenized_train, eval_dataset=tokenized_val,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience_actual)],
        tokenizer=tokenizer_qwen, optimizers=optimizers_tuple,
        sorted_indices_sampler=sorted_original_indices,
        heuristic_config_sampler=heuristic_config_dict_for_sampler,
        num_samples_total_sampler=num_samples_total_for_sampler
    )
    if grad_checkpoint_actual and hasattr(trainer.model, 'gradient_checkpointing_enable'):
        print("Enabling gradient checkpointing for Heuristic Trainer model...")
        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args_hf.gradient_checkpointing_kwargs)
        if hasattr(trainer.model.config, "use_cache"): trainer.model.config.use_cache = False

    start_time_train = time.time()
    if calculated_max_steps > 0:
        train_result = trainer.train()
        total_steps_taken = train_result.global_step
        train_metrics_dict = train_result.metrics
    else:
        print("Warning: Calculated max_steps is 0. Skipping Heuristic training.")
        total_steps_taken = 0
        train_metrics_dict = {"train_loss": 0.0, "train_runtime": 0.0, "train_samples_per_second": 0.0,
                              "train_steps_per_second": 0.0}

    end_time_train = time.time()
    training_time = end_time_train - start_time_train

    trainer.save_model(os.path.join(run_output_dir, "best_model"))
    tokenizer_qwen.save_pretrained(os.path.join(run_output_dir, "best_model"))

    print(f"Heuristic Trainer training metrics: {train_metrics_dict}")
    trainer.log_metrics("train_final", train_metrics_dict);
    trainer.save_metrics("train_final", train_metrics_dict)

    eval_results_on_val = trainer.evaluate(eval_dataset=tokenized_val, metric_key_prefix="eval")
    print(f"Heuristic Trainer validation metrics (best model from training): {eval_results_on_val}")
    trainer.log_metrics("eval_final_best", eval_results_on_val);
    trainer.save_metrics("eval_final_best", eval_results_on_val)

    test_results = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print(f"Heuristic Trainer test metrics (best model from training): {test_results}")
    trainer.log_metrics("test_final", test_results);
    trainer.save_metrics("test_final", test_results)

    summary = {
        "difficulty_measurer": args.difficulty_measurer, "training_scheduler": args.training_scheduler,
        "best_val_accuracy": eval_results_on_val.get("eval_accuracy",
                                                     eval_results_on_val.get("eval_eval_accuracy", -1.0)),
        "test_accuracy": test_results.get("test_accuracy", -1.0),
        "training_time_s": training_time, "total_steps": total_steps_taken,
        "output_dir": run_output_dir
    }
    return summary


def run_training_with_pudf_scheduler(args, train_full_hf_with_difficulty, val_hf, test_hf, tokenizer_qwen,
                                     run_output_dir):
    print(f"\nStarting PUDF Scheduler Run: Diff='{args.difficulty_measurer}', Sched='pudf_theta'")
    print(f"Output Dir: {run_output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = False
    bf16_ready = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if device.type == 'cuda' and bf16_ready and args.use_bf16:
        use_bf16 = True
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    num_epochs_actual = args.num_epochs if args.num_epochs is not None else DEFAULT_NUM_TRAIN_EPOCHS_PUDF
    pudf_config_obj = types.SimpleNamespace(
        strategy=DEFAULT_PUDF_STRATEGY,
        ordering=args.pudf_ordering if args.pudf_ordering is not None else DEFAULT_PUDF_ORDERING,
        num_obs_theta=args.pudf_num_obs_theta if args.pudf_num_obs_theta is not None else DEFAULT_THETA_ESTIMATION_SET_SIZE,
        min_train_length=args.pudf_min_train_length if args.pudf_min_train_length is not None else DEFAULT_PUDF_MIN_TRAIN_LENGTH,
        lower_bound=args.pudf_lower_bound if args.pudf_lower_bound is not None else DEFAULT_PUDF_LOWER_BOUND,
        upper_bound=args.pudf_upper_bound if args.pudf_upper_bound is not None else DEFAULT_PUDF_UPPER_BOUND,
        balanced=False, use_length=False, use_word_rarity=False,
        task_name_for_pudf="ag_news_pudf",
        num_epochs=num_epochs_actual,
        competency=num_epochs_actual / 2.0
    )

    if len(train_full_hf_with_difficulty) <= pudf_config_obj.num_obs_theta:
        original_num_obs = pudf_config_obj.num_obs_theta
        pudf_config_obj.num_obs_theta = max(1, int(len(train_full_hf_with_difficulty) * 0.1))
        print(
            f"Warning: Requested pudf_num_obs_theta ({original_num_obs}) is too large. Adjusted to {pudf_config_obj.num_obs_theta}.")
        if pudf_config_obj.num_obs_theta == 0 and len(
            train_full_hf_with_difficulty) > 0: pudf_config_obj.num_obs_theta = 1
        if len(train_full_hf_with_difficulty) <= pudf_config_obj.num_obs_theta and len(
                train_full_hf_with_difficulty) > 0:
            pudf_config_obj.num_obs_theta = max(1, len(train_full_hf_with_difficulty) - 1)
            print(f"Further adjusted pudf_num_obs_theta to {pudf_config_obj.num_obs_theta} due to small dataset size.")
            if pudf_config_obj.num_obs_theta == 0 and len(train_full_hf_with_difficulty) == 1:
                raise ValueError("Cannot split dataset for theta estimation if only 1 sample exists.")

    train_theta_hf_split = train_full_hf_with_difficulty.train_test_split(
        test_size=pudf_config_obj.num_obs_theta, seed=args.seed, shuffle=True
    )
    actual_train_hf = train_theta_hf_split['train']
    theta_estimation_hf = train_theta_hf_split['test']
    if len(actual_train_hf) == 0 and len(train_full_hf_with_difficulty) > 0:
        raise ValueError(
            f"Actual training set (actual_train_hf) is empty after splitting. Total: {len(train_full_hf_with_difficulty)}, Theta Split: {pudf_config_obj.num_obs_theta}")

    print(f"PUDF splits: Actual Train={len(actual_train_hf)}, Theta Estimation Set={len(theta_estimation_hf)}")
    num_labels = train_full_hf_with_difficulty.features['label'].num_classes

    def tokenize_fn_pudf(ex):
        tokenized = tokenizer_qwen(ex["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        return tokenized

    tokenized_datasets_dict = {}
    map_num_procs = args.num_workers if args.num_workers > 0 else None

    for name, ds_split in [("actual_train", actual_train_hf), ("theta_estimation", theta_estimation_hf),
                           ("validation_main", val_hf), ("test", test_hf)]:
        print(f"Tokenizing split for PUDF: {name}")
        current_cols_to_remove = [col for col in ["text"] if col in ds_split.column_names]
        tokenized_ds = ds_split.map(
            tokenize_fn_pudf, batched=True, num_proc=map_num_procs, remove_columns=current_cols_to_remove
        )
        if 'label' in tokenized_ds.column_names:
            tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_datasets_dict[name] = tokenized_ds

    global_tokenized_test_pudf = tokenized_datasets_dict["test"]
    actual_train_tds, actual_train_col_order = create_qwen_tensor_dataset(tokenized_datasets_dict["actual_train"],
                                                                          "actual_train_pool_pudf", True)
    theta_est_tds, theta_est_cols = create_qwen_tensor_dataset(tokenized_datasets_dict["theta_estimation"],
                                                               "theta_est_set_pudf", True)
    val_has_difficulty = 'difficulty' in tokenized_datasets_dict["validation_main"].column_names
    val_main_tds, val_main_cols = create_qwen_tensor_dataset(tokenized_datasets_dict["validation_main"],
                                                             "val_main_eval_pudf", val_has_difficulty)

    eval_batch_size_actual = args.eval_batch_size if args.eval_batch_size is not None else DEFAULT_PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE_PUDF
    theta_est_dl = DataLoader(theta_est_tds, eval_batch_size_actual, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    main_val_dl = DataLoader(val_main_tds, eval_batch_size_actual, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    model_cfg = AutoConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels, trust_remote_code=True)
    if tokenizer_qwen.pad_token_id is not None:
        model_cfg.pad_token_id = tokenizer_qwen.pad_token_id
    else:
        print("Warning: tokenizer_qwen.pad_token_id is None before PUDF model config assignment.")

    model_load_kwargs = {"config": model_cfg, "trust_remote_code": True, "attn_implementation": "flash_attention_2"}
    if use_bf16: model_load_kwargs["torch_dtype"] = torch.bfloat16
    print(
        f"Loading model {MODEL_CHECKPOINT} for PUDF with torch_dtype: {model_load_kwargs.get('torch_dtype', 'default (float32)')} and attn_implementation: {model_load_kwargs.get('attn_implementation')}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, **model_load_kwargs)

    if hasattr(model, 'resize_token_embeddings') and len(tokenizer_qwen) > model.config.vocab_size:
        print(f"Resizing PUDF model token embeddings from {model.config.vocab_size} to {len(tokenizer_qwen)}")
        model.resize_token_embeddings(len(tokenizer_qwen))

    if model.config.pad_token_id != tokenizer_qwen.pad_token_id:
        print(f"Syncing PUDF model.config.pad_token_id to tokenizer's: {tokenizer_qwen.pad_token_id}")
        model.config.pad_token_id = tokenizer_qwen.pad_token_id
    elif model.config.pad_token_id is None and tokenizer_qwen.pad_token_id is not None:
        print(f"PUDF Model config pad_token_id is None, setting from tokenizer: {tokenizer_qwen.pad_token_id}")
        model.config.pad_token_id = tokenizer_qwen.pad_token_id

    if model.config.pad_token_id is None:
        raise ValueError("PUDF Model config pad_token_id is None after all sync attempts.")

    use_grad_checkpointing_actual = args.use_gradient_checkpointing if args.use_gradient_checkpointing is not None else DEFAULT_USE_GRADIENT_CHECKPOINTING_PUDF
    if use_grad_checkpointing_actual and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False} if packaging.version.parse(
                torch.__version__) >= packaging.version.parse("2.0.0") else {})
        if hasattr(model.config, "use_cache"): model.config.use_cache = False
    model.to(device)

    lr_adafactor_actual = args.learning_rate if args.learning_rate is not None else DEFAULT_LEARNING_RATE_ADAFACTOR_PUDF
    weight_decay_actual = args.weight_decay if args.weight_decay is not None else DEFAULT_WEIGHT_DECAY_PUDF
    optimizer = Adafactor(model.parameters(), lr=lr_adafactor_actual, scale_parameter=False, relative_step=False,
                          warmup_init=False, weight_decay=weight_decay_actual)

    train_batch_size_physical_actual = args.train_batch_size if args.train_batch_size is not None else DEFAULT_PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE_PUDF
    grad_accum_steps_actual = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else DEFAULT_GRADIENT_ACCUMULATION_STEPS_PUDF

    if len(actual_train_tds) == 0 and train_batch_size_physical_actual * grad_accum_steps_actual > 0:
        num_approx_update_steps_epoch = 0
    elif train_batch_size_physical_actual * grad_accum_steps_actual == 0 and len(actual_train_tds) > 0:
        raise ValueError("Effective batch size for PUDF scheduler is 0 but there is training data.")
    elif len(actual_train_tds) == 0 and train_batch_size_physical_actual * grad_accum_steps_actual == 0:
        num_approx_update_steps_epoch = 0
    else:
        num_approx_update_steps_epoch = ceil(
            len(actual_train_tds) / (train_batch_size_physical_actual * grad_accum_steps_actual))

    total_train_steps_approx = max(1, num_approx_update_steps_epoch * num_epochs_actual)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=max(1, int(0.06 * total_train_steps_approx)),
                                                   num_training_steps=total_train_steps_approx)
    scaler = GradScaler(enabled=torch.cuda.is_available() and not use_bf16)
    best_val_acc = 0.0;
    early_stop_count = 0
    patience_early_stop_actual = args.patience_early_stopping if args.patience_early_stopping is not None else DEFAULT_PATIENCE_EARLY_STOPPING_PUDF
    train_stats_pudf = [];
    prev_cap = -5.0;
    cur_cap = 0.0
    overall_start_time_pudf = time.time();
    epochs_completed = 0

    for epoch in range(num_epochs_actual):
        epochs_completed = epoch + 1
        print(f"\n======== PUDF Epoch {epochs_completed}/{num_epochs_actual} ========")
        model.train()

        est_theta_guidance = prev_cap
        if len(theta_est_dl.dataset) > 0:
            print("Estimating capacity (theta) for PUDF scheduler...")
            current_est_theta, _ = evaluate_and_estimate_qwen_theta_pudf(
                model, theta_est_dl, device, theta_est_cols, num_labels,
                pudf_config_obj.num_obs_theta, f"Ep{epochs_completed} ThetaEst"
            )
            model.train()
            est_theta_guidance = current_est_theta
            if est_theta_guidance > prev_cap:
                cur_cap = est_theta_guidance
            else:
                cur_cap = max(prev_cap + 0.05, est_theta_guidance + 0.1)
            prev_cap = est_theta_guidance
        else:
            print("Warning: Theta estimation set is empty. Progressing capacity with a fallback.")
            min_difficulty_assumed = -2.5
            max_difficulty_assumed = 2.5
            cur_cap = min_difficulty_assumed + (max_difficulty_assumed - min_difficulty_assumed) * (
                        epochs_completed / max(1, num_epochs_actual))
            est_theta_guidance = cur_cap

        pudf_args_for_epoch = copy.deepcopy(pudf_config_obj);
        pudf_args_for_epoch.epoch = epoch
        filtered_data = get_epoch_training_data(
            actual_train_tds, pudf_args_for_epoch, epoch, pudf_config_obj.task_name_for_pudf,
            theta_hat=cur_cap, lower_offset=pudf_config_obj.lower_bound, upper_offset=pudf_config_obj.upper_bound
        )
        n_epoch_samples = len(filtered_data['labels']) if 'labels' in filtered_data and filtered_data[
            'labels'] is not None else 0

        if n_epoch_samples == 0:
            print(
                f"Warning: PUDF selected 0 samples for training in epoch {epochs_completed}. Avg epoch loss set to 0.");
            avg_epoch_loss = 0.0
            val_acc, val_loss_val, theta_main, _ = evaluate_qwen_main_val_pudf(
                model, main_val_dl, device, val_main_cols, num_labels, f"Ep{epochs_completed} MainVal_PUDF (NoTrain)",
                pudf_config_obj
            )
            train_stats_pudf.append({
                'epoch': epochs_completed, 'train_loss': avg_epoch_loss, 'val_loss': val_loss_val, 'val_acc': val_acc,
                'cur_cap_used': cur_cap, 'theta_guidance_est': est_theta_guidance, 'theta_main_val': theta_main,
                'n_samples': n_epoch_samples
            })
            if val_acc <= best_val_acc: early_stop_count += 1
            if early_stop_count >= patience_early_stop_actual: print(
                "PUDF Early stopping due to no improvement."); break
            continue

        print(f"PUDF selected {n_epoch_samples} samples for epoch {epochs_completed}. Capacity used: {cur_cap:.4f}")
        epoch_tensors = [filtered_data['input_ids'], filtered_data['attention_mask'], filtered_data['labels'],
                         filtered_data['difficulty']]
        epoch_ds_filtered = TensorDataset(*epoch_tensors)
        epoch_dl = DataLoader(epoch_ds_filtered, train_batch_size_physical_actual, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
        epoch_loss_sum = 0;
        optim_steps = 0;
        optimizer.zero_grad()
        prog_bar = tqdm(epoch_dl, f"Epoch {epochs_completed} PUDF Training", leave=False)
        for step, batch_data in enumerate(prog_bar):
            if len(batch_data) < 3:
                print(
                    f"Warning: Skipping malformed batch in PUDF training (expected >=3 tensors, got {len(batch_data)}). Batch content: {batch_data}")
                continue
            ids, mask, lbls = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)

            with autocast(device.type, dtype=amp_dtype, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
                loss_val = outputs.loss
            if loss_val is None or torch.isnan(loss_val): print(
                f"NaN/None loss step {step}. Skip."); optimizer.zero_grad(set_to_none=True); continue

            loss_val_acc = loss_val / grad_accum_steps_actual
            if scaler.is_enabled():
                scaler.scale(loss_val_acc).backward()
            else:
                loss_val_acc.backward()
            epoch_loss_sum += loss_val.item()

            if (step + 1) % grad_accum_steps_actual == 0 or (step + 1) == len(epoch_dl):
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step();
                optimizer.zero_grad(set_to_none=True);
                optim_steps += 1
            prog_bar.set_postfix(
                {'loss': f'{epoch_loss_sum / (step + 1):.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'})

        avg_epoch_loss = epoch_loss_sum / max(1, len(epoch_dl))

        val_acc, val_loss_val, theta_main, _ = evaluate_qwen_main_val_pudf(
            model, main_val_dl, device, val_main_cols, num_labels, f"Ep{epochs_completed} MainVal_PUDF", pudf_config_obj
        )
        train_stats_pudf.append({
            'epoch': epochs_completed, 'train_loss': avg_epoch_loss, 'val_loss': val_loss_val, 'val_acc': val_acc,
            'cur_cap_used': cur_cap, 'theta_guidance_est': est_theta_guidance, 'theta_main_val': theta_main,
            'n_samples': n_epoch_samples
        })
        if val_acc > best_val_acc:
            print(f"PUDF Val acc improved ({best_val_acc:.4f} -> {val_acc:.4f}). Saving model...");
            best_val_acc = val_acc;
            early_stop_count = 0
            model.save_pretrained(os.path.join(run_output_dir, "best_model"))
            tokenizer_qwen.save_pretrained(os.path.join(run_output_dir, "best_model"))
        else:
            early_stop_count += 1
            if early_stop_count >= patience_early_stop_actual: print("PUDF Early stopping."); break
        gc.collect();
        torch.cuda.empty_cache()

    total_loop_t_pudf = time.time() - overall_start_time_pudf
    print(f"\n--- PUDF Training Finished ({epochs_completed} epochs completed) ---")
    print(f"Total PUDF training loop duration: {total_loop_t_pudf:.2f}s")

    test_acc_pudf, test_loss_pudf = -1.0, -1.0
    best_model_path_pudf = os.path.join(run_output_dir, "best_model")
    if os.path.exists(best_model_path_pudf) and any(
            f.startswith(("pytorch_model", "model.safetensors")) for f in os.listdir(best_model_path_pudf)):
        print("\nLoading best PUDF model for test eval...")
        model_test_pudf_load_args = {"trust_remote_code": True}
        if use_bf16: model_test_pudf_load_args["torch_dtype"] = torch.bfloat16

        try:
            loaded_config = AutoConfig.from_pretrained(best_model_path_pudf, trust_remote_code=True)
            model_test_pudf_load_args["config"] = loaded_config
            model_test_pudf_load_args["attn_implementation"] = "flash_attention_2"
            model_test_pudf = AutoModelForSequenceClassification.from_pretrained(best_model_path_pudf,
                                                                                 **model_test_pudf_load_args)
            model_test_pudf.to(device)

            test_tds_pudf, test_cols_pudf = create_qwen_tensor_dataset(global_tokenized_test_pudf, "test_final_pudf",
                                                                       include_difficulty=False)
            test_dl_pudf = DataLoader(test_tds_pudf, eval_batch_size_actual, num_workers=args.num_workers,
                                      pin_memory=True)
            test_acc_pudf, test_loss_pudf, _, _ = evaluate_qwen_main_val_pudf(model_test_pudf, test_dl_pudf, device,
                                                                              test_cols_pudf, num_labels,
                                                                              "Final Test Eval PUDF")
            print(f"PUDF Test Results: Acc={test_acc_pudf:.4f}, Loss={test_loss_pudf:.4f}")
        except Exception as load_err:
            print(f"Error loading best PUDF model for test: {load_err}")
            traceback.print_exc()
    else:
        print(f"No best model checkpoint found at {best_model_path_pudf} for PUDF run.")

    summary = {
        "difficulty_measurer": args.difficulty_measurer, "training_scheduler": "pudf_theta",
        "best_val_accuracy": best_val_acc, "test_accuracy": test_acc_pudf,
        "training_time_s": total_loop_t_pudf, "epochs_completed": epochs_completed,
        "output_dir": run_output_dir
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Qwen AGNews Ablation Study Script with Scaled Difficulties")
    parser.add_argument('--difficulty_measurer', type=str, required=True,
                        choices=['pudf_irt', 'sentence_length', 'word_rarity'], help='Difficulty measurement method.')
    parser.add_argument('--training_scheduler', type=str, required=True, choices=['linear', 'root', 'pudf_theta'],
                        help='Training curriculum scheduler.')
    parser.add_argument('--output_dir_root', type=str, default="./qwen_agnews_ablation_runs_scaled",
                        help='Root directory for saving results.')
    parser.add_argument('--pudf_difficulty_file', type=str, default=DEFAULT_PUDF_DIFFICULTY_FILE_PATH,
                        help='Path to pre-computed PUDF IRT difficulty scores JSON file.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for reproducibility.')
    parser.add_argument('--use_bf16', action='store_true', default=False, help='Use BF16 if available.')

    # Changed default for num_workers to 4 as requested
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers (0 for main process, >0 for multiprocessing in DataLoader and .map()).')

    parser.add_argument('--num_epochs', type=int, default=None, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for Adafactor.')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimizer.')
    parser.add_argument('--train_batch_size', type=int, default=None, help='Per device physical training batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Per device evaluation batch size.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='Gradient accumulation steps.')
    parser.add_argument('--use_gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='Enable gradient checkpointing (True/False).')
    parser.add_argument('--patience_early_stopping', type=int, default=None, help='Patience for early stopping.')

    parser.add_argument('--pudf_ordering', type=str, choices=['easiest', 'hardest', 'middleout'], default=None,
                        help=f'Data ordering for PUDF scheduler (default: {DEFAULT_PUDF_ORDERING}).')
    parser.add_argument('--pudf_num_obs_theta', type=int, default=None,
                        help=f'Number of observations for theta estimation in PUDF (default: {DEFAULT_THETA_ESTIMATION_SET_SIZE}).')
    parser.add_argument('--pudf_min_train_length', type=int, default=None,
                        help=f'Min training samples per epoch for PUDF (default: {DEFAULT_PUDF_MIN_TRAIN_LENGTH}).')
    parser.add_argument('--pudf_lower_bound', type=float, default=None,
                        help=f'Lower difficulty bound offset for PUDF theta strategy (default: {DEFAULT_PUDF_LOWER_BOUND}).')
    parser.add_argument('--pudf_upper_bound', type=float, default=None,
                        help=f'Upper difficulty bound offset for PUDF theta strategy (default: {DEFAULT_PUDF_UPPER_BOUND}).')

    parser.add_argument('--competency_param_heuristic', type=float, default=None,
                        help=f'Competency parameter for heuristic schedulers (default: {DEFAULT_COMPETENCY_PARAM_HEURISTIC}).')
    parser.add_argument('--min_train_percent_heuristic', type=float, default=None,
                        help=f'Minimum percentage of data for heuristic schedulers (default: {DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC}).')
    parser.add_argument('--c_init_heuristic', type=float, default=None,
                        help=f'Initial data percentage for heuristic schedulers (default: {DEFAULT_C_INIT_HEURISTIC}).')

    args = parser.parse_args()

    run_specific_output_dir = os.path.join(args.output_dir_root,
                                           f"{args.difficulty_measurer}_{args.training_scheduler}_seed{args.seed}")
    setup_environment(run_specific_output_dir)
    set_seed(args.seed)

    tokenizer_qwen = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True, use_fast=True,
                                                   cache_dir=os.environ.get("TRANSFORMERS_CACHE"))
    if tokenizer_qwen.pad_token is None:
        if tokenizer_qwen.eos_token is not None:
            tokenizer_qwen.pad_token = tokenizer_qwen.eos_token
            if tokenizer_qwen.pad_token_id is None: tokenizer_qwen.pad_token_id = tokenizer_qwen.eos_token_id
        else:
            pad_token_str = '[PAD]'
            tokenizer_qwen.add_special_tokens({'pad_token': pad_token_str})
            if tokenizer_qwen.pad_token_id is None: tokenizer_qwen.pad_token_id = tokenizer_qwen.convert_tokens_to_ids(
                pad_token_str)
        print(f"Set Qwen pad_token to: '{tokenizer_qwen.pad_token}' (ID: {tokenizer_qwen.pad_token_id})")
    elif tokenizer_qwen.pad_token_id is None and tokenizer_qwen.pad_token is not None:
        tokenizer_qwen.pad_token_id = tokenizer_qwen.convert_tokens_to_ids(tokenizer_qwen.pad_token)
        print(
            f"Set Qwen pad_token_id for existing pad_token '{tokenizer_qwen.pad_token}' to: {tokenizer_qwen.pad_token_id}")

    if tokenizer_qwen.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id could not be set. This is required.")

    print("Loading AG News dataset...")
    raw_ds = load_dataset(DATASET_ID, cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    complete_ds_prov = raw_ds['complete']
    if 'news_story' in complete_ds_prov.column_names and 'text' not in complete_ds_prov.column_names:
        complete_ds_prov = complete_ds_prov.rename_column("news_story", "text")
    if 'labeling' in complete_ds_prov.column_names and 'label' not in complete_ds_prov.column_names:
        complete_ds_prov = complete_ds_prov.rename_column("labeling", "label")
    if 'text' not in complete_ds_prov.column_names or 'label' not in complete_ds_prov.column_names:
        raise ValueError(f"Essential 'text' or 'label' not found. Columns: {complete_ds_prov.column_names}")

    def cast_label_to_int(example):
        example['label'] = int(example['label']); return example

    map_num_procs_for_main = args.num_workers if args.num_workers > 0 else None
    complete_ds_processed = complete_ds_prov.map(cast_label_to_int, num_proc=map_num_procs_for_main)

    complete_ds_shuffled = complete_ds_processed.shuffle(seed=args.seed)
    train_val_split = complete_ds_shuffled.train_test_split(test_size=0.2, seed=args.seed)
    train_full_hf = train_val_split['train']
    temp_hf = train_val_split['test']
    val_test_split = temp_hf.train_test_split(test_size=0.5, seed=args.seed)
    val_hf = val_test_split['train'];
    test_hf = val_test_split['test']
    print(f"Dataset splits: Train_Full={len(train_full_hf)}, Validation={len(val_hf)}, Test={len(test_hf)}")

    target_min_difficulty_scale = -2.5
    target_max_difficulty_scale = 2.5
    pudf_scores_for_range = None
    try:
        print(f"Loading PUDF IRT difficulty scores from: {args.pudf_difficulty_file} to determine target scale.")
        if not os.path.exists(args.pudf_difficulty_file):
            print(
                f"Warning: PUDF difficulty file not found at {args.pudf_difficulty_file}. Using default scale [{target_min_difficulty_scale:.4f}, {target_max_difficulty_scale:.4f}] for scaling heuristic difficulties if needed.")
        else:
            with open(os.path.abspath(args.pudf_difficulty_file), 'r') as f:
                diff_data = json.load(f)
            pudf_scores_for_range = np.array(diff_data[DIFFICULTY_JSON_KEY], dtype=float)
            if pudf_scores_for_range.size > 0:
                target_min_difficulty_scale = pudf_scores_for_range.min()
                target_max_difficulty_scale = pudf_scores_for_range.max()
                if target_min_difficulty_scale == target_max_difficulty_scale:
                    print(
                        f"Warning: All loaded PUDF IRT scores are identical ({target_min_difficulty_scale:.4f}). Adjusting target scale slightly to create a range.")
                    target_min_difficulty_scale -= 0.5
                    target_max_difficulty_scale += 0.5
                    if target_min_difficulty_scale == target_max_difficulty_scale:
                        target_min_difficulty_scale = -1.0;
                        target_max_difficulty_scale = 1.0
                print(
                    f"Determined target difficulty scale from PUDF file: [{target_min_difficulty_scale:.4f}, {target_max_difficulty_scale:.4f}]")
            else:
                print(
                    f"Warning: PUDF difficulty file loaded but contained no scores. Using default scale [{target_min_difficulty_scale:.4f}, {target_max_difficulty_scale:.4f}].")
    except Exception as e:
        print(
            f"Warning: Could not load or process PUDF difficulty file '{args.pudf_difficulty_file}' to determine target scale: {e}. Using default scale [{target_min_difficulty_scale:.4f}, {target_max_difficulty_scale:.4f}].")

    if args.difficulty_measurer == 'pudf_irt':
        if pudf_scores_for_range is not None and len(pudf_scores_for_range) == len(train_full_hf):
            print("Using PUDF IRT scores directly as difficulty.")
            difficulty_values_for_column = pudf_scores_for_range.tolist()
        else:
            err_msg = f"PUDF IRT scores selected ('{args.difficulty_measurer}') but could not be loaded correctly, file was empty, or length mismatch. "
            err_msg += f"Expected {len(train_full_hf)} scores, "
            err_msg += f"got {len(pudf_scores_for_range) if pudf_scores_for_range is not None else 'None'} from file '{args.pudf_difficulty_file}'."
            raise ValueError(err_msg)
    elif args.difficulty_measurer in ['sentence_length', 'word_rarity']:
        print(f"Calculating raw '{args.difficulty_measurer}' scores...")
        raw_heuristic_scores = calculate_heuristic_difficulty_scores(train_full_hf, args.difficulty_measurer, 'text')
        raw_scores_np = np.array(raw_heuristic_scores)
        if raw_scores_np.size == 0:
            if len(train_full_hf) > 0:
                raise ValueError(
                    f"Heuristic difficulty measurer '{args.difficulty_measurer}' resulted in empty scores for a non-empty dataset.")
            else:
                print(
                    f"Warning: Heuristic difficulty measurer '{args.difficulty_measurer}' resulted in empty scores because the input dataset is empty.")
                difficulty_values_for_column = []
        else:
            print(
                f"Scaling '{args.difficulty_measurer}' scores to target range [{target_min_difficulty_scale:.4f}, {target_max_difficulty_scale:.4f}]...")
            scaled_heuristic_scores_np = min_max_scale(raw_scores_np, target_min_difficulty_scale,
                                                       target_max_difficulty_scale)
            difficulty_values_for_column = scaled_heuristic_scores_np.tolist()
            print(
                f"Original '{args.difficulty_measurer}' range: [{raw_scores_np.min():.2f}, {raw_scores_np.max():.2f}]")
            print(
                f"Scaled '{args.difficulty_measurer}' range: [{min(difficulty_values_for_column) if difficulty_values_for_column else 'N/A':.2f}, {max(difficulty_values_for_column) if difficulty_values_for_column else 'N/A':.2f}]")
    else:
        raise ValueError(f"Unsupported difficulty_measurer: {args.difficulty_measurer}")

    if len(train_full_hf) > 0:
        if len(difficulty_values_for_column) != len(train_full_hf):
            raise ValueError(
                f"Length of difficulty scores ({len(difficulty_values_for_column)}) does not match dataset length ({len(train_full_hf)}).")
        train_full_hf_with_difficulty = train_full_hf.add_column("difficulty", difficulty_values_for_column)
    else:
        print("Warning: train_full_hf is empty. Difficulty column not added.")
        train_full_hf_with_difficulty = train_full_hf

    print(
        f"Successfully prepared train_full_hf with '{args.difficulty_measurer}' difficulty scores (scaled where appropriate).")

    final_summary = None
    if len(train_full_hf_with_difficulty) == 0:
        print(
            "ERROR: No training data available after processing (train_full_hf_with_difficulty is empty). Aborting run.")
        final_summary = {
            "difficulty_measurer": args.difficulty_measurer, "training_scheduler": args.training_scheduler,
            "error": "No training data available for main training loop.",
            "best_val_accuracy": 0.0, "test_accuracy": 0.0, "training_time_s": 0, "epochs_completed": 0,
            "output_dir": run_specific_output_dir
        }
    elif args.training_scheduler in ['linear', 'root']:
        final_summary = run_training_with_heuristic_sampler(args, train_full_hf_with_difficulty, val_hf, test_hf,
                                                            tokenizer_qwen, run_specific_output_dir)
    elif args.training_scheduler == 'pudf_theta':
        final_summary = run_training_with_pudf_scheduler(args, train_full_hf_with_difficulty, val_hf, test_hf,
                                                         tokenizer_qwen, run_specific_output_dir)
    else:
        raise ValueError(f"Unsupported training_scheduler: {args.training_scheduler}")

    if final_summary:
        summary_file_path = os.path.join(run_specific_output_dir, "final_run_summary.json")
        with open(summary_file_path, 'w') as f:
            # Ensure all numpy types are converted for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            serializable_summary = json.loads(json.dumps(final_summary, default=convert_numpy_types))
            json.dump(serializable_summary, f, indent=4)

        print(f"Final summary for the run saved to: {summary_file_path}")
    print(f"\n===== Ablation Run Finished: {args.difficulty_measurer} + {args.training_scheduler} =====")


if __name__ == "__main__":
    main()