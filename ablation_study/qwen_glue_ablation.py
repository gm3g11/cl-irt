# glue_qwen_ablation.py
import numpy as np
import random
import os
import time
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import evaluate
import types
import gc
import traceback
from tqdm.auto import tqdm
from huggingface_hub import whoami
from math import ceil
import argparse
import re  # For word_rarity tokenizer
import shutil  # Added for deleting model directories

# --- Assumed External Custom Modules ---
# ADVICE: To reduce verbose debug output, please comment out or remove
# print statements within your build_features_Qwen.py script.
try:
    from build_features_Qwen import get_epoch_training_data  # Make sure this is the fixed version
    from irt_scoring import calculate_theta
except ImportError:
    print("ERROR: build_features_Qwen.py and/or irt_scoring.py not found.")
    print("Please ensure these files are in the correct location.")
    exit(1)


# --- Environment and Cache Setup ---
def setup_hf_environment(hf_home_path):
    os.environ["HF_HOME"] = hf_home_path
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home_path, "models")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home_path, "datasets")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


# --- Global Settings ---
transformers.logging.set_verbosity_error()
GLUETASKS_DEFAULT = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp']
TASK_MAX_LENGTHS = {"mrpc": 72, "rte": 150, "mnli": 72, "qqp": 56, "sst2": 32, "qnli": 80}
DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_EVAL_BATCH_SIZE = 256
CHECKPOINT_SUBDIR_NAME = "best_model_checkpoint"  # Name for the subdirectory to save model/tokenizer


# --- Integrated get_epoch_training_data function (Fixed and with reduced prints) ---
def get_epoch_training_data(ts, args, epoch, task, theta_hat=None, diffs_sorted_idx=None, lower_offset=-np.inf,
                            upper_offset=0):
    strategy = getattr(args, 'strategy', 'baseline')
    ordering = getattr(args, 'ordering', 'easiest')
    is_balanced = getattr(args, 'balanced', False)
    min_train_len = getattr(args, 'min_train_length', 100)
    competency_param = getattr(args, 'competency', 5)

    if not isinstance(ts, TensorDataset):
        print(f"Error: Expected TensorDataset, got {type(ts)}")
        return {}

    tensors = ts.tensors
    # Assuming train_col_order ensures correct structure for Qwen (ids, mask, labels, difficulty)
    if len(tensors) != 4:
        print(f"Error: get_epoch_training_data expects 4 tensors (ids, mask, labels, diff), got {len(tensors)}")
        return {}

    input_ids, attention_mask, labels, difficulties_tensor = tensors
    num_total_examples = len(input_ids)

    if strategy == 'baseline':
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
                'difficulty': difficulties_tensor}

    if difficulties_tensor is None or difficulties_tensor.numel() == 0:
        print("Warning: Difficulties tensor is None or empty for curriculum scheduling.")
        return {}
    difficulties_np = difficulties_tensor.cpu().numpy()
    if difficulties_np.size == 0: return {}

    current_diffs_sorted_idx = diffs_sorted_idx
    if current_diffs_sorted_idx is None:
        if ordering == 'easiest':
            current_diffs_sorted_idx = np.argsort(difficulties_np)
        elif ordering == 'hardest':
            current_diffs_sorted_idx = np.argsort(difficulties_np)[::-1]
        elif ordering == 'middleout':
            safe_difficulties = np.nan_to_num(difficulties_np); current_diffs_sorted_idx = np.argsort(
                np.abs(safe_difficulties))
        else:
            current_diffs_sorted_idx = np.argsort(difficulties_np)

    if current_diffs_sorted_idx is None or len(current_diffs_sorted_idx) == 0: return {}

    train_2 = {
        'input_ids': input_ids[current_diffs_sorted_idx],
        'attention_mask': attention_mask[current_diffs_sorted_idx],
        'labels': labels[current_diffs_sorted_idx],
        'difficulty': difficulties_tensor[current_diffs_sorted_idx]
    }

    if is_balanced:
        unique_labels_balanced, counts_balanced = torch.unique(train_2['labels'], return_counts=True)
        if len(unique_labels_balanced) > 1 and len(unique_labels_balanced) < 10:
            min_class_count = torch.min(counts_balanced).item();
            balanced_indices_list = []
            for label_val in unique_labels_balanced:
                label_indices = (train_2['labels'] == label_val).nonzero(as_tuple=True)[0]
                sampled_indices = label_indices[torch.randperm(len(label_indices))[:min_class_count]] if len(
                    label_indices) > min_class_count else label_indices
                balanced_indices_list.append(sampled_indices)
            if balanced_indices_list:
                final_balanced_indices = torch.cat(balanced_indices_list);
                final_balanced_indices = final_balanced_indices[torch.randperm(len(final_balanced_indices))]
                train_2 = {key: val[final_balanced_indices] for key, val in train_2.items()}

    num_examples_in_train_2 = len(train_2['input_ids'])
    if num_examples_in_train_2 == 0: return {}
    num_train = 0

    if strategy == 'ordered':
        return train_2
    elif strategy == 'simple':
        num_epochs_total_for_simple = getattr(args, 'num_epochs', 20)
        data_per_epoch = num_total_examples / (
                    num_epochs_total_for_simple / 2.0) if num_epochs_total_for_simple > 0 else num_total_examples
        num_train = min(int(data_per_epoch * (epoch + 1)), num_total_examples) if epoch % 2 == 0 else min(
            int(data_per_epoch * epoch), num_total_examples)
        num_train = min(num_train, num_examples_in_train_2)
        effective_min_train_len = min(int(min_train_len), num_examples_in_train_2)
        num_train = max(num_train, effective_min_train_len)
    elif strategy in ['naacl-linear', 'naacl-root']:
        c_init = 0.01;
        competency_epochs = max(1, int(competency_param));
        current_progress_epoch = epoch + 1
        if current_progress_epoch >= competency_epochs:
            epoch_pacing_ratio = 1.0
        else:
            if strategy == 'naacl-linear':
                epoch_pacing_ratio = c_init + (1.0 - c_init) * (current_progress_epoch / competency_epochs)
            else:
                epoch_pacing_ratio = c_init + (1.0 - c_init) * np.sqrt(current_progress_epoch / competency_epochs)
        num_train = int(epoch_pacing_ratio * num_total_examples)
        num_train = max(int(min_train_len), num_train)
        num_train = min(num_train, num_examples_in_train_2)
    elif strategy == 'theta':
        if theta_hat is None: raise ValueError("Theta strategy requires theta_hat.")
        difficulties_sorted_for_theta = train_2['difficulty']
        lower_bound_val = theta_hat + getattr(args, 'lower_bound', -np.inf)
        upper_bound_val = theta_hat + getattr(args, 'upper_bound', 0.0)
        train_idx_mask = (difficulties_sorted_for_theta.cpu() >= lower_bound_val) & (
                    difficulties_sorted_for_theta.cpu() <= upper_bound_val)
        final_indices_theta = torch.where(train_idx_mask)[0].cpu()
        num_selected_by_theta = len(final_indices_theta)
        effective_min_train_len_theta = min(int(min_train_len), num_examples_in_train_2)
        if num_selected_by_theta < effective_min_train_len_theta:
            current_difficulties_np_theta = train_2['difficulty'].cpu().numpy()
            distances_theta = np.abs(current_difficulties_np_theta - theta_hat)
            closest_indices_in_train_2 = np.argsort(distances_theta)[:effective_min_train_len_theta]
            final_indices_theta = torch.tensor(closest_indices_in_train_2, dtype=torch.long)
        num_train = len(final_indices_theta)
        if num_train == 0: return {}
        final_dict = {key: val[final_indices_theta] for key, val in train_2.items()}
        return final_dict
    else:
        raise NotImplementedError(f"Strategy '{strategy}' not implemented.")
    if num_train == 0 and strategy not in ['ordered', 'baseline', 'theta']: return {}
    final_dict = {key: val[:num_train] for key, val in train_2.items()}
    return final_dict


# --- Helper Functions (Continued - from previous script version) ---
# (set_random_seed, _tokenize_for_rarity, get_example_rarities_combined, scale_difficulties,
#  load_and_prepare_data, tokenize_function_qwen, create_dataset_qwen, evaluate_and_estimate_qwen
#  are assumed to be here and are identical to the previous response, so they are omitted for brevity
#  in this specific diff, but they ARE PART of the full script below)

def set_random_seed(seed):
    print(f"Setting global random seed to: {seed}")
    torch.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _tokenize_for_rarity(sent):  # Identical to previous
    if not isinstance(sent, str): sent = str(sent)
    tokens = [x.strip() for x in re.split(r'([^\w\'])', sent) if x and x.strip()]
    if not tokens and sent: tokens = [sent]
    return tokens


def get_example_rarities_combined(texts_list_primary, texts_list_secondary=None):  # Identical to previous
    if texts_list_secondary and len(texts_list_primary) != len(texts_list_secondary):
        new_secondary = [None] * len(texts_list_primary)
        for i in range(min(len(texts_list_primary), len(texts_list_secondary))): new_secondary[i] = \
        texts_list_secondary[i]
        texts_list_secondary = new_secondary
    corpus_tokens = [];
    example_token_lists = [[] for _ in range(len(texts_list_primary))]
    for i in range(len(texts_list_primary)):
        s1_tokens = _tokenize_for_rarity(texts_list_primary[i])
        current_example_tokens = [t for t in s1_tokens if t]
        if texts_list_secondary and i < len(texts_list_secondary) and texts_list_secondary[i] is not None:
            s2_tokens = _tokenize_for_rarity(texts_list_secondary[i])
            current_example_tokens.extend([t for t in s2_tokens if t])
        corpus_tokens.extend(current_example_tokens);
        example_token_lists[i] = current_example_tokens
    counts = {};
    N = len(corpus_tokens)
    for tok in corpus_tokens: counts.setdefault(tok, 0); counts[tok] += 1
    if N == 0: return [0.0] * len(texts_list_primary)
    epsilon = 1e-9;
    rarity_scores = []
    for tokens in example_token_lists:
        if not tokens:
            p_hat = 0.0
        else:
            log_probs = [np.log(counts.get(tok, 0) / N + epsilon) for tok in tokens]; p_hat = -np.mean(
                log_probs) if log_probs else 0.0
        rarity_scores.append(p_hat)
    return rarity_scores


def scale_difficulties(heuristic_difficulties_np, task_name, diff_dir_base, default_min=-3.0,
                       default_max=3.0):  # Identical
    reference_irt_file = os.path.join(diff_dir_base, f'{task_name.lower()}-1pl', 'best_parameters.json')
    target_min, target_max = default_min, default_max
    if not isinstance(heuristic_difficulties_np, np.ndarray): heuristic_difficulties_np = np.array(
        heuristic_difficulties_np, dtype=np.float32)
    if heuristic_difficulties_np.size == 0: return heuristic_difficulties_np
    source_min, source_max = np.min(heuristic_difficulties_np), np.max(heuristic_difficulties_np)
    try:
        if os.path.exists(reference_irt_file):
            with open(reference_irt_file, 'r') as f:
                irt_data = json.load(f)
            if 'diff' in irt_data and isinstance(irt_data['diff'], list) and len(irt_data['diff']) > 0:
                ref_diffs_np = np.array(irt_data['diff'])
                if ref_diffs_np.size > 0:
                    t_min_ref, t_max_ref = np.min(ref_diffs_np), np.max(ref_diffs_np)
                    if t_max_ref > t_min_ref: target_min, target_max = t_min_ref, t_max_ref
    except Exception:
        pass  # Use default if any error
    if (source_max - source_min) < 1e-9: return np.full_like(heuristic_difficulties_np, target_min)
    scaled_values = target_min + (heuristic_difficulties_np - source_min) * (target_max - target_min) / (
                source_max - source_min)
    return scaled_values


def load_and_prepare_data(args, task):  # Identical to previous (with refined prints removed)
    raw_datasets = load_dataset('glue', task, cache_dir=args.cache_dir)
    train_raw = raw_datasets['train'];
    difficulties_list = []
    if args.difficulty_measurer == 'pudf_irt':
        diff_file_path = os.path.join(args.diff_dir, f'{task.lower()}-1pl', 'best_parameters.json')
        try:
            with open(diff_file_path, 'r') as f:
                irt_data = json.load(f)
            if 'diff' not in irt_data or not isinstance(irt_data['diff'], list) or len(irt_data['diff']) != len(
                train_raw): raise ValueError("IRT file error or length mismatch.")
            difficulties_list = irt_data['diff']
        except Exception as e:
            print(f"ERROR PUDL-IRT load: {diff_file_path}, {e}"); raise
    elif args.difficulty_measurer in ['sentence_length', 'word_rarity']:
        s1_key, s2_key = ("premise", "hypothesis") if task == "mnli" else ("sentence1", "sentence2") if task in ["mrpc",
                                                                                                                 "rte"] else (
        "question1", "question2") if task == "qqp" else ("question", "sentence") if task == "qnli" else (
        "sentence", None) if task == "sst2" else (None, None)
        if not s1_key: raise ValueError(f"Task {task} keys undefined.")
        if args.difficulty_measurer == 'sentence_length':
            for i in range(len(train_raw)):
                s1_len = len(str(train_raw[i][s1_key]).split()) if train_raw[i][s1_key] is not None else 0
                s2_len = len(str(train_raw[i][s2_key]).split()) if s2_key and train_raw[i][s2_key] is not None else 0
                difficulties_list.append(s1_len + s2_len)
        elif args.difficulty_measurer == 'word_rarity':
            primary_texts = [str(ex[s1_key]) if ex[s1_key] is not None else "" for ex in train_raw]
            secondary_texts = [str(ex[s2_key]) if ex[s2_key] is not None else "" for ex in
                               train_raw] if s2_key else None
            difficulties_list = get_example_rarities_combined(primary_texts, secondary_texts)
        if args.training_scheduler == 'pudf_theta':
            difficulties_np = np.array(difficulties_list, dtype=np.float32);
            scaled_difficulties_np = scale_difficulties(difficulties_np, task, args.diff_dir);
            difficulties_list = scaled_difficulties_np.tolist()
    elif args.difficulty_measurer == 'baseline_diff':
        difficulties_list = [0.0] * len(train_raw)
    else:
        raise ValueError(f"Unknown difficulty_measurer: {args.difficulty_measurer}")
    if 'difficulty' in train_raw.column_names: train_raw = train_raw.remove_columns(['difficulty'])
    train_with_diff = train_raw.add_column('difficulty', difficulties_list)
    train_val_split = train_with_diff.train_test_split(test_size=0.1, seed=args.random_seed);
    train_dataset_hf, val_dataset_hf = train_val_split['train'], train_val_split['test']
    val_split_name = 'validation_matched' if task == 'mnli' else 'validation';
    test_dataset_hf = raw_datasets[val_split_name]
    return train_dataset_hf, val_dataset_hf, test_dataset_hf


def tokenize_function_qwen(examples, task, tokenizer):  # Identical
    max_length = TASK_MAX_LENGTHS.get(task, 128)
    text_a_key, text_b_key = ("premise", "hypothesis") if task == "mnli" else ("sentence1", "sentence2") if task in [
        "mrpc", "rte"] else ("question1", "question2") if task == "qqp" else (
    "question", "sentence") if task == "qnli" else ("sentence", None) if task == "sst2" else (None, None)
    if not text_a_key: raise ValueError(f"Task {task} keys undefined for tokenization.")
    processed_text_a = [str(t) if t is not None else "" for t in examples[text_a_key]]
    if text_b_key and examples.get(text_b_key):
        processed_text_b = [str(t) if t is not None else "" for t in examples[text_b_key]]
        return tokenizer(text=processed_text_a, text_pair=processed_text_b, padding="max_length", truncation=True,
                         max_length=max_length)
    return tokenizer(text=processed_text_a, padding="max_length", truncation=True, max_length=max_length)


def create_dataset_qwen(dataset_hf, task, tokenizer, include_difficulty=True):  # Identical
    sample_tokenization = tokenizer(text="dummy", text_pair="dummy pair" if task not in ["sst2"] else None)
    tokenized_cols_to_keep = ['input_ids', 'attention_mask']
    if 'token_type_ids' in sample_tokenization: tokenized_cols_to_keep.append('token_type_ids')
    text_keys_to_remove = ["premise", "hypothesis", "idx"] if task == "mnli" else ["sentence1", "sentence2",
                                                                                   "idx"] if task in ["mrpc",
                                                                                                      "rte"] else [
        "question1", "question2", "idx", "id"] if task == "qqp" else ["question", "sentence",
                                                                      "idx"] if task == "qnli" else ["sentence",
                                                                                                     "idx"] if task == "sst2" else []
    cols_to_remove_during_map = [k for k in text_keys_to_remove if k in dataset_hf.column_names]
    tokenized_dataset_hf = dataset_hf.map(lambda exs: tokenize_function_qwen(exs, task, tokenizer), batched=True,
                                          remove_columns=cols_to_remove_during_map, desc=f"Tokenizing {task}")
    if 'label' in tokenized_dataset_hf.column_names:
        tokenized_dataset_hf = tokenized_dataset_hf.rename_column('label', 'labels')
    elif 'labels' not in tokenized_dataset_hf.column_names:
        raise ValueError("Label column missing.")
    final_cols_ordered = list(tokenized_cols_to_keep);
    final_cols_ordered.append('labels')
    if include_difficulty:
        if 'difficulty' not in tokenized_dataset_hf.column_names: tokenized_dataset_hf = tokenized_dataset_hf.add_column(
            'difficulty', [0.0] * len(tokenized_dataset_hf))
        final_cols_ordered.append('difficulty')
    tokenized_dataset_hf.set_format(type='torch', columns=final_cols_ordered)
    return TensorDataset(*[tokenized_dataset_hf[col] for col in final_cols_ordered]), final_cols_ordered


def evaluate_and_estimate_qwen(model, dataloader, device, column_order, num_obs_theta=-1, mode='eval',
                               amp_dtype=torch.float16):  # Identical
    global use_amp;
    val_loss_sum = 0.0;
    current_accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    preds_list, labels_list, difficulties_list_for_theta = [], [], [];
    model.eval();
    num_batches = 0
    desc = "Val/Test" if mode == 'eval' else "ThetaEst" if mode == 'estimate' else "Val+ThetaEst"
    try:
        ids_idx = column_order.index('input_ids');
        mask_idx = column_order.index('attention_mask');
        tti_idx = column_order.index('token_type_ids') if 'token_type_ids' in column_order else -1;
        lbl_idx = column_order.index('labels');
        diff_idx = -1
        if mode in ['estimate', 'eval_estimate'] and 'difficulty' in column_order: diff_idx = column_order.index(
            'difficulty')
    except ValueError as e:
        print(f"FATAL: Eval col missing: {e}"); raise
    for batch in tqdm(dataloader, desc=desc, leave=False, disable=os.environ.get("TQDM_DISABLE") == "1"):
        num_batches += 1;
        input_ids = batch[ids_idx].to(device, non_blocking=True);
        attention_mask = batch[mask_idx].to(device, non_blocking=True);
        token_type_ids = batch[tti_idx].to(device, non_blocking=True) if tti_idx != -1 else None;
        labels = batch[lbl_idx].to(device, non_blocking=True);
        diff_tensor_cpu = None
        if mode in ['estimate', 'eval_estimate'] and diff_idx != -1: diff_tensor_cpu = batch[diff_idx]
        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype,
                                       enabled=(use_amp and device.type == 'cuda')):
            m_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels};
            _ = [m_kwargs.update({'token_type_ids': token_type_ids}) if token_type_ids is not None else None];
            outputs = model(**m_kwargs);
            logits = outputs.logits;
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
        val_loss_sum += loss.item();
        predictions = torch.argmax(logits, dim=-1);
        current_accuracy_metric.add_batch(predictions=predictions.detach().cpu(), references=labels.detach().cpu())
        if mode in ['estimate', 'eval_estimate']: preds_list.append(
            logits.detach().float().cpu().numpy());labels_list.append(labels.detach().cpu().numpy());_ = [
            difficulties_list_for_theta.append(diff_tensor_cpu.cpu().numpy()) if diff_tensor_cpu is not None else None]
    avg_val_loss = val_loss_sum / num_batches if num_batches > 0 else 0.0;
    validation_accuracy = 0.0
    try:
        eval_score = current_accuracy_metric.compute() if num_batches > 0 else None;validation_accuracy = eval_score[
            'accuracy'] if eval_score else 0.0
    except:
        pass
    if mode == 'eval': return validation_accuracy, avg_val_loss
    theta_hat, model_cap_time = 0.0, 0.0;
    can_est_theta = (mode in ['estimate', 'eval_estimate']) and difficulties_list_for_theta and diff_idx != -1
    if can_est_theta:
        all_logits_np = np.concatenate(preds_list);
        all_labels_np = np.concatenate(labels_list);
        all_difficulties_np = np.concatenate(difficulties_list_for_theta)
        if len(all_difficulties_np) == len(all_labels_np) and len(all_difficulties_np) > 0:
            resp_pat = [1 if p == t else -1 for p, t in zip(np.argmax(all_logits_np, axis=1), all_labels_np)];
            time_s = time.time()
            try:
                theta_res = calculate_theta(all_difficulties_np, resp_pat, num_obs=num_obs_theta); theta_hat = \
                theta_res[0] if isinstance(theta_res, (list, np.ndarray)) and len(theta_res) > 0 else float(theta_res)
            except Exception as e:
                print(f"ERROR theta calc: {e}");theta_hat = 0.0
            model_cap_time = time.time() - time_s
    if mode == 'estimate': return theta_hat, model_cap_time
    return validation_accuracy, avg_val_loss, theta_hat, model_cap_time


# --- Main Training Function ---
use_amp = False


def train_glue_task(args, task, output_dir_task):
    global use_amp
    checkpoint_save_path = os.path.join(output_dir_task, CHECKPOINT_SUBDIR_NAME)  # Path for actual model files

    print(f"\n===== Starting Task: {task} =====");
    print(f"  Difficulty: {args.difficulty_measurer}, Scheduler: {args.training_scheduler}");
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu');
    print(f"Using device: {device}")
    bf16_ready = (torch.cuda.is_available() and torch.cuda.is_bf16_supported());
    amp_dtype = torch.bfloat16 if bf16_ready else torch.float16
    use_amp = torch.cuda.is_available();
    print(f"Using AMP: {use_amp} with dtype: {amp_dtype}")

    try:
        train_hf, dev_hf, test_hf = load_and_prepare_data(args, task); print(
            f"Data sizes: Train={len(train_hf)}, Val={len(dev_hf)}, Test={len(test_hf)}")
    except Exception as e:
        print(f"FATAL: Data loading for {task}: {e}"); traceback.print_exc(); return 0.0, 0.0, 0.0, 0.0
    num_labels = 3 if task.startswith("mnli") else 2;
    print(f"Loading model: {args.model_name} (Labels: {num_labels})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True,
                                                  trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels,
                                                                   cache_dir=args.cache_dir, trust_remote_code=True,
                                                                   ignore_mismatched_sizes=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'}); model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id: model.config.pad_token_id = tokenizer.pad_token_id
        assert model.config.pad_token_id is not None and tokenizer.pad_token_id is not None, "Pad token ID not set."
    except Exception as e:
        print(f"FATAL: Loading model/tokenizer failed: {e}"); traceback.print_exc(); return 0.0, 0.0, 0.0, 0.0

    model.to(device)
    if args.use_gradient_checkpointing: print("Enabling gradient checkpointing."); model.gradient_checkpointing_enable()
    model.config.use_cache = False
    try:
        train_dataset_pt, train_col_order = create_dataset_qwen(train_hf, task, tokenizer, include_difficulty=True)
        dev_dataset_pt, dev_col_order = create_dataset_qwen(dev_hf, task, tokenizer, include_difficulty=True)
        test_dataset_pt, test_col_order = create_dataset_qwen(test_hf, task, tokenizer, include_difficulty=False)
    except Exception as e:
        print(f"FATAL: PyTorch Dataset creation failed: {e}"); traceback.print_exc(); return 0.0, 0.0, 0.0, 0.0

    dl_kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    dev_dataloader = DataLoader(dev_dataset_pt, batch_size=args.eval_batch_size, shuffle=False, **dl_kwargs)
    test_dataloader = DataLoader(test_dataset_pt, batch_size=args.eval_batch_size, shuffle=False,
                                 **dl_kwargs) if test_dataset_pt and len(test_dataset_pt) > 0 else None
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate, scale_parameter=False, relative_step=False,
                          warmup_init=False, weight_decay=args.weight_decay)
    num_steps_epoch_full = ceil(len(train_dataset_pt) / (args.train_batch_size * args.gradient_accumulation_steps))
    num_total_steps_est = num_steps_epoch_full * args.num_epochs
    num_warmup = max(1, int(args.warmup_ratio * num_total_steps_est))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup,
                                                num_training_steps=num_total_steps_est)
    scaler = GradScaler(enabled=use_amp)
    best_val_acc = 0.0;
    early_stop_cnt = 0;
    train_stats_epochs = [];
    overall_start_time = time.time();
    epochs_run = 0
    prev_theta, cur_theta = -5.0, 0.0;
    total_model_save_time_task = 0.0

    cfg_data_sel = types.SimpleNamespace(**vars(args));
    cfg_data_sel.task = task
    if args.training_scheduler == 'pudf_theta':
        cfg_data_sel.strategy = 'theta'
    elif args.training_scheduler == 'linear':
        cfg_data_sel.strategy = 'naacl-linear'
    elif args.training_scheduler == 'root':
        cfg_data_sel.strategy = 'naacl-root'
    elif args.training_scheduler == 'baseline_sched':
        cfg_data_sel.strategy = 'baseline'
    else:
        raise ValueError(f"Unsupported scheduler for strategy: {args.training_scheduler}")

    for epoch in range(args.num_epochs):
        epochs_run = epoch + 1;
        epoch_s_time = time.time();
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_loss_sum = 0.0;
        optim_steps_epoch = 0;
        batches_epoch = 0;
        avg_epoch_loss = 0.0;
        num_epoch_ex = 0
        epoch_model_save_time = 0.0

        if args.training_scheduler == 'pudf_theta':
            try:
                est_th, cap_time = evaluate_and_estimate_qwen(model, dev_dataloader, device, dev_col_order,
                                                              num_obs_theta=args.num_obs_theta, mode='estimate',
                                                              amp_dtype=amp_dtype)
                print(f"  Epoch {epoch + 1} Theta estimated: {est_th:.4f} ({cap_time:.2f}s)")
                if est_th > prev_theta:
                    cur_theta = est_th
                else:
                    cur_theta += 0.1
                prev_theta = cur_theta;
            except Exception as e:
                print(
                    f"  Warning: Theta estimation failed for epoch {epoch + 1}: {e}. Using previous: {cur_theta:.4f}"); traceback.print_exc()

        try:
            filt_data_dict = get_epoch_training_data(train_dataset_pt, cfg_data_sel, epoch, task, theta_hat=cur_theta)
        except Exception as e:
            print(
                f"  ERROR get_epoch_training_data epoch {epoch + 1}: {e}. Skip epoch."); traceback.print_exc(); continue

        if filt_data_dict and 'labels' in filt_data_dict and len(filt_data_dict['labels']) > 0: num_epoch_ex = len(
            filt_data_dict['labels'])

        if num_epoch_ex == 0:
            print(f"  Warning: No data selected for epoch {epoch + 1}. Skip training phase.");
        else:
            print(f"  Selected {num_epoch_ex} examples for epoch {epoch + 1}.");
            tensors_epoch_dl = [filt_data_dict[cn] for cn in train_col_order]
            train_ds_epoch_pt = TensorDataset(*tensors_epoch_dl);
            cur_epoch_bs = min(args.train_batch_size, num_epoch_ex) if num_epoch_ex > 0 else args.train_batch_size
            if cur_epoch_bs == 0: print("  Skipping training as current epoch batch size is 0."); continue

            train_dl_epoch = DataLoader(train_ds_epoch_pt, shuffle=True, batch_size=cur_epoch_bs,
                                        drop_last=(
                                                    args.gradient_accumulation_steps > 1 and num_epoch_ex > 0 and num_epoch_ex % cur_epoch_bs != 0 and num_epoch_ex % cur_epoch_bs < args.gradient_accumulation_steps),
                                        **dl_kwargs)
            model.train();
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_dl_epoch, desc=f"Epoch {epoch + 1} Train", leave=False,
                        disable=os.environ.get("TQDM_DISABLE") == "1")
            for step, batch_ep in enumerate(pbar):
                batches_epoch += 1
                try:
                    ids_tr = batch_ep[train_col_order.index('input_ids')].to(device, non_blocking=True);
                    mask_tr = batch_ep[train_col_order.index('attention_mask')].to(device, non_blocking=True)
                    tti_idx_tr = train_col_order.index('token_type_ids') if 'token_type_ids' in train_col_order else -1
                    tti_tr = batch_ep[tti_idx_tr].to(device, non_blocking=True) if tti_idx_tr != -1 else None
                    lbl_tr = batch_ep[train_col_order.index('labels')].to(device, non_blocking=True)
                except Exception as e:
                    print(f"  ERROR unpack train batch {step} in epoch {epoch + 1}: {e}"); continue
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == 'cuda')):
                    mkwargs_tr = {'input_ids': ids_tr, 'attention_mask': mask_tr, 'labels': lbl_tr}
                    if tti_tr is not None: mkwargs_tr['token_type_ids'] = tti_tr
                    out_tr = model(**mkwargs_tr);
                    loss_t = out_tr.loss
                if loss_t is None or torch.isnan(loss_t): optimizer.zero_grad(set_to_none=True); continue
                loss_t_val = loss_t.item();
                loss_t = loss_t / args.gradient_accumulation_steps
                scaler.scale(loss_t).backward();
                epoch_loss_sum += loss_t_val
                if ((step + 1) % args.gradient_accumulation_steps == 0) or ((step + 1) == len(train_dl_epoch)):
                    scaler.unscale_(optimizer);
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer);
                    scaler.update();
                    scheduler.step();
                    optimizer.zero_grad(set_to_none=True)
                    optim_steps_epoch += 1
                if step % (max(1, len(train_dl_epoch) // 10 if len(train_dl_epoch) > 20 else 1)) == 0: pbar.set_postfix(
                    {'loss': f'{loss_t_val:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
            avg_epoch_loss = epoch_loss_sum / batches_epoch if batches_epoch > 0 else 0.0
            print(f"  Epoch {epoch + 1} Avg Train Loss: {avg_epoch_loss:.4f} ({optim_steps_epoch} optim steps)")

        val_acc_ep, val_loss_ep, th_after_ep, _ = evaluate_and_estimate_qwen(model, dev_dataloader, device,
                                                                             dev_col_order,
                                                                             num_obs_theta=args.num_obs_theta,
                                                                             mode='eval_estimate', amp_dtype=amp_dtype)
        print(f"  Epoch {epoch + 1} Val: Acc={val_acc_ep:.4f}, Loss={val_loss_ep:.4f}, Theta={th_after_ep:.4f}")
        epoch_stat_entry = {'epoch': epoch + 1, 'Train Loss': avg_epoch_loss, 'Val Loss': val_loss_ep,
                            'Val Acc': val_acc_ep, 'Est Theta': th_after_ep, 'Num Train Ex': num_epoch_ex,
                            'Optim Steps': optim_steps_epoch, 'Model Save Time': 0.0}
        if val_acc_ep > best_val_acc:
            print(f"  Val acc improved ({best_val_acc:.4f} -> {val_acc_ep:.4f}). Saving to {checkpoint_save_path}...");
            best_val_acc = val_acc_ep;
            early_stop_cnt = 0
            os.makedirs(checkpoint_save_path, exist_ok=True)  # Ensure checkpoint subdir exists
            try:
                save_s_time = time.time()
                m_to_save = getattr(model, '_orig_mod', model);
                m_to_save.save_pretrained(checkpoint_save_path);
                tokenizer.save_pretrained(checkpoint_save_path)
                save_e_time = time.time()
                epoch_model_save_time = save_e_time - save_s_time
                total_model_save_time_task += epoch_model_save_time
                print(f"    Model saved in {epoch_model_save_time:.2f}s.")
                epoch_stat_entry['Model Save Time'] = epoch_model_save_time
            except Exception as e:
                print(f"  Warn: Error saving best model: {e}")
        else:
            early_stop_cnt += 1;
            print(
                f"  Val acc ({val_acc_ep:.4f}) vs best ({best_val_acc:.4f}). Early stop: {early_stop_cnt}/{args.early_stopping_patience}")
            if early_stop_cnt >= args.early_stopping_patience: print("Early stopping."); break
        train_stats_epochs.append(epoch_stat_entry)
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    total_train_time_s = time.time() - overall_start_time
    print(f"\n--- Training Loop Finished for {task} ---");
    print(
        f"  Actual epochs: {epochs_run}, Total Train Time: {total_train_time_s:.2f}s, Total Model Save Time: {total_model_save_time_task:.2f}s, Best Val Acc: {best_val_acc:.4f}")
    test_acc, test_loss = 0.0, 0.0
    model_loaded_for_test = False
    if test_dataloader:
        print("\n--- Final Test Evaluation ---");
        weights_ok = any(os.path.exists(os.path.join(checkpoint_save_path, fN)) for fN in
                         ["pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"])
        if weights_ok and best_val_acc > 0:
            print(f"  Loading best model from: {checkpoint_save_path}...");
            try:
                model_test = AutoModelForSequenceClassification.from_pretrained(checkpoint_save_path,
                                                                                num_labels=num_labels,
                                                                                trust_remote_code=True).to(device)
                model_loaded_for_test = True
                test_acc, test_loss = evaluate_and_estimate_qwen(model_test, test_dataloader, device, test_col_order,
                                                                 mode='eval', amp_dtype=amp_dtype)
                print(f'  Final Test Accuracy: {test_acc:.4f}, Final Test Loss: {test_loss:.4f}');
                del model_test
            except Exception as e:
                print(f"  ERROR test eval: {e}"); traceback.print_exc(); test_acc, test_loss = -1.0, -1.0
        elif not weights_ok:
            print(f"  Best model weights not found in {checkpoint_save_path}."); test_acc, test_loss = -2.0, -2.0
        else:
            print(f"  Skip test (no model improved on val to be saved)."); test_acc, test_loss = -4.0, -4.0
    else:
        print("  Test dataloader None. Skip test."); test_acc, test_loss = -3.0, -3.0

    task_sum_stats = {"task": task, "difficulty_measurer": args.difficulty_measurer,
                      "training_scheduler": args.training_scheduler, "best_val_acc": best_val_acc, "test_acc": test_acc,
                      "test_loss": test_loss, "total_train_time_s": round(total_train_time_s, 2),
                      "total_model_save_time_s": round(total_model_save_time_task, 2), "epochs_run": epochs_run,
                      "cfg_args_snap": {k: str(v) for k, v in vars(args).items()}, "epoch_stats": train_stats_epochs}
    stats_fname = os.path.join(output_dir_task, f"summary_stats_{task}.json");
    print(f"  Saving task summary: {stats_fname}")
    try:
        with open(stats_fname, "w") as f:
            json.dump(task_sum_stats, f, indent=4,
                      default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(o))
    except Exception as e:
        print(f"  Warn: Error saving task summary: {e}")

    # Delete the saved model checkpoint to save space
    if model_loaded_for_test or (weights_ok and best_val_acc > 0):  # If we attempted to load or know it was saved
        if os.path.exists(checkpoint_save_path):
            try:
                print(f"  Deleting model checkpoint directory: {checkpoint_save_path}")
                shutil.rmtree(checkpoint_save_path)
                print(f"    Successfully deleted {checkpoint_save_path}.")
            except Exception as e:
                print(f"  Warning: Error deleting model checkpoint directory {checkpoint_save_path}: {e}")
        else:
            print(
                f"  Model checkpoint directory {checkpoint_save_path} not found for deletion (already deleted or never saved).")

    del model, tokenizer, optimizer, scheduler, scaler;
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"===== Finished Task Processing: {task} =====")
    return best_val_acc, test_acc, total_train_time_s, total_model_save_time_task


def run_ablations():
    parser = argparse.ArgumentParser(description="Qwen GLUE Ablation Study Script")
    parser.add_argument('--difficulty_measurer', type=str, required=True,
                        choices=['pudf_irt', 'sentence_length', 'word_rarity', 'baseline_diff'])
    parser.add_argument('--training_scheduler', type=str, required=True,
                        choices=['linear', 'root', 'pudf_theta', 'baseline_sched'])
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument('--diff_dir', type=str,
                        default=GLUE_DIFFICULTY_DIR,
                        help="Base directory for task-specific IRT difficulty files. Expected: diff_dir/task-1pl/best_parameters.json")
    parser.add_argument('--cache_dir', type=str, default=HF_HOME,
                        help="Hugging Face cache directory (sets HF_HOME).")
    parser.add_argument('--output_dir_root', type=str, default="./my_qwen_glue_ablations_output")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument('--eval_batch_size', type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID (-1 for CPU).")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=63)
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest')
    parser.add_argument('--min_train_length', type=int, default=1000)
    parser.add_argument('--competency', type=int, default=5)
    parser.add_argument('--lower_bound', type=float, default=-np.inf)
    parser.add_argument('--upper_bound', type=float, default=0.0)
    parser.add_argument('--balanced', action='store_true', default=False)
    parser.add_argument('--num_obs_theta', type=int, default=1000)
    parser.add_argument('--tasks', nargs='+', default=GLUETASKS_DEFAULT, choices=GLUETASKS_DEFAULT)
    args = parser.parse_args()

    setup_hf_environment(args.cache_dir);
    set_random_seed(args.random_seed)
    try:
        user_info = whoami()
        print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
    except Exception:
        print(f"HF login check failed.")
    cfg_name = f"diff_{args.difficulty_measurer}_sched_{args.training_scheduler}"
    run_out_dir = os.path.join(args.output_dir_root, cfg_name);
    os.makedirs(run_out_dir, exist_ok=True)
    print(f"Outputs for run (Diff: {args.difficulty_measurer}, Sched: {args.training_scheduler}) in: {run_out_dir}")
    overall_sum = {
        "configuration": {k: str(v) if not isinstance(v, (list, dict)) else v for k, v in vars(args).items()},
        "tasks": {}}
    total_model_save_time_all_tasks = 0.0

    for task in args.tasks:
        task_spec_out_dir = os.path.join(run_out_dir, task);
        os.makedirs(task_spec_out_dir, exist_ok=True)
        # print(f"Starting task {task} in output directory: {task_spec_out_dir}") # Reduced verbosity
        best_val, test_acc, total_t, task_model_save_time = train_glue_task(args, task, task_spec_out_dir)
        total_model_save_time_all_tasks += task_model_save_time
        overall_sum["tasks"][task] = {"best_val_acc": best_val, "test_acc": test_acc, "total_train_time_s": total_t,
                                      "task_model_save_time_s": task_model_save_time}

    overall_sum["total_model_save_time_all_tasks_s"] = round(total_model_save_time_all_tasks, 2)
    overall_sum_fname = os.path.join(run_out_dir, "overall_run_summary.json")
    print(f"\nSaving overall run summary: {overall_sum_fname}")
    try:
        def conv_json(o):
            if isinstance(o, (np.integer, np.int_)):
                return int(o)
            elif isinstance(o, (np.floating, np.float_)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, types.SimpleNamespace):
                return vars(o)
            if o == np.inf: return "Infinity"
            if o == -np.inf: return "-Infinity"
            return str(o)

        with open(overall_sum_fname, "w") as f:
            json.dump(overall_sum, f, indent=4, default=conv_json)
    except Exception as e:
        print(f"Warn: Error saving overall summary: {e}")
    # print("\nAll tasks for this configuration finished.") # Reduced verbosity
    print_final_summary_for_user(overall_sum)


def print_final_summary_for_user(summary_data):
    print("\n\n===== FINAL SUMMARY OF THIS RUN =====")
    cfg = summary_data.get('configuration', {})
    print(f"Difficulty Measurer: {cfg.get('difficulty_measurer', 'N/A')}")
    print(f"Training Scheduler:  {cfg.get('training_scheduler', 'N/A')}")
    print("----------------------------------------------------")
    print("Task         | Test Acc. | Train Time (s) | Save Time (s)")
    print("----------------------------------------------------")
    total_t_all_tasks = 0;
    valid_accs = []
    tasks_d = summary_data.get('tasks', {})
    task_order = cfg.get('tasks', GLUETASKS_DEFAULT)
    if not isinstance(task_order, list): task_order = GLUETASKS_DEFAULT

    for task_n in task_order:
        if task_n in tasks_d:
            res = tasks_d[task_n]
            t_acc = res.get('test_acc', 0.0);
            tr_time = res.get('total_train_time_s', 0.0)
            save_time = res.get('task_model_save_time_s', 0.0)
            print(f"{task_n:<12} | {t_acc:<9.4f} | {tr_time:<14.2f} | {save_time:<10.2f}")
            if isinstance(t_acc, (float, int)) and t_acc >= -0.0001: valid_accs.append(t_acc)
            total_t_all_tasks += tr_time
    print("----------------------------------------------------")
    if valid_accs: avg_acc = np.mean(valid_accs) if valid_accs else 0.0; print(
        f"Avg Test Acc (valid tasks): {avg_acc:.4f}")
    print(f"Total Train Time (all tasks): {total_t_all_tasks:.2f}s")
    print(f"Total Model Save Time (all tasks): {summary_data.get('total_model_save_time_all_tasks_s', 0.0):.2f}s")
    run_sum_path = os.path.join(cfg.get('output_dir_root', 'N/A'),
                                f"diff_{cfg.get('difficulty_measurer', 'N_A')}_sched_{cfg.get('training_scheduler', 'N_A')}",
                                "overall_run_summary.json")
    print(f"Full configuration & results saved in: {run_sum_path}")
    print("=====================================")


if __name__ == '__main__':
    # import sys
    # if not sys.stdout.isatty(): os.environ["TQDM_DISABLE"] = "1" # Optional: Disable tqdm for non-TTY
    run_ablations()