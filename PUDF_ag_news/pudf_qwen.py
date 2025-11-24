# pudf_qwen_agnews.py
import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from transformers.optimization import Adafactor
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
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

# Assuming these custom modules are in the same directory or Python path
from build_features_Qwen import get_epoch_training_data
from irt_scoring import calculate_theta

# --- Configuration ---
OUTPUT_DIR = "Qwen2.5_7B_PUDF_AGNews_3"
MODEL_CHECKPOINT = "Qwen/Qwen2.5-7B"
DATASET_ID = "contemmcm/ag_news"
AG_NEWS_DIFFICULTY_FILE_PATH = "../gen_difficulty/merged_jsonlines_output/test-1pl/best_parameters.json"
DIFFICULTY_JSON_KEY = "diff"

RANDOM_SEED = 63
MAX_LENGTH = 128
THETA_ESTIMATION_SET_SIZE = 1000
NUM_TRAIN_EPOCHS = 15
LEARNING_RATE_ADAFACTOR = 2e-5
WEIGHT_DECAY = 0.01

PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 16
PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE = 64

PUDF_CONFIG = types.SimpleNamespace(
    strategy='theta',
    ordering='easiest',
    num_obs_theta=1000,
    min_train_length=500,
    lower_bound=-np.inf,
    upper_bound=0.0,
    balanced=False,
    use_length=False,
    use_word_rarity=False,
    task_name_for_pudf="ag_news_pudf",
    num_epochs=NUM_TRAIN_EPOCHS,
    competency=NUM_TRAIN_EPOCHS / 2.0
)
PATIENCE_EARLY_STOPPING = 3
USE_GRADIENT_CHECKPOINTING = True


# --- Environment Setup ---
def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
        print("Removed HF_TOKEN environment variable to use cached token.")
    try:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        user_info = whoami(token=hf_token)
        print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
    except Exception as e:
        print(f"Hugging Face login check warning: {e}. Public models should still work.")


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed_value}")


# --- Helper: Create Qwen Tensor Dataset ---
def create_qwen_tensor_dataset(hf_tokenized_dataset_split,
                               partition_name="unknown",
                               include_difficulty=True):
    print(f"Creating TensorDataset for Qwen {partition_name} (include_difficulty={include_difficulty})...")
    actual_columns = hf_tokenized_dataset_split.column_names
    print(f"  DEBUG ({partition_name}): Input HF Dataset columns: {actual_columns}")

    final_columns_ordered = ['input_ids', 'attention_mask']
    if 'token_type_ids' in actual_columns and 'token_type_ids' in hf_tokenized_dataset_split.features:
        print(f"  WARNING ({partition_name}): 'token_type_ids' found, will be included (unexpected for Qwen).")
        final_columns_ordered.append('token_type_ids')
    else:
        print(f"  INFO ({partition_name}): 'token_type_ids' will NOT be included (expected for Qwen).")

    if 'labels' in actual_columns:
        final_columns_ordered.append('labels')
    elif partition_name in ["actual_train_pool_for_pudf", "theta_estimation_set", "validation_main_eval"]:
        raise ValueError(f"'labels' column expected for {partition_name} but not found in {actual_columns}.")

    if include_difficulty:
        if 'difficulty' in actual_columns:
            final_columns_ordered.append('difficulty')
        else:
            raise RuntimeError(
                f"Difficulty column required for {partition_name} (include_difficulty=True) but missing from {actual_columns}.")
    else:
        if 'difficulty' in actual_columns:
            print(
                f"Note ({partition_name}): 'difficulty' column present but EXCLUDED from TensorDataset.")

    for col in final_columns_ordered:
        if col not in actual_columns:
            raise ValueError(f"Column '{col}' for {partition_name} not in columns: {actual_columns}")

    try:
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
            tensors_to_extract.append(tensor_data)

        print(f"  Shapes of tensors for TensorDataset ({partition_name}):")
        for i, col_n in enumerate(final_columns_ordered): print(f"    {col_n}: {tensors_to_extract[i].shape}")
        print(f"Final tensor order for {partition_name} TensorDataset: {final_columns_ordered}")
        return TensorDataset(*tensors_to_extract), final_columns_ordered
    except Exception as e:
        print(f"ERROR creating TensorDataset for {partition_name}: {e}")
        traceback.print_exc()
        raise


# --- Helper: Evaluate Qwen Theta ---
def evaluate_and_estimate_qwen_theta(model, dataloader, device, column_order, num_labels,
                                     num_obs_theta=-1, desc_prefix="QwenThetaEst"):
    print(f"Estimating Qwen Theta ({desc_prefix})...")
    model.eval()
    all_preds_logits_list = []
    all_labels_list = []
    all_difficulties_list = []
    use_bf16_eval = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype_eval = torch.bfloat16 if use_bf16_eval else torch.float16

    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        labels_idx = column_order.index('labels')
        difficulty_idx = column_order.index('difficulty')
    except ValueError as e:
        print(f"FATAL ERROR ({desc_prefix}): Column missing for theta. Expected order: {column_order}. Error: {e}")
        traceback.print_exc();
        return 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc_prefix, leave=False):
            try:
                input_ids = batch[input_ids_idx].to(device, non_blocking=True)
                attention_mask = batch[attention_mask_idx].to(device, non_blocking=True)
                labels_batch = batch[labels_idx].to(device, non_blocking=True)
                difficulty_tensor_for_batch = batch[difficulty_idx].cpu().numpy()
            except Exception as e:
                print(f"ERROR ({desc_prefix}) unpacking batch: {e}");
                traceback.print_exc();
                continue

            with autocast(device.type, dtype=amp_dtype_eval,
                          enabled=use_bf16_eval or (torch.cuda.is_available() and not use_bf16_eval)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
            all_preds_logits_list.append(logits.cpu().numpy())
            all_labels_list.append(labels_batch.cpu().numpy())
            all_difficulties_list.append(difficulty_tensor_for_batch)

    theta_hat, model_capacity_time = 0.0, 0.0
    if not all_difficulties_list:
        print(f"Warning ({desc_prefix}): No difficulty scores. Theta defaults to 0.0.")
    else:
        final_preds_logits, final_labels, concatenated_difficulties = map(np.concatenate,
                                                                          [all_preds_logits_list, all_labels_list,
                                                                           all_difficulties_list])
        if not (len(final_preds_logits) == len(final_labels) == len(concatenated_difficulties)):
            print(f"ERROR ({desc_prefix}): Mismatch lengths. Skip theta.")
        elif len(final_labels) == 0:
            print(f"Warning ({desc_prefix}): No data for theta estimation.")
        else:
            time_s = time.time()
            predictions = np.argmax(final_preds_logits, axis=1)
            response_pattern = (predictions == final_labels).astype(int) * 2 - 1
            num_total, indices = len(concatenated_difficulties), np.arange(len(concatenated_difficulties))
            if num_obs_theta > 0 and num_obs_theta < num_total: indices = np.random.choice(num_total, num_obs_theta,
                                                                                           replace=False)
            try:
                theta_hat = calculate_theta(concatenated_difficulties[indices], response_pattern[indices])[0]
            except Exception as e:
                print(f"ERROR ({desc_prefix}) theta calc: {e}. Theta=0.0");
                traceback.print_exc();
                theta_hat = 0.0
            model_capacity_time = time.time() - time_s
    print(f"Theta estimated ({desc_prefix}): {theta_hat:.4f} in {model_capacity_time:.2f}s")
    return theta_hat, model_capacity_time


# --- Helper: Evaluate Qwen on Main Validation (Acc, Loss) ---
def evaluate_qwen_main_val(model, dataloader, device, column_order, num_labels, desc_prefix="QwenMainVal"):
    print(f"Evaluating Qwen on Main Validation ({desc_prefix})...")
    model.eval()
    accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
    total_loss, num_batches = 0.0, 0
    use_bf16_eval = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype_eval = torch.bfloat16 if use_bf16_eval else torch.float16
    all_preds, all_labels, all_logits_for_theta, all_difficulties_main_val = [], [], [], []

    try:
        input_ids_idx, attention_mask_idx, labels_idx = map(column_order.index,
                                                            ['input_ids', 'attention_mask', 'labels'])
        difficulty_idx_main_val = column_order.index('difficulty') if 'difficulty' in column_order else -1
        dataset_has_difficulty = (difficulty_idx_main_val != -1)
    except ValueError as e:
        print(f"FATAL ({desc_prefix}): Column missing for main val. Error: {e}");
        return 0.0, float('inf'), 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc_prefix, leave=False):
            num_batches += 1
            try:
                input_ids, attention_mask, labels_batch = batch[input_ids_idx].to(device), batch[attention_mask_idx].to(
                    device), batch[labels_idx].to(device)
                if dataset_has_difficulty: all_difficulties_main_val.append(
                    batch[difficulty_idx_main_val].cpu().numpy())
            except Exception as e:
                print(f"ERROR ({desc_prefix}) unpacking: {e}");
                traceback.print_exc();
                continue

            with autocast(device.type, dtype=amp_dtype_eval,
                          enabled=use_bf16_eval or (torch.cuda.is_available() and not use_bf16_eval)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss, logits = outputs.loss, outputs.logits.float()
            if loss is not None: total_loss += loss.item()
            all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.append(labels_batch.cpu().numpy())
            if dataset_has_difficulty: all_logits_for_theta.append(logits.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    val_accuracy = 0.0
    if all_labels and all_preds:
        final_preds, final_labels_np = np.concatenate(all_preds), np.concatenate(all_labels)
        if len(final_preds) == len(final_labels_np) and len(final_labels_np) > 0:
            val_accuracy = accuracy_metric.compute(predictions=final_preds, references=final_labels_np)['accuracy']

    theta_main, theta_time = 0.0, 0.0
    if dataset_has_difficulty and all_difficulties_main_val and all_logits_for_theta and len(all_logits_for_theta) > 0:
        s_time = time.time()
        try:
            logits_c, labels_c, diffs_c = map(np.concatenate,
                                              [all_logits_for_theta, all_labels,
                                               all_difficulties_main_val])  # Use all_labels here
            if len(logits_c) == len(labels_c) == len(diffs_c) and len(labels_c) > 0:
                preds_idx = np.argmax(logits_c, axis=1);
                rps = (preds_idx == labels_c).astype(int) * 2 - 1
                num_obs = PUDF_CONFIG.num_obs_theta if PUDF_CONFIG.num_obs_theta > 0 and PUDF_CONFIG.num_obs_theta < len(
                    diffs_c) else -1
                theta_main = calculate_theta(diffs_c, rps, num_obs)[0]
        except Exception as e_th:
            print(f"Error calc theta for {desc_prefix}: {e_th}");
            theta_main = 0.0;
            traceback.print_exc()
        theta_time = time.time() - s_time
    elif not dataset_has_difficulty:
        print(f"INFO ({desc_prefix}): No diffs in main val, Theta(MainVal)=0.0")

    print(
        f"Main Validation Results ({desc_prefix}): Acc={val_accuracy:.4f}, Loss={avg_loss:.4f}, Theta={theta_main:.4f} ({theta_time:.2f}s)")
    return val_accuracy, avg_loss, theta_main, theta_time


# --- Main Training Function ---
def train_qwen_pudf():
    setup_environment()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_bf16 = False
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        if torch.cuda.is_bf16_supported():
            use_bf16 = True;
            print("BF16 Supported: True.")
        else:
            print("BF16 Supported: False. Using FP16/FP32 on CUDA.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("Loading AG News dataset...")
    raw_ds = load_dataset(DATASET_ID, cache_dir=os.environ["HF_DATASETS_CACHE"])
    complete_ds_prov = raw_ds['complete']
    print(f"Initial columns in 'complete' split: {complete_ds_prov.column_names}")
    if 'news_story' in complete_ds_prov.column_names and 'text' not in complete_ds_prov.column_names:
        complete_ds_prov = complete_ds_prov.rename_column("news_story", "text")
    if 'labeling' in complete_ds_prov.column_names and 'label' not in complete_ds_prov.column_names:
        complete_ds_prov = complete_ds_prov.rename_column("labeling", "label")
    if 'text' not in complete_ds_prov.column_names or 'label' not in complete_ds_prov.column_names:
        raise ValueError(f"Essential 'text' or 'label' not found. Columns: {complete_ds_prov.column_names}")
    complete_ds = complete_ds_prov.shuffle(seed=RANDOM_SEED)

    train_val_split = complete_ds.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    train_full_hf = train_val_split['train']
    temp_hf = train_val_split['test']
    val_test_split = temp_hf.train_test_split(test_size=0.5, seed=RANDOM_SEED)
    validation_main_hf = val_test_split['train']
    test_hf = val_test_split['test']

    print(f"Loading difficulty scores from: {AG_NEWS_DIFFICULTY_FILE_PATH}")
    try:
        with open(os.path.abspath(AG_NEWS_DIFFICULTY_FILE_PATH), 'r') as f:
            diff_data = json.load(f)
        diff_scores = diff_data[DIFFICULTY_JSON_KEY]
        if len(diff_scores) != len(train_full_hf): raise ValueError("Difficulty scores mismatch train_full_hf size.")
        train_full_hf = train_full_hf.add_column("difficulty", diff_scores)
    except Exception as e:
        print(f"FATAL: Diff scores error: {e}");
        traceback.print_exc();
        return

    if len(train_full_hf) <= THETA_ESTIMATION_SET_SIZE: raise ValueError("Train pool too small for theta set.")
    if (len(train_full_hf) - THETA_ESTIMATION_SET_SIZE) < PUDF_CONFIG.min_train_length:
        print(f"Warning: Actual train samples after split < min_train_length.")
    train_theta_hf_split = train_full_hf.train_test_split(test_size=THETA_ESTIMATION_SET_SIZE, seed=RANDOM_SEED,
                                                          shuffle=True)
    actual_train_hf = train_theta_hf_split['train']
    theta_estimation_hf = train_theta_hf_split['test']
    print(f"PUDF splits: Actual Train={len(actual_train_hf)}, Theta Estimation Set={len(theta_estimation_hf)}")

    print(f"Loading tokenizer for: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=os.environ["TRANSFORMERS_CACHE"], token=True,
                                              trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token;
            print(f"Set pad_token to eos_token ('{tokenizer.pad_token}')")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'});
            print("Added [PAD] as pad_token")

    num_cpus = os.cpu_count()
    num_proc_map = min(max(1, (num_cpus // 2) if num_cpus else 1), 16)  # Default, can be increased
    print(f"Using {num_proc_map} processes for dataset mapping.")

    def tokenize_fn(ex):
        return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    tokenized_datasets_dict = {}
    base_columns_to_keep = ['label', 'difficulty']

    for name, ds_split in [("actual_train", actual_train_hf),
                           ("theta_estimation", theta_estimation_hf),
                           ("validation_main", validation_main_hf),
                           ("test", test_hf)]:

        cols_in_split = ds_split.column_names
        cols_to_remove = ["text"]
        for col_to_check in ['source', 'url', 'title', 'image', 'category', 'description', 'rank', 'pubdate']:
            if col_to_check in cols_in_split and col_to_check not in base_columns_to_keep:  # ensure not removing label/difficulty if they share names
                cols_to_remove.append(col_to_check)

        final_cols_to_remove = [col for col in cols_to_remove if col in cols_in_split]  # Only remove existing columns

        print(f"Tokenizing split: {name}. Will remove columns: {final_cols_to_remove}")
        tokenized_datasets_dict[name] = ds_split.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc_map,
            remove_columns=final_cols_to_remove
        )
        if 'label' in tokenized_datasets_dict[name].column_names:
            tokenized_datasets_dict[name] = tokenized_datasets_dict[name].rename_column("label", "labels")
            print(f"Renamed 'label' to 'labels' for {name} dataset.")
        if name in ["actual_train", "theta_estimation", "validation_main"] and 'labels' not in tokenized_datasets_dict[
            name].column_names:
            raise ValueError(
                f"'labels' column missing in {name} after processing. Columns: {tokenized_datasets_dict[name].column_names}")

    # Store tokenized test set for final evaluation
    global_tokenized_test = tokenized_datasets_dict["test"]

    num_labels = complete_ds.features['label'].num_classes
    print(f"Number of labels: {num_labels}")

    actual_train_tds, actual_train_col_order = create_qwen_tensor_dataset(tokenized_datasets_dict["actual_train"],
                                                                          "actual_train_pool", True)
    theta_est_tds, theta_est_cols = create_qwen_tensor_dataset(tokenized_datasets_dict["theta_estimation"],
                                                               "theta_est_set", True)
    theta_est_dl = DataLoader(theta_est_tds, PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=True)

    main_val_has_diff = 'difficulty' in tokenized_datasets_dict["validation_main"].column_names
    val_main_tds, val_main_cols = create_qwen_tensor_dataset(tokenized_datasets_dict["validation_main"],
                                                             "val_main_eval", main_val_has_diff)
    main_val_dl = DataLoader(val_main_tds, PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False, num_workers=2,
                             pin_memory=True)

    print(f"Loading model: {MODEL_CHECKPOINT}")
    model_cfg = AutoConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels, trust_remote_code=True)
    if tokenizer.pad_token_id is not None: model_cfg.pad_token_id = tokenizer.pad_token_id

    model_load_kwargs = {"config": model_cfg, "trust_remote_code": True}
    if use_bf16: model_load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, **model_load_kwargs)
    if hasattr(model, 'resize_token_embeddings') and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id != tokenizer.pad_token_id:  # Ensure sync after potential resize
        print(f"Re-syncing model.config.pad_token_id to {tokenizer.pad_token_id}")
        model.config.pad_token_id = tokenizer.pad_token_id

    if USE_GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"): model.config.use_cache = False
        print("Gradient checkpointing enabled, use_cache=False.")
    model.to(device)

    print("Setting up Adafactor optimizer.")
    optimizer = Adafactor(model.parameters(), lr=LEARNING_RATE_ADAFACTOR, scale_parameter=False, relative_step=False,
                          warmup_init=False, weight_decay=WEIGHT_DECAY)
    num_approx_update_steps_epoch = ceil(
        len(actual_train_tds) / (PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    total_train_steps = max(1, num_approx_update_steps_epoch * NUM_TRAIN_EPOCHS)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.06 * total_train_steps)),
                                                   num_training_steps=total_train_steps)

    scaler_enabled_flag = torch.cuda.is_available() and not use_bf16
    scaler = GradScaler(enabled=scaler_enabled_flag)
    print(
        f"GradScaler enabled: {scaler.is_enabled()} (use_bf16: {use_bf16}, cuda_available: {torch.cuda.is_available()})")

    print(f"\nStarting PUDF Qwen Training Loop ({PUDF_CONFIG.strategy} strategy)...")
    best_val_acc = 0.0;
    early_stop_count = 0;
    train_stats = []
    pudf_overhead = 0.0;
    prev_cap = -5.0;
    cur_cap = 0.0;
    est_theta_guidance = 0.0;
    epochs_completed = 0
    overall_start_time = time.time()

    for epoch in range(NUM_TRAIN_EPOCHS):
        epochs_completed = epoch + 1;
        print(f"\n======== Epoch {epochs_completed}/{NUM_TRAIN_EPOCHS} ========")
        epoch_s_time = time.time();
        model.train()
        theta_est_t, filter_t, avg_epoch_loss = 0.0, 0.0, 0.0

        if PUDF_CONFIG.strategy == 'theta':
            print("Estimating capacity (theta)...")
            est_theta_guidance, theta_est_t = evaluate_and_estimate_qwen_theta(model, theta_est_dl, device,
                                                                               theta_est_cols, num_labels,
                                                                               PUDF_CONFIG.num_obs_theta,
                                                                               f"Ep{epochs_completed} ThetaEst")
            model.train();
            pudf_overhead += theta_est_t
            if est_theta_guidance > prev_cap:
                cur_cap = est_theta_guidance
            else:
                cur_cap = prev_cap + 0.1 if prev_cap > -4.9 else 0.1;
                print(
                    f"  Theta ({est_theta_guidance:.4f}) not > prev ({prev_cap:.4f}). Adjusted cur_cap: {cur_cap:.4f}")
        elif PUDF_CONFIG.strategy == 'baseline':
            cur_cap = np.inf;
            est_theta_guidance = 'N/A'
        else:
            cur_cap = np.inf;
            est_theta_guidance = 'N/A';
            print(
                f"Warning: Strategy '{PUDF_CONFIG.strategy}' using default cur_cap.")

        filter_s_time = time.time()
        pudf_args = copy.deepcopy(PUDF_CONFIG);
        pudf_args.epoch = epoch
        filtered_data = get_epoch_training_data(actual_train_tds, pudf_args, epoch, PUDF_CONFIG.task_name_for_pudf,
                                                theta_hat=cur_cap if PUDF_CONFIG.strategy in ['theta',
                                                                                              'theta-hard'] and isinstance(
                                                    cur_cap, float) else None, lower_offset=PUDF_CONFIG.lower_bound,
                                                upper_offset=PUDF_CONFIG.upper_bound)
        filter_t = time.time() - filter_s_time;
        pudf_overhead += filter_t
        print(
            f"Data filtering epoch {epochs_completed} took {filter_t:.2f}s. Capacity: {cur_cap if isinstance(cur_cap, (float, np.floating)) else 'N/A'}.")

        n_epoch_samples = len(filtered_data['labels'])
        if n_epoch_samples == 0:
            print("Warning: No data for training.");
            avg_epoch_loss = 0.0
        else:
            print(f"PUDF selected {n_epoch_samples} samples for epoch {epochs_completed}.")
            epoch_tensors = [filtered_data['input_ids'], filtered_data['attention_mask'], filtered_data['labels'],
                             filtered_data['difficulty']]
            epoch_ds_filtered = TensorDataset(*epoch_tensors)
            epoch_dl = DataLoader(epoch_ds_filtered, PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=True, num_workers=2,
                                  pin_memory=True)

            epoch_loss_sum = 0;
            optim_steps = 0;
            optimizer.zero_grad()
            prog_bar = tqdm(epoch_dl, f"Epoch {epochs_completed} Training", leave=False)
            for step, batch_data in enumerate(prog_bar):
                ids, mask, lbls = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
                with autocast(device.type, dtype=amp_dtype,
                              enabled=use_bf16 or (torch.cuda.is_available() and not use_bf16)):
                    outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    loss_val = outputs.loss
                if loss_val is None or torch.isnan(loss_val): print(
                    f"NaN/None loss step {step}. Skip."); optimizer.zero_grad(set_to_none=True); continue

                loss_val_acc = loss_val / GRADIENT_ACCUMULATION_STEPS
                if scaler.is_enabled():
                    scaler.scale(loss_val_acc).backward()
                else:
                    loss_val_acc.backward()
                epoch_loss_sum += loss_val.item()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(epoch_dl):
                    if scaler.is_enabled(): scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer);
                        scaler.update()
                    else:
                        optimizer.step()
                    lr_scheduler.step();
                    optimizer.zero_grad(set_to_none=True);
                    optim_steps += 1
                prog_bar.set_postfix(
                    {'loss': f'{epoch_loss_sum / (step + 1):.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'})
            avg_epoch_loss = epoch_loss_sum / len(epoch_dl) if len(epoch_dl) > 0 else 0
            print(f"Epoch {epochs_completed} Avg Train Loss: {avg_epoch_loss:.4f} ({optim_steps} optim steps)")

        val_acc, val_loss_val, theta_main, theta_main_t = evaluate_qwen_main_val(model, main_val_dl, device,
                                                                                 val_main_cols, num_labels,
                                                                                 f"Ep{epochs_completed} MainVal")
        if PUDF_CONFIG.strategy == 'theta' and isinstance(est_theta_guidance, float): prev_cap = est_theta_guidance

        train_stats.append(
            {'epoch': epochs_completed, 'train_loss': avg_epoch_loss, 'val_loss': val_loss_val, 'val_acc': val_acc,
             'cur_cap': cur_cap if isinstance(cur_cap, float) else 'N/A',
             'theta_guidance': est_theta_guidance if isinstance(est_theta_guidance, float) else 'N/A',
             'theta_main_val': theta_main, 'theta_est_t': theta_est_t, 'filter_t': filter_t,
             'pudf_main_val_theta_time': theta_main_t, 'n_samples': n_epoch_samples})

        if val_acc > best_val_acc:
            print(f"Val acc improved ({best_val_acc:.4f} -> {val_acc:.4f}). Saving model...");
            best_val_acc = val_acc;
            early_stop_count = 0
            model_to_save = getattr(model, '_orig_mod', model)
            model_to_save.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"));
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            pudf_config_save_path = os.path.join(OUTPUT_DIR, "best_model", "pudf_training_config.json")
            try:
                with open(pudf_config_save_path, 'w') as f_cfg:
                    save_cfg = {
                        k: str(v) if isinstance(v, (float, np.floating)) and (v == np.inf or v == -np.inf) else v for
                        k, v in PUDF_CONFIG.__dict__.items() if not k.startswith('__')}
                    save_cfg['LEARNING_RATE'] = LEARNING_RATE_ADAFACTOR  # Store actual LR used
                    json.dump(save_cfg, f_cfg, indent=4, default=str)
            except Exception as e_cfg:
                print(f"Could not save PUDF config: {e_cfg}")
        else:
            early_stop_count += 1;
            print(
                f"Val acc not improved. Early stop: {early_stop_count}/{PATIENCE_EARLY_STOPPING}");
        if early_stop_count >= PATIENCE_EARLY_STOPPING: print("Early stopping."); break
        print(f"Epoch {epochs_completed} duration: {time.time() - epoch_s_time:.2f}s");
        gc.collect();
        torch.cuda.empty_cache()

    total_loop_t = time.time() - overall_start_time
    # MODIFIED: Added print for total training time
    print(f"\n--- Training Finished ({epochs_completed} epochs) ---")
    print(f"Total training loop duration: {total_loop_t:.2f}s ({total_loop_t / 3600:.2f} hours)")

    stats_p = os.path.join(OUTPUT_DIR, "train_stats.json");
    json.dump(train_stats, open(stats_p, 'w'), indent=4, default=str)
    print(f"Training statistics saved to {stats_p}")

    # MODIFIED: Robust best model check and test evaluation
    best_model_p = os.path.join(OUTPUT_DIR, "best_model")
    test_acc, test_loss_val = -10.0, -10.0  # Default values

    can_load_model = False
    if os.path.isdir(best_model_p):
        if (os.path.exists(os.path.join(best_model_p, "pytorch_model.bin")) or
                os.path.exists(os.path.join(best_model_p, "model.safetensors")) or
                os.path.exists(os.path.join(best_model_p, "model.safetensors.index.json")) or
                os.path.exists(os.path.join(best_model_p, "pytorch_model.bin.index.json"))):
            can_load_model = True
        else:
            files_in_dir = os.listdir(best_model_p)
            if any(fname.startswith("pytorch_model-") or fname.startswith("model-") for fname in files_in_dir) and \
                    any(fname.endswith((".bin", ".safetensors")) for fname in files_in_dir):
                print(f"Found sharded model files in {best_model_p} without a primary index file, attempting to load.")
                can_load_model = True
            elif files_in_dir and any(f.endswith((".json", ".bin", ".safetensors")) for f in files_in_dir):
                print(f"Warning: {best_model_p} exists and contains files. Attempting to load from directory.")
                can_load_model = True

    if can_load_model:
        print("\nLoading best model for test eval...")
        try:
            model_cfg_best = AutoConfig.from_pretrained(best_model_p, trust_remote_code=True)
            model_load_kwargs_best = {"config": model_cfg_best, "trust_remote_code": True}
            if use_bf16:
                model_load_kwargs_best["torch_dtype"] = torch.bfloat16
            else:
                model_load_kwargs_best["torch_dtype"] = model_load_dtype  # From earlier in train_qwen_pudf

            model_for_test_eval = AutoModelForSequenceClassification.from_pretrained(best_model_p,
                                                                                     **model_load_kwargs_best)
            model_for_test_eval.to(device)

            # Ensure global_tokenized_test is used here. It was defined in train_qwen_pudf data prep.
            test_tds, test_cols = create_qwen_tensor_dataset(global_tokenized_test, "test_final", False)
            test_dl = DataLoader(test_tds, PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE, num_workers=2, pin_memory=True)
            test_acc, test_loss_val, _, _ = evaluate_qwen_main_val(model_for_test_eval, test_dl, device, test_cols,
                                                                   num_labels, "Final Test Eval")
            print(f"Test Results: Acc={test_acc:.4f}, Loss={test_loss_val:.4f}")
            del model_for_test_eval  # Clean up
        except Exception as e_load_test:
            print(f"Error loading best model or evaluating on test set: {e_load_test}")
            traceback.print_exc()
            test_acc, test_loss_val = -1.0, -1.0
    else:
        print(f"No valid best model checkpoint found in {best_model_p}. Cannot evaluate on test set.")
        test_acc, test_loss_val = -2.0, -2.0

    cfg_log = {k: str(v) if isinstance(v, (float, np.floating)) and (v == np.inf or v == -np.inf) else v for k, v in
               PUDF_CONFIG.__dict__.items() if not k.startswith('__')}
    cfg_log.update({'LEARNING_RATE': LEARNING_RATE_ADAFACTOR,
                    'TRAIN_BATCH_SIZE_PHYSICAL': PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE,
                    'GRAD_ACCUM_STEPS': GRADIENT_ACCUMULATION_STEPS,
                    'EVAL_BATCH_SIZE': PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE})
    summary = {"best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss_val,
               "epochs_run": epochs_completed, "total_train_time_s": round(total_loop_t, 2),
               "pudf_overhead_s": round(pudf_overhead, 2), "config": cfg_log}
    json.dump(summary, open(os.path.join(OUTPUT_DIR, "final_summary.json"), 'w'), indent=4, default=str)
    print(f"Final run summary saved to {os.path.join(OUTPUT_DIR, 'final_summary.json')}")
    print("===== PUDF Qwen AGNews Task Finished =====")


if __name__ == "__main__":
    train_qwen_pudf()