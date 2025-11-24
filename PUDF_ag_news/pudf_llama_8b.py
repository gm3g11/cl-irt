# pudf_llama_agnews.py
import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,  # For loading config of saved model
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
from build_features_Llama import get_epoch_training_data
from irt_scoring import calculate_theta

# --- Configuration ---
OUTPUT_DIR = "Llama3.1_8B_PUDF_AGNews_Adafactor_BS64_3"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
AG_NEWS_DIFFICULTY_FILE_PATH = "../gen_difficulty/merged_jsonlines_output/test-1pl/best_parameters.json"
DIFFICULTY_JSON_KEY = "diff"

RANDOM_SEED = 63
MAX_LENGTH = 128
THETA_ESTIMATION_SET_SIZE = 10000
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5  # For Adafactor
WEIGHT_DECAY = 0.01

# Batch size configuration as per user's baseline
PER_DEVICE_TRAIN_BATCH_SIZE_CONFIG = 64
GRADIENT_ACCUMULATION_STEPS_CONFIG = 16
PER_DEVICE_EVAL_BATCH_SIZE = 64

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
    num_epochs=NUM_EPOCHS,
    competency=NUM_EPOCHS / 2.0
)
PATIENCE_EARLY_STOPPING = 3
USE_GRADIENT_CHECKPOINTING = True  # Common for large models


# --- Environment Setup ---
def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
        print("Removed HF_TOKEN environment variable to use cached token.")
    try:
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        user_info = whoami(token=hf_token)
        print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
    except Exception as e:
        print(
            f"Hugging Face login check failed: {e}. Public models should still work if already cached or no auth required.")


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed_value}")


# --- Helper: Create Llama Tensor Dataset ---
def create_llama_tensor_dataset(hf_tokenized_dataset_split,
                                partition_name="unknown",
                                include_difficulty=True):
    print(f"Creating TensorDataset for Llama {partition_name} (include_difficulty={include_difficulty})...")
    actual_columns = hf_tokenized_dataset_split.column_names
    print(f"  DEBUG ({partition_name}): Input HF Dataset columns: {actual_columns}")

    final_columns_ordered = ['input_ids', 'attention_mask']
    if 'token_type_ids' in actual_columns and 'token_type_ids' in hf_tokenized_dataset_split.features:
        final_columns_ordered.append('token_type_ids')
        print(f"  INFO ({partition_name}): 'token_type_ids' will be included (unexpected for Llama).")
    else:
        print(f"  INFO ({partition_name}): 'token_type_ids' will NOT be included (expected for Llama).")

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


# --- Helper: Evaluate Llama Theta ---
def evaluate_and_estimate_llama_theta(model, dataloader, device, column_order, num_labels,
                                      num_obs_theta=-1, desc_prefix="LlamaThetaEst"):
    print(f"Estimating Llama Theta ({desc_prefix})...")
    model.eval()
    all_preds_logits_list = []
    all_labels_list = []
    all_difficulties_list = []
    bf16_ready_eval = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype_eval = torch.bfloat16 if bf16_ready_eval else torch.float16

    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        labels_idx = column_order.index('labels')
        difficulty_idx = column_order.index('difficulty')
    except ValueError as e:
        print(
            f"FATAL ERROR ({desc_prefix}): Column missing for theta estimation. Expected 'input_ids', 'attention_mask', 'labels', 'difficulty' in column_order: {column_order}. Error: {e}")
        traceback.print_exc()
        return 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc_prefix, leave=False):
            try:
                input_ids = batch[input_ids_idx].to(device, non_blocking=True)
                attention_mask = batch[attention_mask_idx].to(device, non_blocking=True)
                labels_batch = batch[labels_idx].to(device, non_blocking=True)
                difficulty_tensor_for_batch = batch[difficulty_idx].cpu().numpy()
            except IndexError as e:
                print(f"ERROR ({desc_prefix}): IndexError unpacking batch. Error: {e}")
                traceback.print_exc();
                continue
            except Exception as e:
                print(f"ERROR ({desc_prefix}): Unexpected error unpacking batch: {e}")
                traceback.print_exc();
                continue

            with autocast(device.type, dtype=amp_dtype_eval,
                          enabled=bf16_ready_eval or (torch.cuda.is_available() and not bf16_ready_eval)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

            all_preds_logits_list.append(logits.cpu().numpy())
            all_labels_list.append(labels_batch.cpu().numpy())
            all_difficulties_list.append(difficulty_tensor_for_batch)

    theta_hat, model_capacity_time = 0.0, 0.0
    if not all_difficulties_list:
        print(f"Warning ({desc_prefix}): No difficulty scores collected. Theta defaults to 0.0.")
    else:
        final_preds_logits = np.concatenate(all_preds_logits_list)
        final_labels = np.concatenate(all_labels_list)
        concatenated_difficulties = np.concatenate(all_difficulties_list)

        if not (len(final_preds_logits) == len(final_labels) == len(concatenated_difficulties)):
            print(f"ERROR ({desc_prefix}): Mismatch in lengths. Theta estimation skipped.")
        elif len(final_labels) == 0:
            print(f"Warning ({desc_prefix}): No data processed for theta estimation.")
        else:
            time_model_s = time.time()
            predictions_for_metric = np.argmax(final_preds_logits, axis=1)
            response_pattern_irt = (predictions_for_metric == final_labels).astype(int) * 2 - 1
            num_total_obs = len(concatenated_difficulties)
            indices_for_theta = np.arange(num_total_obs)
            if num_obs_theta > 0 and num_obs_theta < num_total_obs:
                indices_for_theta = np.random.choice(num_total_obs, num_obs_theta, replace=False)
            try:
                theta_hat = \
                    calculate_theta(concatenated_difficulties[indices_for_theta],
                                    response_pattern_irt[indices_for_theta])[
                        0]
            except Exception as e:
                print(f"ERROR ({desc_prefix}) during theta calculation: {e}. Theta defaults to 0.0.")
                traceback.print_exc();
                theta_hat = 0.0
            model_capacity_time = time.time() - time_model_s
    print(f"Theta estimated ({desc_prefix}): {theta_hat:.4f} in {model_capacity_time:.2f}s")
    return theta_hat, model_capacity_time


# --- Helper: Evaluate Llama on Main Validation (Acc, Loss) ---
def evaluate_llama_main_val(model, dataloader, device, column_order, num_labels, desc_prefix="LlamaMainVal"):
    print(f"Evaluating Llama on Main Validation ({desc_prefix})...")
    model.eval()
    accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
    total_loss = 0.0
    num_batches = 0
    bf16_ready_eval = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype_eval = torch.bfloat16 if bf16_ready_eval else torch.float16

    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        labels_idx = column_order.index('labels')
    except ValueError as e:
        print(f"FATAL ERROR ({desc_prefix}): Column missing for main validation. Error: {e}")
        traceback.print_exc();
        return 0.0, float('inf'), 0.0, 0.0

    all_preds_for_metric = []
    all_labels_for_metric = []
    all_logits_for_theta_calc = []

    difficulty_idx_main_val = column_order.index('difficulty') if 'difficulty' in column_order else -1
    dataset_has_difficulty_main_val = (difficulty_idx_main_val != -1)
    all_difficulties_main_val = []
    theta_main_val, theta_time_main_val = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc_prefix, leave=False):
            num_batches += 1
            try:
                input_ids = batch[input_ids_idx].to(device, non_blocking=True)
                attention_mask = batch[attention_mask_idx].to(device, non_blocking=True)
                labels_batch = batch[labels_idx].to(device, non_blocking=True)
                if dataset_has_difficulty_main_val:
                    all_difficulties_main_val.append(batch[difficulty_idx_main_val].cpu().numpy())
            except Exception as e:
                print(f"ERROR ({desc_prefix}) unpacking batch: {e}")
                traceback.print_exc();
                continue

            with autocast(device.type, dtype=amp_dtype_eval,
                          enabled=bf16_ready_eval or (torch.cuda.is_available() and not bf16_ready_eval)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            logits = outputs.logits.float()

            if loss is not None: total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_preds_for_metric.append(predictions.cpu().numpy())
            all_labels_for_metric.append(labels_batch.cpu().numpy())
            if dataset_has_difficulty_main_val:
                all_logits_for_theta_calc.append(logits.cpu().numpy())

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    val_accuracy = 0.0
    if all_labels_for_metric and all_preds_for_metric:
        final_predictions = np.concatenate(all_preds_for_metric)
        final_references = np.concatenate(all_labels_for_metric)
        if len(final_predictions) == len(final_references) and len(final_references) > 0:
            val_accuracy = accuracy_metric.compute(predictions=final_predictions, references=final_references)[
                'accuracy']
        else:
            print(f"Warning ({desc_prefix}): Mismatch or no data for accuracy calculation.")

    if dataset_has_difficulty_main_val and all_difficulties_main_val and all_logits_for_theta_calc and len(
            all_logits_for_theta_calc) > 0:
        print(f"Attempting theta calculation for {desc_prefix} as difficulties were found.")
        s_time = time.time()
        try:
            logits_concat = np.concatenate(all_logits_for_theta_calc)
            labels_concat = np.concatenate(all_labels_for_metric)  # Use all_labels collected in this loop
            difficulties_concat = np.concatenate(all_difficulties_main_val)
            if not (len(logits_concat) == len(labels_concat) == len(difficulties_concat)):
                print(f"ERROR ({desc_prefix}): Mismatch in lengths for theta. Skipping theta calc.")
            elif len(labels_concat) == 0:
                print(f"Warning ({desc_prefix}): No labels for theta calculation on main val set.")
            else:
                preds_indices_theta = np.argmax(logits_concat, axis=1)
                rps_theta = (preds_indices_theta == labels_concat).astype(int) * 2 - 1
                theta_main_val = calculate_theta(difficulties_concat, rps_theta,
                                                 num_obs=PUDF_CONFIG.num_obs_theta if PUDF_CONFIG.num_obs_theta > 0 else -1)[
                    0]
        except Exception as e_theta:
            print(f"Error calculating theta for {desc_prefix}: {e_theta}");
            theta_main_val = 0.0
            traceback.print_exc()
        theta_time_main_val = time.time() - s_time
    elif dataset_has_difficulty_main_val:  # Difficulties expected but not enough data collected (e.g. all_logits_for_theta_calc is empty)
        print(f"INFO ({desc_prefix}): Difficulties column present but not enough data collected for theta calculation.")
    else:  # No difficulty column in the dataset
        print(f"INFO ({desc_prefix}): No difficulties in main val set, Theta(MainVal) will be 0.0")

    print(
        f"Main Validation Results ({desc_prefix}): Accuracy={val_accuracy:.4f}, Avg Loss={avg_loss:.4f}, Theta(MainVal)={theta_main_val:.4f} ({theta_time_main_val:.2f}s)")
    return val_accuracy, avg_loss, theta_main_val, theta_time_main_val


# --- Main Training Function ---
def train_llama_pudf():
    setup_environment()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_bf16 = False
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("CUDA BF16 Supported: True. Will use BF16.")
        else:
            print("CUDA BF16 Supported: False. Will use FP16 if CUDA available, else FP32 on CPU.")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("Loading AG News dataset...")
    ag_news_dataset_hf = load_dataset("contemmcm/ag_news", cache_dir=os.environ["HF_DATASETS_CACHE"])
    complete_dataset_provisional = ag_news_dataset_hf['complete']
    print(f"Initial columns in 'complete' split: {complete_dataset_provisional.column_names}")
    if 'news_story' in complete_dataset_provisional.column_names and 'text' not in complete_dataset_provisional.column_names:
        print("Renaming 'news_story' to 'text'.")
        complete_dataset_provisional = complete_dataset_provisional.rename_column("news_story", "text")

    if 'labeling' in complete_dataset_provisional.column_names and 'label' not in complete_dataset_provisional.column_names:
        print("Renaming 'labeling' to 'label'.")
        complete_dataset_provisional = complete_dataset_provisional.rename_column("labeling", "label")

    if 'text' not in complete_dataset_provisional.column_names:
        raise ValueError(f"Essential column 'text' not found. Columns: {complete_dataset_provisional.column_names}")
    if 'label' not in complete_dataset_provisional.column_names:
        raise ValueError(f"Essential column 'label' not found. Columns: {complete_dataset_provisional.column_names}")
    complete_dataset = complete_dataset_provisional
    print(
        f"Using columns for processing: 'text' and 'label'. Final columns in complete_dataset: {complete_dataset.column_names}")

    train_val_split = complete_dataset.train_test_split(test_size=0.2, seed=RANDOM_SEED, shuffle=True)
    train_full_hf = train_val_split['train']
    temp_hf = train_val_split['test']
    val_test_split = temp_hf.train_test_split(test_size=0.5, seed=RANDOM_SEED, shuffle=True)
    validation_main_hf = val_test_split['train']
    test_hf = val_test_split['test']

    print(f"Loading difficulty scores from: {AG_NEWS_DIFFICULTY_FILE_PATH}")
    try:
        with open(os.path.abspath(AG_NEWS_DIFFICULTY_FILE_PATH), 'r') as f:
            difficulty_data = json.load(f)
        difficulty_scores = difficulty_data[DIFFICULTY_JSON_KEY]
        if len(difficulty_scores) != len(train_full_hf):
            raise ValueError(f"Diff scores ({len(difficulty_scores)}) != train_full ({len(train_full_hf)}).")
        train_full_hf = train_full_hf.add_column("difficulty", difficulty_scores)
    except Exception as e:
        print(f"FATAL error loading diff scores: {e}")
        traceback.print_exc();
        return

    if len(train_full_hf) <= THETA_ESTIMATION_SET_SIZE:
        raise ValueError(f"Train pool ({len(train_full_hf)}) too small for theta set ({THETA_ESTIMATION_SET_SIZE}).")
    if (len(train_full_hf) - THETA_ESTIMATION_SET_SIZE) < PUDF_CONFIG.min_train_length:
        print(
            f"Warning: After taking {THETA_ESTIMATION_SET_SIZE} for theta, remaining train samples ({len(train_full_hf) - THETA_ESTIMATION_SET_SIZE}) < min_train_length ({PUDF_CONFIG.min_train_length}).")

    train_theta_hf_split = train_full_hf.train_test_split(test_size=THETA_ESTIMATION_SET_SIZE, seed=RANDOM_SEED,
                                                          shuffle=True)
    actual_train_hf = train_theta_hf_split['train']
    theta_estimation_hf = train_theta_hf_split['test']
    print(f"PUDF splits: Actual Train={len(actual_train_hf)}, Theta Estimation Set={len(theta_estimation_hf)}")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=os.environ["TRANSFORMERS_CACHE"], token=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print("Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Warning: No EOS token found. Adding a new pad token '[PAD]'")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    print("Tokenizing datasets...")
    num_cpus = os.cpu_count()
    num_proc_map = min(max(1, (num_cpus // 2) if num_cpus else 1), 16)
    print(f"Using {num_proc_map} processes for dataset mapping.")

    tokenized_datasets_dict = {}
    cols_to_remove_base = [col for col in complete_dataset.column_names if col not in ['text', 'label', 'difficulty']]

    for name, ds_split in [("actual_train", actual_train_hf),
                           ("theta_estimation", theta_estimation_hf),
                           ("validation_main", validation_main_hf),
                           ("test", test_hf)]:
        current_cols_to_remove = ["text"] + [col for col in cols_to_remove_base if col in ds_split.column_names]
        tokenized_datasets_dict[name] = ds_split.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc_map,
            remove_columns=current_cols_to_remove
        )
        if 'label' in tokenized_datasets_dict[name].column_names:
            tokenized_datasets_dict[name] = tokenized_datasets_dict[name].rename_column("label", "labels")
            print(f"Renamed 'label' to 'labels' for {name} dataset.")
        if name in ["actual_train", "theta_estimation", "validation_main"] and 'labels' not in tokenized_datasets_dict[
            name].column_names:
            raise ValueError(
                f"'labels' column missing in {name} dataset after renaming attempt. Columns: {tokenized_datasets_dict[name].column_names}")

    tokenized_actual_train = tokenized_datasets_dict["actual_train"]
    tokenized_theta_estimation = tokenized_datasets_dict["theta_estimation"]
    tokenized_validation_main = tokenized_datasets_dict["validation_main"]
    # Keep a reference to the tokenized test set for final evaluation
    global_tokenized_test = tokenized_datasets_dict["test"]  # Store this for later use

    num_labels = complete_dataset.features['label'].num_classes
    print(f"Number of labels: {num_labels}")

    actual_train_tensordataset, _ = create_llama_tensor_dataset(tokenized_actual_train, "actual_train_pool_for_pudf",
                                                                True)
    theta_est_tensords, theta_est_col_order = create_llama_tensor_dataset(tokenized_theta_estimation,
                                                                          "theta_estimation_set", True)
    theta_estimation_dataloader = DataLoader(theta_est_tensords, batch_size=PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False,
                                             num_workers=2, pin_memory=True)

    main_val_set_has_difficulty_flag = 'difficulty' in tokenized_validation_main.column_names
    val_main_tensords, val_main_col_order = create_llama_tensor_dataset(tokenized_validation_main,
                                                                        "validation_main_eval",
                                                                        main_val_set_has_difficulty_flag)
    main_validation_dataloader = DataLoader(val_main_tensords, batch_size=PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False,
                                            num_workers=2, pin_memory=True)

    print(f"Loading model: {MODEL_NAME}")
    model_load_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels, token=True)
    if tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id
    else:
        raise ValueError("Tokenizer pad_token_id is None, critical for model config.")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config, cache_dir=os.environ["TRANSFORMERS_CACHE"],
        token=True, torch_dtype=model_load_dtype
    )

    if hasattr(model, 'resize_token_embeddings') and len(tokenizer) > model.config.vocab_size:
        print(f"Resizing model token embeddings from {model.config.vocab_size} to: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        if model.config.pad_token_id != tokenizer.pad_token_id:
            print(f"Re-setting model.config.pad_token_id to {tokenizer.pad_token_id} after embedding resize.")
            model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, 'gradient_checkpointing_enable'): model.gradient_checkpointing_enable(); print(
        "Gradient checkpointing enabled.")
    if hasattr(model.config, 'use_cache'): model.config.use_cache = False; print("Set model.config.use_cache = False")
    model.to(device)

    print("Setting up Adafactor optimizer.")
    optimizer = Adafactor(model.parameters(), lr=LEARNING_RATE, scale_parameter=False, relative_step=False,
                          warmup_init=False, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch_approx = ceil(
        len(actual_train_tensordataset) / (PER_DEVICE_TRAIN_BATCH_SIZE_CONFIG * GRADIENT_ACCUMULATION_STEPS_CONFIG))
    total_training_steps = num_update_steps_per_epoch_approx * NUM_EPOCHS
    if total_training_steps == 0 and len(actual_train_tensordataset) > 0: total_training_steps = NUM_EPOCHS
    if total_training_steps == 0: total_training_steps = 1

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_training_steps)) if total_training_steps > 0 else 0,
        num_training_steps=max(1, total_training_steps)
    )
    scaler_enabled_flag = torch.cuda.is_available() and not use_bf16
    scaler = GradScaler(enabled=scaler_enabled_flag)
    print(
        f"GradScaler enabled: {scaler.is_enabled()} (use_bf16: {use_bf16}, cuda_available: {torch.cuda.is_available()})")

    print(f"\nStarting PUDF Llama Training Loop ({PUDF_CONFIG.strategy} strategy)...")
    best_val_accuracy = 0.0;
    early_stop_counter = 0;
    training_stats_list = []
    total_pudf_overhead_time = 0.0;
    prev_cap = -5.0;
    cur_cap_for_filtering = 0.0
    estimated_theta_current_epoch = 0.0;
    actual_epochs_run = 0

    overall_train_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        actual_epochs_run = epoch + 1
        print(f"\n======== Epoch {actual_epochs_run} / {NUM_EPOCHS} ========")
        epoch_start_time = time.time();
        model.train()
        epoch_theta_est_time = 0.0;
        epoch_filter_time = 0.0;
        avg_epoch_train_loss = 0.0

        if PUDF_CONFIG.strategy == 'theta':
            print("Estimating capacity (theta) from dedicated training subset...")
            estimated_theta_current_epoch, epoch_theta_est_time = evaluate_and_estimate_llama_theta(
                model, theta_estimation_dataloader, device, theta_est_col_order, num_labels,
                num_obs_theta=PUDF_CONFIG.num_obs_theta, desc_prefix=f"Epoch {actual_epochs_run} ThetaEst"
            )
            model.train()
            total_pudf_overhead_time += epoch_theta_est_time
            if estimated_theta_current_epoch > prev_cap:
                cur_cap_for_filtering = estimated_theta_current_epoch
            else:
                cur_cap_for_filtering = prev_cap + 0.1 if prev_cap > -4.9 else 0.1
                print(
                    f"  Theta_guidance ({estimated_theta_current_epoch:.4f}) not > prev ({prev_cap:.4f}). Adjusted cur_cap: {cur_cap_for_filtering:.4f}")
        elif PUDF_CONFIG.strategy == 'baseline':
            cur_cap_for_filtering = np.inf
            estimated_theta_current_epoch = 'N/A'
        else:
            cur_cap_for_filtering = np.inf
            estimated_theta_current_epoch = 'N/A'
            print(f"Warning: Strategy '{PUDF_CONFIG.strategy}' - cur_cap logic might need specific handling.")

        filter_start_time = time.time()
        pudf_args_for_filter = copy.deepcopy(PUDF_CONFIG)
        pudf_args_for_filter.epoch = epoch

        filtered_data_dict = get_epoch_training_data(
            actual_train_tensordataset, pudf_args_for_filter, epoch, PUDF_CONFIG.task_name_for_pudf,
            theta_hat=cur_cap_for_filtering if PUDF_CONFIG.strategy in ['theta', 'theta-hard'] and isinstance(
                cur_cap_for_filtering, float) else None,
            lower_offset=PUDF_CONFIG.lower_bound, upper_offset=PUDF_CONFIG.upper_bound
        )
        epoch_filter_time = time.time() - filter_start_time;
        total_pudf_overhead_time += epoch_filter_time
        print(
            f"Data filtering for epoch {actual_epochs_run} took {epoch_filter_time:.2f}s. Capacity: {cur_cap_for_filtering if isinstance(cur_cap_for_filtering, (float, np.floating)) else 'N/A'}.")

        num_epoch_train_samples = len(filtered_data_dict['labels'])
        if num_epoch_train_samples == 0:
            print("Warning: No data for training this epoch.");
            avg_epoch_train_loss = 0.0
        else:
            print(f"PUDF selected {num_epoch_train_samples} samples for training in epoch {actual_epochs_run}.")
            epoch_train_tensors = [
                filtered_data_dict['input_ids'], filtered_data_dict['attention_mask'],
                filtered_data_dict['labels'], filtered_data_dict['difficulty']
            ]
            epoch_train_dataset_filtered = TensorDataset(*epoch_train_tensors)
            epoch_train_dataloader = DataLoader(epoch_train_dataset_filtered,
                                                batch_size=PER_DEVICE_TRAIN_BATCH_SIZE_CONFIG,
                                                shuffle=True, num_workers=2, pin_memory=True)

            epoch_total_loss_sum = 0;
            num_optimizer_steps_this_epoch = 0
            optimizer.zero_grad()
            progress_bar = tqdm(epoch_train_dataloader, desc=f"Epoch {actual_epochs_run} Training", leave=False)
            for step, batch in enumerate(progress_bar):
                batch_input_ids = batch[0].to(device);
                batch_attention_mask = batch[1].to(device)
                batch_labels = batch[2].to(device)

                with autocast(device.type, dtype=amp_dtype,
                              enabled=use_bf16 or (torch.cuda.is_available() and not use_bf16)):
                    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
                    loss = outputs.loss
                if loss is None or torch.isnan(loss): print(
                    f"Warning: NaN/None loss at step {step}. Skipping."); optimizer.zero_grad(
                    set_to_none=True); continue

                loss_acc = loss / GRADIENT_ACCUMULATION_STEPS_CONFIG

                if scaler.is_enabled():
                    scaler.scale(loss_acc).backward()
                else:
                    loss_acc.backward()

                epoch_total_loss_sum += loss.item()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS_CONFIG == 0 or (step + 1) == len(epoch_train_dataloader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    num_optimizer_steps_this_epoch += 1

                current_avg_batch_loss = epoch_total_loss_sum / (step + 1)
                progress_bar.set_postfix(
                    {'loss': f'{current_avg_batch_loss:.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'})
            avg_epoch_train_loss = epoch_total_loss_sum / len(epoch_train_dataloader) if len(
                epoch_train_dataloader) > 0 else 0
            print(
                f"Epoch {actual_epochs_run} Avg Training Loss: {avg_epoch_train_loss:.4f} ({num_optimizer_steps_this_epoch} optimizer steps)")

        val_accuracy, val_loss, theta_on_main_val, theta_time_main_val = evaluate_llama_main_val(
            model, main_validation_dataloader, device, val_main_col_order, num_labels,
            desc_prefix=f"Epoch {actual_epochs_run} MainVal"
        )

        if PUDF_CONFIG.strategy == 'theta' and isinstance(estimated_theta_current_epoch, float):
            prev_cap = estimated_theta_current_epoch

        training_stats_list.append({
            'epoch': actual_epochs_run, 'train_loss': avg_epoch_train_loss, 'val_loss': val_loss,
            'val_acc': val_accuracy,
            'cur_cap_for_filter': cur_cap_for_filtering if isinstance(cur_cap_for_filtering,
                                                                      (float, np.floating)) else 'N/A',
            'theta_guidance': estimated_theta_current_epoch if isinstance(estimated_theta_current_epoch, (
                float, np.floating)) else 'N/A',
            'theta_on_main_val_set': theta_on_main_val,
            'pudf_theta_est_time': epoch_theta_est_time, 'pudf_filter_time': epoch_filter_time,
            'pudf_main_val_theta_time': theta_time_main_val,
            'n_train_samples_epoch': num_epoch_train_samples
        })

        if val_accuracy > best_val_accuracy:
            print(f"Val acc improved ({best_val_accuracy:.4f} --> {val_accuracy:.4f}). Saving best model...")
            best_val_accuracy = val_accuracy;
            early_stop_counter = 0
            model_to_save = getattr(model, '_orig_mod', model)
            model_to_save.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            pudf_config_save_path = os.path.join(OUTPUT_DIR, "best_model", "pudf_training_config.json")
            try:
                with open(pudf_config_save_path, 'w') as f_cfg:
                    save_cfg = {
                        k: str(v) if isinstance(v, (float, np.floating)) and (v == np.inf or v == -np.inf) else v for
                        k, v in PUDF_CONFIG.__dict__.items() if not k.startswith('__')}
                    save_cfg['LEARNING_RATE'] = LEARNING_RATE
                    json.dump(save_cfg, f_cfg, indent=4, default=str)
            except Exception as e_cfg:
                print(f"Could not save PUDF config: {e_cfg}")
        else:
            early_stop_counter += 1
            print(f"Val acc not improved. Early stop count: {early_stop_counter}/{PATIENCE_EARLY_STOPPING}")
            if early_stop_counter >= PATIENCE_EARLY_STOPPING: print("Early stopping."); break

        print(f"Epoch {actual_epochs_run} duration: {time.time() - epoch_start_time:.2f}s");
        gc.collect();
        torch.cuda.empty_cache()

    total_train_loop_duration = time.time() - overall_train_start_time
    print("\n--- PUDF Llama Training Loop Finished ---")
    print(
        f"Total training loop duration: {total_train_loop_duration:.2f}s ({total_train_loop_duration / 3600:.2f} hours)")  # ADDED PRINT

    stats_path = os.path.join(OUTPUT_DIR, "training_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats_list, f, indent=4, default=str)
    print(f"Training statistics saved to {stats_path}")

    # Initialize test results for summary to ensure they are defined
    test_accuracy, test_loss = -10.0, -10.0  # Using distinct values to indicate not run or error

    best_model_path = os.path.join(OUTPUT_DIR, "best_model")
    can_load_model = False
    if os.path.isdir(best_model_path):
        if (os.path.exists(os.path.join(best_model_path, "pytorch_model.bin")) or
                os.path.exists(os.path.join(best_model_path, "model.safetensors")) or
                os.path.exists(os.path.join(best_model_path, "model.safetensors.index.json")) or
                os.path.exists(os.path.join(best_model_path, "pytorch_model.bin.index.json"))):
            can_load_model = True
        else:
            files_in_dir = os.listdir(best_model_path)
            if any(fname.startswith("pytorch_model-") or fname.startswith("model-") for fname in files_in_dir) and \
                    any(fname.endswith((".bin", ".safetensors")) for fname in files_in_dir):
                print(
                    f"Found sharded model files in {best_model_path} without a primary index file, attempting to load.")
                can_load_model = True
            elif files_in_dir and any(f.endswith((".json", ".bin", ".safetensors")) for f in files_in_dir):
                print(f"Warning: {best_model_path} exists and contains files. Attempting to load from directory.")
                can_load_model = True

    if can_load_model:
        print(f"\nLoading best model from: {best_model_path} for final test evaluation...")
        try:
            # For Llama, trust_remote_code is usually False.
            model_config_best = AutoConfig.from_pretrained(best_model_path,
                                                           token=True)  # Use token if needed for gated models

            model_load_kwargs_best = {"config": model_config_best, "token": True}
            if use_bf16:
                model_load_kwargs_best["torch_dtype"] = torch.bfloat16
            else:
                model_load_kwargs_best["torch_dtype"] = model_load_dtype  # Original model_load_dtype

            loaded_model_for_test = AutoModelForSequenceClassification.from_pretrained(best_model_path,
                                                                                       **model_load_kwargs_best)
            loaded_model_for_test.to(device)

            # Ensure tokenized_test (global_tokenized_test) is correctly used
            test_tensords, test_col_order = create_llama_tensor_dataset(global_tokenized_test, "test_final_eval",
                                                                        include_difficulty=False)
            test_dataloader = DataLoader(test_tensords, batch_size=PER_DEVICE_EVAL_BATCH_SIZE, num_workers=2,
                                         pin_memory=True)

            test_accuracy, test_loss, _, _ = evaluate_llama_main_val(loaded_model_for_test, test_dataloader, device,
                                                                     test_col_order, num_labels, "Final Test Eval")
            print(f"Final Test Results: Accuracy={test_accuracy:.4f}, Loss={test_loss:.4f}")

        except Exception as e_load_eval:
            print(f"Error loading best model or evaluating on test set: {e_load_eval}")
            traceback.print_exc()
            test_accuracy, test_loss = -1.0, -1.0

    else:
        print(
            f"No valid best model checkpoint found in {best_model_path}. This may happen if no improvement was seen or saving failed.")
        test_accuracy, test_loss = -2.0, -2.0

    final_config_log = {k: str(v) if isinstance(v, (float, np.floating)) and (v == np.inf or v == -np.inf) else v
                        for k, v in PUDF_CONFIG.__dict__.items() if not k.startswith('__')}
    final_config_log['NUM_EPOCHS'] = NUM_EPOCHS
    final_config_log['LEARNING_RATE'] = LEARNING_RATE
    final_config_log['PER_DEVICE_TRAIN_BATCH_SIZE_CONFIG'] = PER_DEVICE_TRAIN_BATCH_SIZE_CONFIG
    final_config_log['GRADIENT_ACCUMULATION_STEPS_CONFIG'] = GRADIENT_ACCUMULATION_STEPS_CONFIG
    final_config_log['PER_DEVICE_EVAL_BATCH_SIZE'] = PER_DEVICE_EVAL_BATCH_SIZE

    results_summary = {"best_validation_accuracy": best_val_accuracy, "final_test_accuracy": test_accuracy,
                       "final_test_loss": test_loss, "epochs_run": actual_epochs_run,
                       "total_training_loop_time_seconds": round(total_train_loop_duration, 2),
                       "total_pudf_overhead_time_seconds": round(total_pudf_overhead_time, 2),
                       "config": final_config_log}
    summary_path = os.path.join(OUTPUT_DIR, "final_run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4, default=str)
    print(f"Final run summary saved to {summary_path}")

    print("===== PUDF Llama AGNews Task Finished =====")


if __name__ == "__main__":
    train_llama_pudf()