# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import time
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
# Use Adafactor Optimizer
from transformers.optimization import Adafactor
# Mixed Precision
from torch.cuda.amp import GradScaler # Standard import style
from torch.amp import autocast # Updated import style
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset
import evaluate
# Assuming these custom modules are in the same directory or Python path
from build_features_Qwen import get_epoch_training_data # Import the corrected version
from irt_scoring import calculate_theta
import types # To create a simple namespace object for config
import gc # Garbage Collection
import traceback # For detailed error logging
from tqdm.auto import tqdm # Import tqdm for progress bars
from huggingface_hub import login, whoami # Keep for checking login status
from math import ceil # For calculating number of batches

# --- Environment, Setup, Config, Helper Functions ---
# ... (Keep all the code from HF_HOME down to evaluate_and_estimate as it was) ...
# --- Environment and Cache Setup ---
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE # MODIFY IF NEEDED
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Hugging Face Auth Check (Optional but good practice) ---
try: user_info = whoami(); print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
except Exception as e: print(f"HF Login Check Failed: {e}. Public models should still work.")

# --- Global Random Seed ---
random_seed = 63
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

# --- Task Definitions ---
GLUETASKS = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp'] # Restore full list
task_max_lengths = {"mrpc": 72, "rte": 150, "mnli": 72, "qqp": 56, "sst2": 32, "qnli": 80}

# --- Global Training Configuration ---
# Batch sizes for Qwen 7B - Adjust based on your GPU VRAM
train_batch_size = 256 # Per device step (e.g., 16, 8, 4 if OOM)
effective_batch_size = 256 # Target effective batch size
gradient_accumulation_steps = max(1, effective_batch_size // train_batch_size) # Adjusts automatically
eval_batch_size = 256 # Can likely handle larger eval batches (e.g., 64, 128)
test_batch_size = 256

print(f"Train Batch Size (per device step): {train_batch_size}")
print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
print(f"Effective Batch Size (Train): {train_batch_size * gradient_accumulation_steps}")
print(f"Eval/Test Batch Size: {eval_batch_size}")

transformers.logging.set_verbosity_error()

# --- BF16 Check ---
bf16_ready = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
amp_dtype = torch.bfloat16 if bf16_ready else torch.float16
print(f"AMP Dtype selected: {amp_dtype}")

# --- Helper Functions ---
def load_and_prepare_data(task, diff_dir, cache_dir):
    """Loads GLUE dataset, adds difficulty scores, performs train/val split."""
    print(f"Loading dataset for task: {task}")
    try: raw_datasets = load_dataset('glue', task, cache_dir=cache_dir)
    except Exception as e: print(f"ERROR loading dataset '{task}': {e}"); raise
    # Handle MNLI validation split name difference
    validation_split_name = 'validation_matched' if task == 'mnli' else 'validation'
    if validation_split_name not in raw_datasets:
        print(f"Warning: Expected validation split '{validation_split_name}' not found for task '{task}'. Trying 'validation'.")
        validation_split_name = 'validation' # Fallback
        if validation_split_name not in raw_datasets:
             raise ValueError(f"Cannot find a suitable validation/test split for task '{task}'")

    train_diff_file = f'{diff_dir}/{task.lower()}-1pl/best_parameters.json'; print(f"Loading difficulty scores from: {train_diff_file}")
    try:
        with open(train_diff_file, 'r') as file: difficulty_data = json.load(file)
        if 'diff' not in difficulty_data or not isinstance(difficulty_data['diff'], list): raise ValueError("Difficulty file needs list under 'diff' key.")
    except Exception as e: print(f"ERROR loading/parsing difficulty file {train_diff_file}: {e}"); raise
    train = raw_datasets['train']
    if len(difficulty_data['diff']) != len(train): raise ValueError(f"Difficulty count ({len(difficulty_data['diff'])}) != dataset size ({len(train)}) for {task}.")
    print("Adding difficulty scores...");
    if 'difficulty' in train.column_names: print("Warning: Replacing 'difficulty' column."); train = train.remove_columns(['difficulty'])
    train = train.add_column('difficulty', difficulty_data['diff'])
    print("Splitting train data (90/10)...")
    train_val_split = train.train_test_split(test_size=0.1, seed=random_seed)
    train_dataset, val_dataset = train_val_split['train'], train_val_split['test']

    # Determine test set: Use the designated validation split (matched for MNLI, standard validation otherwise)
    test_dataset = raw_datasets[validation_split_name]
    print(f"Using '{validation_split_name}' split for testing.")

    print("Data loading complete.")
    return train_dataset, val_dataset, test_dataset


def tokenize_function(examples, task, tokenizer):
    """Tokenizes examples based on task structure and max length."""
    max_length = task_max_lengths.get(task);
    if max_length is None: print(f"Warning: max_length not defined for {task}. Using 128."); max_length = 128
    text_a, text_b = None, None # Initialize
    if task == "mnli": text_a = examples["premise"]; text_b = examples["hypothesis"]
    elif task in ["mrpc", "rte", "qqp"]: text_a = examples["sentence1" if task != "qqp" else "question1"]; text_b = examples["sentence2" if task != "qqp" else "question2"]
    elif task == "qnli": text_a = examples["question"]; text_b = examples["sentence"]
    elif task == "sst2": text_a = examples["sentence"]; text_b = None
    else: raise ValueError(f"Task {task} not supported by tokenize_function.")

    # Handle None case for single sentences properly
    if text_b:
        return tokenizer(text=text_a, text_pair=text_b, padding="max_length", truncation=True, max_length=max_length)
    else:
        return tokenizer(text=text_a, padding="max_length", truncation=True, max_length=max_length)


def create_dataset(dataset, task, tokenizer, include_difficulty=True):
    """Tokenizes dataset and converts to PyTorch TensorDataset."""
    print(f"Tokenizing dataset (include_difficulty={include_difficulty})...")
    try:
        # Determine required columns based on tokenizer output (Qwen typically doesn't use token_type_ids)
        # Tokenize a dummy example to see output columns
        # Ensure the dummy example matches the task structure (single sentence vs pair)
        if task in ["sst2"]: dummy_example = dataset[:1]
        else: dummy_example = dataset[:1] # Assumes paired structure for others, adjust if needed

        sample_tokenization = tokenize_function(dummy_example, task, tokenizer)
        tokenized_cols = ['input_ids', 'attention_mask']
        if 'token_type_ids' in sample_tokenization:
             print("Note: Tokenizer generated token_type_ids, including them.")
             tokenized_cols.append('token_type_ids')

        # Columns to keep: tokenized columns + label (+ difficulty if needed)
        cols_to_keep = set(tokenized_cols + ['label']) # Base dataset uses 'label'
        if include_difficulty:
            if 'difficulty' not in dataset.column_names:
                 raise ValueError(f"Difficulty column missing from dataset for task {task} but include_difficulty=True.")
            cols_to_keep.add('difficulty')

        cols_to_remove = [col for col in dataset.column_names if col not in cols_to_keep]

        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, task, tokenizer),
            batched=True,
            remove_columns=cols_to_remove,
            desc="Running tokenizer"
        )

        # Define the final order of columns for TensorDataset
        # *** CRITICAL: Ensure this order matches the unpacking in get_epoch_training_data ***
        # *** Using PLURAL 'labels' here to be consistent with what get_epoch returns ***
        final_columns_ordered = tokenized_cols + ['labels'] # Use PLURAL 'labels'
        if include_difficulty:
            final_columns_ordered.append('difficulty')

        print(f"Setting format to torch. Columns: {final_columns_ordered}")
        # Rename 'label' column to 'labels' before setting format IF NEEDED
        if 'label' in tokenized_dataset.column_names and 'labels' not in tokenized_dataset.column_names:
             print("Renaming 'label' column to 'labels' for consistency.")
             tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        elif 'label' in tokenized_dataset.column_names and 'labels' in tokenized_dataset.column_names:
             print("Warning: Both 'label' and 'labels' columns exist. Removing 'label'.")
             tokenized_dataset = tokenized_dataset.remove_columns(['label'])


        tokenized_dataset.set_format(type='torch', columns=final_columns_ordered)

        # Extract tensors in the specified order
        tensors_to_extract = [tokenized_dataset[col] for col in final_columns_ordered]

        return TensorDataset(*tensors_to_extract), final_columns_ordered # Return column order for clarity

    except Exception as e: print(f"ERROR tokenizing/formatting: {e}"); traceback.print_exc(); raise

# --- evaluate_and_estimate function ---
# ... (Keep the evaluate_and_estimate function as it was - it correctly uses column_order) ...
accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
def evaluate_and_estimate(model, dataloader, device, column_order, num_obs_theta=-1, mode='eval'):
    """Evaluates model, calculates loss/accuracy, optionally estimates theta."""
    val_loss = 0.0; metric = accuracy_metric; preds_list, labels_list, difficulties_list = [], [], []
    model.eval(); num_batches = 0
    eval_desc = "Validation Eval" if mode != 'estimate' else "Theta Estimation Eval"
    amp_eval_dtype = torch.bfloat16 if bf16_ready else torch.float16 # Use global bf16_ready

    # Determine indices from column_order
    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        token_type_ids_idx = column_order.index('token_type_ids') if 'token_type_ids' in column_order else -1
        # *** Use PLURAL 'labels' here to match column_order ***
        label_idx = column_order.index('labels')
        difficulty_idx = column_order.index('difficulty') if 'difficulty' in column_order else -1
    except ValueError as e:
        print(f"FATAL ERROR: Column missing in expected order {column_order}. Error: {e}")
        raise

    # Print structure once if debugging needed
    # print(f"Eval Batch Structure Indices: ids={input_ids_idx}, mask={attention_mask_idx}, type_ids={token_type_ids_idx}, label={label_idx}, diff={difficulty_idx}")

    for batch in tqdm(dataloader, desc=eval_desc, leave=False):
        num_batches += 1
        try: # Unpack batch using determined indices
            input_ids = batch[input_ids_idx].to(device, non_blocking=True)
            attention_mask = batch[attention_mask_idx].to(device, non_blocking=True)
            token_type_ids = batch[token_type_ids_idx].to(device, non_blocking=True) if token_type_ids_idx != -1 else None
            labels = batch[label_idx].to(device, non_blocking=True) # This uses the correct index for 'labels'

            difficulty_tensor = None
            should_have_difficulty = (mode == 'estimate' or mode == 'eval_estimate')
            if should_have_difficulty and difficulty_idx != -1:
                 difficulty_tensor = batch[difficulty_idx] # Keep on CPU until needed
            elif should_have_difficulty and difficulty_idx == -1:
                 print(f"Warning: Mode requires difficulty, but it's missing from batch structure (idx={difficulty_idx}).")

        except IndexError as e: print(f"ERROR: IndexError unpacking eval batch (len={len(batch)}). Expected indices based on {column_order}. Error: {e}"); continue
        except Exception as e: print(f"ERROR: Unexpected error unpacking eval batch: {e}"); continue

        with torch.no_grad(), autocast(device_type='cuda', dtype=amp_eval_dtype, enabled=torch.cuda.is_available()):
            try: # Model forward pass - model expects 'labels' argument name
                model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                if token_type_ids is not None:
                    model_kwargs['token_type_ids'] = token_type_ids
                outputs = model(**model_kwargs)
                logits = outputs.logits
                loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device) # Handle cases where loss isn't returned
            except Exception as e: print(f"\nERROR eval forward: {e}"); print(f"Shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}"); continue # Add token_type_ids shape if relevant

        preds_list.append(logits.detach().float().cpu().numpy()) # Store logits for potential analysis
        labels_list.append(labels.detach().cpu().numpy()) # Use the variable 'labels'
        if difficulty_tensor is not None and should_have_difficulty:
            difficulties_list.append(difficulty_tensor.cpu().numpy()) # Append NumPy array

        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions.detach().cpu(), references=labels.detach().cpu()) # Use the variable 'labels'
        val_loss += loss.item()

    # --- Aggregation after loop ---
    preds_np = np.concatenate(preds_list) if preds_list else np.array([])
    out_label_ids_np = np.concatenate(labels_list) if labels_list else np.array([])
    all_difficulties_np = np.concatenate(difficulties_list) if difficulties_list else None # Already numpy

    # Final loss calculation
    val_loss /= num_batches if num_batches > 0 else 1

    # Final accuracy calculation
    try:
        eval_score = metric.compute() if num_batches > 0 else None
        validation_accuracy = eval_score['accuracy'] if eval_score else 0.0
    except Exception as e:
        print(f"Warning: metric compute failed: {e}. Accuracy set to 0.0.")
        validation_accuracy = 0.0

    # --- Return based on mode ---
    if mode == 'eval':
        return validation_accuracy, val_loss

    # Check if theta estimation is possible
    can_estimate_theta = (all_difficulties_np is not None and len(all_difficulties_np) > 0 and (mode == 'estimate' or mode == 'eval_estimate'))

    if not can_estimate_theta:
         if mode == 'estimate' or mode == 'eval_estimate': print("Warning: Cannot estimate theta - difficulties missing or empty.")
         default_theta, default_time = 0.0, 0.0
         return (default_theta, default_time) if mode == 'estimate' else (validation_accuracy, val_loss, default_theta, default_time)

    # Estimate Theta
    time_model_s = time.time()
    if len(all_difficulties_np) != len(out_label_ids_np):
        print(f"ERROR: Mismatch between difficulty count ({len(all_difficulties_np)}) and response count ({len(out_label_ids_np)}). Cannot estimate theta.");
        default_theta, default_time = 0.0, 0.0
        return (default_theta, default_time) if mode == 'estimate' else (validation_accuracy, val_loss, default_theta, default_time)
    else:
        # Calculate response pattern (1 for correct, -1 for incorrect)
        response_pattern = [1 if pred_label == true_label else -1 for pred_label, true_label in zip(np.argmax(preds_np, axis=1), out_label_ids_np)]
        try:
            # Pass difficulties and responses to IRT scoring function
            # Ensure num_obs is handled correctly (-1 means use all data)
            effective_num_obs = num_obs_theta if num_obs_theta > 0 else len(response_pattern)
            theta_hat = calculate_theta(all_difficulties_np, response_pattern, num_obs=effective_num_obs)[0] # Assuming calculate_theta returns a tuple/list
        except Exception as e:
            print(f"ERROR during theta calculation: {e}. Setting theta to 0.0."); traceback.print_exc(); theta_hat = 0.0
        time_model_e = time.time(); model_capacity_time = time_model_e - time_model_s
        # Return appropriate tuple based on mode
        return (theta_hat, model_capacity_time) if mode == 'estimate' else (validation_accuracy, val_loss, theta_hat, model_capacity_time)

    # Fallback if mode is unrecognized (shouldn't happen with current logic)
    print(f"Warning: Unrecognized evaluation mode '{mode}'. Returning accuracy and loss only.");
    return validation_accuracy, val_loss

# --- Main Training Function ---
def train(config, output_dir):
    """Main training loop integrating IRT curriculum."""
    print(f"\n===== Starting Training for Task: {config.task} =====");
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu >= 0 else 'cpu')
    print(f"Using device: {device}");
    use_amp = torch.cuda.is_available(); print(f"Using AMP: {use_amp} with dtype: {amp_dtype}") # Use global amp_dtype

    # --- Data Loading ---
    try:
        train_dataset_hf, dev_dataset_hf, test_dataset_hf = load_and_prepare_data(config.task, config.diff_dir, config.cache_dir)
        train_size, val_size, test_size = len(train_dataset_hf), len(dev_dataset_hf), len(test_dataset_hf)
        print(f"Data sizes: Train={train_size}, Val={val_size}, Test={test_size}"); assert train_size > 0 and val_size > 0
    except Exception as e: print(f"FATAL: Data loading failed: {e}"); traceback.print_exc(); return 0.0, 0.0

    # --- Model/Tokenizer Loading ---
    # ... (Keep model/tokenizer loading and padding token setup as before) ...
    num_labels = 3 if config.task.startswith("mnli") else 1 if config.task == "stsb" else 2; model_name = config.model_name
    print(f"Loading: {model_name} (Labels: {num_labels})");
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir, use_fast=True, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=config.cache_dir, trust_remote_code=True)

        # *** Explicit Padding Token Setup ***
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                print("Warning: pad_token is None. Setting pad_token = eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
                if getattr(model.config, 'pad_token_id', None) is None:
                     print(f"Setting model.config.pad_token_id to tokenizer's eos_token_id: {tokenizer.eos_token_id}")
                     model.config.pad_token_id = tokenizer.eos_token_id
                else:
                     print(f"Model config already has pad_token_id: {model.config.pad_token_id}. Not overriding with eos_token.")

            else:
                print("Warning: Both pad_token and eos_token are None. Adding a new [PAD] token.")
                pad_token_str = '[PAD]'
                added_tokens = tokenizer.add_special_tokens({'pad_token': pad_token_str})
                print(f"Added {added_tokens} new pad token: '{pad_token_str}' with id: {tokenizer.pad_token_id}")
                if added_tokens > 0:
                    print("Resizing model token embeddings to accommodate new pad token...")
                    model.resize_token_embeddings(len(tokenizer))
                if getattr(model.config, 'pad_token_id', None) is None:
                     print(f"Setting model.config.pad_token_id to newly added pad_token_id: {tokenizer.pad_token_id}")
                     model.config.pad_token_id = tokenizer.pad_token_id
                elif model.config.pad_token_id != tokenizer.pad_token_id:
                     print(f"Warning: Model config had pad_token_id {model.config.pad_token_id}, but tokenizer now has {tokenizer.pad_token_id}. Sticking with tokenizer's ID.")
                     model.config.pad_token_id = tokenizer.pad_token_id

        if getattr(model.config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
             print(f"Setting model.config.pad_token_id to tokenizer.pad_token_id ({tokenizer.pad_token_id}) as a final check.")
             model.config.pad_token_id = tokenizer.pad_token_id
        elif getattr(model.config, 'pad_token_id', None) is None and tokenizer.pad_token_id is None:
             raise ValueError("Critical Error: Could not determine or set a pad_token_id for the model/tokenizer.")

        print(f"Final Tokenizer - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
        print(f"Final Model config - pad_token_id: {model.config.pad_token_id}")

    except Exception as e: print(f"FATAL: Loading model/tokenizer failed: {e}"); traceback.print_exc(); return 0.0, 0.0


    model.to(device)
    # Memory saving techniques
    if config.use_gradient_checkpointing:
        print("Enabling gradient checkpointing.")
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if hasattr(model.config, "sliding_window") and model.config.sliding_window is not None:
        print(f"Note: Disabling model.config.sliding_window (was {model.config.sliding_window}).")
        model.config.sliding_window = None

    # Torch Compile (Optional)
    if hasattr(torch, 'compile') and torch.cuda.is_available() and config.use_torch_compile:
        try: print("Attempting torch.compile(model)..."); model = torch.compile(model); print("Model compiled successfully.")
        except Exception as e: print(f"Torch compile failed: {e}. Continuing without compilation.")
    elif config.use_torch_compile: print("Warning: Compile requested but torch.compile is unavailable/disabled.")

    # --- Output Directories ---
    task_output_dir = os.path.join(output_dir, config.task); best_model_dir = os.path.join(task_output_dir, "best_model")
    os.makedirs(task_output_dir, exist_ok=True); os.makedirs(best_model_dir, exist_ok=True)

    # --- Dataset Creation & Dataloaders ---
    print("Creating/tokenizing datasets...");
    try:
        # Get column orders during creation - create_dataset now uses 'labels'
        train_dataset, train_col_order = create_dataset(train_dataset_hf, config.task, tokenizer, include_difficulty=True)
        dev_dataset, dev_col_order = create_dataset(dev_dataset_hf, config.task, tokenizer, include_difficulty=True) # Dev needs difficulty
        test_dataset, test_col_order = create_dataset(test_dataset_hf, config.task, tokenizer, include_difficulty=False) # Test doesn't need difficulty
        print(f"Train column order: {train_col_order}") # Should show 'labels'
        print(f"Dev column order: {dev_col_order}")   # Should show 'labels'
        print(f"Test column order: {test_col_order}")  # Should show 'labels'
    except Exception as e: print(f"FATAL: Dataset creation failed: {e}"); traceback.print_exc(); return 0.0, 0.0

    print("Creating dataloaders...");
    # Use DataLoader kwargs for potential speedups
    dl_kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    base_train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False, **dl_kwargs)
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, **dl_kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **dl_kwargs) if len(test_dataset) > 0 else None

    # --- Optimizer & Scheduler ---
    print("Setting up Adafactor optimizer.")
    optimizer = Adafactor( model.parameters(), lr=config.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=config.weight_decay )
    num_update_steps_per_epoch_base = ceil(len(base_train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps_estimate = num_update_steps_per_epoch_base * config.num_epochs
    num_warmup_steps = max(1, int(config.warmup_ratio * num_training_steps_estimate))
    print(f"Scheduler: Est Total Steps={num_training_steps_estimate}, Warmup Steps={num_warmup_steps} ({config.warmup_ratio*100}%)")
    scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps_estimate )
    scaler = GradScaler(enabled=use_amp);

    # --- Training Loop Initialization ---
    best_accuracy = 0.0; early_stop_count = 0; patience = config.early_stopping_patience; training_stats = []; total_pudf_overhead_time = 0.0;
    prev_cap = -5.0; cur_cap = 0.0
    print(f"\nStarting training loop: Max {config.num_epochs} epochs, Patience {patience}..."); start_train_loop_time = time.time(); actual_epochs = 0

    # --- Epoch Loop ---
    for epoch in range(config.num_epochs):
        actual_epochs = epoch + 1; print(f"\n======== Epoch {epoch + 1} / {config.num_epochs} ========"); epoch_start_time = time.time()
        epoch_total_loss = 0.0; model_capacity_time_est = 0.0; filter_time = 0.0; estimated_theta_hat = None
        num_optimizer_steps = 0

        # --- Theta Estimation (if strategy requires it) ---
        # ... (Keep theta estimation logic as before - uses dev_dataloader and dev_col_order) ...
        if config.strategy == 'theta':
             print("Estimating model capacity (theta) using dev set...");
             try:
                 estimated_theta_hat, model_capacity_time_est = evaluate_and_estimate(model, dev_dataloader, device, dev_col_order, num_obs_theta=config.num_obs, mode='estimate')
                 total_pudf_overhead_time += model_capacity_time_est; print(f"Theta estimated: {estimated_theta_hat:.4f} ({model_capacity_time_est:.2f}s)")
                 if estimated_theta_hat > prev_cap:
                      cur_cap = estimated_theta_hat
                      print(f"Theta improved. Setting cur_cap = {cur_cap:.4f}")
                 else:
                      cur_cap += 0.1 # Adjust this increment as needed
                      print(f"Theta ({estimated_theta_hat:.4f}) <= prev ({prev_cap:.4f}). Adjusted cur_cap: {cur_cap:.4f}")
             except Exception as e: print(f"Warning: Capacity estimation failed: {e}. Using previous cur_cap={cur_cap:.4f}"); traceback.print_exc()
        else:
            cur_cap = np.inf
            print(f"Strategy is not 'theta', setting cur_cap to {cur_cap} (effectively no capacity filtering)")

        # --- Data Filtering / Selection ---
        print(f"Filtering training data (Capacity: {cur_cap:.4f})...");
        filter_time_s = time.time()
        filtered_training_data_dict = {} # Initialize empty dict
        try:
            # --- Debug Prints Around Call ---
            print(f"\n[train Debug] Calling get_epoch_training_data for Epoch {epoch+1}.")
            print(f"[train Debug]   Input TensorDataset tensor count: {len(train_dataset.tensors)}")
            if len(train_dataset.tensors) >= 3: # Check if at least 3 tensors exist
                 # Ensure using the correct index based on train_col_order
                 labels_idx_in_train_ds = train_col_order.index('labels')
                 print(f"[train Debug]   Input labels tensor (index {labels_idx_in_train_ds}) shape: {train_dataset.tensors[labels_idx_in_train_ds].shape}")
            else:
                 print(f"[train Debug]   WARNING: Input TensorDataset has only {len(train_dataset.tensors)} tensors! Expected structure might be wrong.")

            # Call the function
            filtered_training_data_dict = get_epoch_training_data(
                 train_dataset, config, epoch, config.task, cur_cap,
                 diffs_sorted_idx=None,
                 lower_offset=config.lower_bound, upper_offset=config.upper_bound
            )

            print(f"[train Debug] Returned dict keys from get_epoch: {filtered_training_data_dict.keys()}")
            # --- CORRECTED DEBUG CHECK using PLURAL 'labels' ---
            if not filtered_training_data_dict: print("[train Debug] WARNING: Returned dictionary is empty!")
            elif 'labels' not in filtered_training_data_dict: print("[train Debug] WARNING: Returned dictionary is MISSING 'labels' key!")
            elif isinstance(filtered_training_data_dict.get('labels'), torch.Tensor): print(f"[train Debug] Returned 'labels' tensor shape: {filtered_training_data_dict['labels'].shape}")

        except Exception as e: print(f"ERROR: get_epoch_training_data failed: {e}. Skipping epoch."); traceback.print_exc(); continue
        filter_time_e = time.time(); filter_time = filter_time_e - filter_time_s; total_pudf_overhead_time += filter_time

        # --- Create Epoch Dataloader ---
        # --- CORRECTED Check using PLURAL 'labels' ---
        epoch_labels = filtered_training_data_dict.get('labels', []) # Use PLURAL 'labels'
        if not filtered_training_data_dict or 'labels' not in filtered_training_data_dict or len(epoch_labels) == 0:
             if filtered_training_data_dict and 'labels' not in filtered_training_data_dict: # Use PLURAL 'labels'
                 print(f"[train Debug] Error condition check: 'labels' key missing in non-empty dict. Keys: {filtered_training_data_dict.keys()}")
             print("Warning: No data selected after filtering OR 'labels' key missing/empty. Skipping training phase for this epoch.");
             num_epoch_examples = 0; avg_train_loss = 0.0
             # No optimizer steps this epoch if no data
        else:
            # This block should now execute correctly
            num_epoch_examples = len(filtered_training_data_dict['labels']); print(f"Selected {num_epoch_examples} examples for training ({filter_time:.2f}s filtering).")
            try:
                 # Expects train_col_order to contain 'labels'
                 tensors_for_epoch = [filtered_training_data_dict[col_name] for col_name in train_col_order]
                 train_dataset_epoch = TensorDataset(*tensors_for_epoch)
                 train_dataloader_epoch = DataLoader(train_dataset_epoch, shuffle=True, batch_size=train_batch_size, **dl_kwargs)
                 print(f"Created epoch dataloader with {len(train_dataloader_epoch)} batches.")
            except KeyError as e: print(f"ERROR creating epoch dataloader: Key {e} not found in filtered data dict. Check get_epoch_training_data output. Skipping."); traceback.print_exc(); continue
            except Exception as e: print(f"ERROR creating epoch dataloader: {e}. Skipping."); traceback.print_exc(); continue

            # --- Training Phase ---
            print(f"Training epoch {epoch + 1}..."); model.train(); optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_dataloader_epoch, desc=f"Epoch {epoch+1} Training", leave=False)
            num_steps_in_epoch = len(pbar)
            epoch_step_losses = []

            for step, batch in enumerate(pbar):
                 try: # Unpack batch using train_col_order indices (which includes 'labels')
                      input_ids = batch[train_col_order.index('input_ids')].to(device, non_blocking=True)
                      attention_mask = batch[train_col_order.index('attention_mask')].to(device, non_blocking=True)
                      token_type_ids = batch[train_col_order.index('token_type_ids')].to(device, non_blocking=True) if 'token_type_ids' in train_col_order else None
                      labels = batch[train_col_order.index('labels')].to(device, non_blocking=True) # Get 'labels' tensor
                 except IndexError as e: print(f"ERROR: IndexError unpacking train batch (step {step}, len={len(batch)}). Expected indices based on {train_col_order}. Error: {e}"); continue
                 except Exception as e: print(f"ERROR: Unexpected error unpacking train batch (step {step}): {e}"); continue

                 try:
                     with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                         # Prepare model inputs - model expects 'labels' kwarg
                         model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                         if token_type_ids is not None:
                             model_kwargs['token_type_ids'] = token_type_ids
                         outputs = model(**model_kwargs)
                         loss = outputs.loss

                     # ... (Rest of training step: loss check, backward, scaler, optimizer step) ...
                     if loss is None: continue
                     if torch.isnan(loss): optimizer.zero_grad(set_to_none=True); continue
                     loss = loss / config.gradient_accumulation_steps
                     scaler.scale(loss).backward();
                     epoch_step_losses.append(loss.item() * config.gradient_accumulation_steps)
                     if ((step + 1) % config.gradient_accumulation_steps == 0) or ((step + 1) == num_steps_in_epoch):
                         scaler.unscale_(optimizer)
                         torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                         scaler.step(optimizer)
                         scaler.update()
                         scheduler.step()
                         optimizer.zero_grad(set_to_none=True)
                         num_optimizer_steps += 1
                         current_avg_loss = np.mean(epoch_step_losses) if epoch_step_losses else 0
                         pbar.set_postfix({'avg_loss': f'{current_avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

                 except Exception as e:
                      print(f"\nERROR during training step {step}: {e}")
                      traceback.print_exc()
                      optimizer.zero_grad(set_to_none=True)
                      print("Attempting to continue to next step...")
                      continue

            avg_train_loss = np.mean(epoch_step_losses) if num_optimizer_steps > 0 else 0.0
            print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f} ({num_optimizer_steps} optimizer steps)")

        # --- Validation Phase ---
        # ... (Validation logic remains the same - uses dev_col_order which should contain 'labels') ...
        print("Evaluating on validation set..."); model_capacity_time_eval = 0.0
        try:
            # Pass dev column order to evaluation function
            dev_acc, val_loss, epoch_theta_estimate, model_capacity_time_eval = evaluate_and_estimate(
                model, dev_dataloader, device, dev_col_order, num_obs_theta=config.num_obs, mode='eval_estimate'
            )
            total_pudf_overhead_time += model_capacity_time_eval;
            print(f"Epoch {epoch+1} Validation: Acc={dev_acc:.4f}, Loss={val_loss:.4f}, Theta={epoch_theta_estimate:.4f} ({model_capacity_time_eval:.2f}s eval)")

            # Update previous capacity if theta improved
            if config.strategy == 'theta':
                 if epoch_theta_estimate > prev_cap:
                      prev_cap = epoch_theta_estimate
                      print(f"Validation theta improved. Updated prev_cap for next epoch: {prev_cap:.4f}")
                 else:
                      print(f"Validation theta ({epoch_theta_estimate:.4f}) <= prev_cap ({prev_cap:.4f}).")

            # Store stats for this epoch
            training_stats.append({
                'epoch': epoch + 1, 'Train Loss': avg_train_loss, 'Val Loss': val_loss, 'Val Acc': dev_acc,
                'cur_cap': cur_cap if config.strategy == 'theta' else 'N/A', 'theta_est': epoch_theta_estimate,
                'eval_time': model_capacity_time_est + model_capacity_time_eval, 'filter_time': filter_time,
                'n_train_epoch': num_epoch_examples, 'optimizer_steps_epoch': num_optimizer_steps
                })

            # --- Checkpointing and Early Stopping (remains the same) ---
            if dev_acc > best_accuracy:
                print(f"Val acc improved ({best_accuracy:.4f} --> {dev_acc:.4f}). Saving best model to {best_model_dir}...");
                best_accuracy = dev_acc; early_stop_count = 0
                try:
                    model_to_save = getattr(model, '_orig_mod', model);
                    model_to_save.save_pretrained(best_model_dir);
                    tokenizer.save_pretrained(best_model_dir)
                    saved_files = os.listdir(best_model_dir);
                    weights_found = ("pytorch_model.bin" in saved_files or "model.safetensors" in saved_files or "model.safetensors.index.json" in saved_files)
                    if weights_found: print("Model save confirmed (found standard weights or sharded index).")
                    else: print(f"Warning: NO standard weights file OR sharded index found after save in {best_model_dir}. Files: {saved_files}")
                except Exception as e:
                    print(f"Warning: Error saving best model: {e}")
                    traceback.print_exc()
            else:
                early_stop_count += 1;
                print(f"Val acc ({dev_acc:.4f}) did not improve vs best ({best_accuracy:.4f}). Early stop count: {early_stop_count}/{patience}")
                if early_stop_count >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.");
                    break # Exit epoch loop

        except Exception as e: print(f"ERROR during validation epoch {epoch+1}: {e}"); traceback.print_exc(); print("Stopping training for this task due to validation error."); break # Stop task on validation error

        # --- Epoch End Cleanup ---
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        epoch_end_time = time.time(); print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f}s.")


    # --- End of Training Loop ---
    # ... (Save Training Stats) ...
    # ... (Final Test Evaluation - uses test_col_order which should contain 'labels') ...
    # ... (Save Final Summary) ...
    # ... (Explicit Cleanup) ...
    # --- End of Training Loop ---
    end_train_loop_time = time.time(); train_loop_duration = end_train_loop_time - start_train_loop_time
    print("\n--- Training Loop Finished ---"); print(f"Actual epochs run: {actual_epochs}, Total Time: {train_loop_duration:.2f}s, PUDF Overhead: {total_pudf_overhead_time:.2f}s, Best Val Acc: {best_accuracy:.4f}")

    # --- Save Training Stats ---
    model_checkpoint_name = config.model_name.replace('/', '_'); stats_filename_base = f"{model_checkpoint_name}_{config.task}"
    training_stats_filename = os.path.join(task_output_dir, f"training_stats_{stats_filename_base}.json")
    print(f"Saving training stats to: {training_stats_filename}");
    try:
        with open(training_stats_filename, "w") as f: json.dump(training_stats, f, indent=4)
    except Exception as e: print(f"Warning: Error saving training stats: {e}")

    # --- Final Test Evaluation ---
    print("\n--- Final Test Evaluation ---"); test_acc, test_loss, test_time_seconds = 0.0, 0.0, 0.0
    test_time_start = time.time()
    if test_dataloader is None:
        print("Test dataloader is None. Skipping final test evaluation."); test_acc, test_loss = -3.0, -3.0 # Use specific code for skipped test
    else:
        # *** MODIFIED CHECK: Look for standard weights OR sharded index file ***
        weights_file_bin = os.path.join(best_model_dir, "pytorch_model.bin")
        weights_file_safetensors = os.path.join(best_model_dir, "model.safetensors")
        weights_index_safetensors = os.path.join(best_model_dir, "model.safetensors.index.json") # Sharded index

        best_model_exists = os.path.isdir(best_model_dir) and \
                            (os.path.exists(weights_file_bin) or \
                             os.path.exists(weights_file_safetensors) or \
                             os.path.exists(weights_index_safetensors)) # Check index file too

        if best_model_exists:
            print(f"Loading best model from: {best_model_dir} for final test evaluation...");
            try:
                model_loaded = AutoModelForSequenceClassification.from_pretrained(best_model_dir, trust_remote_code=True).to(device);
                tokenizer_loaded = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True, trust_remote_code=True) # Load tokenizer too
                # Pass test column order (which includes 'labels')
                test_acc, test_loss = evaluate_and_estimate(model_loaded, test_dataloader, device, test_col_order, mode='eval') # Mode is 'eval' for test set
                print(f'Final Test Accuracy: {test_acc:.4f}, Final Test Loss: {test_loss:.4f}');
                del model_loaded, tokenizer_loaded # Clean up loaded model
            except Exception as e:
                print(f"ERROR during final test evaluation: {e}"); traceback.print_exc();
                test_acc, test_loss = -1.0, -1.0 # Use specific code for test error
        else:
            print(f"Best model weights not found (checked for bin, safetensors, and index.json) in {best_model_dir}. Cannot run final test evaluation.");
            test_acc, test_loss = -2.0, -2.0 # Use specific code for model not found

    test_time_end = time.time(); test_time_seconds = test_time_end - test_time_start
    print(f"Test evaluation phase took: {test_time_seconds:.2f} seconds")

    # --- Save Final Summary ---
    # ... (Saving logic as before, using convert_numpy) ...
    final_stats_filename = os.path.join(task_output_dir, f"final_stats_{stats_filename_base}_Acc_{test_acc:.4f}_TestTime_{test_time_seconds:.0f}s.json")
    print(f"Saving final summary results to: {final_stats_filename}");
    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        elif obj == -np.inf: return "-Infinity"
        return obj
    final_summary = {
        "task": config.task, "model": config.model_name, "strategy": config.strategy, "ordering": config.ordering,
        "lower_bound": convert_numpy(config.lower_bound), "upper_bound": convert_numpy(config.upper_bound),
        "min_train_length": config.min_train_length, "num_obs_theta": config.num_obs,
        "num_epochs_set": config.num_epochs, "num_epochs_run": actual_epochs,
        "best_validation_accuracy": best_accuracy, "final_test_accuracy": test_acc, "final_test_loss": test_loss,
        "total_training_loop_time_seconds": round(train_loop_duration, 2),
        "final_test_time_seconds": round(test_time_seconds, 2),
        "total_pudf_overhead_seconds": round(total_pudf_overhead_time, 2),
        "config": {k: convert_numpy(v) for k, v in config.__dict__.items()},
        "training_stats_summary": training_stats
    }
    try:
        with open(final_stats_filename, "w") as f: json.dump(final_summary, f, indent=4, default=convert_numpy)
    except Exception as e: print(f"Warning: Error saving final summary JSON: {e}")

    # --- Explicit Cleanup ---
    # ... (Cleanup logic as before) ...
    print("Cleaning up task resources...")
    try: del model, tokenizer, optimizer, scheduler, scaler
    except NameError: pass
    try: del train_dataset, dev_dataset, test_dataset, train_dataset_epoch
    except NameError: pass
    try: del base_train_dataloader, dev_dataloader, test_dataloader, train_dataloader_epoch
    except NameError: pass
    try: del train_dataset_hf, dev_dataset_hf, test_dataset_hf
    except NameError: pass
    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"===== Finished Task: {config.task} =====")
    return best_accuracy, test_acc # Return final metrics


# --- run() function ---
# ... (Keep run() function as it was) ...
def run():
    # --- Configuration Namespace ---
    config = types.SimpleNamespace()

    # --- Model and Data Paths ---
    config.model_name = "Qwen/Qwen2.5-7B"
    config.diff_dir = GLUE_DIFFICULTY_DIR # VERIFY PATH
    config.cache_dir = os.environ.get("TRANSFORMERS_CACHE", "./hf_cache/models")

    # --- Training Hyperparameters ---
    config.num_epochs = 20
    config.learning_rate = 1e-5
    config.weight_decay = 0.01
    config.warmup_ratio = 0.06
    config.max_grad_norm = 1.0

    # --- Batching and Hardware ---
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.gpu = 0
    config.num_workers = 4

    # --- PUDF Strategy Parameters ---
    config.strategy = 'theta'
    config.ordering = 'easiest'
    config.num_obs = 1000
    config.min_train_length = 1000
    config.lower_bound = -np.inf
    config.upper_bound = 0.0
    config.balanced = False

    # --- Misc / Compatibility ---
    config.use_length = False
    config.use_word_rarity = False
    config.use_torch_compile = False
    config.use_gradient_checkpointing = True
    config.early_stopping_patience = 5

    # --- End Configuration ---

    # --- Run Setup ---
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_short_name = config.model_name.split('/')[-1]
    output_dir_base = f"./glue_PUDF_{model_short_name}_{config.strategy}_{run_timestamp}"
    print(f"Base output directory for this run: {output_dir_base}")
    os.makedirs(output_dir_base, exist_ok=True)

    # --- Task Loop ---
    results = {}
    current_glue_tasks = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp']
    for task in tqdm(current_glue_tasks, desc="Overall Task Progress"):
        config.task = task; print(f"\n\n>>>>>>>> Starting Task: {config.task} <<<<<<<<")
        print(f"--- Task Configuration ({config.task}) ---");
        for key, value in config.__dict__.items():
            val_str = "-Infinity" if value == -np.inf else str(value)
            print(f"  {key}: {val_str}")
        print(f"-------------------------")

        task_start_time = time.time()
        try:
            top_dev, test_acc = train(config, output_dir_base)
            results[task] = {"best_dev_acc": top_dev, "test_acc": test_acc}
        except Exception as e:
            print(f"\n!FATAL ERROR during Task {task} execution!");
            traceback.print_exc();
            results[task] = {"best_dev_acc": "FATAL_ERROR", "test_acc": "FATAL_ERROR"}
        task_end_time = time.time(); print(f">>>>>>>> Finished Task: {config.task} in {task_end_time - task_start_time:.2f}s <<<<<<<<")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Print Summary Results ---
    # ... (Summary printing as before) ...
    print("\n\n===== Run Summary ====="); print(f"Model: {config.model_name}, Strategy: {config.strategy}"); print(f"Output Dir: {output_dir_base}")
    print("Task Results (Best Dev Acc / Test Acc):")
    all_tasks_successful = True
    for task in sorted(results.keys()):
        res = results[task];
        dev_res = f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'], float) else str(res['best_dev_acc'])
        test_res = f"{res['test_acc']:.4f}" if isinstance(res['test_acc'], float) else str(res['test_acc'])
        print(f"  - {task}: {dev_res} / {test_res}")
        if isinstance(res['test_acc'], str) or res['test_acc'] < -0.5:
             all_tasks_successful = False
    print("=====================")
    if not all_tasks_successful: print("WARNING: One or more tasks failed to produce a final test accuracy.")

    # --- Save Overall Summary ---
    # ... (Summary saving as before) ...
    summary_file = os.path.join(output_dir_base, "run_summary.json")
    try:
        def convert_numpy(obj): # Re-define helper here just in case
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, np.bool_): return bool(obj)
            elif obj == -np.inf: return "-Infinity"
            return obj
        with open(summary_file, "w") as f: json.dump(results, f, indent=4, default=convert_numpy)
        print(f"Overall run summary saved: {summary_file}")
    except Exception as e: print(f"Warning: Failed to save overall run summary: {e}")


if __name__ == '__main__':
    # Ensure the custom modules are importable
    # import sys
    # sys.path.append('/path/to/your/modules') # Example
    run()