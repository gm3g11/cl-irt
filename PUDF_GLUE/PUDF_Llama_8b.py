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
# *** Use Adafactor Optimizer ***
from transformers.optimization import Adafactor
# Mixed Precision
from torch.cuda.amp import GradScaler # Standard import style
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset
import evaluate
# Assuming these custom modules are in the same directory or Python path
from build_features_Llama import get_epoch_training_data # Use the Llama-specific version
from irt_scoring import calculate_theta
import types
import gc
import traceback
from tqdm.auto import tqdm
from huggingface_hub import login, whoami
from math import ceil

# --- Environment and Cache Setup ---
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Hugging Face Auth ---
if "HF_TOKEN" in os.environ: del os.environ["HF_TOKEN"]; print("Removed HF_TOKEN env var.")
try: user_info = whoami(); print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
except Exception as e: print(f"HF Login Check Failed: {e}. Ensure login.")

# --- Global Random Seed ---
random_seed = 63
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

# --- Task Definitions ---
GLUETASKS = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp']
task_max_lengths = {"mrpc": 72, "rte": 150, "mnli": 72, "qqp": 56, "sst2": 32, "qnli": 80}

# --- Global Training Configuration ---
# *** MODIFIED BATCH SIZES - Trying BS=16 ***
train_batch_size = 256 # Increased per-device batch size. Try 64 if this works.
effective_batch_size = 256 # Keep target effective batch size (adjust if desired)
gradient_accumulation_steps = max(1, effective_batch_size // train_batch_size) # = 64 // 32 = 2
eval_batch_size = 256 # Increased eval/test batch size
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
    print(f"Loading dataset for task: {task}")
    try: raw_datasets = load_dataset('glue', task, cache_dir=cache_dir)
    except Exception as e: print(f"ERROR loading dataset '{task}': {e}"); raise
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
    test_split_name = 'validation_matched' if task == 'mnli' else 'validation'
    if test_split_name not in raw_datasets: print(f"Warning: Test split '{test_split_name}' not found for {task}. Using 'validation'."); test_split_name = 'validation'
    test_dataset = raw_datasets[test_split_name]; print("Data loading complete.")
    return train_dataset, val_dataset, test_dataset

def tokenize_function(examples, task, tokenizer):
    max_length = task_max_lengths.get(task);
    if max_length is None: print(f"Warning: max_length not defined for {task}. Using 128."); max_length = 128
    if task == "mnli": text_a = examples["premise"]; text_b = examples["hypothesis"]
    elif task in ["mrpc", "rte"]: text_a = examples["sentence1"]; text_b = examples["sentence2"]
    elif task == "qnli": text_a = examples["question"]; text_b = examples["sentence"]
    elif task == "qqp": text_a = examples["question1"]; text_b = examples["question2"]
    elif task == "sst2": text_a = examples["sentence"]; text_b = None
    else: raise ValueError(f"Task {task} not supported.")
    return tokenizer(text=text_a, text_pair=text_b, padding="max_length", truncation=True, max_length=max_length)

def create_dataset(dataset, task, tokenizer, include_difficulty=True):
    print(f"Tokenizing dataset (include_difficulty={include_difficulty})...")
    try:
        # Llama models typically don't use token_type_ids
        tokenized_cols = ['input_ids', 'attention_mask']
        cols_to_remove = [col for col in dataset.column_names if col not in ['label', 'difficulty']]
        tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, task, tokenizer), batched=True, remove_columns=cols_to_remove, desc="Running tokenizer")

        final_columns = tokenized_cols + ['label']
        if include_difficulty:
            if 'difficulty' not in tokenized_dataset.column_names:
                 # If difficulty is somehow missing after map (shouldn't happen if in cols_to_remove logic is right)
                 if 'difficulty' in dataset.column_names:
                     print("Re-adding difficulty column after tokenization map.")
                     tokenized_dataset = tokenized_dataset.add_column('difficulty', dataset['difficulty'])
                 else:
                    raise RuntimeError("Difficulty column missing and not found in original dataset.")
            final_columns.append('difficulty')

        # Rename 'label' to 'labels' if needed for consistency (though TensorDataset doesn't care)
        # We will rely on positional access later, so renaming isn't strictly needed here.
        # if 'label' in tokenized_dataset.column_names:
        #     tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        #     if 'labels' not in final_columns: final_columns[final_columns.index('label')] = 'labels'

        tokenized_dataset.set_format(type='torch', columns=final_columns)
        tensors_to_extract = [tokenized_dataset[col] for col in final_columns]
        print(f"Final tensor order for TensorDataset: {final_columns}") # Debug print
        return TensorDataset(*tensors_to_extract)
    except Exception as e: print(f"ERROR tokenizing/formatting: {e}"); traceback.print_exc(); raise

accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
def evaluate_and_estimate(model, dataloader, device, num_obs_theta=-1, mode='eval'):
    val_loss = 0.0; metric = accuracy_metric; preds_list, labels_list, difficulties_list = [], [], []
    model.eval(); num_batches = 0
    eval_desc = "Validation Eval" if mode != 'estimate' else "Theta Estimation Eval"
    is_first_batch = True; # No need for has_token_type_ids check for Llama
    amp_eval_dtype = torch.bfloat16 if bf16_ready else torch.float16

    for batch in tqdm(dataloader, desc=eval_desc, leave=False):
        num_batches += 1; batch_len = len(batch)
        if batch_len < 3: print(f"Warning: Skipping batch length {batch_len} (expected at least input_ids, attention_mask, labels)"); continue

        try: # Unpack batch (assuming order: input_ids, attention_mask, labels, [difficulty])
            input_ids=batch[0].to(device, non_blocking=True)
            attention_mask=batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True) # Expect label at index 2

            difficulty_tensor = None; difficulty_index = 3 # Expect difficulty at index 3
            should_have_difficulty = (mode == 'estimate' or mode == 'eval_estimate')

            if should_have_difficulty:
                if batch_len > difficulty_index:
                    difficulty_tensor = batch[difficulty_index]
                else:
                    print(f"Warning: Expected difficulty for mode '{mode}' but batch length is {batch_len}. Check dataset creation.")

        except IndexError as e: print(f"ERROR: IndexError unpacking eval batch (len={batch_len}). Error: {e}"); continue

        with torch.no_grad(), autocast(device_type='cuda', dtype=amp_eval_dtype, enabled=torch.cuda.is_available()):
            try: # Model forward - NO token_type_ids for Llama
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits; loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
            except Exception as e: print(f"\nERROR eval forward: {e}"); print(f"Shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}"); continue

        preds_list.append(logits.detach().float().cpu().numpy()); labels_list.append(labels.detach().cpu().numpy())
        if difficulty_tensor is not None and should_have_difficulty: difficulties_list.append(difficulty_tensor.cpu().numpy())

        predictions = torch.argmax(logits, dim=-1); metric.add_batch(predictions=predictions.detach().cpu(), references=labels.detach().cpu())
        val_loss += loss.item()

    preds = np.concatenate(preds_list) if preds_list else np.array([]); out_label_ids = np.concatenate(labels_list) if labels_list else np.array([])
    all_difficulties = np.concatenate(difficulties_list) if difficulties_list else None
    val_loss /= num_batches if num_batches > 0 else 1
    try: eval_score = metric.compute() if num_batches > 0 else None; validation_accuracy = eval_score['accuracy'] if eval_score else 0.0
    except Exception as e: print(f"Warning: metric compute failed: {e}. Acc=0."); validation_accuracy = 0.0

    if mode == 'eval': return validation_accuracy, val_loss

    if all_difficulties is None and (mode == 'estimate' or mode == 'eval_estimate'):
         print("Warning: Cannot estimate theta - difficulties missing."); default_theta, default_time = 0.0, 0.0
         return (default_theta, default_time) if mode == 'estimate' else (validation_accuracy, val_loss, default_theta, default_time)

    if mode == 'estimate' or mode == 'eval_estimate':
        time_model_s = time.time()
        if len(all_difficulties) != len(out_label_ids):
             print(f"ERROR: Mismatch length difficulties({len(all_difficulties)}) vs labels({len(out_label_ids)}). Cannot estimate theta.");
             default_theta, default_time = 0.0, 0.0
             return (default_theta, default_time) if mode == 'estimate' else (validation_accuracy, val_loss, default_theta, default_time)
        elif len(all_difficulties) == 0:
            print("Warning: No data for theta estimation."); default_theta, default_time = 0.0, 0.0
            return (default_theta, default_time) if mode == 'estimate' else (validation_accuracy, val_loss, default_theta, default_time)
        else:
            # Ensure responses are binary (1 for correct, -1 for incorrect)
            # Sometimes labels might be floats, ensure comparison works
            preds_indices = np.argmax(preds, axis=1)
            labels_indices = out_label_ids.astype(int) # Ensure labels are int for comparison
            rps = np.where(preds_indices == labels_indices, 1, -1)
            # rps = [1 if p == c else -1 for p, c in zip(np.argmax(preds, axis=1), out_label_ids)] # Original potentially fragile comparison
            try:
                theta_hat = calculate_theta(all_difficulties, rps, num_obs=num_obs_theta)[0]
            except Exception as e: print(f"ERROR theta calc: {e}. Theta=0."); traceback.print_exc(); theta_hat = 0.0
            time_model_e = time.time(); model_capacity_time = time_model_e - time_model_s
            return (theta_hat, model_capacity_time) if mode == 'estimate' else (validation_accuracy, val_loss, theta_hat, model_capacity_time)

    print(f"Warning: Unrecognized mode '{mode}'."); return validation_accuracy, val_loss

# --- Main Training Function ---
def train(config, output_dir):
    """Main training loop integrating IRT curriculum."""
    print(f"\n===== Starting Training for Task: {config.task} ====="); device = torch.device('cuda' if torch.cuda.is_available() and config.gpu >= 0 else 'cpu')
    print(f"Using device: {device}"); use_amp = torch.cuda.is_available() and bf16_ready # Use bf16 if supported
    print(f"Using AMP: {use_amp} with dtype: {amp_dtype}") # Print selected dtype

    try: # Data Loading
        train_dataset_hf, dev_dataset_hf, test_dataset_hf = load_and_prepare_data(config.task, config.diff_dir, config.cache_dir)
        train_size, val_size, test_size = len(train_dataset_hf), len(dev_dataset_hf), len(test_dataset_hf)
        print(f"Data sizes: Train={train_size}, Val={val_size}, Test={test_size}"); assert train_size > 0 and val_size > 0
    except Exception as e: print(f"FATAL: Data loading failed: {e}"); return 0.0, 0.0

    num_labels = 3 if config.task.startswith("mnli") else 1 if config.task == "stsb" else 2; model_name = config.model_name
    print(f"Loading: {model_name} (Labels: {num_labels})"); # Model/Tokenizer Loading
    try:
        # Use token=True instead of use_auth_token=True for newer Transformers versions
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir, use_fast=True, token=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=config.cache_dir, token=True)

        # --- Pad Token Handling ---
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                print("Setting pad_token = eos_token")
                tokenizer.pad_token = tokenizer.eos_token
                # Explicitly set model pad token id AFTER potentially modifying tokenizer
                model.config.pad_token_id = tokenizer.eos_token_id
            else:
                # Fallback: Add a new pad token if no EOS token exists
                print("Warning: No EOS token found. Adding a new pad token '[PAD]'")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize model embeddings
                model.resize_token_embeddings(len(tokenizer))
                # Set the model's pad token id to the new token's id
                if model.config.pad_token_id is None: # Avoid overwriting if already set
                    model.config.pad_token_id = tokenizer.pad_token_id
        else:
             # If pad token exists, ensure model config matches tokenizer
             print(f"Using existing pad_token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
             if model.config.pad_token_id != tokenizer.pad_token_id:
                  print(f"Warning: Model pad_token_id ({model.config.pad_token_id}) differs from Tokenizer ({tokenizer.pad_token_id}). Setting model config.")
                  model.config.pad_token_id = tokenizer.pad_token_id

        print(f"Final Tokenizer - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
        print(f"Final Model config - pad_token_id: {model.config.pad_token_id}")
        # --- End Pad Token Handling ---

    except Exception as e: print(f"FATAL: Loading model or tokenizer failed: {e}"); traceback.print_exc(); return 0.0, 0.0

    model.to(device)
    # Enable gradient checkpointing BEFORE potential compilation
    model.gradient_checkpointing_enable();
    model.config.use_cache = False # Required for gradient checkpointing

    if hasattr(torch, 'compile') and torch.cuda.is_available() and config.use_torch_compile: # torch.compile
        try: print("Attempting model compilation..."); model = torch.compile(model); print("Model compiled successfully.")
        except Exception as e: print(f"Torch compile failed: {e}. Continuing without compilation.")
    elif config.use_torch_compile: print("Compile requested but unavailable/disabled.")

    task_output_dir = os.path.join(output_dir, config.task); best_model_dir = os.path.join(task_output_dir, "best_model")
    os.makedirs(task_output_dir, exist_ok=True); os.makedirs(best_model_dir, exist_ok=True)

    print("Creating/tokenizing datasets..."); # Dataset Creation
    try:
        train_dataset = create_dataset(train_dataset_hf, config.task, tokenizer, include_difficulty=True)
        dev_dataset = create_dataset(dev_dataset_hf, config.task, tokenizer, include_difficulty=True)
        # Test dataset should not include difficulty for final evaluation format
        test_dataset = create_dataset(test_dataset_hf, config.task, tokenizer, include_difficulty=False)
        print(f"Dataset tensor counts: Train={len(train_dataset.tensors)}, Dev={len(dev_dataset.tensors)}, Test={len(test_dataset.tensors)}")
    except Exception as e: print(f"FATAL: Dataset creation failed: {e}"); return 0.0, 0.0

    print("Creating dataloaders..."); # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) if len(test_dataset) > 0 and len(test_dataset.tensors) > 0 else None


    # *** Use Adafactor Optimizer ***
    print("Setting up Adafactor optimizer.")
    optimizer = Adafactor(
            model.parameters(),
            lr=config.learning_rate, # Use configured LR
            scale_parameter=False,  # Recommended
            relative_step=False,    # Recommended
            warmup_init=False,      # Recommended
            weight_decay=0.01       # Optional weight decay
        )

    # Estimate steps based on FULL training dataloader initially for scheduler, curriculum loader changes per epoch
    num_update_steps_per_full_epoch = ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps_estimate = num_update_steps_per_full_epoch * config.num_epochs
    num_warmup_steps = max(1, int(0.06 * num_training_steps_estimate)); # Scheduler
    print(f"Scheduler: Est Total Steps={num_training_steps_estimate}, Warmup Steps={num_warmup_steps} ({num_warmup_steps/num_training_steps_estimate*100:.1f}%)")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps_estimate)

    # Use standard GradScaler init for selected dtype
    print(f"Initializing GradScaler (enabled={use_amp})...")
    # Correct GradScaler initialization for PyTorch 1.6+ AMP API
    # scaler = GradScaler(enabled=use_amp);
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)


    best_accuracy = 0.0; early_stop_count = 0; patience = 5; training_stats = []; total_pudf_overhead_time = 0.0; # Init
    prev_cap = -5.0; cur_cap = -5.0 # Initialize prev_cap low
    print(f"\nStarting training loop: Max {config.num_epochs} epochs, Patience {patience}..."); start_train_loop_time = time.time(); actual_epochs = 0

    # --- Epoch Loop ---
    for epoch in range(config.num_epochs):
        actual_epochs = epoch + 1; print(f"\n======== Epoch {epoch + 1} / {config.num_epochs} ========"); epoch_start_time = time.time()
        epoch_total_loss = 0.0; model_capacity_time_est = 0.0; filter_time = 0.0; estimated_theta_hat = None

        # --- Capacity Estimation (Theta Strategy) ---
        if config.strategy == 'theta':
             print("Estimating model capacity (theta) using dev set...");
             try:
                 # Use mode='estimate' which expects difficulty scores in dev_dataset
                 estimated_theta_hat, model_capacity_time_est = evaluate_and_estimate(model, dev_dataloader, device, num_obs_theta=config.num_obs, mode='estimate')
                 total_pudf_overhead_time += model_capacity_time_est; print(f"                                                                                                                                 Theta estimated: {estimated_theta_hat:.4f} ({model_capacity_time_est:.2f}s)")
                 if estimated_theta_hat > prev_cap:
                      cur_cap = estimated_theta_hat; print(f"Theta improved. Setting cur_cap = {cur_cap:.4f}")
                 else:
                      cur_cap += 0.1 # Simple heuristic: nudge capacity up if estimation didn't improve
                      print(f"Theta ({estimated_theta_hat:.4f}) <= prev_cap ({prev_cap:.4f}). Nudging cur_cap to: {cur_cap:.4f}")
             except Exception as e: print(f"Warning: Capacity estimation failed: {e}. Using previous cur_cap={cur_cap:.4f}"); traceback.print_exc()
        else: # Handle other strategies or default behavior if needed
             cur_cap = 0.0 # Example default if not theta strategy

        # --- Data Filtering/Selection ---
        print(f"Filtering training data (Capacity: {cur_cap:.4f})...");
        filter_time_s = time.time()
        try:
             # Ensure train_dataset is passed correctly (it's the TensorDataset)
             filtered_training_data = get_epoch_training_data(train_dataset, config, epoch, config.task, cur_cap, diffs_sorted_idx=None, lower_offset=config.lower_bound, upper_offset=config.upper_bound)
        except Exception as e: print(f"ERROR: get_epoch_training_data failed: {e}. Skipping epoch."); traceback.print_exc(); continue
        filter_time_e = time.time(); filter_time = filter_time_e - filter_time_s; total_pudf_overhead_time += filter_time

        # --- Create Epoch Dataloader ---
        if 'labels' not in filtered_training_data or len(filtered_training_data['labels']) == 0:
             print("Warning: No data selected for training this epoch. Skipping training phase."); num_epoch_examples = 0; avg_train_loss = 0.0
             # Need to decide if we should still validate if no training happened
        else:
            num_epoch_examples = len(filtered_training_data['labels']); print(f"Selected {num_epoch_examples} examples for training ({filter_time:.2f}s filtering).")
            try:
                 # filtered_training_data should be a dict of tensors: {'input_ids': ..., 'attention_mask': ..., 'labels': ..., 'difficulty': ...}
                 tensors_for_epoch = [
                     filtered_training_data['input_ids'],
                     filtered_training_data['attention_mask'],
                     filtered_training_data['labels'],
                     filtered_training_data['difficulty'] # Difficulty needed for potential future use, even if not directly in loss
                 ]
                 train_dataset_epoch = TensorDataset(*tensors_for_epoch)
                 train_dataloader_epoch = DataLoader(train_dataset_epoch, shuffle=True, batch_size=train_batch_size, num_workers=config.num_workers, pin_memory=True)
                 print(f"Created epoch dataloader with {len(train_dataloader_epoch)} batches.")
            except KeyError as e: print(f"ERROR creating epoch TensorDataset: Missing key {e}. Check get_epoch_training_data output."); continue
            except Exception as e: print(f"ERROR creating epoch dataloader: {e}. Skipping training phase."); traceback.print_exc(); continue

            # --- Training Phase ---
            print(f"Training epoch {epoch + 1}..."); model.train(); optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_dataloader_epoch, desc=f"Epoch {epoch+1} Training", leave=False)
            num_steps_in_epoch = len(pbar)
            num_optimizer_steps_this_epoch = 0

            for step, batch in enumerate(pbar):
                 # Batch order: input_ids, attention_mask, labels, difficulty
                 if len(batch) < 3: print(f"Warning: Skipping short batch (len={len(batch)}) in training."); continue
                 input_ids=batch[0].to(device, non_blocking=True)
                 attention_mask=batch[1].to(device, non_blocking=True)
                 labels = batch[2].to(device, non_blocking=True) # Labels at index 2

                 try:
                     with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels); # Llama expects no token_type_ids
                         loss = outputs.loss
                     if loss is None: raise ValueError("Model returned None loss") # More specific error
                     if torch.isnan(loss): print(f"Warning: NaN loss encountered at step {step}. Skipping batch."); optimizer.zero_grad(set_to_none=True); continue

                     loss = loss / config.gradient_accumulation_steps # Scale loss for accumulation
                     scaler.scale(loss).backward(); # Scaled backward pass
                     epoch_total_loss += loss.item() * config.gradient_accumulation_steps # Accumulate unscaled loss

                     # --- Optimizer Step ---
                     if ((step + 1) % config.gradient_accumulation_steps == 0) or ((step + 1) == num_steps_in_epoch):
                         # Only unscale and step after sufficient accumulation steps or at the end
                         scaler.unscale_(optimizer); # Unscale gradients before clipping
                         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
                         scaler.step(optimizer); # Optimizer step (checks inf/NaN)
                         scaler.update(); # Update scaler for next iteration
                         scheduler.step(); # Update learning rate
                         optimizer.zero_grad(set_to_none=True) # Clear gradients
                         num_optimizer_steps_this_epoch += 1

                         # Update progress bar
                         current_avg_loss = epoch_total_loss / num_optimizer_steps_this_epoch if num_optimizer_steps_this_epoch > 0 else 0
                         pbar.set_postfix({'avg_loss': f'{current_avg_loss:.4f}'})

                 except Exception as e:
                      print(f"\nERROR during training step {step}: {e}");
                      # print(f"Input Shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}")
                      optimizer.zero_grad(set_to_none=True); # Clear potentially corrupted grads
                      print("Skipping step and continuing epoch..."); continue # Try to continue epoch

            # Calculate average loss based on actual optimizer steps taken
            avg_train_loss = epoch_total_loss / num_optimizer_steps_this_epoch if num_optimizer_steps_this_epoch > 0 else 0.0
            print(f"                                                                                                                                 Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f} ({num_optimizer_steps_this_epoch} optimizer steps)")

        # --- Validation Phase ---
        print("Evaluating on validation set..."); model_capacity_time_eval = 0.0
        try:
            # Use mode='eval_estimate' to get accuracy, loss, and theta from validation run
            dev_acc, val_loss, epoch_theta_estimate, model_capacity_time_eval = evaluate_and_estimate(model, dev_dataloader, device, num_obs_theta=config.num_obs, mode='eval_estimate')
            total_pudf_overhead_time += model_capacity_time_eval; print(f"                                                                                                                                 Epoch {epoch+1} Validation: Acc={dev_acc:.4f}, Loss={val_loss:.4f}, Theta={epoch_theta_estimate:.4f} ({model_capacity_time_eval:.2f}s eval)")

            # Update previous capacity if theta estimate improved
            if epoch_theta_estimate > prev_cap:
                 prev_cap = epoch_theta_estimate; print(f"Validation theta improved. Updated prev_cap for next epoch: {prev_cap:.4f}")
            else: print(f"Theta ({epoch_theta_estimate:.4f}) <= prev_cap ({prev_cap:.4f}).")

            # Store stats
            training_stats.append({'epoch': epoch + 1, 'Train Loss': avg_train_loss, 'Val Loss': val_loss, 'Val Acc': dev_acc, 'cur_cap': cur_cap, 'theta_est': epoch_theta_estimate, 'eval_time': model_capacity_time_est + model_capacity_time_eval, 'filter_time': filter_time, 'n_train': num_epoch_examples})

            # --- Early Stopping & Model Saving ---
            if dev_acc > best_accuracy:
                print(f"Val acc improved ({best_accuracy:.4f} --> {dev_acc:.4f}). Saving best model to {best_model_dir}..."); best_accuracy = dev_acc; early_stop_count = 0
                try:
                    # Use recommended way to get the underlying model if compiled/wrapped
                    model_to_save = getattr(model, '_orig_mod', model);
                    model_to_save.save_pretrained(best_model_dir);
                    tokenizer.save_pretrained(best_model_dir)
                    # Save check (optional but good practice)
                    saved_files = os.listdir(best_model_dir);
                    if "config.json" in saved_files and \
                       (os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
                        os.path.exists(os.path.join(best_model_dir, "model.safetensors")) or \
                        os.path.exists(os.path.join(best_model_dir, "model.safetensors.index.json"))):
                         print("Model save confirmed (found standard weights or sharded index).")
                    else: print(f"Warning: Standard weights or index file not found after save in {best_model_dir}. Files: {saved_files}")
                except Exception as e: print(f"Warning: Error saving best model: {e}"); traceback.print_exc()
            else:
                early_stop_count += 1; print(f"Val acc ({dev_acc:.4f}) vs best ({best_accuracy:.4f}). Early stop count: {early_stop_count}/{patience}")
                if early_stop_count >= patience: print("Early stopping triggered."); break # Exit epoch loop

        except Exception as e: print(f"ERROR during validation for epoch {epoch+1}: {e}"); traceback.print_exc(); print("Stopping training for this task."); break # Exit epoch loop

        # --- Cleanup per Epoch ---
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        epoch_end_time = time.time(); print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f}s.")

    # --- End of Training Loop ---
    end_train_loop_time = time.time(); train_loop_duration = end_train_loop_time - start_train_loop_time
    print("\n--- Training Loop Finished ---"); print(f"Actual epochs run: {actual_epochs}, Total Time: {train_loop_duration:.2f}s, PUDF Overhead: {total_pudf_overhead_time:.2f}s, Best Val Acc: {best_accuracy:.4f}")

    # --- Save Training Stats ---
    model_checkpoint_name = config.model_name.replace('/', '_'); # Sanitize name
    stats_filename_base = f"{model_checkpoint_name}_{config.task}"
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
    # *** FIXED CHECK FOR SAVED MODEL ***
    elif os.path.isdir(best_model_dir) and \
         (os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
          os.path.exists(os.path.join(best_model_dir, "model.safetensors")) or \
          os.path.exists(os.path.join(best_model_dir, "model.safetensors.index.json"))): # Check for sharded index too
        print(f"Loading best model from: {best_model_dir}...");
        try:
            # Use token=True for consistency if needed by HF hub auth
            model_loaded = AutoModelForSequenceClassification.from_pretrained(best_model_dir, token=True).to(device);
            tokenizer_loaded = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True, token=True)
            print("Evaluating best model on test set...");
            # Use mode='eval' which doesn't expect/use difficulty scores
            test_acc, test_loss = evaluate_and_estimate(model_loaded, test_dataloader, device, mode='eval')
            print(f'Final Test Accuracy: {test_acc:.4f}, Final Test Loss: {test_loss:.4f}');
            del model_loaded, tokenizer_loaded # Cleanup loaded model
        except Exception as e:
            print(f"ERROR during final test evaluation: {e}"); traceback.print_exc(); test_acc, test_loss = -1.0, -1.0 # Error during eval
    else:
        print(f"Best model not found or incomplete in {best_model_dir}. Cannot run final test evaluation."); test_acc, test_loss = -2.0, -2.0 # Model not found code

    test_time_end = time.time(); test_time_seconds = test_time_end - test_time_start
    print(f"Test evaluation duration: {test_time_seconds:.2f} seconds")

    # --- Save Final Summary ---
    final_stats_filename = os.path.join(task_output_dir, f"final_stats_{stats_filename_base}_Acc_{test_acc:.4f}_TestTime_{test_time_seconds:.0f}s.json")
    print(f"Saving final summary stats to: {final_stats_filename}");
    final_summary = {
        "task": config.task, "model": config.model_name, "strategy": config.strategy, "ordering": config.ordering,
        "lower_bound": str(config.lower_bound), "upper_bound": config.upper_bound, # Stringify -inf
        "min_train_length": config.min_train_length,
        "num_obs_theta": config.num_obs, "num_epochs_set": config.num_epochs, "num_epochs_run": actual_epochs,
        "best_validation_accuracy": best_accuracy, "final_test_accuracy": test_acc, "final_test_loss": test_loss,
        "total_training_loop_time_seconds": train_loop_duration,
        "final_test_time_seconds": test_time_seconds,
        "total_pudf_overhead_seconds": total_pudf_overhead_time,
        "config": {k: (str(v) if isinstance(v, float) and v == -np.inf else v) for k, v in config.__dict__.items()}, # Stringify -inf in config too
        "training_stats_summary": training_stats }
    try:
        with open(final_stats_filename, "w") as f: json.dump(final_summary, f, indent=4, default=str) # Use default=str for safety
    except Exception as e: print(f"Warning: Error saving final summary JSON: {e}")

    # --- Explicit Cleanup ---
    print("Cleaning up task resources...")
    try: del model, tokenizer, optimizer, scheduler, scaler
    except NameError: pass
    try: del train_dataset, dev_dataset, test_dataset
    except NameError: pass
    try: del train_dataloader, dev_dataloader, test_dataloader
    except NameError: pass
    try: del train_dataset_hf, dev_dataset_hf, test_dataset_hf
    except NameError: pass
    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"===== Finished Task: {config.task} =====")
    return best_accuracy, test_acc # Return results


def run():
    # --- Configuration ---
    config = types.SimpleNamespace()
    config.model_name = "meta-llama/Meta-Llama-3.1-8B"
    config.diff_dir = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/GLUE_output_difficulty_jsonlines"
    config.cache_dir = os.environ.get("TRANSFORMERS_CACHE", "./hf_cache/models")
    config.num_epochs = 20
    config.learning_rate = 1e-5 # Adjusted LR for Llama
    config.gpu = 0 # Set GPU index (0 for first GPU)
    config.num_workers = 4 # Number of workers for DataLoader
    # Use calculated gradient accumulation steps based on global batch sizes
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.strategy = 'theta' # PUDF strategy
    config.ordering = 'easiest' # PUDF ordering
    config.num_obs = 1000 # Number of observations for theta estimation (-1 for all)
    config.min_train_length = 1000 # Minimum examples for training per epoch
    config.lower_bound = -np.inf # Lower bound for PUDF filtering
    config.upper_bound = 0.0 # Upper bound for PUDF filtering (relative to theta_hat)
    config.balanced = False # Whether to balance data in PUDF (usually False)
    # Ensure required attributes for build_features exist, even if not used by Llama version
    config.use_length = False
    config.use_word_rarity = False
    # Speedup Option
    config.use_torch_compile = False # Set to True to try torch.compile (requires PyTorch 2.0+)
    # --- End Configuration ---

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_short_name = config.model_name.split('/')[-1]
    output_dir_base = f"./glue_PUDF_{model_short_name}_{config.strategy}_{run_timestamp}"
    print(f"Base output directory: {output_dir_base}")
    os.makedirs(output_dir_base, exist_ok=True)

    # --- Loop through specified GLUE tasks ---
    results = {}
    for task in tqdm(GLUETASKS, desc="Overall Task Progress"):
        config.task = task; print(f"\n\n>>>>>>>> Starting Task: {config.task} <<<<<<<<")
        print(f"--- Using Configuration ---");
        # Print config, handling -np.inf for display
        for key, value in config.__dict__.items(): print(f"  {key}: {'-Infinity' if isinstance(value, float) and value == -np.inf else value}")
        print(f"-------------------------")
        task_start_time = time.time()
        try:
            top_dev, test_acc = train(config, output_dir_base)
            results[task] = {"best_dev_acc": top_dev, "test_acc": test_acc}
        except Exception as e: print(f"\n!FATAL ERROR during Task {task} execution!"); traceback.print_exc(); results[task] = {"best_dev_acc": "FATAL_ERROR", "test_acc": "FATAL_ERROR"}
        task_end_time = time.time(); print(f">>>>>>>> Finished Task: {config.task} in {task_end_time - task_start_time:.2f}s <<<<<<<<")

    # --- Print Summary Results ---
    print("\n\n===== Run Summary ====="); print(f"Model: {config.model_name}, Strategy: {config.strategy}"); print(f"Output Dir: {output_dir_base}")
    print("Task Results (Best Dev Acc / Test Acc):")
    for task in sorted(results.keys()): # Sort for consistent output
        res = results[task]; dev_res = f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'], float) else str(res['best_dev_acc'])
        test_res = f"{res['test_acc']:.4f}" if isinstance(res['test_acc'], float) else str(res['test_acc'])
        print(f"  - {task}: {dev_res} / {test_res}")
    print("=====================")
    summary_file = os.path.join(output_dir_base, "run_summary.json")
    try:
        with open(summary_file, "w") as f: json.dump(results, f, indent=4, default=str) # Use default=str
        print(f"Summary saved: {summary_file}")
    except Exception as e: print(f"Warning: Failed to save run summary JSON: {e}")

if __name__ == '__main__':
    run()