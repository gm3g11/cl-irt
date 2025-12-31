# -*- coding: utf-8 -*-
# Qwen_7B.py (Corrected Again + Manual Loss Eval Fix)
import numpy as np
import random
import os
import time
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
# Use Adafactor Optimizer (Matches PUDF Script)
from transformers.optimization import Adafactor
# Mixed Precision
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset
import evaluate
# No external CL imports - integrated below
import types # To create a simple namespace object for config
import gc # Garbage Collection
import traceback # For detailed error logging
from tqdm.auto import tqdm # Import tqdm for progress bars
from huggingface_hub import login, whoami # Keep for checking login status
from math import ceil # For calculating number of batches
import re # For tokenization helper
import copy # For get_epoch_training_data adaptation

# --- Environment and Cache Setup (Matches PUDF Script) ---
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE # MODIFY IF NEEDED
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Matches PUDF Script

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Hugging Face Auth Check (Matches PUDF Script) ---
try: user_info = whoami(); print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
except Exception as e: print(f"HF Login Check Failed: {e}. Public models should still work.")

# --- Global Random Seed (Matches PUDF Script) ---
random_seed = 63
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

# --- Task Definitions (Matches PUDF Script) ---
GLUETASKS = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp']
task_max_lengths = {"mrpc": 72, "rte": 150, "mnli": 72, "qqp": 56, "sst2": 32, "qnli": 80}

# --- Global Training Configuration (ALIGNED WITH PUDF Script) ---
# Using PUDF batch sizes - User confirmed OOM is not a concern currently.
train_batch_size = 256 # Per device step (FROM PUDF SCRIPT)
effective_batch_size = 256 # Target effective batch size (FROM PUDF SCRIPT)
gradient_accumulation_steps = max(1, effective_batch_size // train_batch_size) # = 1
eval_batch_size = 256 # FROM PUDF SCRIPT
test_batch_size = 256 # FROM PUDF SCRIPT

print(f"Train Batch Size (per device step): {train_batch_size}")
print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
print(f"Effective Batch Size (Train): {train_batch_size * gradient_accumulation_steps}")
print(f"Eval/Test Batch Size: {eval_batch_size}")

transformers.logging.set_verbosity_error()

# --- BF16 Check (Matches PUDF Script) ---
bf16_ready = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
amp_dtype = torch.bfloat16 if bf16_ready else torch.float16
print(f"AMP Dtype selected: {amp_dtype}")

# --- Integrated Helper Functions ---

def tokenize(sent):
    """Basic tokenization splitting on non-word characters."""
    if not isinstance(sent, str): sent = str(sent)
    # Keep apostrophes, handle other non-alphanumeric as split points
    tokens = [x.strip() for x in re.split(r'([^\w\'])', sent) if x and x.strip()]
    # Fallback if split results in nothing (e.g., only punctuation)
    if not tokens and sent: tokens = [sent]
    return ' '.join(tokens) if tokens else ""


def get_example_rarities(list_of_strings):
    """Calculates word rarity (mean negative log prob) for a list of strings."""
    if not isinstance(list_of_strings, list) or not all(isinstance(s, str) for s in list_of_strings):
         # Add handling for potential None or non-string elements gracefully
         print("Warning: Non-string elements found in input to get_example_rarities. Converting to string.")
         list_of_strings = [str(s) if s is not None else "" for s in list_of_strings]
         # raise ValueError("Input to get_example_rarities must be a list of strings.") # Made less strict
    print(f"Calculating word rarity for {len(list_of_strings)} examples...")
    result = []
    tokenized_corpus = [tokenize(text).split(' ') for text in list_of_strings]
    counts = dict(); N = 0
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]; N += len(valid_tokens)
        for tok in valid_tokens: counts.setdefault(tok, 0); counts[tok] += 1
    if N == 0: print("Warning: Corpus resulted in 0 tokens."); return [0.0] * len(list_of_strings)
    epsilon = 1e-9
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]
        if not valid_tokens: p_hat = 0.0
        else: log_probs = [np.log(counts.get(tok, 0) / N + epsilon) for tok in valid_tokens]; p_hat = -np.mean(log_probs) if log_probs else 0.0
        result.append(p_hat)
    print("Word rarity calculation finished.")
    return result

# --- Modified Data Loading and Preparation (Calculates Difficulty) ---

def load_and_prepare_data(task, difficulty_measurer, cache_dir):
    """Loads GLUE dataset, CALCULATES difficulty, performs train/val split."""
    print(f"Loading dataset for task: {task}")
    try: raw_datasets = load_dataset('glue', task, cache_dir=cache_dir)
    except Exception as e: print(f"ERROR loading dataset '{task}': {e}"); raise

    validation_split_name = 'validation_matched' if task == 'mnli' else 'validation'
    if validation_split_name not in raw_datasets:
        print(f"Warning: Expected split '{validation_split_name}'. Trying 'validation'.")
        validation_split_name = 'validation';
        if validation_split_name not in raw_datasets: raise ValueError(f"Cannot find validation/test split for task '{task}'")

    train = raw_datasets['train']
    print(f"Calculating difficulty scores using: {difficulty_measurer}")
    diff_scores = []; texts_for_rarity = []
    s1_key, s2_key = None, None
    if task in ['sst2']: s1_key = 'sentence'
    elif task in ['mrpc', 'rte']: s1_key, s2_key = 'sentence1', 'sentence2'
    elif task in ['qqp']: s1_key, s2_key = 'question1', 'question2'
    elif task in ['qnli']: s1_key, s2_key = 'question', 'sentence'
    elif task == 'mnli': s1_key, s2_key = 'premise', 'hypothesis'
    else: raise ValueError(f"Task {task} structure unknown.")

    for i in range(len(train)):
        s1 = str(train[i][s1_key])
        s2 = str(train[i][s2_key]) if s2_key and s2_key in train.features else None
        if difficulty_measurer == 'sentence_length':
            diff_scores.append(len(s1) + (len(s2) if s2 else 0))
        elif difficulty_measurer == 'word_rarity':
            texts_for_rarity.append(s1 + (" " + s2 if s2 else ""))
        else: raise ValueError(f"Unsupported difficulty_measurer: {difficulty_measurer}")

    if difficulty_measurer == 'word_rarity': diff_scores = get_example_rarities(texts_for_rarity)
    if len(diff_scores) != len(train): raise ValueError(f"Difficulty count mismatch.")

    print("Adding difficulty scores...");
    if 'difficulty' in train.column_names: train = train.remove_columns(['difficulty'])
    train = train.add_column('difficulty', diff_scores)
    print("Splitting train data (90/10)...")
    train_val_split = train.train_test_split(test_size=0.1, seed=random_seed)
    train_dataset, val_dataset = train_val_split['train'], train_val_split['test']
    test_dataset = raw_datasets[validation_split_name]
    print(f"Using '{validation_split_name}' split for testing.")
    print("Data loading complete.")
    return train_dataset, val_dataset, test_dataset

# --- Tokenization and Dataset Creation ---

def tokenize_function(examples, task, tokenizer):
    """Tokenizes examples based on task structure and max length."""
    max_length = task_max_lengths.get(task, 128)
    s1_key, s2_key = None, None
    if task == "mnli": s1_key, s2_key = "premise", "hypothesis"
    elif task in ["mrpc", "rte"]: s1_key, s2_key = "sentence1", "sentence2"
    elif task == "qqp": s1_key, s2_key = "question1", "question2"
    elif task == "qnli": s1_key, s2_key = "question", "sentence"
    elif task == "sst2": s1_key = "sentence"
    else: raise ValueError(f"Task {task} not supported.")

    # Handle potential None values before converting to string
    text_a_raw = examples[s1_key]
    text_b_raw = examples[s2_key] if s2_key else None

    text_a = [str(t) if t is not None else "" for t in text_a_raw] if isinstance(text_a_raw, list) else [str(text_a_raw) if text_a_raw is not None else ""]
    text_b = None
    if text_b_raw:
        text_b = [str(t) if t is not None else "" for t in text_b_raw] if isinstance(text_b_raw, list) else [str(text_b_raw) if text_b_raw is not None else ""]


    if text_b: return tokenizer(text=text_a, text_pair=text_b, padding="max_length", truncation=True, max_length=max_length)
    else: return tokenizer(text=text_a, padding="max_length", truncation=True, max_length=max_length)

# ***** MODIFIED create_dataset TO ALWAYS INCLUDE DIFFICULTY (even dummy) *****
def create_dataset(dataset, task, tokenizer, is_test_set=False):
    """Tokenizes dataset and converts to PyTorch TensorDataset, ensuring difficulty column."""
    print(f"Tokenizing dataset (is_test_set={is_test_set})...")
    try:
        dummy_example = dataset[:1]
        sample_tokenization = tokenize_function(dummy_example, task, tokenizer)
        tokenized_cols = ['input_ids', 'attention_mask']
        has_token_type_ids = 'token_type_ids' in sample_tokenization
        if has_token_type_ids: tokenized_cols.append('token_type_ids'); print("Note: Including token_type_ids.")
        else: print("Note: No token_type_ids generated.")

        cols_to_keep_initial = set(tokenized_cols + ['label']) # Start with needed columns
        cols_to_remove = [col for col in dataset.column_names if col not in cols_to_keep_initial and col != 'difficulty'] # Keep difficulty initially if present

        tokenized_dataset = dataset.map(lambda exs: tokenize_function(exs, task, tokenizer), batched=True, remove_columns=cols_to_remove, desc="Running tokenizer")

        # Rename label -> labels if necessary
        if 'label' in tokenized_dataset.column_names:
            print("Renaming 'label' column to 'labels'.")
            tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
        elif 'labels' not in tokenized_dataset.column_names:
             # Check if 'label' exists as fallback before erroring
             if 'label' not in dataset.column_names: # Check original dataset too
                 raise ValueError("'labels' or 'label' column missing.")
             else: # If original had 'label' but map removed it somehow? Highly unlikely.
                 print("Warning: 'label' column existed but 'labels' not found after map. Proceeding cautiously.")


        # Ensure difficulty column exists, add dummy if needed
        if 'difficulty' not in tokenized_dataset.column_names:
            if is_test_set:
                 print("Adding dummy difficulty column to test set for structural consistency...")
                 dummy_difficulty = [0.0] * len(tokenized_dataset)
                 tokenized_dataset = tokenized_dataset.add_column('difficulty', dummy_difficulty)
            else:
                 # This implies load_and_prepare_data failed for train/dev
                 raise RuntimeError(f"Difficulty column missing in non-test dataset for task {task}.")

        # Define final order, assuming difficulty is last
        final_columns_ordered = tokenized_cols + ['labels', 'difficulty']
        # Verify all columns exist
        for col in final_columns_ordered:
             if col not in tokenized_dataset.column_names:
                  raise ValueError(f"Expected column '{col}' not found in tokenized dataset.")

        print(f"Setting format. Final columns order: {final_columns_ordered}")
        tokenized_dataset.set_format(type='torch', columns=final_columns_ordered)
        tensors_to_extract = [tokenized_dataset[col] for col in final_columns_ordered]
        print(f"TensorDataset created with {len(tensors_to_extract)} tensors.")
        return TensorDataset(*tensors_to_extract), final_columns_ordered
    except Exception as e: print(f"ERROR tokenizing/formatting: {e}"); traceback.print_exc(); raise


# --- Evaluation Function (Manual Loss Calculation) ---
accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])

# ***** MODIFIED: Evaluate function to ALWAYS calculate loss manually *****
def evaluate_model(model, dataloader, device, column_order):
    """Evaluates model, calculates loss/accuracy MANUALLY."""
    val_loss = 0.0; metric = accuracy_metric; preds_list, labels_list = [], []
    model.eval(); num_batches = 0; eval_desc = "Validation/Test Eval"
    amp_eval_dtype = torch.bfloat16 if bf16_ready else torch.float16
    loss_fct = torch.nn.CrossEntropyLoss() # Define loss function

    try: # Determine indices from the provided column order
        ids_idx = column_order.index('input_ids'); mask_idx = column_order.index('attention_mask')
        tti_idx = column_order.index('token_type_ids') if 'token_type_ids' in column_order else -1
        lbl_idx = column_order.index('labels')
        # diff_idx = column_order.index('difficulty') # Not needed for calculation but confirms presence
        print(f"Eval Indices: ids={ids_idx}, mask={mask_idx}, tti={tti_idx}, lbl={lbl_idx}")
    except ValueError as e: print(f"FATAL: Column missing in eval order {column_order}. Error: {e}"); raise

    for batch in tqdm(dataloader, desc=eval_desc, leave=False):
        num_batches += 1
        try: # Unpack batch using determined indices
            input_ids=batch[ids_idx].to(device, non_blocking=True); attention_mask=batch[mask_idx].to(device, non_blocking=True)
            token_type_ids = batch[tti_idx].to(device, non_blocking=True) if tti_idx != -1 else None
            labels = batch[lbl_idx].to(device, non_blocking=True)
        except Exception as e: print(f"ERROR unpacking eval batch: {e}"); traceback.print_exc(); continue

        with torch.no_grad(), autocast(device_type='cuda', dtype=amp_eval_dtype, enabled=torch.cuda.is_available()):
            try:
                # *** CORE CHANGE: Build kwargs WITHOUT labels ***
                kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                if token_type_ids is not None: kwargs['token_type_ids'] = token_type_ids

                # *** CORE CHANGE: Call model without labels ***
                outputs = model(**kwargs)
                logits = outputs.logits

                 # *** CORE CHANGE: Calculate loss manually ***
                if labels is not None and logits is not None:
                     # Ensure model config has num_labels if needed
                     num_labels = getattr(model.config, "num_labels", logits.shape[-1])
                     loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                else:
                     loss = torch.tensor(0.0, device=device) # Handle missing labels/logits

            except Exception as e:
                print(f"\nERROR eval forward/manual loss: {e}")
                traceback.print_exc() # Add detailed traceback!
                continue # Skip batch on error

        # Proceed with metrics using manually obtained logits and loss
        if logits is not None and labels is not None:
            preds_list.append(logits.detach().float().cpu().numpy()) # Use .float() for safety if AMP out is diff dtype
            labels_list.append(labels.detach().cpu().numpy())
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions.detach().cpu(), references=labels.detach().cpu())
        val_loss += loss.item() # Use manually calculated loss

    val_loss /= num_batches if num_batches > 0 else 1
    try:
        eval_score = metric.compute() if num_batches > 0 else None
        acc = eval_score['accuracy'] if eval_score else 0.0
        if num_batches == 0: print("Warning: No batches were successfully processed during evaluation.")
    except Exception as e: print(f"Warning: metric compute failed: {e}. Setting Acc=0."); acc = 0.0
    return acc, val_loss


# --- Integrated Curriculum Learning Logic (Corrected - Assuming difficulty is last) ---
def get_epoch_training_data(ts, config, epoch, train_col_order):
    """Selects training data subset based on heuristic scheduler."""
    scheduler_type = getattr(config, 'training_scheduler', 'baseline')
    ordering_type = getattr(config, 'ordering', 'easiest')
    is_balanced = getattr(config, 'balanced', False)
    min_len = getattr(config, 'min_train_length', 128)
    competency_param = getattr(config, 'competency', 5)

    if not isinstance(ts, torch.utils.data.TensorDataset): raise TypeError(f"Input 'ts' must be TensorDataset.")

    num_total_examples = len(ts); tensors = ts.tensors; num_tensors = len(tensors)
    print(f"[CL Func] Input TensorDataset has {num_tensors} tensors. Expected order: {train_col_order}")
    if len(train_col_order) != num_tensors:
         raise ValueError(f"Mismatch between train_col_order length ({len(train_col_order)}) and num_tensors ({num_tensors})")

    # Get tensors based on names from train_col_order
    tensor_dict = {name: tensors[i] for i, name in enumerate(train_col_order)}
    input_ids = tensor_dict['input_ids']
    attention_mask = tensor_dict['attention_mask']
    labels = tensor_dict['labels']
    difficulties = tensor_dict['difficulty']
    token_type_ids = tensor_dict.get('token_type_ids') # Will be None if not in order
    has_token_type_ids = token_type_ids is not None

    if scheduler_type == 'baseline':
        print("[CL Func] Scheduler: baseline - Using full dataset.")
        # Return dict directly using extracted tensors
        return tensor_dict # Already has all necessary tensors by name

    c_init = 0.01
    # Create empty dicts based on keys present in tensor_dict
    train_sorted = {key: None for key in tensor_dict}
    train_epoch = {key: None for key in tensor_dict}


    print(f"[CL Func] Sorting data (Ordering: {ordering_type})...")
    difficulty_np = difficulties.cpu().numpy(); safe_difficulties = np.nan_to_num(difficulty_np)
    if ordering_type == 'easiest': diffs_sorted_idx = np.argsort(safe_difficulties)
    elif ordering_type == 'hardest': diffs_sorted_idx = np.argsort(safe_difficulties)[::-1]
    elif ordering_type == 'middleout': diffs_sorted_idx = np.argsort(np.abs(safe_difficulties))
    else: raise NotImplementedError(f"Ordering '{ordering_type}' unknown.")

    if is_balanced:
        print("[CL Func] Applying balanced sorting...")
        per_label_lists = {}; unique_labels = torch.unique(labels).cpu().numpy()
        for ul in unique_labels: per_label_lists[ul] = []
        for idx in diffs_sorted_idx: label_item = labels[idx].item();
        if label_item in per_label_lists: per_label_lists[label_item].append(idx)
        max_length = max(len(v) for v in per_label_lists.values()) if per_label_lists else 0; balanced_idx = []
        for l in range(max_length):
            for k in sorted(per_label_lists.keys()): v = per_label_lists[k];
            if l < len(v): balanced_idx.append(v[l])
        if not balanced_idx: print("Warning: Balancing yielded no indices.")
        else: diffs_sorted_idx = np.array(balanced_idx)

    # Populate train_sorted using keys from tensor_dict
    for key, tensor in tensor_dict.items():
        train_sorted[key] = tensor[diffs_sorted_idx]

    num_train = 0; competency_epoch = max(1, competency_param)
    if scheduler_type == 'linear':
        if epoch >= competency_epoch: epoch_competency = 1.0
        else: epoch_competency = c_init + (1.0 - c_init) * (epoch / competency_epoch)
        num_train = int(epoch_competency * num_total_examples); print(f"[CL Func] Scheduler: linear - Epoch {epoch+1} competency={epoch_competency:.3f}")
    elif scheduler_type == 'root':
        if epoch >= competency_epoch: epoch_competency = 1.0
        else: epoch_competency = c_init + (1.0 - c_init) * np.sqrt(epoch / competency_epoch)
        num_train = int(epoch_competency * num_total_examples); print(f"[CL Func] Scheduler: root - Epoch {epoch+1} competency={epoch_competency:.3f}")
    else: raise NotImplementedError(f"Scheduler '{scheduler_type}' unknown.")

    num_train = max(min_len, num_train); num_available = len(train_sorted['input_ids']); num_train = min(num_train, num_available)
    print(f"[CL Func] Selecting {num_train} examples (Min: {min_len}, Available: {num_available}).")

    # Slice sorted data using keys from train_sorted
    for key, tensor in train_sorted.items():
         if tensor is not None: train_epoch[key] = tensor[:num_train]
    return train_epoch


# --- Main Training Function (Adapted for Heuristics & PUDF Config) ---
def train(config, output_dir_base):
    """Main training loop adapted for heuristic CL runs with PUDF config."""
    task_output_dir = os.path.join(output_dir_base, config.task)
    best_model_dir = os.path.join(task_output_dir, "best_model")
    os.makedirs(task_output_dir, exist_ok=True)

    print(f"\n===== Starting Training for Task: {config.task} ====="); print(f"    Difficulty: {config.difficulty_measurer}"); print(f"    Scheduler: {config.training_scheduler}"); print(f"    Output Dir: {task_output_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu >= 0 else 'cpu'); print(f"Using device: {device}"); use_amp = torch.cuda.is_available(); print(f"Using AMP: {use_amp} with dtype: {amp_dtype}")

    try: train_dataset_hf, dev_dataset_hf, test_dataset_hf = load_and_prepare_data(config.task, config.difficulty_measurer, config.cache_dir)
    except Exception as e: print(f"FATAL: Data loading failed: {e}"); traceback.print_exc(); return 0.0, 0.0
    train_size, val_size, test_size = len(train_dataset_hf), len(dev_dataset_hf), len(test_dataset_hf); print(f"Data sizes: Train={train_size}, Val={val_size}, Test={test_size}"); assert train_size > 0 and val_size > 0

    num_labels = 3 if config.task.startswith("mnli") else 1 if config.task == "stsb" else 2; model_name = config.model_name; print(f"Loading: {model_name} (Labels: {num_labels})");
    try: # Model/Tokenizer loading with padding setup
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir, use_fast=True, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=config.cache_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token; print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'}); model.resize_token_embeddings(len(tokenizer)); print("Added [PAD] token.") # Add pad token if missing
        model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config matches
        print(f"Using pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    except Exception as e: print(f"FATAL: Loading model/tokenizer failed: {e}"); traceback.print_exc(); return 0.0, 0.0

    model.to(device);
    if config.use_gradient_checkpointing: model.gradient_checkpointing_enable(); print("Gradient Checkpointing Enabled.")
    # Qwen2 doesn't typically use use_cache=False setting like older models, but keep if needed
    # model.config.use_cache = False;
    # Qwen2 doesn't have sliding_window in base config usually
    # if hasattr(model.config, "sliding_window"): model.config.sliding_window = None

    print("Creating/tokenizing datasets (difficulty always included)...");
    try: # Call create_dataset with is_test_set flag
        # include_difficulty is implicitly handled by adding dummy in create_dataset now
        train_dataset, train_col_order = create_dataset(train_dataset_hf, config.task, tokenizer, is_test_set=False)
        dev_dataset, dev_col_order = create_dataset(dev_dataset_hf, config.task, tokenizer, is_test_set=False)
        # Pass is_test_set=True for the test dataset
        test_dataset, test_col_order = create_dataset(test_dataset_hf, config.task, tokenizer, is_test_set=True)
        print(f"Train columns: {train_col_order}") # Should include 'labels' and 'difficulty'
        print(f"Dev columns: {dev_col_order}")
        print(f"Test columns: {test_col_order}")
    except Exception as e: print(f"FATAL: Dataset creation failed: {e}"); traceback.print_exc(); return 0.0, 0.0

    print("Creating dataloaders...");
    dl_kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    base_train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False, **dl_kwargs)
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, **dl_kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **dl_kwargs) if test_dataset and len(test_dataset) > 0 else None

    print("Setting up Adafactor optimizer.");
    optimizer = Adafactor(model.parameters(), lr=config.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=config.weight_decay)
    num_update_steps_per_epoch_base = ceil(len(base_train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps_estimate = num_update_steps_per_epoch_base * config.num_epochs
    num_warmup_steps = max(1, int(config.warmup_ratio * num_training_steps_estimate))
    print(f"Scheduler: Est Total Steps={num_training_steps_estimate}, Warmup Steps={num_warmup_steps} ({config.warmup_ratio*100}%)")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps_estimate)
    scaler = GradScaler(enabled=use_amp);

    best_accuracy = 0.0; early_stop_count = 0; patience = config.early_stopping_patience; training_stats = [];
    print(f"\nStarting training loop: Max {config.num_epochs} epochs, Patience {patience}...");
    start_train_loop_time = time.time(); actual_epochs = 0

    # --- Epoch Loop ---
    for epoch in range(config.num_epochs):
        actual_epochs = epoch + 1; print(f"\n======== Epoch {epoch + 1} / {config.num_epochs} ========"); epoch_start_time = time.time()
        filter_time = 0.0; num_optimizer_steps = 0; avg_train_loss = 0.0

        print(f"Selecting training data (Scheduler: {config.training_scheduler})...");
        filter_time_s = time.time()
        try:
            # Pass train_col_order to get_epoch_training_data
            filtered_data_dict = get_epoch_training_data(train_dataset, config, epoch, train_col_order)
        except Exception as e: print(f"ERROR: get_epoch_training_data: {e}. Skipping epoch."); traceback.print_exc(); continue
        filter_time = time.time() - filter_time_s

        # Check if data was selected
        if not filtered_data_dict or 'labels' not in filtered_data_dict or len(filtered_data_dict.get('labels', [])) == 0:
            print("Warning: No data selected/available after filtering. Skipping training phase.");
            train_dataloader_epoch = None
            num_epoch_examples = 0
        else:
            num_epoch_examples = len(filtered_data_dict['labels']) # Get length from tensor
            print(f"Selected {num_epoch_examples} examples ({filter_time:.2f}s filtering).")
            try: # Create Epoch Loader using train_col_order
                 tensors_for_epoch = []
                 # Reconstruct tensor list in the correct order expected by DataLoader
                 for col_name in train_col_order:
                     if col_name not in filtered_data_dict: raise KeyError(f"Column '{col_name}' missing in filtered data.")
                     tensors_for_epoch.append(filtered_data_dict[col_name])

                 train_dataset_epoch = TensorDataset(*tensors_for_epoch)
                 current_epoch_batch_size = min(train_batch_size, num_epoch_examples) if num_epoch_examples > 0 else train_batch_size
                 # drop_last=True might be important if BS > num_examples and using grad accum
                 use_drop_last = config.gradient_accumulation_steps > 1 and num_epoch_examples < current_epoch_batch_size
                 train_dataloader_epoch = DataLoader(train_dataset_epoch, shuffle=True, batch_size=current_epoch_batch_size, drop_last=use_drop_last, **dl_kwargs)
                 print(f"Created epoch dataloader with {len(train_dataloader_epoch)} batches (batch size: {current_epoch_batch_size}, drop_last={use_drop_last}).")
            except Exception as e: print(f"ERROR creating epoch dataloader: {e}. Skipping."); traceback.print_exc(); continue

        # --- Training Phase ---
        if train_dataloader_epoch: # Check if dataloader was created
            model.train(); optimizer.zero_grad(set_to_none=True);
            pbar = tqdm(train_dataloader_epoch, desc=f"Epoch {epoch+1} Training", leave=False)
            num_steps_in_epoch = len(pbar); epoch_step_losses = []

            for step, batch in enumerate(pbar):
                 try: # Unpack using train_col_order indices
                      batch_tensors = {name: batch[i] for i, name in enumerate(train_col_order)}
                      input_ids=batch_tensors['input_ids'].to(device, non_blocking=True)
                      attention_mask=batch_tensors['attention_mask'].to(device, non_blocking=True)
                      token_type_ids = batch_tensors.get('token_type_ids') # Returns None if key missing
                      if token_type_ids is not None: token_type_ids = token_type_ids.to(device, non_blocking=True)
                      labels = batch_tensors['labels'].to(device, non_blocking=True)
                 except Exception as e: print(f"ERROR unpack train batch {step}: {e}"); traceback.print_exc(); continue

                 try: # Training Step
                     with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                         # Pass labels during training, assuming internal loss works ok here
                         kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                         if token_type_ids is not None: kwargs['token_type_ids'] = token_type_ids
                         outputs = model(**kwargs); loss = outputs.loss
                     if loss is None or torch.isnan(loss): print(f"Warning: Skip step {step} None/NaN loss."); optimizer.zero_grad(set_to_none=True); continue

                     loss_val = loss.item() # Get loss value before dividing
                     loss = loss / config.gradient_accumulation_steps
                     scaler.scale(loss).backward();
                     epoch_step_losses.append(loss_val) # Store original loss value for avg

                     if ((step + 1) % config.gradient_accumulation_steps == 0) or ((step + 1) == num_steps_in_epoch):
                         scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                         scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                         num_optimizer_steps += 1;
                     current_avg_loss = np.mean(epoch_step_losses) if epoch_step_losses else 0
                     if step % 50 == 0 or (step+1) == num_steps_in_epoch: # Log less frequently for large models
                         pbar.set_postfix({'avg_loss': f'{current_avg_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

                 except Exception as e: print(f"\nERROR training step {step}: {e}"); traceback.print_exc(); optimizer.zero_grad(set_to_none=True); continue

            avg_train_loss = np.mean(epoch_step_losses) if num_optimizer_steps > 0 else 0.0
            print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f} ({num_optimizer_steps} optimizer steps)")

        # --- Validation Phase ---
        print("Evaluating on validation set...")
        try:
            # Call MODIFIED evaluate_model, passing the dev_col_order
            dev_acc, val_loss = evaluate_model(model, dev_dataloader, device, dev_col_order)
            print(f"Epoch {epoch+1} Validation: Acc={dev_acc:.4f}, Loss={val_loss:.4f}")
            training_stats.append({'epoch': epoch + 1, 'Train Loss': avg_train_loss, 'Val Loss': val_loss, 'Val Acc': dev_acc, 'filter_time': filter_time, 'n_train_epoch': num_epoch_examples, 'optimizer_steps_epoch': num_optimizer_steps})
            if dev_acc > best_accuracy:
                print(f"Val acc improved ({best_accuracy:.4f} --> {dev_acc:.4f}). Saving model to {best_model_dir}...");
                best_accuracy = dev_acc; early_stop_count = 0
                try:
                    os.makedirs(best_model_dir, exist_ok=True);
                    # Handle potential model wrappers like FSDP if used later
                    model_to_save = getattr(model, 'module', model) # Basic check
                    model_to_save = getattr(model_to_save, '_orig_mod', model_to_save) # For torch.compile maybe
                    model_to_save.save_pretrained(best_model_dir);
                    tokenizer.save_pretrained(best_model_dir);
                    print("Model save executed.")
                except Exception as e: print(f"Warning: Error saving best model: {e}"); traceback.print_exc();
            else:
                early_stop_count += 1; print(f"Val acc ({dev_acc:.4f}) did not improve vs best ({best_accuracy:.4f}). Early stop count: {early_stop_count}/{patience}")
                if early_stop_count >= patience: print("Early stopping triggered."); break
        except Exception as e: print(f"ERROR validation epoch {epoch+1}: {e}"); traceback.print_exc(); break

        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        epoch_end_time = time.time(); print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f}s.")

    # --- Post Training ---
    end_train_loop_time = time.time(); train_loop_duration = end_train_loop_time - start_train_loop_time; print("\n--- Training Loop Finished ---"); print(f"Actual epochs: {actual_epochs}, Total Time: {train_loop_duration:.2f}s, Best Val Acc: {best_accuracy:.4f}")

    # Save Training Stats
    model_short = config.model_name.split('/')[-1].replace('.', '-'); stats_base = f"{model_short}_{config.task}_{config.difficulty_measurer}_{config.training_scheduler}"
    stats_file = os.path.join(task_output_dir, f"training_stats_{stats_base}.json"); print(f"Saving training stats: {stats_file}");
    try:
        with open(stats_file, "w") as f: json.dump(training_stats, f, indent=4)
    except Exception as e: print(f"Warning: Error saving stats: {e}")

    # Final Test Evaluation
    print("\n--- Final Test Evaluation ---"); test_acc, test_loss, test_time_seconds = 0.0, 0.0, 0.0; test_time_start = time.time()
    if test_dataloader is None: print("Test dataloader None. Skip."); test_acc, test_loss = -3.0, -3.0
    else:
        weights_bin = os.path.join(best_model_dir, "pytorch_model.bin"); weights_sf = os.path.join(best_model_dir, "model.safetensors"); idx_sf = os.path.join(best_model_dir, "model.safetensors.index.json")
        # Check for actual weights files or index file
        model_found = os.path.isdir(best_model_dir) and (os.path.exists(weights_bin) or os.path.exists(weights_sf) or os.path.exists(idx_sf))
        if model_found:
            print(f"Loading best model: {best_model_dir}...");
            try:
                # Load model and tokenizer from best dir
                model_loaded = AutoModelForSequenceClassification.from_pretrained(best_model_dir, trust_remote_code=True).to(device);
                # No need to reload tokenizer usually, but added for completeness if needed
                # tokenizer_loaded = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True, trust_remote_code=True)

                # Call MODIFIED evaluate_model, passing test_col_order
                test_acc, test_loss = evaluate_model(model_loaded, test_dataloader, device, test_col_order)
                print(f'Final Test Accuracy: {test_acc:.4f}, Final Test Loss: {test_loss:.4f}'); del model_loaded
            except Exception as e: print(f"ERROR during test eval: {e}"); traceback.print_exc(); test_acc, test_loss = -1.0, -1.0
        else: print(f"Best model weights not found in {best_model_dir}. Skip test."); test_acc, test_loss = -2.0, -2.0
    test_time_seconds = time.time() - test_time_start; print(f"Test eval took: {test_time_seconds:.2f}s")

    # Save Final Summary
    summary_file = os.path.join(task_output_dir, f"final_stats_{stats_base}_Acc_{test_acc:.4f}.json"); print(f"Saving final summary: {summary_file}");
    def convert_np(obj): # Helper for JSON serialization
        if isinstance(obj, np.integer): return int(obj);
        elif isinstance(obj, np.floating): return float(obj);
        elif isinstance(obj, np.ndarray): return obj.tolist();
        return obj
    final_summary = { "task": config.task, "model": config.model_name, "difficulty_measurer": config.difficulty_measurer, "training_scheduler": config.training_scheduler, "ordering": config.ordering, "competency": getattr(config, 'competency', 'N/A'), "min_train_length": config.min_train_length, "num_epochs_set": config.num_epochs, "num_epochs_run": actual_epochs, "best_validation_accuracy": best_accuracy, "final_test_accuracy": test_acc, "final_test_loss": test_loss, "total_training_loop_time_seconds": round(train_loop_duration, 2), "final_test_time_seconds": round(test_time_seconds, 2), "config_snapshot": {k: convert_np(v) for k, v in config.__dict__.items()}, "training_stats_summary": training_stats }
    try:
        with open(summary_file, "w") as f: json.dump(final_summary, f, indent=4, default=convert_np)
    except Exception as e: print(f"Warning: Error saving final summary: {e}")

    print("Cleaning up task resources...");
    try: del model, tokenizer, optimizer, scheduler, scaler # Add model_loaded if defined? No, deleted earlier
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
    return best_accuracy, test_acc


# --- run() function with Heuristic Loops & Aligned Config ---
def run():
    # --- Base Configuration (ALIGNED WITH PUDF SCRIPT) ---
    base_config = types.SimpleNamespace()
    base_config.model_name = "Qwen/Qwen2.5-7B" # Corrected model name based on common HF naming
    base_config.cache_dir = os.environ.get("TRANSFORMERS_CACHE", "./hf_cache/models")
    base_config.num_epochs = 20 # Matches PUDF
    base_config.learning_rate = 1e-5 # Matches PUDF
    base_config.weight_decay = 0.01 # Matches PUDF
    base_config.warmup_ratio = 0.06 # Matches PUDF
    base_config.max_grad_norm = 1.0 # Matches PUDF
    base_config.gradient_accumulation_steps = gradient_accumulation_steps # Use calculated value (will be 1)
    base_config.gpu = 0 # Matches PUDF
    base_config.num_workers = 4 # Matches PUDF
    base_config.ordering = 'easiest' # Matches PUDF
    base_config.min_train_length = 1000 # Matches PUDF
    base_config.competency = 5 # Keep for linear/root (Not in PUDF)
    base_config.balanced = False # Matches PUDF
    # base_config.use_torch_compile = False # Removed as likely not needed/used
    base_config.use_gradient_checkpointing = True # Matches PUDF
    base_config.early_stopping_patience = 5 # Matches PUDF
    # --- End Aligned Configuration ---

    # --- Experiment Loops ---
    difficulty_measures_to_run = ['sentence_length', 'word_rarity']
    schedulers_to_run = ['linear', 'root']
    # *** UPDATED BASE DIRECTORY NAME ***
    model_folder_name = base_config.model_name.split('/')[-1].replace('.','-') # e.g., Qwen1-5-7B
    base_output_dir_root = f"./{model_folder_name}" # Include model name and fix method
    print(f"*** Using Manual Loss Calculation in evaluate_model ***")
    print(f"*** Outputting results to: {base_output_dir_root} ***")

    overall_results = {}

    for diff_measure in difficulty_measures_to_run:
        for scheduler in schedulers_to_run:
            config = copy.deepcopy(base_config)
            config.difficulty_measurer = diff_measure
            config.training_scheduler = scheduler
            # Remove PUDF-specific params implicitly by not adding/using them
            # config.strategy is effectively replaced by config.training_scheduler

            run_id = f"{config.difficulty_measurer}_{config.training_scheduler}"
            current_output_dir_base = os.path.join(base_output_dir_root, run_id)
            print(f"\n\n===== Starting Run: {run_id} ====="); print(f"Output base: {current_output_dir_base}")
            os.makedirs(current_output_dir_base, exist_ok=True)
            run_results = {}

            for task in tqdm(GLUETASKS, desc=f"Task Progress ({run_id})"):
                config.task = task
                print(f"\n\n>>> Task: {config.task} ({run_id}) <<<"); print("Config:", {k: v for k, v in config.__dict__.items()}) # Print config dict
                task_start_time = time.time()
                try:
                    top_dev, test_acc = train(config, current_output_dir_base)
                    run_results[task] = {"best_dev_acc": top_dev, "test_acc": test_acc}
                except Exception as e: print(f"!FATAL ERROR Task {task} Run {run_id}!"); traceback.print_exc(); run_results[task] = {"best_dev_acc": "FATAL_ERROR", "test_acc": "FATAL_ERROR"}
                task_end_time = time.time(); print(f">>> Finished Task: {config.task} in {task_end_time - task_start_time:.2f}s <<<")
                gc.collect();
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            print(f"\n\n===== Run Summary ({run_id}) ====="); # Print run summary
            print(f"Difficulty: {config.difficulty_measurer}, Scheduler: {config.training_scheduler}, Output: {current_output_dir_base}")
            print("Task Results (Dev Acc / Test Acc):")
            for task in sorted(run_results.keys()):
                res=run_results[task]; dev=f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'],float) else res['best_dev_acc']; test=f"{res['test_acc']:.4f}" if isinstance(res['test_acc'],float) else res['test_acc']
                print(f"  - {task}: {dev} / {test}")
            print("==================================="); overall_results[run_id] = run_results
            summary_file = os.path.join(current_output_dir_base, f"run_summary_{run_id}.json"); # Save run summary
            try:
                with open(summary_file, "w") as f: json.dump(run_results, f, indent=4, default=str)
                print(f"Run summary saved: {summary_file}")
            except Exception as e: print(f"Warning: Failed to save run summary: {e}")

    # Print Overall Summary
    print("\n\n===== Overall Run Summary =====");
    for run_id, run_results in overall_results.items():
         print(f"\n--- Results for: {run_id} ---")
         for task in sorted(run_results.keys()):
              res=run_results[task]; dev=f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'],float) else res['best_dev_acc']; test=f"{res['test_acc']:.4f}" if isinstance(res['test_acc'],float) else res['test_acc']
              print(f"  - {task}: {dev} / {test}")
    print("=============================")
    overall_summary_file = os.path.join(base_output_dir_root, "overall_summary.json"); # Save overall summary
    try:
        with open(overall_summary_file, "w") as f: json.dump(overall_results, f, indent=4, default=str)
        print(f"Overall summary saved: {overall_summary_file}")
    except Exception as e: print(f"Warning: Failed to save overall summary: {e}")

if __name__ == '__main__':
    run()