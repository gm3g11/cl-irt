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
from torch.optim import AdamW
# Mixed Precision - Updated Import Style
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset
import evaluate
# Removed build_features/irt_scoring imports - will integrate needed functions
import types # To create a simple namespace object for config
import gc # Garbage Collection
import traceback # For detailed error logging
from tqdm.auto import tqdm # Import tqdm for progress bars
import re # For tokenization helper
import copy # For get_epoch_training_data

# --- Environment and Cache Setup ---
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache" # MODIFY IF NEEDED
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Global Random Seed ---
random_seed = 63
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)

# --- Task Definitions ---
GLUETASKS = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp']
# Use user-provided max lengths
task_max_lengths = {"mrpc": 72, "rte": 150, "mnli": 72, "qqp": 56, "sst2": 32, "qnli": 80}

# --- Global Training Configuration ---
batch_size = 256 # Keep consistent batch size
print(f"Using global batch size: {batch_size}")
transformers.logging.set_verbosity_error()

# --- Helper Functions ---

# ***** NEW: Tokenization helper for difficulty calculation *****
def tokenize(sent):
    # Simple whitespace and punctuation splitting
    return ' '.join([x.strip() for x in re.split(r'(\W+)', str(sent)) if x.strip()])

# ***** NEW: Word Rarity Calculation Function *****
def get_example_rarities(examples_texts):
    """for an input list of texts, return the sentence rarity difficulty heuristic"""
    if not examples_texts or not isinstance(examples_texts[0], str):
        print("Warning: Input to get_example_rarities doesn't seem to be a list of strings.")
        # Try to handle pairs if passed accidentally (less robust)
        if isinstance(examples_texts[0], list) and len(examples_texts[0]) == 2:
             print("Handling potential list of pairs...")
             examples_texts = [str(e[0]) + " " + str(e[1]) for e in examples_texts]
        else:
             return [0.0] * len(examples_texts) # Return default on unexpected input

    result = []
    tokenized_corpus = [tokenize(text).split(' ') for text in examples_texts]

    vocab = set()
    counts = dict()
    N = 0
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t] # Filter out empty strings
        vocab.update(valid_tokens)
        N += len(valid_tokens)
        for tok in valid_tokens:
            counts.setdefault(tok, 0)
            counts[tok] += 1

    if N == 0: # Handle empty corpus case
        print("Warning: Corpus resulted in 0 tokens for rarity calculation.")
        return [0.0] * len(examples_texts)

    # Calculate rarity for each example
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]
        if not valid_tokens:
             p_hat = 0.0 # Assign 0 rarity to empty sentences
        else:
             # Use small epsilon to avoid log(0) for words unseen at test time (though here we build vocab on input)
             epsilon = 1e-9
             log_probs = [np.log(counts.get(tok, 0) / N + epsilon) for tok in valid_tokens]
             # Average log probability seems more stable than sum for varying lengths
             # Negative average log probability (higher means rarer words on average)
             p_hat = -np.mean(log_probs) if log_probs else 0.0
             # Original sum log prob version:
             # p_hats = [(counts.get(tok, 0) / N) + epsilon for tok in valid_tokens]
             # p_hat = np.sum(np.log(p_hats)) * -1 if p_hats else 0.0
        result.append(p_hat)

    return result

# ***** MODIFIED: Data Loading to CALCULATE difficulty *****
def load_and_prepare_data(task, difficulty_measurer, cache_dir):
    """Loads GLUE dataset, adds difficulty scores, performs train/val split."""
    print(f"Loading dataset for task: {task}")
    try: raw_datasets = load_dataset('glue', task, cache_dir=cache_dir)
    except Exception as e: print(f"ERROR loading dataset '{task}': {e}"); raise

    train = raw_datasets['train']
    print(f"Calculating difficulty scores using: {difficulty_measurer}")
    diff_scores = []
    if difficulty_measurer == 'sentence_length':
        if task in ['sst2']:
            diff_scores = [len(str(p)) for p in train['sentence']]
        elif task in ['mrpc', 'rte']:
            diff_scores = [len(str(p1)) + len(str(p2)) for p1, p2 in zip(train['sentence1'], train['sentence2'])]
        elif task in ['qqp']:
            diff_scores = [len(str(p1)) + len(str(p2)) for p1, p2 in zip(train['question1'], train['question2'])]
        elif task in ['qnli']:
            diff_scores = [len(str(p1)) + len(str(p2)) for p1, p2 in zip(train['question'], train['sentence'])]
        elif task == 'mnli':
            diff_scores = [len(str(p1)) + len(str(p2)) for p1, p2 in zip(train['premise'], train['hypothesis'])]
        else: raise ValueError(f"Task {task} not supported for sentence_length difficulty.")

    elif difficulty_measurer == 'word_rarity':
        texts_to_process = []
        if task in ['sst2']: texts_to_process = [str(s) for s in train['sentence']]
        elif task in ['mrpc', 'rte']: texts_to_process = [str(s1) + " " + str(s2) for s1, s2 in zip(train['sentence1'], train['sentence2'])]
        elif task in ['qqp']: texts_to_process = [str(q1) + " " + str(q2) for q1, q2 in zip(train['question1'], train['question2'])]
        elif task in ['qnli']: texts_to_process = [str(q) + " " + str(s) for q, s in zip(train['question'], train['sentence'])]
        elif task == 'mnli': texts_to_process = [str(p) + " " + str(h) for p, h in zip(train['premise'], train['hypothesis'])]
        else: raise ValueError(f"Task {task} not supported for word_rarity difficulty.")
        diff_scores = get_example_rarities(texts_to_process)
    else:
        raise ValueError(f"Unsupported difficulty_measurer for this script: {difficulty_measurer}")

    if len(diff_scores) != len(train): raise ValueError(f"Calculated difficulty count ({len(diff_scores)}) != dataset size ({len(train)}) for {task}.")

    print("Adding difficulty scores...");
    if 'difficulty' in train.column_names: print("Warning: Replacing 'difficulty' column."); train = train.remove_columns(['difficulty'])
    train = train.add_column('difficulty', diff_scores)
    print("Splitting train data (90/10)...")
    # Ensure split happens AFTER adding difficulty
    train_val_split = train.train_test_split(test_size=0.1, seed=random_seed)
    train_dataset, val_dataset = train_val_split['train'], train_val_split['test']
    test_split_name = 'validation_matched' if task == 'mnli' else 'validation'
    if test_split_name not in raw_datasets: print(f"Warning: Test split '{test_split_name}' not found for {task}. Using 'validation'."); test_split_name = 'validation'
    test_dataset = raw_datasets[test_split_name]; print("Data loading complete.")
    # Difficulty is now in train_dataset and val_dataset from the split
    return train_dataset, val_dataset, test_dataset

def tokenize_function(examples, task, tokenizer):
    # Using user-provided max lengths
    max_length = task_max_lengths.get(task)
    if max_length is None: print(f"Warning: max_length not defined for {task}. Using 128."); max_length = 128
    if task == "mnli": return tokenizer(text=examples["premise"], text_pair=examples["hypothesis"], padding="max_length", truncation=True, max_length=max_length)
    if task in ["mrpc", "rte"]: return tokenizer(text=examples["sentence1"], text_pair=examples["sentence2"], padding="max_length", truncation=True, max_length=max_length)
    if task == "qnli": return tokenizer(text=examples["question"], text_pair=examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    if task == "qqp": return tokenizer(text=examples["question1"], text_pair=examples["question2"], padding="max_length", truncation=True, max_length=max_length)
    if task == "sst2": return tokenizer(text=examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    raise ValueError(f"Task {task} not supported.")

# ***** MODIFIED create_dataset TO ALWAYS INCLUDE DIFFICULTY (even dummy) *****
def create_dataset(dataset, task, tokenizer, is_test_set=False):
    """Tokenizes dataset and ensures difficulty column is present for consistent structure."""
    print(f"Tokenizing dataset (is_test_set={is_test_set})...")
    try:
        sample_tokenization = tokenize_function(dataset[:1], task, tokenizer)
        has_token_type_ids = 'token_type_ids' in sample_tokenization
        tokenized_cols = ['input_ids', 'attention_mask']
        if has_token_type_ids: tokenized_cols.append('token_type_ids')

        # Keep 'difficulty' if present during tokenization map
        cols_to_remove = [col for col in dataset.column_names if col not in ['label', 'difficulty']]
        tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, task, tokenizer), batched=True, remove_columns=cols_to_remove, desc="Running tokenizer")

        final_columns = tokenized_cols + ['label']

        # Ensure difficulty column exists, add dummy if needed (esp. for test)
        if 'difficulty' not in tokenized_dataset.column_names:
            if is_test_set:
                 print("Adding dummy difficulty column to test set for structural consistency...")
                 dummy_difficulty = [0.0] * len(tokenized_dataset)
                 tokenized_dataset = tokenized_dataset.add_column('difficulty', dummy_difficulty)
            else:
                 # This shouldn't happen for train/dev if load_and_prepare_data worked
                 raise RuntimeError(f"Difficulty column missing in non-test dataset for task {task}.")

        final_columns.append('difficulty') # Difficulty is now always expected
        print(f"Difficulty column included/added. Final columns: {final_columns}")


        tokenized_dataset.set_format(type='torch', columns=final_columns)
        tensors_to_extract = [tokenized_dataset[col] for col in final_columns]
        print(f"Final tensor order for TensorDataset: {final_columns}") # Should now always include difficulty
        # Example order: ['input_ids', 'attention_mask', 'token_type_ids', 'label', 'difficulty']
        # Or: ['input_ids', 'attention_mask', 'label', 'difficulty']
        return TensorDataset(*tensors_to_extract)
    except Exception as e:
        print(f"ERROR tokenizing/formatting: {e}");
        traceback.print_exc()
        raise


accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])

# ***** MODIFIED: Evaluate function to ALWAYS calculate loss manually *****
# This avoids potential issues with the model's internal loss calculation
# when the input batch structure differs slightly (like missing difficulty originally).
def evaluate_model(model, dataloader, device, mode='eval'): # Mode param kept but unused in logic change
    """Evaluates model, calculates loss/accuracy MANUALLY."""
    val_loss = 0.0; metric = accuracy_metric; preds_list, labels_list = [], []
    model.eval(); num_batches = 0
    eval_desc = "Validation Eval" if mode=='eval' else "Test Eval" # Adjust description based on mode maybe?
    is_first_batch = True; has_token_type_ids = False
    # No need to track difficulty_present_in_batch anymore as we assume it's always there

    # Define loss function here (assuming classification)
    # Adjust if your task requires a different loss (e.g., regression)
    loss_fct = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc=eval_desc, leave=False):
        num_batches += 1
        batch_len = len(batch) # Should now be consistently 4 or 5

        # Determine structure on first batch
        if is_first_batch:
            # Expecting (ids, mask, [tti], label, difficulty) -> len 4 or 5
            if batch_len == 5: has_token_type_ids = True
            elif batch_len == 4: has_token_type_ids = False
            else: raise ValueError(f"Unexpected batch length {batch_len}. Expected 4 or 5.")
            is_first_batch = False
            print(f"Eval batch struct determined: Len={batch_len}, HasTokID={has_token_type_ids}")

        # Unpack batch - Indices are now fixed relative to has_token_type_ids
        # 0: input_ids, 1: attention_mask
        # 2: token_type_ids (if present), label (if not)
        # 3: label (if tti present), difficulty (if not)
        # 4: difficulty (if tti present)
        try:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            token_type_ids = batch[2].to(device, non_blocking=True) if has_token_type_ids else None
            label_index = 3 if has_token_type_ids else 2
            if batch_len <= label_index: raise IndexError(f"Label index {label_index} out of range for batch length {batch_len}")
            labels = batch[label_index].to(device, non_blocking=True)
            # We don't strictly need 'difficulty' for eval, but it's index 4 or 3
        except IndexError as e:
            print(f"ERROR: IndexError unpacking eval batch. Len:{batch_len}, HasTokID:{has_token_type_ids}, LabelIdx:{label_index}. Error: {e}")
            traceback.print_exc()
            continue

        with torch.no_grad():
            try:
                # ***** CORE CHANGE: Do NOT pass labels to model *****
                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids) # No labels=labels argument
                logits = outputs.logits

                # ***** CORE CHANGE: Calculate loss manually *****
                if labels is not None and logits is not None:
                     # Ensure model config has num_labels if needed
                     num_labels = getattr(model.config, "num_labels", logits.shape[-1])
                     loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                else:
                     loss = torch.tensor(0.0, device=device) # Handle missing labels/logits if applicable

            except Exception as e:
                # Add detailed traceback!
                print(f"\nERROR during model forward pass or manual loss calc in eval: {e}")
                traceback.print_exc() # Get the exact failing line
                print("Skipping batch.")
                continue # Skip batch on error

        # Proceed with metrics using manually obtained logits and loss
        if logits is not None and labels is not None:
            preds_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions.detach().cpu(), references=labels.detach().cpu())
        val_loss += loss.item() # Use manually calculated loss

    # Concatenate results & calculate final metrics
    val_loss /= num_batches if num_batches > 0 else 1
    try:
        eval_score = metric.compute() if num_batches > 0 else None
        validation_accuracy = eval_score['accuracy'] if eval_score else 0.0
        if num_batches == 0:
             print("Warning: No batches were successfully processed during evaluation.")
    except Exception as e:
        print(f"Warning: metric compute failed: {e}. Setting Acc=0.")
        validation_accuracy = 0.0

    return validation_accuracy, val_loss


# ***** NEW: get_epoch_training_data adapted for 'linear'/'root' and config object *****
def get_epoch_training_data(ts, config, epoch):
    """Selects training data subset based on scheduler type ('linear', 'root', 'baseline')."""
    scheduler_type = config.training_scheduler
    ordering_type = config.ordering
    num_total_examples = len(ts) # Get length from TensorDataset

    if scheduler_type == 'baseline':
        print("Scheduler: baseline - Using full dataset.")
        # Return the full dataset as a dictionary of tensors
        # Assumes ts.tensors includes difficulty now
        tensors = ts.tensors
        input_ids = tensors[0]
        attention_mask = tensors[1]
        has_token_type_ids = len(tensors) == 5 # ids, mask, tti, label, diff
        token_type_ids = tensors[2] if has_token_type_ids else None
        labels = tensors[3] if has_token_type_ids else tensors[2]
        difficulties = tensors[4] if has_token_type_ids else tensors[3]

        data_dict = { 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'difficulty': difficulties }
        if token_type_ids is not None: data_dict['token_type_ids'] = token_type_ids # Include TTI if present
        return data_dict

    # --- Logic for 'linear', 'root' ---
    c_init = 0.01  # Initial competency proportion (can be made a config param if needed)

    # Data structure to hold sorted tensors
    train_sorted = { 'input_ids': None, 'attention_mask': None, 'token_type_ids': None, 'labels': None, 'difficulty': None }
    # Data structure for the final selected subset
    train_epoch = { 'input_ids': None, 'attention_mask': None, 'token_type_ids': None, 'labels': None, 'difficulty': None }

    # Extract tensors (handle optional token_type_ids) - assumes difficulty is last
    tensors = ts.tensors
    input_ids = tensors[0]
    attention_mask = tensors[1]
    has_token_type_ids = len(tensors) == 5 # Infer based on expected length (ids, mask, [tti], labels, diff)
    token_type_ids = tensors[2] if has_token_type_ids else None
    labels = tensors[3] if has_token_type_ids else tensors[2]
    difficulties = tensors[4] if has_token_type_ids else tensors[3] # Difficulty is last

    print(f"Sorting data based on difficulty (Ordering: {ordering_type})...")
    # Sort examples based on difficulty and ordering strategy
    difficulty_np = difficulties.cpu().numpy()
    if ordering_type == 'easiest': diffs_sorted_idx = np.argsort(difficulty_np)
    elif ordering_type == 'hardest': diffs_sorted_idx = np.argsort(difficulty_np)[::-1]
    elif ordering_type == 'middleout': diffs_sorted_idx = np.argsort(np.abs(difficulty_np))
    else: raise NotImplementedError(f"Ordering '{ordering_type}' not implemented.")

    # Apply balancing if requested (simplified, assumes binary/multi-class labels are numeric)
    if config.balanced:
        print("Applying balanced sorting...")
        per_label_lists = {}
        unique_labels = torch.unique(labels).cpu().numpy()
        for ul in unique_labels: per_label_lists[ul] = []

        for idx in diffs_sorted_idx:
            label_item = labels[idx].item()
            if label_item in per_label_lists: per_label_lists[label_item].append(idx)

        max_length = max(len(v) for v in per_label_lists.values()) if per_label_lists else 0
        train_2_idx = []
        for l in range(max_length):
            for k in sorted(per_label_lists.keys()):
                v = per_label_lists[k]
                if l < len(v): train_2_idx.append(v[l])

        if not train_2_idx: print("Warning: No indices selected after balancing. Using original sort.")
        else: diffs_sorted_idx = np.array(train_2_idx) # Use balanced indices

    # Populate train_sorted with sorted tensors
    train_sorted['input_ids'] = input_ids[diffs_sorted_idx]
    train_sorted['attention_mask'] = attention_mask[diffs_sorted_idx]
    if has_token_type_ids: train_sorted['token_type_ids'] = token_type_ids[diffs_sorted_idx]
    train_sorted['labels'] = labels[diffs_sorted_idx]
    train_sorted['difficulty'] = difficulties[diffs_sorted_idx]


    # --- Calculate number of examples for the epoch based on scheduler ---
    num_train = 0
    competency_param = getattr(config, 'competency', 5) # Get competency or default to 5

    if scheduler_type == 'linear':
        competency_epoch = max(1, competency_param) # Avoid division by zero
        if epoch >= competency_epoch: epoch_competency = 1.0
        else: epoch_competency = c_init + (1.0 - c_init) * (epoch / competency_epoch)
        num_train = int(epoch_competency * num_total_examples)
        print(f"Scheduler: linear - Epoch {epoch+1} competency={epoch_competency:.3f}")

    elif scheduler_type == 'root':
        competency_epoch = max(1, competency_param) # Avoid division by zero
        if epoch >= competency_epoch: epoch_competency = 1.0
        else: epoch_competency = c_init + (1.0 - c_init) * np.sqrt(epoch / competency_epoch)
        num_train = int(epoch_competency * num_total_examples)
        print(f"Scheduler: root - Epoch {epoch+1} competency={epoch_competency:.3f}")

    else:
        raise NotImplementedError(f"Scheduler '{scheduler_type}' logic not defined here.")

    # Ensure minimum length and not exceeding available data
    min_len = getattr(config, 'min_train_length', 128) # Get min_len or default
    num_train = max(min_len, num_train)
    num_train = min(num_train, len(train_sorted['input_ids'])) # Use length of sorted data
    print(f"Selecting {num_train} examples (Min: {min_len}).")

    # Slice the sorted data to get the epoch's training set
    train_epoch['input_ids'] = train_sorted['input_ids'][:num_train]
    train_epoch['attention_mask'] = train_sorted['attention_mask'][:num_train]
    if has_token_type_ids: train_epoch['token_type_ids'] = train_sorted['token_type_ids'][:num_train]
    train_epoch['labels'] = train_sorted['labels'][:num_train]
    train_epoch['difficulty'] = train_sorted['difficulty'][:num_train] # Also slice difficulty

    return train_epoch


# --- Main Training Function (MODIFIED) ---
def train(config, output_dir_base): # Takes base output dir now
    """Main training loop for a given task using specified config."""
    # Construct specific output dir for this task run
    task_output_dir = os.path.join(output_dir_base, config.task)
    best_model_dir = os.path.join(task_output_dir, "best_model")
    os.makedirs(task_output_dir, exist_ok=True)
    # Don't necessarily make best_model_dir here, let save_pretrained handle it

    print(f"\n===== Starting Training for Task: {config.task} =====")
    print(f"    Difficulty: {config.difficulty_measurer}")
    print(f"    Scheduler: {config.training_scheduler}")
    print(f"    Output Dir: {task_output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu >= 0 else 'cpu')
    print(f"Using device: {device}"); use_amp = torch.cuda.is_available(); print(f"Using AMP: {use_amp}")
    try: # Data Loading uses config.difficulty_measurer now
        train_dataset_hf, dev_dataset_hf, test_dataset_hf = load_and_prepare_data(
            config.task, config.difficulty_measurer, config.cache_dir
        )
        train_size, val_size, test_size = len(train_dataset_hf), len(dev_dataset_hf), len(test_dataset_hf)
        print(f"Data sizes: Train={train_size}, Val={val_size}, Test={test_size}"); assert train_size > 0 and val_size > 0
    except Exception as e: print(f"FATAL: Data loading failed: {e}"); traceback.print_exc(); return 0.0, 0.0

    num_labels = 3 if config.task.startswith("mnli") else 1 if config.task == "stsb" else 2
    model_name = 'microsoft/deberta-v3-base'
    print(f"Loading: {model_name} (Labels: {num_labels})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=config.cache_dir)
    except Exception as e: print(f"FATAL: Loading model failed: {e}"); traceback.print_exc(); return 0.0, 0.0
    model.to(device)

    # torch.compile option removed for simplicity based on original script state

    print("Creating/tokenizing datasets (difficulty will be included/added)...")
    try:
        # Create datasets - difficulty column is now always added by create_dataset
        train_dataset = create_dataset(train_dataset_hf, config.task, tokenizer, is_test_set=False)
        dev_dataset = create_dataset(dev_dataset_hf, config.task, tokenizer, is_test_set=False)
        test_dataset = create_dataset(test_dataset_hf, config.task, tokenizer, is_test_set=True) # Mark as test set
    except Exception as e: print(f"FATAL: Dataset creation failed: {e}"); return 0.0, 0.0

    print("Creating dataloaders...")
    # Use global batch_size
    # Dataloaders now expect batches with consistent structure (including difficulty)
    train_dataloader_full = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True) if len(test_dataset) > 0 else None

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.999))
    # Estimate steps based on FULL dataloader length for scheduler
    num_training_steps_estimate = len(train_dataloader_full) * config.num_epochs
    num_warmup_steps = max(1, int(0.06 * num_training_steps_estimate))
    print(f"Scheduler: Est steps={num_training_steps_estimate}, Warmup={num_warmup_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps_estimate)
    scaler = GradScaler(enabled=use_amp)

    best_accuracy = 0.0; early_stop_count = 0; patience = 5; training_stats = []
    print(f"\nStarting training loop: Max {config.num_epochs} epochs, Patience {patience}...")
    start_train_loop_time = time.time(); actual_epochs = 0

    # --- Epoch Loop ---
    for epoch in range(config.num_epochs):
        actual_epochs = epoch + 1; print(f"\n======== Epoch {epoch + 1} / {config.num_epochs} ========"); epoch_start_time = time.time()
        epoch_loss = 0.0; filter_time = 0.0;

        # Get data for the epoch using the CL strategy
        print(f"Filtering data (Scheduler: {config.training_scheduler}, Ordering: {config.ordering})...");
        filter_time_s = time.time()
        try:
            # Pass the full TensorDataset (train_dataset)
            filtered_training_data = get_epoch_training_data(train_dataset, config, epoch)
        except Exception as e: print(f"ERROR: get_epoch_training_data failed: {e}. Skipping epoch."); traceback.print_exc(); continue
        filter_time_e = time.time(); filter_time = filter_time_e - filter_time_s

        if 'labels' not in filtered_training_data or len(filtered_training_data['labels']) == 0:
             print("Warning: No data selected. Skipping training phase."); num_epoch_examples = 0; avg_train_loss = 0.0
             current_train_dataloader = None # No loader if no data
        else:
            num_epoch_examples = len(filtered_training_data['labels'])
            print(f"Selected {num_epoch_examples} examples ({filter_time:.2f}s filtering).")
            try: # Create Epoch Loader
                 tensors_for_epoch = [filtered_training_data['input_ids'], filtered_training_data['attention_mask']]
                 # Check if TTI exists in the dictionary returned by get_epoch_training_data
                 has_token_type_ids_epoch = 'token_type_ids' in filtered_training_data and filtered_training_data['token_type_ids'] is not None
                 if has_token_type_ids_epoch: tensors_for_epoch.append(filtered_training_data['token_type_ids'])
                 # Labels and difficulty are expected to be present
                 tensors_for_epoch.extend([filtered_training_data['labels'], filtered_training_data['difficulty']])
                 train_dataset_epoch = TensorDataset(*tensors_for_epoch)
                 # Use global batch size, but drop_last might be relevant if batch size > num examples
                 effective_batch_size = min(batch_size, num_epoch_examples) if num_epoch_examples > 0 else batch_size
                 # Ensure drop_last=True if effective_batch_size > num_epoch_examples? Usually handled by DataLoader.
                 current_train_dataloader = DataLoader(train_dataset_epoch, shuffle=True, batch_size=effective_batch_size, num_workers=config.num_workers, pin_memory=True)
            except Exception as e: print(f"ERROR creating epoch dataloader: {e}. Skipping."); traceback.print_exc(); current_train_dataloader = None; continue

        # --- Train Phase ---
        if current_train_dataloader:
            print(f"Training epoch {epoch + 1}..."); model.train(); epoch_loss = 0.0; optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(current_train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
            for step, batch in enumerate(pbar):
                 # Determine batch structure based on length (should be 4 or 5)
                 batch_len = len(batch)
                 has_token_type_ids_batch = batch_len == 5 # ids, mask, tti, label, diff
                 label_index_epoch = 3 if has_token_type_ids_batch else 2 # Index of label tensor

                 input_ids=batch[0].to(device, non_blocking=True); attention_mask=batch[1].to(device, non_blocking=True)
                 token_type_ids = batch[2].to(device, non_blocking=True) if has_token_type_ids_batch else None; labels = batch[label_index_epoch].to(device, non_blocking=True)
                 try:
                     with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                         # Pass labels during training as internal loss is expected to work here
                         outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels);
                         loss = outputs.loss
                     if loss is None: # Check if loss calculation failed internally
                         print(f"Warning: Internal loss calculation returned None at step {step}. Check model config/inputs."); continue
                     if torch.isnan(loss): print(f"Warning: NaN loss step {step}. Skip."); optimizer.zero_grad(set_to_none=True); continue

                     scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                     scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                     current_loss = loss.item(); epoch_loss += current_loss; pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                 except Exception as e:
                     print(f"ERROR training step {step}: {e}");
                     traceback.print_exc() # Add traceback here too
                     optimizer.zero_grad(set_to_none=True); print("Skip step."); continue
            avg_train_loss = epoch_loss / len(pbar) if len(pbar) > 0 else 0.0; print(f"Epoch {epoch+1} Avg Training Loss: {avg_train_loss:.4f}")
        else:
             avg_train_loss = 0.0 # No training occurred

        # --- Validation Phase ---
        print("Evaluating on validation set...")
        try:
            # Use the MODIFIED evaluate_model function (calculates loss manually)
            dev_acc, val_loss = evaluate_model(model, dev_dataloader, device, mode='eval')
            print(f"Epoch {epoch+1} Validation: Acc={dev_acc:.4f}, Loss={val_loss:.4f}")

            training_stats.append({
                'epoch': epoch + 1, 'Train Loss': avg_train_loss, 'Val Loss': val_loss, 'Val Acc': dev_acc,
                'filter_time': filter_time, 'n_train': num_epoch_examples
            })

            if dev_acc > best_accuracy: # Early Stopping & Saving
                print(f"Val acc improved ({best_accuracy:.4f} --> {dev_acc:.4f}). Saving model to {best_model_dir}...");
                best_accuracy = dev_acc; early_stop_count = 0
                try:
                    # Ensure parent directory exists before saving
                    os.makedirs(best_model_dir, exist_ok=True)
                    model_to_save = getattr(model, '_orig_mod', model); model_to_save.save_pretrained(best_model_dir); tokenizer.save_pretrained(best_model_dir)
                    # Add save confirmation check
                    saved_files = os.listdir(best_model_dir)
                    if "pytorch_model.bin" in saved_files or "model.safetensors" in saved_files: print("Model save confirmed.")
                    else: print(f"Warning: Save command executed, but weights file not found in {best_model_dir}. Files: {saved_files}")
                except Exception as e: print(f"Warning: Error saving best model: {e}")
            else:
                early_stop_count += 1; print(f"Val acc ({dev_acc:.4f}) vs best ({best_accuracy:.4f}). Early stop count: {early_stop_count}/{patience}")
                if early_stop_count >= patience: print("Early stopping."); break
        except Exception as e: print(f"ERROR during validation epoch {epoch+1}: {e}"); traceback.print_exc(); print("Stopping task."); break

        gc.collect(); # Cleanup
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        epoch_end_time = time.time(); print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f}s.")

    # --- End of Training Loop ---
    end_train_loop_time = time.time(); train_loop_duration = end_train_loop_time - start_train_loop_time
    print("\n--- Training Loop Finished ---")
    print(f"Actual epochs: {actual_epochs}, Total Time: {train_loop_duration:.2f}s, Best Val Acc: {best_accuracy:.4f}")

    # --- Save Stats ---
    model_checkpoint_name = "deberta-v3-base"
    # Use config fields in stats filename for clarity
    stats_filename_base = f"{model_checkpoint_name}_{config.task}_{config.difficulty_measurer}_{config.training_scheduler}"
    training_stats_filename = os.path.join(task_output_dir, f"training_stats_{stats_filename_base}.json")
    print(f"Saving training stats: {training_stats_filename}");
    try:
        with open(training_stats_filename, "w") as f: json.dump(training_stats, f, indent=4)
    except Exception as e: print(f"Warning: Error saving training stats: {e}")

    # --- Final Test Evaluation ---
    print("\n--- Final Test Evaluation ---"); test_acc, test_loss = 0.0, 0.0
    if test_dataloader is None: print("Test dataloader None. Skip."); test_acc, test_loss = -3.0, -3.0
    elif os.path.isdir(best_model_dir) and \
         (os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
          os.path.exists(os.path.join(best_model_dir, "model.safetensors"))):
        print(f"Loading best model: {best_model_dir}...");
        try:
            # Load the best saved model
            model_loaded = AutoModelForSequenceClassification.from_pretrained(best_model_dir);
            # Tokenizer needed if tokenizer specific changes were saved, good practice to load it too
            tokenizer_loaded = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True)
            model_loaded.to(device);
            print("Evaluating on test set using evaluate_model (manual loss)...");
            # Use the MODIFIED evaluate_model, which now calculates loss manually
            test_acc, test_loss = evaluate_model(model_loaded, test_dataloader, device, mode='test') # Use mode='test' for description
            print(f'Final Test Acc: {test_acc:.4f}, Loss: {test_loss:.4f}');
            del model_loaded, tokenizer_loaded # Clean up loaded model
        except Exception as e:
            print(f"ERROR during final test eval: {e}");
            traceback.print_exc() # Crucial traceback
            test_acc, test_loss = -1.0, -1.0
    else:
        print(f"Best model directory/weights not found or invalid: {best_model_dir}. Cannot test.");
        test_acc, test_loss = -2.0, -2.0

    # --- Save Final Summary ---
    final_stats_filename = os.path.join(task_output_dir, f"final_stats_{stats_filename_base}_Acc_{test_acc:.4f}.json")
    print(f"Saving final summary: {final_stats_filename}");
    final_summary = {
        "task": config.task, "model": model_checkpoint_name,
        "difficulty_measurer": config.difficulty_measurer, # Add specific config
        "training_scheduler": config.training_scheduler, # Add specific config
        "ordering": config.ordering,
        "competency": getattr(config, 'competency', 'N/A'), # Add competency if used
        "min_train_length": config.min_train_length,
        "num_epochs_set": config.num_epochs, "num_epochs_run": actual_epochs,
        "best_validation_accuracy": best_accuracy, "final_test_accuracy": test_acc, "final_test_loss": test_loss,
        "total_training_loop_time_seconds": train_loop_duration,
        "config_snapshot": {k: str(v) for k, v in config.__dict__.items()}, # Save snapshot
        "training_stats_summary": training_stats
    }
    try:
        with open(final_stats_filename, "w") as f: json.dump(final_summary, f, indent=4, default=str)
    except Exception as e: print(f"Warning: Error saving final summary: {e}")

    # Explicit cleanup
    print("Cleaning up task resources...")
    try: del model, tokenizer, optimizer, scheduler, scaler
    except NameError: pass
    try: del train_dataset, dev_dataset, test_dataset
    except NameError: pass
    try: del train_dataloader_full, dev_dataloader, test_dataloader, current_train_dataloader
    except NameError: pass
    try: del train_dataset_hf, dev_dataset_hf, test_dataset_hf
    except NameError: pass
    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"===== Finished Task: {config.task} =====")
    return best_accuracy, test_acc


# ***** MODIFIED: Run function with loops *****
def run():
    # --- Base Configuration ---
    base_config = types.SimpleNamespace()
    # Removed diff_dir - difficulty is calculated
    base_config.cache_dir = os.environ.get("TRANSFORMERS_CACHE", "./hf_cache/models")
    base_config.num_epochs = 20 # Consistent epochs
    base_config.learning_rate = 2e-5
    base_config.gpu = 0
    base_config.num_workers = 4
    base_config.ordering = 'easiest' # Common ordering for curriculum
    base_config.min_train_length = 128 # Consistent minimum length
    base_config.competency = 5 # Default competency for linear/root
    base_config.balanced = False

    # --- Experiment Loops ---
    difficulty_measures_to_run = ['sentence_length', 'word_rarity']
    schedulers_to_run = ['linear', 'root']
    base_output_dir_root = "./DebertaV3_ManualLossEval" # Changed output dir name slightly
    print(f"*** Using Manual Loss Calculation in evaluate_model ***")
    print(f"*** Outputting results to: {base_output_dir_root} ***")


    overall_results = {} # Store results across all runs

    for diff_measure in difficulty_measures_to_run:
        for scheduler in schedulers_to_run:

            # --- Create Specific Config & Output Dir ---
            config = copy.deepcopy(base_config) # Start with base config
            config.difficulty_measurer = diff_measure
            config.training_scheduler = scheduler

            run_id = f"{config.difficulty_measurer}_{config.training_scheduler}"
            current_output_dir_base = os.path.join(base_output_dir_root, run_id)
            print(f"\n\n===== Starting Run: {run_id} =====")
            print(f"Outputting to base directory: {current_output_dir_base}")
            os.makedirs(current_output_dir_base, exist_ok=True)

            run_results = {} # Store results for this specific run combination

            # --- Loop through specified GLUE tasks ---
            for task in tqdm(GLUETASKS, desc=f"Task Progress ({run_id})"):
                config.task = task # Set task in config for the train function
                print(f"\n\n>>>>>>>> Starting Task: {config.task} ({run_id}) <<<<<<<<")
                print(f"--- Using Configuration ---")
                for key, value in config.__dict__.items(): print(f"  {key}: {value}")
                print(f"-------------------------")
                task_start_time = time.time()
                try:
                    # Pass the base dir for this run (e.g., DebertaV3/sentence_length_linear)
                    top_dev, test_acc = train(config, current_output_dir_base)
                    run_results[task] = {"best_dev_acc": top_dev, "test_acc": test_acc}
                except Exception as e:
                    print(f"\n!FATAL ERROR Task {task} in Run {run_id}!"); traceback.print_exc()
                    run_results[task] = {"best_dev_acc": "FATAL_ERROR", "test_acc": "FATAL_ERROR"}
                task_end_time = time.time()
                print(f">>>>>>>> Finished Task: {config.task} in {task_end_time - task_start_time:.2f}s <<<<<<<<")

            # --- Print Summary for this Run Combination ---
            print(f"\n\n===== Run Summary ({run_id}) =====")
            print(f"Difficulty: {config.difficulty_measurer}")
            print(f"Scheduler: {config.training_scheduler}")
            print(f"Output Dir Base: {current_output_dir_base}")
            print("Task Results (Best Dev Acc / Test Acc):")
            for task in sorted(run_results.keys()):
                res = run_results[task]; dev_res = f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'], float) else res['best_dev_acc']
                test_res = f"{res['test_acc']:.4f}" if isinstance(res['test_acc'], float) else res['test_acc']
                print(f"  - {task}: {dev_res} / {test_res}")
            print("===================================")
            overall_results[run_id] = run_results # Store results for this run

            # Save summary for this specific run
            summary_file = os.path.join(current_output_dir_base, f"run_summary_{run_id}.json")
            try:
                with open(summary_file, "w") as f: json.dump(run_results, f, indent=4, default=str)
                print(f"Run summary saved: {summary_file}")
            except Exception as e: print(f"Warning: Failed to save run summary: {e}")

    # --- Print Overall Summary ---
    print("\n\n===== Overall Run Summary =====")
    for run_id, run_results in overall_results.items():
         print(f"\n--- Results for: {run_id} ---")
         for task in sorted(run_results.keys()):
              res = run_results[task]; dev_res = f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'], float) else res['best_dev_acc']
              test_res = f"{res['test_acc']:.4f}" if isinstance(res['test_acc'], float) else res['test_acc']
              print(f"  - {task}: {dev_res} / {test_res}")
    print("=============================")
    overall_summary_file = os.path.join(base_output_dir_root, "overall_summary.json")
    try:
        with open(overall_summary_file, "w") as f: json.dump(overall_results, f, indent=4, default=str)
        print(f"Overall summary saved: {overall_summary_file}")
    except Exception as e: print(f"Warning: Failed to save overall summary: {e}")


if __name__ == '__main__':
    run()