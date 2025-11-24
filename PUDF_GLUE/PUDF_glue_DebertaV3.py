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
from torch.amp import GradScaler  # Updated import for torch.amp.GradScaler
from torch.amp import autocast  # For CUDA
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset
import evaluate
from build_features import get_epoch_training_data  # Assuming this is the corrected version
from irt_scoring import calculate_theta
import types  # To create a simple namespace object for config
import gc  # Garbage Collection
import traceback  # For detailed error logging
from tqdm.auto import tqdm  # Import tqdm for progress bars

# --- Environment and Cache Setup ---
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Global Random Seed ---
random_seed = 63
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# --- Task Definitions ---
GLUETASKS = ['mrpc', 'rte', 'sst2', 'mnli', 'qnli', 'qqp']
task_max_lengths = {"mrpc": 72, "rte": 150, "mnli": 72, "qqp": 56, "sst2": 32, "qnli": 80}

# --- Global Training Configuration ---
batch_size = 256  # This is a global default, can be overridden by config if needed elsewhere
print(f"Using global batch size for DataLoaders (can be overridden per epoch): {batch_size}")
transformers.logging.set_verbosity_error()


# --- Helper Functions ---
def load_and_prepare_data(task, diff_dir, cache_dir):
    """Loads GLUE dataset, adds difficulty scores, performs train/val split."""
    print(f"Loading dataset for task: {task}")
    try:
        raw_datasets = load_dataset('glue', task, cache_dir=cache_dir)
    except Exception as e:
        print(f"ERROR loading dataset '{task}': {e}");
        raise

    train_diff_file = f'{diff_dir}/{task.lower()}-1pl/best_parameters.json'
    print(f"Loading difficulty scores from: {train_diff_file}")
    try:
        with open(train_diff_file, 'r') as file:
            difficulty_data = json.load(file)
        if 'diff' not in difficulty_data or not isinstance(difficulty_data['diff'], list):
            raise ValueError("Difficulty file needs a list under the 'diff' key.")
    except Exception as e:
        print(f"ERROR loading/parsing difficulty file {train_diff_file}: {e}");
        raise

    train_hf_dataset = raw_datasets['train']  # Use a more descriptive name
    if len(difficulty_data['diff']) != len(train_hf_dataset):
        raise ValueError(
            f"Difficulty count ({len(difficulty_data['diff'])}) != "
            f"dataset size ({len(train_hf_dataset)}) for {task}."
        )

    print("Adding difficulty scores...")
    if 'difficulty' in train_hf_dataset.column_names:
        print("Warning: Replacing existing 'difficulty' column.")
        train_hf_dataset = train_hf_dataset.remove_columns(['difficulty'])
    train_hf_dataset = train_hf_dataset.add_column('difficulty', difficulty_data['diff'])

    print("Splitting train data (90/10) for validation...")
    train_val_split = train_hf_dataset.train_test_split(test_size=0.1, seed=random_seed, shuffle=True)
    final_train_dataset = train_val_split['train']
    final_val_dataset = train_val_split['test']

    test_split_name = 'validation_matched' if task == 'mnli' else 'validation'
    if test_split_name not in raw_datasets:
        print(f"Warning: Test split '{test_split_name}' not found for {task}. Using 'validation' as test split.")
        test_split_name = 'validation'  # Fallback if specific test split not found
    final_test_dataset = raw_datasets[test_split_name]

    print("Data loading and preparation complete.")
    return final_train_dataset, final_val_dataset, final_test_dataset


def tokenize_function(examples, task, tokenizer):
    max_length = task_max_lengths.get(task)
    if max_length is None:
        print(f"Warning: max_length not defined for {task}. Using 128.");
        max_length = 128

    if task == "mnli":
        return tokenizer(text=examples["premise"], text_pair=examples["hypothesis"], padding="max_length",
                         truncation=True, max_length=max_length)
    if task in ["mrpc", "rte", "qqp"]:  # QQP is pair
        # Assuming qqp uses sentence1 and sentence2 if not question1/question2 explicitly in examples
        # Based on original code, QQP uses question1/question2. Let's stick to that.
        if task == "qqp":
            return tokenizer(text=examples["question1"], text_pair=examples["question2"], padding="max_length",
                             truncation=True, max_length=max_length)
        return tokenizer(text=examples["sentence1"], text_pair=examples["sentence2"], padding="max_length",
                         truncation=True, max_length=max_length)
    if task == "qnli":
        return tokenizer(text=examples["question"], text_pair=examples["sentence"], padding="max_length",
                         truncation=True, max_length=max_length)
    if task == "sst2":
        return tokenizer(text=examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    raise ValueError(f"Task {task} not supported by tokenize_function.")


def create_dataset(hf_dataset, task, tokenizer, include_difficulty=True, partition_name="unknown"):
    """Creates a PyTorch TensorDataset from a Hugging Face Dataset."""
    print(f"Tokenizing dataset for {partition_name} (include_difficulty={include_difficulty})...")
    try:
        # Determine if tokenizer output includes token_type_ids by tokenizing a small sample
        # This is safer than assuming based on task type if tokenizer behavior changes.
        sample_keys = list(hf_dataset.features.keys())
        sample_text_keys = {}
        if task == "mnli":
            sample_text_keys = {"premise": hf_dataset[0]["premise"], "hypothesis": hf_dataset[0]["hypothesis"]}
        elif task in ["mrpc", "rte"]:
            sample_text_keys = {"sentence1": hf_dataset[0]["sentence1"], "sentence2": hf_dataset[0]["sentence2"]}
        elif task == "qnli":
            sample_text_keys = {"question": hf_dataset[0]["question"], "sentence": hf_dataset[0]["sentence"]}
        elif task == "qqp":
            sample_text_keys = {"question1": hf_dataset[0]["question1"], "question2": hf_dataset[0]["question2"]}
        elif task == "sst2":
            sample_text_keys = {"sentence": hf_dataset[0]["sentence"]}
        else:
            raise ValueError(f"Task {task} sample tokenization keys not defined in create_dataset.")

        # Create a dummy example dictionary for sample tokenization
        dummy_example_for_token_type_check = {k: [v] for k, v in sample_text_keys.items()}

        sample_tokenization = tokenize_function(dummy_example_for_token_type_check, task, tokenizer)
        has_token_type_ids = 'token_type_ids' in sample_tokenization

        tokenized_cols = ['input_ids', 'attention_mask']
        if has_token_type_ids:
            tokenized_cols.append('token_type_ids')

        # Define columns to remove from the original dataset after tokenization
        # We keep 'label' and 'difficulty' (if it exists and is needed)
        # All other original text columns used for tokenization should be removed.
        cols_to_remove_after_map = [key for key in sample_text_keys.keys() if key in hf_dataset.column_names]

        # Also remove other miscellaneous columns not needed for model input or PUDF
        original_cols = list(hf_dataset.column_names)
        cols_to_keep_during_map = ['label']
        if 'difficulty' in original_cols:  # Keep difficulty if it exists
            cols_to_keep_during_map.append('difficulty')

        # Identify columns to remove that are not text inputs, label, or difficulty
        for col in original_cols:
            if col not in sample_text_keys and col not in cols_to_keep_during_map and col not in cols_to_remove_after_map:
                cols_to_remove_after_map.append(col)
        cols_to_remove_after_map = list(set(cols_to_remove_after_map))  # Unique list

        tokenized_dataset_hf = hf_dataset.map(
            lambda examples: tokenize_function(examples, task, tokenizer),
            batched=True,
            remove_columns=cols_to_remove_after_map,  # Remove original text and other unneeded cols
            desc=f"Running tokenizer on {partition_name}"
        )

        final_pytorch_columns = tokenized_cols + ['label']  # Standard model inputs + label
        if include_difficulty:
            if 'difficulty' not in tokenized_dataset_hf.column_names:
                # This might happen for the test set if include_difficulty=False was intended
                # but called with True, or if difficulty wasn't added.
                raise RuntimeError(
                    f"Difficulty column missing in {partition_name} when expected (include_difficulty=True).")
            final_pytorch_columns.append('difficulty')

        tokenized_dataset_hf.set_format(type='torch', columns=final_pytorch_columns)

        tensors_to_extract = [tokenized_dataset_hf[col] for col in final_pytorch_columns]
        print(f"Final tensor order for {partition_name} TensorDataset: {final_pytorch_columns}")
        return TensorDataset(*tensors_to_extract)
    except Exception as e:
        print(f"ERROR tokenizing/formatting {partition_name}: {e}");
        traceback.print_exc();
        raise


accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])


def evaluate_and_estimate(model, dataloader, device, num_obs_theta=-1, mode='eval'):
    val_loss = 0.0
    # metric = accuracy_metric # Use a fresh metric object per call to avoid state issues
    metric_computer = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
    preds_list, labels_list, difficulties_list = [], [], []
    model.eval()
    num_batches = 0

    eval_desc = "Validation Eval"
    if mode == 'estimate':
        eval_desc = "Theta Estimation Eval"
    elif mode == 'eval_estimate':
        eval_desc = "Validation Eval & Theta Est"

    # Determine batch structure based on the dataloader's underlying TensorDataset
    # Expected tensor order in batch: input_ids, attention_mask, [token_type_ids], labels, [difficulty]
    # This relies on create_dataset producing tensors in a consistent order.
    dataset_tensors = dataloader.dataset.tensors
    num_tensors_in_batch_source = len(dataset_tensors)

    # Infer structure based on create_dataset's output order
    # (input_ids, attention_mask, [token_type_ids], label, [difficulty])
    has_token_type_ids = False
    has_difficulty_in_batch = False

    if num_tensors_in_batch_source == 5:  # ids, mask, ttti, label, difficulty
        has_token_type_ids = True
        has_difficulty_in_batch = True  # Assumes difficulty is present for 5 tensors
    elif num_tensors_in_batch_source == 4:
        # Could be: ids, mask, ttti, label (no difficulty) OR ids, mask, label, difficulty (no ttti)
        # The create_dataset function adds token_type_ids before label, and difficulty after label.
        # So, if 4 tensors:
        # Case 1: ids, mask, ttti, label (include_difficulty=False in create_dataset)
        # Case 2: ids, mask, label, difficulty (no ttti from tokenizer)
        # We need a robust way to check. For now, assume if mode needs difficulty, it's Case 2.
        # And if not, it's Case 1 if 'token_type_ids' was found by create_dataset's sample tokenization.
        # This part is tricky due to ambiguity.
        # Let's assume create_dataset correctly determines if TTTI are produced by tokenizer.
        # If create_dataset included TTTI and we have 4 tensors, then difficulty must be missing.
        # If create_dataset did NOT include TTTI and we have 4 tensors, then difficulty must be present.

        # Let's try to infer based on what `create_dataset` would produce.
        # This is still a bit indirect. A more robust way would be to pass the structure.
        # For now, stick to len(batch) but be aware of its limitations.
        # The original code's logic:
        if mode in ['estimate', 'eval_estimate']:  # These modes REQUIRE difficulty
            # If 4 tensors and difficulty is required, means no TTTI
            has_token_type_ids = False  # Assumes ids, mask, label, diff
            has_difficulty_in_batch = True
        else:  # 'eval' mode, difficulty not strictly required by this function for its primary output
            # If 4 tensors, could be (ids, mask, ttti, label) OR (ids, mask, label, difficulty if val set has it)
            # This part of the original code was: has_token_type_ids = True (ids, mask, ttti, label)
            # This assumes if it's 4 and not estimating, it has TTTI and no difficulty in batch.
            # This might be problematic if val_dataset was created with difficulty but no TTTI.
            # Let's refine: check if the last element could be a difficulty tensor.
            # However, without knowing the structure from create_dataset, this is guesswork.
            # The original code's check was:
            # elif batch_len == 4: has_token_type_ids = True # Assume ids, mask, type_ids, label (no diff needed) unless estimating
            # This implies for 'eval' mode, a 4-tensor batch is (ids, mask, ttti, label)
            has_token_type_ids = True  # Default assumption for 4-tensor in eval
            has_difficulty_in_batch = False
            # A better check for this case might be needed if val sets can have difficulty and no ttti
            # For PUDF, dev_dataset is created with include_difficulty=True.
            # So for 'eval_estimate', if len is 4, it must be (ids, mask, label, difficulty).
            if mode == 'eval_estimate':  # Recorrect for eval_estimate
                has_token_type_ids = False
                has_difficulty_in_batch = True

    elif num_tensors_in_batch_source == 3:  # ids, mask, label
        has_token_type_ids = False
        has_difficulty_in_batch = False
    else:
        raise ValueError(f"Unexpected number of tensors ({num_tensors_in_batch_source}) from DataLoader for {mode}.")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=eval_desc, leave=False)):
        num_batches += 1
        # This batch unpacking logic needs to be extremely careful and match create_dataset's output order.
        # Order from create_dataset: input_ids, attention_mask, [token_type_ids], label, [difficulty]
        input_ids = batch[0].to(device, non_blocking=True)
        attention_mask = batch[1].to(device, non_blocking=True)
        current_idx = 2
        token_type_ids_val = None
        if has_token_type_ids:
            if len(batch) <= current_idx: raise IndexError(
                f"Batch too short for token_type_ids. Len: {len(batch)}, Idx: {current_idx}")
            token_type_ids_val = batch[current_idx].to(device, non_blocking=True);
            current_idx += 1

        if len(batch) <= current_idx: raise IndexError(
            f"Batch too short for labels. Len: {len(batch)}, Idx: {current_idx}")
        labels_val = batch[current_idx].to(device, non_blocking=True);
        current_idx += 1

        difficulty_tensor_val = None
        should_have_difficulty_for_mode = mode in ['estimate', 'eval_estimate']
        if has_difficulty_in_batch:  # If TensorDataset was created with difficulty
            if len(batch) <= current_idx:
                if should_have_difficulty_for_mode:  # Critical if needed for mode
                    raise IndexError(f"Batch too short for difficulty. Len: {len(batch)}, Idx: {current_idx}")
                # else, not critical for 'eval' mode, can be None
            else:
                difficulty_tensor_val = batch[current_idx]  # Stays on CPU if it came like that
        elif should_have_difficulty_for_mode:
            print(f"Warning: Mode '{mode}' expects difficulty, but batch structure indicates it's missing.")

        with torch.no_grad():
            try:
                # Pass labels to model for its internal loss calculation
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids_val,
                                labels=labels_val)
                logits = outputs.logits
                loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0,
                                                                                  device=device)  # Handle cases where model might not return loss
            except Exception as e:
                print(f"\nERROR during model forward pass in eval (batch {batch_idx}): {e}");
                print(f"Shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels_val.shape}");
                traceback.print_exc();
                continue

        preds_list.append(logits.detach().cpu().numpy())
        labels_list.append(labels_val.detach().cpu().numpy())
        if difficulty_tensor_val is not None and should_have_difficulty_for_mode:
            difficulties_list.append(difficulty_tensor_val.cpu().numpy())  # Ensure it's numpy and on CPU

        predictions = torch.argmax(logits, dim=-1)
        metric_computer.add_batch(predictions=predictions.detach().cpu(), references=labels_val.detach().cpu())
        val_loss += loss.item()

    preds_np = np.concatenate(preds_list) if preds_list else np.array([])
    out_label_ids_np = np.concatenate(labels_list) if labels_list else np.array([])
    all_difficulties_np = np.concatenate(difficulties_list) if difficulties_list else np.array(
        [])  # Ensure it's an array

    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0

    try:
        eval_score = metric_computer.compute() if num_batches > 0 else None
        validation_accuracy = eval_score['accuracy'] if eval_score else 0.0
    except Exception as e:
        print(f"Warning: metric_computer.compute() failed: {e}. Accuracy set to 0.0.");
        validation_accuracy = 0.0

    if mode == 'eval':
        return validation_accuracy, avg_val_loss

    # Theta Estimation part
    if not all_difficulties_np.size > 0 and should_have_difficulty_for_mode:  # Check if array has elements
        print("Warning: Cannot estimate theta - difficulty scores missing/not collected or empty.")
        default_theta, default_time = 0.0, 0.0
        return (default_theta, default_time) if mode == 'estimate' else (
        validation_accuracy, avg_val_loss, default_theta, default_time)

    if mode == 'estimate' or mode == 'eval_estimate':
        if not all_difficulties_np.size > 0:  # Should have been caught above, but as a safeguard
            print("No difficulties for theta estimation in mode '{mode}'. Returning defaults.")
            default_theta, default_time = 0.0, 0.0
            return (default_theta, default_time) if mode == 'estimate' else (
            validation_accuracy, avg_val_loss, default_theta, default_time)

        time_model_s = time.time()
        if len(all_difficulties_np) != len(out_label_ids_np):
            print(
                f"ERROR: Mismatch len difficulties ({len(all_difficulties_np)}) vs responses ({len(out_label_ids_np)}). Theta estimation skipped.")
            default_theta, default_time = 0.0, 0.0
            return (default_theta, default_time) if mode == 'estimate' else (
            validation_accuracy, avg_val_loss, default_theta, default_time)

        response_pattern_irt = [1 if p == c else -1 for p, c in zip(np.argmax(preds_np, axis=1), out_label_ids_np)]
        try:
            theta_hat_val = calculate_theta(all_difficulties_np, response_pattern_irt, num_obs=num_obs_theta)[0]
        except Exception as e:
            print(f"ERROR during theta calculation: {e}. Setting theta_hat to 0.0");
            traceback.print_exc();
            theta_hat_val = 0.0
        model_capacity_time_val = time.time() - time_model_s

        if mode == 'estimate':
            return theta_hat_val, model_capacity_time_val
        else:  # eval_estimate
            return validation_accuracy, avg_val_loss, theta_hat_val, model_capacity_time_val

    print(f"Warning: Unrecognized mode '{mode}' in evaluate_and_estimate. Returning eval metrics.");
    return validation_accuracy, avg_val_loss


# --- Main Training Function ---
def train(config, output_dir):
    """Main training loop for a given task using specified config."""
    print(f"\n===== Starting Training for Task: {config.task} =====")
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    use_amp = torch.cuda.is_available() and device.type == 'cuda'  # Ensure AMP only on CUDA
    print(f"Using AMP: {use_amp}")

    try:
        train_dataset_hf, dev_dataset_hf, test_dataset_hf = load_and_prepare_data(config.task, config.diff_dir,
                                                                                  config.cache_dir)
        train_size, val_size, test_size = len(train_dataset_hf), len(dev_dataset_hf), len(test_dataset_hf)
        print(f"Data sizes: Train={train_size}, Val={val_size}, Test={test_size}")
        assert train_size > 0 and val_size > 0, "Train or validation dataset is empty."
    except Exception as e:
        print(f"FATAL: Data loading failed: {e}");
        traceback.print_exc();
        return 0.0, 0.0

    num_labels = 3 if config.task.startswith("mnli") else 1 if config.task == "stsb" else 2
    model_name_resolved = getattr(config, 'model_name', 'microsoft/deberta-v3-base')  # Use config if provided
    print(f"Loading: {model_name_resolved} (Labels: {num_labels})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_resolved, cache_dir=config.cache_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_resolved, num_labels=num_labels,
                                                                   cache_dir=config.cache_dir)
        model.to(device)
    except Exception as e:
        print(f"FATAL: Loading model/tokenizer failed: {e}");
        traceback.print_exc();
        return 0.0, 0.0

    if hasattr(torch, 'compile') and use_amp and config.use_torch_compile:  # torch.compile benefits most on CUDA
        try:
            print("Attempting compilation..."); model = torch.compile(model); print("Compiled.")
        except Exception as e:
            print(f"Compile failed: {e}. Continuing.")
    elif config.use_torch_compile:
        print("Compile requested but not available/applicable.")

    task_output_dir = os.path.join(output_dir, config.task);
    best_model_dir = os.path.join(task_output_dir, "best_model")
    os.makedirs(task_output_dir, exist_ok=True);
    os.makedirs(best_model_dir, exist_ok=True)

    print("Creating/tokenizing datasets...")
    try:
        train_dataset_pytorch = create_dataset(train_dataset_hf, config.task, tokenizer, include_difficulty=True,
                                               partition_name="train")
        dev_dataset_pytorch = create_dataset(dev_dataset_hf, config.task, tokenizer, include_difficulty=True,
                                             partition_name="validation")
        test_dataset_pytorch = create_dataset(test_dataset_hf, config.task, tokenizer, include_difficulty=False,
                                              partition_name="test")
    except Exception as e:
        print(f"FATAL: Dataset creation failed: {e}");
        traceback.print_exc();
        return 0.0, 0.0

    print("Creating dataloaders...")
    # Use global batch_size defined at the top of the script
    train_dataloader_full = DataLoader(train_dataset_pytorch, batch_size=batch_size, shuffle=True,
                                       num_workers=config.num_workers, pin_memory=True, drop_last=False)
    dev_dataloader = DataLoader(dev_dataset_pytorch, batch_size=batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset_pytorch, batch_size=batch_size, shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True) if len(
        test_dataset_pytorch) > 0 else None

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.999))

    # Estimate steps based on the full train_dataloader for consistent scheduler setup
    num_training_steps_estimate = len(train_dataloader_full) * config.num_epochs
    num_warmup_steps = max(1, int(0.06 * num_training_steps_estimate)) if num_training_steps_estimate > 0 else 0
    print(f"Scheduler: Est steps={num_training_steps_estimate}, Warmup={num_warmup_steps}")
    if num_training_steps_estimate == 0: print(
        "Warning: Estimated training steps is 0. Scheduler might not work effectively.")

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max(1,
                                                                                                                     num_training_steps_estimate))  # num_training_steps must be > 0

    scaler = torch.amp.GradScaler(enabled=use_amp)  # Updated GradScaler

    best_accuracy = 0.0;
    early_stop_count = 0;
    patience = config.patience if hasattr(config, 'patience') else 5
    training_stats = [];
    total_pudf_overhead_time = 0.0
    prev_cap = -5.0;
    cur_cap = 0.0

    print(f"\nStarting training loop: Max {config.num_epochs} epochs, Patience {patience}...")
    start_train_loop_time = time.time();
    actual_epochs = 0

    for epoch in range(config.num_epochs):
        actual_epochs = epoch + 1;
        print(f"\n======== Epoch {epoch + 1} / {config.num_epochs} ========");
        epoch_start_time = time.time()
        avg_train_loss_epoch = 0.0;
        model_capacity_time_est_epoch = 0.0;
        filter_time_epoch = 0.0

        if config.strategy == 'theta':
            print("Estimating capacity...");
            est_theta_start_time = time.time()
            try:
                estimated_theta_hat, model_capacity_time_est_epoch = evaluate_and_estimate(model, dev_dataloader,
                                                                                           device,
                                                                                           num_obs_theta=config.num_obs,
                                                                                           mode='estimate')
                total_pudf_overhead_time += model_capacity_time_est_epoch
                print(f"Theta estimated: {estimated_theta_hat:.4f} ({model_capacity_time_est_epoch:.2f}s)")
                if estimated_theta_hat > prev_cap:
                    cur_cap = estimated_theta_hat
                else:
                    cur_cap += 0.1; print(
                        f"Theta ({estimated_theta_hat:.4f}) <= prev ({prev_cap:.4f}). Adjusted cur_cap: {cur_cap:.4f}")
            except Exception as e:
                print(f"Warning: Capacity estimation failed: {e}. Using cur_cap={cur_cap:.4f}"); traceback.print_exc()

        print(f"Filtering data (Capacity: {cur_cap:.4f})...");
        filter_time_s = time.time()
        try:
            # train_dataset_pytorch is the full PyTorch TensorDataset
            filtered_training_data_dict = get_epoch_training_data(train_dataset_pytorch, config, epoch, config.task,
                                                                  cur_cap, diffs_sorted_idx=None,
                                                                  lower_offset=config.lower_bound,
                                                                  upper_offset=config.upper_bound)
        except Exception as e:
            print(f"ERROR: get_epoch_training_data failed: {e}. Skipping epoch."); traceback.print_exc(); continue
        filter_time_epoch = time.time() - filter_time_s;
        total_pudf_overhead_time += filter_time_epoch

        num_epoch_examples = len(
            filtered_training_data_dict['labels'])  # Assuming 'labels' key from get_epoch_training_data
        if num_epoch_examples == 0:
            print("Warning: No data selected for training this epoch. Skipping training phase.");
            avg_train_loss_epoch = 0.0
        else:
            print(f"Selected {num_epoch_examples} examples ({filter_time_epoch:.2f}s filtering).")
            try:
                tensors_for_epoch = [filtered_training_data_dict['input_ids'],
                                     filtered_training_data_dict['attention_mask']]
                has_token_type_ids_epoch = filtered_training_data_dict.get('token_type_ids') is not None
                if has_token_type_ids_epoch: tensors_for_epoch.append(filtered_training_data_dict['token_type_ids'])
                tensors_for_epoch.extend(
                    [filtered_training_data_dict['labels'], filtered_training_data_dict['difficulty']])

                train_dataset_epoch_pytorch = TensorDataset(*tensors_for_epoch)
                # Use global batch_size for epoch dataloader, can be overridden if 'effective_batch_size' logic is preferred
                epoch_dataloader_batch_size = min(batch_size,
                                                  num_epoch_examples) if num_epoch_examples > 0 else batch_size
                train_dataloader_epoch = DataLoader(train_dataset_epoch_pytorch, shuffle=True,
                                                    batch_size=epoch_dataloader_batch_size,
                                                    num_workers=config.num_workers, pin_memory=True)
            except Exception as e:
                print(f"ERROR creating epoch dataloader: {e}. Skipping."); traceback.print_exc(); continue

            print(f"Training epoch {epoch + 1}...");
            model.train();
            epoch_train_loss_sum = 0.0;
            num_epoch_train_batches = 0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_dataloader_epoch, desc=f"Epoch {epoch + 1} Training", leave=False)
            for step, batch_train_epoch in enumerate(pbar):
                # Batch unpacking based on how train_dataset_epoch_pytorch was constructed
                # Order: ids, mask, [ttti], labels, difficulty
                input_ids_tr = batch_train_epoch[0].to(device, non_blocking=True)
                attention_mask_tr = batch_train_epoch[1].to(device, non_blocking=True)
                current_idx_tr = 2
                token_type_ids_tr = None
                if has_token_type_ids_epoch:  # Based on filtered_training_data_dict structure
                    token_type_ids_tr = batch_train_epoch[current_idx_tr].to(device, non_blocking=True);
                    current_idx_tr += 1
                labels_tr = batch_train_epoch[current_idx_tr].to(device, non_blocking=True)
                # difficulty is batch_train_epoch[current_idx_tr+1], not directly used in loss

                try:
                    with autocast(device_type=device.type, enabled=use_amp):  # device.type is 'cuda' or 'cpu'
                        outputs = model(input_ids=input_ids_tr, attention_mask=attention_mask_tr,
                                        token_type_ids=token_type_ids_tr, labels=labels_tr)
                        loss = outputs.loss
                    if torch.isnan(loss) or loss is None: print(
                        f"Warning: NaN/None loss step {step}. Skip."); optimizer.zero_grad(set_to_none=True); continue
                    scaler.scale(loss).backward();
                    scaler.unscale_(optimizer);
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer);
                    scaler.update();
                    scheduler.step();
                    optimizer.zero_grad(set_to_none=True)
                    epoch_train_loss_sum += loss.item();
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    num_epoch_train_batches += 1
                except Exception as e:
                    print(f"ERROR training step {step}: {e}"); optimizer.zero_grad(
                        set_to_none=True); traceback.print_exc(); print("Skip step."); continue
            avg_train_loss_epoch = epoch_train_loss_sum / num_epoch_train_batches if num_epoch_train_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} Avg Training Loss: {avg_train_loss_epoch:.4f}")

        print("Evaluating on validation set...");
        model_capacity_time_eval_epoch = 0.0
        try:
            dev_acc, val_loss, epoch_theta_estimate_val, model_capacity_time_eval_epoch = evaluate_and_estimate(model,
                                                                                                                dev_dataloader,
                                                                                                                device,
                                                                                                                num_obs_theta=config.num_obs,
                                                                                                                mode='eval_estimate')
            total_pudf_overhead_time += model_capacity_time_eval_epoch
            print(
                f"Epoch {epoch + 1} Validation: Acc={dev_acc:.4f}, Loss={val_loss:.4f}, Theta={epoch_theta_estimate_val:.4f} ({model_capacity_time_eval_epoch:.2f}s eval)")

            # Update prev_cap based on validation set's theta estimate for the next epoch's filtering decision
            if epoch_theta_estimate_val > prev_cap: prev_cap = epoch_theta_estimate_val; print(
                f"Updated prev_cap for filtering: {prev_cap:.4f}")
            # else: prev_cap remains, cur_cap for next epoch will be based on this prev_cap or nudged.

            training_stats.append(
                {'epoch': epoch + 1, 'Train Loss': avg_train_loss_epoch, 'Val Loss': val_loss, 'Val Acc': dev_acc,
                 'cur_cap_for_epoch_filter': cur_cap,  # The capacity used for THIS epoch's filtering
                 'theta_est_from_val': epoch_theta_estimate_val,  # Theta estimated ON val set THIS epoch
                 'pudf_theta_est_time_epoch': model_capacity_time_est_epoch,  # From explicit capacity estimation step
                 'pudf_filter_time_epoch': filter_time_epoch,
                 'pudf_val_eval_theta_time_epoch': model_capacity_time_eval_epoch,  # From val eval step
                 'n_train_epoch': num_epoch_examples})
            if dev_acc > best_accuracy:
                print(f"Val acc improved ({best_accuracy:.4f} --> {dev_acc:.4f}). Saving...");
                best_accuracy = dev_acc;
                early_stop_count = 0
                try:
                    model_to_save = getattr(model, '_orig_mod', model);
                    model_to_save.save_pretrained(best_model_dir);
                    tokenizer.save_pretrained(best_model_dir)
                    saved_files = os.listdir(best_model_dir)
                    if "pytorch_model.bin" in saved_files or "model.safetensors" in saved_files:
                        print("Model save confirmed.")
                    else:
                        print(
                            f"Warning: Save command executed, but weights file not found in {best_model_dir}. Files: {saved_files}")
                except Exception as e:
                    print(f"Warning: Error saving best model: {e}")
            else:
                early_stop_count += 1;
                print(
                    f"Val acc ({dev_acc:.4f}) vs best ({best_accuracy:.4f}). Early stop count: {early_stop_count}/{patience}")
                if early_stop_count >= patience: print("Early stopping."); break
        except Exception as e:
            print(f"ERROR during validation epoch {epoch + 1}: {e}"); traceback.print_exc(); print(
                "Stopping task."); break

        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        epoch_end_time = time.time();
        print(f"Epoch {epoch + 1} duration: {epoch_end_time - epoch_start_time:.2f}s.")

    end_train_loop_time = time.time();
    train_loop_duration = end_train_loop_time - start_train_loop_time
    print("\n--- Training Loop Finished ---");
    print(
        f"Actual epochs: {actual_epochs}, Total Time: {train_loop_duration:.2f}s, PUDF Overhead: {total_pudf_overhead_time:.2f}s, Best Val Acc: {best_accuracy:.4f}")

    resolved_model_name_for_filename = model_name_resolved.split('/')[-1]  # Use the actual model name for filename
    stats_filename_base = f"{resolved_model_name_for_filename}_{config.task}"
    training_stats_filename = os.path.join(task_output_dir, f"training_stats_{stats_filename_base}.json")
    print(f"Saving training stats: {training_stats_filename}");
    try:
        with open(training_stats_filename, "w") as f:
            json.dump(training_stats, f, indent=4, default=str)  # Use default=str for np.inf
    except Exception as e:
        print(f"Warning: Error saving training stats: {e}")

    print("\n--- Final Test Evaluation ---");
    test_acc, test_loss = 0.0, 0.0
    if test_dataloader is None:
        print("Test dataloader None. Skip."); test_acc, test_loss = -3.0, -3.0
    elif os.path.isdir(best_model_dir) and \
            (os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
             os.path.exists(os.path.join(best_model_dir, "model.safetensors"))):
        print(f"Loading best model: {best_model_dir}...");
        try:
            model_loaded = AutoModelForSequenceClassification.from_pretrained(
                best_model_dir)  # num_labels is in saved config
            tokenizer_loaded = AutoTokenizer.from_pretrained(best_model_dir, use_fast=True)  # Reload tokenizer too
            model_loaded.to(device)
            if config.use_torch_compile and hasattr(torch, 'compile') and use_amp: model_loaded = torch.compile(
                model_loaded)
            print("Evaluating on test set...");
            test_acc, test_loss = evaluate_and_estimate(model_loaded, test_dataloader, device, mode='eval')
            print(f'Final Test Acc: {test_acc:.4f}, Loss: {test_loss:.4f}');
            del model_loaded, tokenizer_loaded
        except Exception as e:
            print(f"ERROR during final test eval: {e}"); traceback.print_exc(); test_acc, test_loss = -1.0, -1.0
    else:
        print(
            f"Best model directory/weights not found: {best_model_dir}. Cannot test."); test_acc, test_loss = -2.0, -2.0

    final_stats_filename = os.path.join(task_output_dir, f"final_stats_{stats_filename_base}_Acc_{test_acc:.4f}.json")
    print(f"Saving final summary: {final_stats_filename}");

    # Prepare config for JSON serialization
    serializable_config = {}
    for k_conf, v_conf in config.__dict__.items():
        if v_conf == -np.inf:
            serializable_config[k_conf] = "-Infinity"
        elif isinstance(v_conf, (list, dict, str, int, float, bool, type(None))):
            serializable_config[k_conf] = v_conf
        else:
            serializable_config[k_conf] = str(v_conf)  # Fallback to string for other types

    final_summary = {
        "task": config.task, "model": resolved_model_name_for_filename, "strategy": config.strategy,
        "ordering": config.ordering,
        "lower_bound": config.lower_bound, "upper_bound": config.upper_bound,
        "min_train_length": config.min_train_length,
        "num_obs_theta": config.num_obs, "num_epochs_set": config.num_epochs, "num_epochs_run": actual_epochs,
        "best_validation_accuracy": best_accuracy, "final_test_accuracy": test_acc, "final_test_loss": test_loss,
        "total_training_loop_time_seconds": train_loop_duration,
        "total_pudf_overhead_seconds": total_pudf_overhead_time,
        "config": serializable_config,
        "training_stats_summary_path": training_stats_filename  # Link to the detailed stats file
    }
    try:
        with open(final_stats_filename, "w") as f:
            json.dump(final_summary, f, indent=4)  # default=str handled by serializable_config
    except Exception as e:
        print(f"Warning: Error saving final summary: {e}")

    print("Cleaning up task resources...")
    try:
        del model, tokenizer, optimizer, scheduler, scaler
    except NameError:
        pass
    try:
        del train_dataset_pytorch, dev_dataset_pytorch, test_dataset_pytorch
    except NameError:
        pass
    try:
        del train_dataloader_full, dev_dataloader, test_dataloader  # train_dataloader_epoch is in loop scope
    except NameError:
        pass
    try:
        del train_dataset_hf, dev_dataset_hf, test_dataset_hf
    except NameError:
        pass
    gc.collect();
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"===== Finished Task: {config.task} =====")
    return best_accuracy, test_acc


def run():
    config = types.SimpleNamespace()
    config.diff_dir = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/GLUE_output_difficulty_jsonlines"
    config.cache_dir = os.environ.get("TRANSFORMERS_CACHE", "./hf_cache/models")
    config.model_name = "microsoft/deberta-v3-base"  # Specify model name in config for flexibility
    config.num_epochs = 20
    config.learning_rate = 2e-5
    config.gpu = 0
    config.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))  # Use Slurm var or default
    config.strategy = 'theta'
    config.ordering = 'easiest'
    config.num_obs = 1000
    config.min_train_length = 1000
    config.lower_bound = -np.inf
    config.upper_bound = 0.0
    config.balanced = False
    config.use_length = False  # Must be present for get_epoch_training_data
    config.use_word_rarity = False  # Must be present
    config.use_torch_compile = False
    config.patience = 5  # Early stopping patience

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir_base = f"./glue_PUDF_{config.model_name.split('/')[-1]}_{config.strategy}_{run_timestamp}"
    print(f"Base output directory for this run: {output_dir_base}")
    os.makedirs(output_dir_base, exist_ok=True)

    results = {}
    for task_item in tqdm(GLUETASKS, desc="Overall Task Progress"):
        config.task = task_item  # Set current task in config
        print(f"\n\n>>>>>>>> Starting Task: {config.task} <<<<<<<<")
        print(f"--- Using Configuration ---")
        for key, value in config.__dict__.items():
            print(f"  {key}: {'-Infinity' if value == -np.inf else value}")
        print(f"-------------------------")
        task_start_time = time.time()
        try:
            top_dev, test_acc_val = train(config, output_dir_base)
            results[config.task] = {"best_dev_acc": top_dev, "test_acc": test_acc_val}
        except Exception as e:
            print(f"\n!FATAL ERROR Task {config.task}!");
            traceback.print_exc()
            results[config.task] = {"best_dev_acc": "FATAL_ERROR", "test_acc": "FATAL_ERROR"}
        task_end_time = time.time()
        print(f">>>>>>>> Finished Task: {config.task} in {task_end_time - task_start_time:.2f}s <<<<<<<<")

    print("\n\n===== Run Summary =====");
    print(f"Strategy: {config.strategy}");
    print(f"Output Dir: {output_dir_base}")
    print("Task Results (Best Dev Acc / Test Acc):")
    for task_key in sorted(results.keys()):
        res = results[task_key]
        dev_res_str = f"{res['best_dev_acc']:.4f}" if isinstance(res['best_dev_acc'], float) else str(
            res['best_dev_acc'])
        test_res_str = f"{res['test_acc']:.4f}" if isinstance(res['test_acc'], float) else str(res['test_acc'])
        print(f"  - {task_key}: {dev_res_str} / {test_res_str}")
    print("=====================")
    summary_file = os.path.join(output_dir_base, "run_summary.json")
    try:
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Summary saved: {summary_file}")
    except Exception as e:
        print(f"Warning: Failed to save summary: {e}")


if __name__ == '__main__':
    run()