import os
import datetime
import random
import numpy as np
import torch
import json  # For saving results
from tqdm import tqdm  # For progress bars in custom evaluation

from datasets import load_dataset, DatasetDict
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForMultipleChoice,
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
    # EarlyStoppingCallback, # Will be removed
)
# from evaluate import load as load_metric # We'll calculate accuracy manually or use numpy
import shutil
import glob
import traceback

# ----- Environment Setup -----
HF_HOME_SPECIFIED = HF_HOME
if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(HF_HOME_SPECIFIED):
    HF_HOME = HF_HOME_SPECIFIED
else:
    print(f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' does not exist. Using default.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ----- Random Seed -----
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
print(f"Using random seed: {random_seed}")

# ----- Timestamp and Config -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_checkpoint_name = "microsoft/deberta-v3-base"  # Renamed for clarity
dataset_id = "GBaker/MedQA-USMLE-4-options"
max_length = 512
per_device_train_batch_size = 16  # From baseline, effectively batch_size for training
per_device_eval_batch_size_custom = 32  # For custom evaluation loop
num_epochs_total = 10  # Train for 10 epochs to get all desired checkpoints

# Output Directories
current_datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
base_output_dir = f"./deberta_v3_base_medqa_run_{current_datetime_str}"
RESULTS_OUTPUT_DIR = os.path.join(base_output_dir, "Deberta_MedQA_results")
training_checkpoints_dir = os.path.join(base_output_dir, "training_checkpoints")

os.makedirs(base_output_dir, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(training_checkpoints_dir, exist_ok=True)

print(f"Base output directory: {base_output_dir}")
print(f"Results JSON directory: {RESULTS_OUTPUT_DIR}")
print(f"Training checkpoints directory: {training_checkpoints_dir}")

# ----- Load Dataset -----
print(f"Loading dataset: {dataset_id}")
try:
    raw_dataset_full = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])
    print("Raw dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    traceback.print_exc()
    exit()

# We need 'train' and 'test' splits. The baseline created a validation split.
# For this new setup, we don't need a validation split for the Trainer.
# We'll use raw_dataset_full["test"] for training and raw_dataset_full["train"] for custom evaluation.
if 'train' not in raw_dataset_full or 'test' not in raw_dataset_full:
    print(f"Error: Dataset {dataset_id} must contain 'train' and 'test' splits.")
    exit()

dataset_for_sft_training_raw = raw_dataset_full["test"]
dataset_for_custom_evaluation_raw = raw_dataset_full["train"]

print(f"Using {len(dataset_for_sft_training_raw)} examples from 'test' split for SFT.")
print(f"Using {len(dataset_for_custom_evaluation_raw)} examples from 'train' split for custom evaluation.")

# ----- Load Tokenizer -----
print(f"Loading tokenizer: {model_checkpoint_name}")
try:
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_checkpoint_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    print("Tokenizer loaded.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    traceback.print_exc()
    exit()

# ----- Label Mapping & Preprocessing -----
ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}  # Renamed for clarity
OPTION_KEYS = ['A', 'B', 'C', 'D']  # Renamed for clarity
NUM_CHOICES = len(OPTION_KEYS)

print(f"Using MAX_LENGTH = {max_length} for tokenization.")


def preprocess_mc_function(examples):
    num_examples_in_batch = len(examples["question"])
    # Each question is repeated NUM_CHOICES times
    first_sentences = [[examples["question"][i]] * NUM_CHOICES for i in range(num_examples_in_batch)]
    second_sentences = []  # Options for each question
    labels = []  # Single integer label per question

    for i in range(num_examples_in_batch):
        options_dict = examples["options"][i]
        answer_idx_key = examples["answer_idx"][i].strip().upper()  # Ensure consistent key format

        # Prepare option texts
        if not isinstance(options_dict, dict):
            # Fallback if options are not a dict (e.g., malformed data)
            current_option_texts = ["[Error: Invalid options format]"] * NUM_CHOICES
        else:
            current_option_texts = [options_dict.get(key, "[Option text unavailable]") for key in OPTION_KEYS]
        second_sentences.append(current_option_texts)

        # Determine label
        if answer_idx_key in ANSWER_MAP:
            labels.append(ANSWER_MAP[answer_idx_key])
        else:
            # This case should ideally be filtered out beforehand or handled
            print(f"Warning: Unmappable answer_idx '{answer_idx_key}' found. Assigning label -100.")
            labels.append(-100)  # Indicates an invalid/unusable label for training/evaluation

    # Flatten lists for tokenizer
    first_sentences_flat = [s for sublist in first_sentences for s in sublist]
    second_sentences_flat = [s for sublist in second_sentences for s in sublist]

    tokenized_examples = tokenizer(
        first_sentences_flat,
        second_sentences_flat,
        max_length=max_length,
        truncation=True,  # Truncate if Q+A pair is too long
        padding=False  # Padding will be handled by DataCollator
    )

    # Unflatten the tokenized outputs (input_ids, attention_mask, etc.)
    unflattened_tokenized = {}
    for k, v in tokenized_examples.items():
        unflattened_tokenized[k] = [v[i: i + NUM_CHOICES] for i in range(0, len(v), NUM_CHOICES)]

    unflattened_tokenized["labels"] = labels  # Add the processed integer labels
    return unflattened_tokenized


print("Tokenizing SFT dataset ('test' split of raw data)...")
try:
    num_proc_tokenizer = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Using {num_proc_tokenizer} processes for SFT tokenization.")
    # Original columns will be removed by TrainingArguments(remove_unused_columns=True)
    # or we can explicitly remove them here.
    # The `map` function will return 'input_ids', 'attention_mask', 'token_type_ids' (for DeBERTa), 'labels'.
    tokenized_sft_dataset = dataset_for_sft_training_raw.map(
        preprocess_mc_function,
        batched=True,
        num_proc=num_proc_tokenizer,
        remove_columns=dataset_for_sft_training_raw.column_names  # Remove original string columns
    )
    print("SFT dataset tokenization complete.")
    print("Columns in tokenized SFT dataset:", tokenized_sft_dataset.column_names)
except Exception as e:
    print(f"Error during SFT tokenization: {e}")
    traceback.print_exc()
    exit()

# ----- Data Collator -----
print("Initializing Data Collator for Multiple Choice...")
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding="longest")


# ----- Evaluation Function (New) -----
def evaluate_deberta_model(model_to_eval, current_tokenizer, eval_dataset_raw, epoch, fine_tuning_duration):
    print(f"\nStarting custom evaluation for epoch {epoch}...")
    model_to_eval.eval()
    model_to_eval.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Ensure model on correct device

    # Tokenize the raw evaluation dataset
    print(f"Tokenizing {len(eval_dataset_raw)} raw evaluation examples for epoch {epoch}...")
    num_proc_eval_tok = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    tokenized_eval_data = eval_dataset_raw.map(
        preprocess_mc_function,
        batched=True,
        num_proc=num_proc_eval_tok,
        remove_columns=eval_dataset_raw.column_names,
        desc=f"Tokenizing eval data for epoch {epoch}"
    )

    all_logits_probs_list = []  # To store [P(A), P(B), P(C), P(D)]
    all_responses_correctness_list = []
    all_true_labels = []  # Store original integer labels for accuracy calculation

    # Manual batching for prediction
    for i in tqdm(range(0, len(tokenized_eval_data), per_device_eval_batch_size_custom),
                  desc=f"Predicting epoch {epoch} batches"):
        batch_data_list = [tokenized_eval_data[j] for j in
                           range(i, min(i + per_device_eval_batch_size_custom, len(tokenized_eval_data)))]

        # Collate batch manually or use DataCollator
        # Features will be lists of lists, need to prepare for model input
        # DataCollatorForMultipleChoice expects a list of dicts, where each dict is one example.
        # The output of .map already gives us dicts with 'input_ids', 'attention_mask', 'labels' etc.
        # where 'input_ids' is (num_choices, seq_len)

        collated_batch = data_collator(batch_data_list)

        input_ids = collated_batch["input_ids"].to(model_to_eval.device)
        attention_mask = collated_batch["attention_mask"].to(model_to_eval.device)
        if "token_type_ids" in collated_batch:  # DeBERTa uses token_type_ids
            token_type_ids = collated_batch["token_type_ids"].to(model_to_eval.device)
        else:
            token_type_ids = None

        batch_labels = collated_batch["labels"]  # These are on CPU, as integers
        all_true_labels.extend(batch_labels.numpy())

        with torch.no_grad():
            if token_type_ids is not None:
                outputs = model_to_eval(input_ids=input_ids, attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
            else:
                outputs = model_to_eval(input_ids=input_ids, attention_mask=attention_mask)

            batch_logits = outputs.logits  # Shape: (batch_size_eval, NUM_CHOICES)

        # Apply softmax to get probabilities
        batch_probs = torch.softmax(batch_logits, dim=-1).cpu().numpy()
        all_logits_probs_list.extend(batch_probs.tolist())

        # Get predictions
        batch_predictions_indices = np.argmax(batch_logits.cpu().numpy(), axis=1)

        for pred_idx, true_label_idx in zip(batch_predictions_indices, batch_labels.numpy()):
            if true_label_idx != -100:  # Consider only valid labels
                all_responses_correctness_list.append(int(pred_idx == true_label_idx))
            # If label is -100, we can't determine correctness.
            # For overall accuracy, we typically only count valid labels.

    # Calculate accuracy only on valid labels
    valid_true_labels = np.array([l for l in all_true_labels if l != -100])
    # `all_responses_correctness_list` already considers only valid labels based on its construction logic.
    # However, if we want overall accuracy from `all_logits_probs_list` and `valid_true_labels`:
    if len(valid_true_labels) > 0:
        # Re-derive predictions for valid labels to ensure alignment
        valid_preds_for_accuracy = []
        current_valid_idx = 0
        for i_pred_probs in range(len(all_logits_probs_list)):
            if all_true_labels[i_pred_probs] != -100:
                valid_preds_for_accuracy.append(np.argmax(all_logits_probs_list[i_pred_probs]))

        accuracy = np.mean(np.array(valid_preds_for_accuracy) == valid_true_labels) if len(
            valid_true_labels) > 0 else 0.0
        # Simpler: accuracy = np.mean(all_responses_correctness_list) if all_responses_correctness_list else 0.0
    else:
        accuracy = 0.0

    # Filename and Output Data Formatting
    model_name_part = model_checkpoint_name.split('/')[-1]  # e.g., "deberta-v3-base"
    dataset_name_part_short = dataset_id.split('/')[-1].split('-')[0]  # "MedQA"
    task_name_for_file = f"{dataset_name_part_short}_deberta"

    output_filename = f"{model_name_part}_{task_name_for_file}_{fine_tuning_duration:.2f}_Acc_{accuracy:.4f}_epochs_{epoch}.json"
    output_filepath = os.path.join(RESULTS_OUTPUT_DIR, output_filename)

    data_to_save = {
        "logits": all_logits_probs_list,  # List of [P(A), P(B), P(C), P(D)]
        "responses": all_responses_correctness_list  # List of 0 or 1 for valid labeled examples
    }
    with open(output_filepath, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved custom evaluation results for epoch {epoch} to {output_filepath}")
    print(f"Epoch {epoch} custom evaluation accuracy: {accuracy:.4f}")
    return accuracy


# ----- Main Script Flow -----
if __name__ == "__main__":
    # 1. Evaluate Pre-trained Model (Epoch 0)
    print("\n--- Evaluating Pre-trained Model (Epoch 0) ---")
    model_epoch0 = DebertaV2ForMultipleChoice.from_pretrained(
        model_checkpoint_name,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    evaluate_deberta_model(model_epoch0, tokenizer, dataset_for_custom_evaluation_raw, epoch=0,
                           fine_tuning_duration=0.0)
    del model_epoch0
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 2. Training
    print("\n--- Starting SFT Training (on 'test' split) for DeBERTa Model ---")

    # Load model for training
    model_for_training = DebertaV2ForMultipleChoice.from_pretrained(
        model_checkpoint_name,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )

    training_args = TrainingArguments(
        output_dir=training_checkpoints_dir,  # Save checkpoints here
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,  # No Trainer internal evaluation
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,  # Adjust if needed, baseline had 1 (batch_size was 16)
        num_train_epochs=num_epochs_total,
        learning_rate=2e-5,  # From baseline
        weight_decay=0.01,  # From baseline
        # logging_dir=f"./{base_output_dir}/{run_name}_logs", # run_name not defined here, use base_output_dir
        logging_dir=os.path.join(base_output_dir, "training_logs"),
        logging_strategy="steps",
        logging_steps=50,  # From baseline
        save_strategy="epoch",
        save_total_limit=None,  # Save all epoch checkpoints
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=random_seed,
        remove_unused_columns=True,  # Trainer will remove columns not used by model's forward pass
        # load_best_model_at_end=False, # Not using Trainer's best model logic
        # metric_for_best_model=None,
    )

    trainer = Trainer(
        model=model_for_training,
        args=training_args,
        train_dataset=tokenized_sft_dataset,  # This is tokenized dataset_for_sft_training_raw
        eval_dataset=None,  # No validation set for Trainer's internal eval loop
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # Not needed for trainer internal eval loop
        # callbacks=None, # No EarlyStopping
    )

    print(f"Starting SFT training for {num_epochs_total} epochs...")
    try:
        train_result = trainer.train()
        total_training_time = train_result.metrics.get("train_runtime", train_result.metrics.get("total_flos",
                                                                                                 0.0))  # Flos as fallback if runtime missing
        print("SFT Training finished.")
        trainer.log_metrics("train", train_result.metrics)  # Log final training metrics
        trainer.save_metrics("train", os.path.join(base_output_dir, "train_results.json"))
        trainer.save_state()  # Save trainer state

        # No need to save "best_model" separately, epoch checkpoints are saved.
        # final_model_path = os.path.join(training_checkpoints_dir, f"final_model_epoch_{num_epochs_total}")
        # trainer.save_model(final_model_path) # This saves the final state
        # print(f"Final model state for epoch {num_epochs_total} potentially saved by Trainer's last step or via checkpointing.")

    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        # sys.exit(1) # Optional: exit if training fails

    # Record total training time for reporting in JSONs
    # If train_result is not available due to error, set to 0
    if 'train_result' not in locals() or train_result is None:
        total_training_time = 0.0
        print("Warning: train_result not available, total_training_time set to 0.")
    else:
        total_training_time = train_result.metrics.get("train_runtime", 0.0)

    del model_for_training, trainer  # Clear VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Evaluate Saved Checkpoints
    print("\n--- Evaluating Specific Epoch Checkpoints (on 'train' split) ---")
    desired_epochs_to_eval = [1, 3, 5, 10]

    for desired_epoch in desired_epochs_to_eval:
        found_checkpoint_for_epoch = False
        potential_ckpt_path = None

        # Trainer saves checkpoints like "checkpoint-X" where X is global steps.
        # We need to map epoch to step or find the checkpoint dir for that epoch.
        # Inspecting trainer_state.json in each checkpoint folder is robust.
        if os.path.exists(training_checkpoints_dir):
            ckpt_folders = sorted([
                os.path.join(training_checkpoints_dir, d)
                for d in os.listdir(training_checkpoints_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_checkpoints_dir, d))
            ])

            for ckpt_path_candidate in ckpt_folders:
                trainer_state_file = os.path.join(ckpt_path_candidate, "trainer_state.json")
                if os.path.exists(trainer_state_file):
                    with open(trainer_state_file, "r") as f:
                        state = json.load(f)
                    # state["epoch"] is float, round it for comparison
                    if round(state.get("epoch", 0.0)) == desired_epoch:
                        potential_ckpt_path = ckpt_path_candidate
                        found_checkpoint_for_epoch = True
                        break  # Found checkpoint for this desired_epoch

        if found_checkpoint_for_epoch and potential_ckpt_path:
            print(f"\nLoading checkpoint for epoch {desired_epoch} from: {potential_ckpt_path}")
            model_for_eval_ckpt = DebertaV2ForMultipleChoice.from_pretrained(potential_ckpt_path)

            evaluate_deberta_model(
                model_for_eval_ckpt, tokenizer, dataset_for_custom_evaluation_raw,
                epoch=desired_epoch, fine_tuning_duration=total_training_time
            )
            del model_for_eval_ckpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(
                f"Warning: Could not find a suitable checkpoint for epoch {desired_epoch} in {training_checkpoints_dir}")

    # Cleanup (Optional, based on baseline script's behavior)
    # The baseline script had aggressive cleanup. For this experiment, keeping checkpoints might be desired.
    # If cleanup is needed, it should be done carefully *after* all evaluations.
    # For now, I'll comment out the aggressive cleanup to ensure checkpoints are available.
    # print(f"\nEvaluation complete. Checkpoints are preserved in {training_checkpoints_dir}.")

    print(f"\nScript finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs in: {base_output_dir}")