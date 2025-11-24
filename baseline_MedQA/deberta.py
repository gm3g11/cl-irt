import os
import datetime
import random
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForMultipleChoice,
    Trainer,
    TrainingArguments,
    DataCollatorForMultipleChoice,
    EarlyStoppingCallback
)
from evaluate import load as load_metric
import shutil
import glob
import traceback

# ----- Environment Setup -----
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
if not os.path.exists(os.path.dirname(HF_HOME)):
    print(f"Warning: Path {os.path.dirname(HF_HOME)} does not exist. Using default Hugging Face cache directory.")
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
print(f"Using random seed: {random_seed}")

# ----- Timestamp and Config -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_name = "microsoft/deberta-v3-base"
dataset_id = "GBaker/MedQA-USMLE-4-options"
max_length = 512
batch_size = 16
num_epochs = 20

# ----- Load Dataset -----
print(f"Loading dataset: {dataset_id}")
try:
    dataset = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])
    print("Raw dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# ----- Create Validation Split (if needed) -----
if 'validation' not in dataset:
    print("Validation split not found. Splitting train set into 80% train / 20% validation...")
    if 'train' not in dataset:
        print("Error: Original train split not found!")
        exit()
    split = dataset['train'].train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
    dataset = DatasetDict({
        'train': split['train'],
        'validation': split['test'],
        'test': dataset['test']
    })
    print("Train set split complete.")
else:
    print("Using predefined train, validation, and test splits.")
print("Final dataset splits:", dataset)

# ----- Load Tokenizer and Model -----
print(f"Loading tokenizer and model: {model_name}")
try:
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    model = DebertaV2ForMultipleChoice.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    print("Tokenizer and model loaded.")
except Exception as e:
    print(f"Error loading tokenizer or model: {e}")
    traceback.print_exc()
    exit()

# ----- Label Mapping & Preprocessing -----
answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
option_keys = ['A', 'B', 'C', 'D']
num_choices = len(option_keys)

print(f"Using MAX_LENGTH = {max_length}")

def preprocess_function(examples):
    num_examples = len(examples["question"])
    first_sentences = [[examples["question"][i]] * num_choices for i in range(num_examples)]
    second_sentences = []
    labels = []
    for i in range(num_examples):
        options_dict = examples["options"][i]
        answer_idx_key = examples["answer_idx"][i]
        if not isinstance(options_dict, dict):
            option_texts = ["[placeholder option]"] * num_choices
        else:
            option_texts = [options_dict.get(key, "[option unavailable]") for key in option_keys]
        second_sentences.append(option_texts)
        if answer_idx_key in answer_map:
            labels.append(answer_map[answer_idx_key])
        else:
            labels.append(-100)
    first_sentences_flat = [s for sublist in first_sentences for s in sublist]
    second_sentences_flat = [s for sublist in second_sentences for s in sublist]
    tokenized_examples = tokenizer(
        first_sentences_flat,
        second_sentences_flat,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    unflattened = {}
    for k, v in tokenized_examples.items():
        unflattened[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
    unflattened["labels"] = labels
    return unflattened

print("Tokenizing dataset...")
try:
    num_proc = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    print(f"Using {num_proc} processes for tokenization.")
    encoded_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names
    )
    print("Tokenization complete.")
    print("Columns in tokenized dataset:", encoded_dataset["train"].column_names)
except Exception as e:
    print(f"Error during tokenization: {e}")
    traceback.print_exc()
    exit()

# ----- Metric -----
print("Loading accuracy metric...")
accuracy_metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    labels = np.array(labels).astype(int)
    valid_indices = (labels != -100)
    valid_preds = predictions[valid_indices]
    valid_labels = labels[valid_indices]
    if len(valid_labels) == 0:
        return {"accuracy": 0.0}
    acc = accuracy_metric.compute(predictions=valid_preds, references=valid_labels)
    return {"accuracy": acc["accuracy"]}

# ----- Data Collator -----
print("Initializing Data Collator...")
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer, padding=True)

# ----- TrainingArguments -----
print(f"Setting up Training Arguments: BS={batch_size}, Epochs={num_epochs}")
run_name = f"deberta_v3_base_medqa_mc_len{max_length}_bs{batch_size}"

training_args = TrainingArguments(
    output_dir=f"./{run_name}_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 2,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir=f"./{run_name}_logs",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=None,  # Keep all checkpoints
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=random_seed,
)

# ----- Trainer -----
print("Initializing Trainer...")
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
except Exception as e:
    print(f"Error initializing Trainer: {e}")
    traceback.print_exc()
    exit()

# ----- Train -----
print("Starting training...")
try:
    train_result = trainer.train()
    print("Training finished.")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print("Best model loaded into trainer for evaluation.")

    # Save the best model
    best_model_path = os.path.join(training_args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model saved to {best_model_path}")
except Exception as e:
    print(f"Error during training: {e}")
    traceback.print_exc()

# ----- Evaluation -----
print("Evaluating on the test set...")
try:
    if 'test' in encoded_dataset:
        test_results = trainer.evaluate(eval_dataset=encoded_dataset["test"])
        print("Test results:", test_results)
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
    else:
        print("Test split not found. Skipping final test evaluation.")
except Exception as e:
    print(f"Error during evaluation: {e}")
    traceback.print_exc()

# ----- Cleanup -----
output_dir_to_clean = training_args.output_dir
print(f"Evaluation complete. Removing unnecessary files from {output_dir_to_clean}...")
try:
    files_to_remove = [
        "pytorch_model.bin", "model.safetensors", "config.json", "training_args.bin",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spm.model"
    ]
    for filename in files_to_remove:
        file_path = os.path.join(output_dir_to_clean, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    checkpoint_dirs = glob.glob(os.path.join(output_dir_to_clean, "checkpoint-*"))
    for checkpoint_dir in checkpoint_dirs:
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
    print("Cleanup complete. Best model is preserved in 'best_model' directory.")
except Exception as e:
    print(f"Error during cleanup: {e}")
    print("You may need to manually remove files/directories from:", output_dir_to_clean)

print(f"Script finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")