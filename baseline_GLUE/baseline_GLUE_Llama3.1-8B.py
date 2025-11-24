import numpy as np
import random
import gc
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import transformers
import time
import os
import datetime
import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
import json
import evaluate
# from torch.amp import autocast, GradScaler   # <--- Removed for FP32
from huggingface_hub import login, whoami

# Output directory
output_dir = "glue_baseline_Llama3.1_8B_3"
os.makedirs(output_dir, exist_ok=True)

# Set environment variable for expandable memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Environment setup
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Remove HF_TOKEN to use cached token if set
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]
    print("Removed HF_TOKEN environment variable to use cached token.")

# Verify Hugging Face login
try:
    user_info = whoami()
    print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
except Exception as e:
    print(f"Error checking token: {e}")
    print("Continuing without token verification.")

# Set random seed for reproducibility
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Model and tasks
models = ["meta-llama/Meta-Llama-3.1-8B"]
GLUE_TASKS = [ "qqp", "mrpc",  "sst2", "mnli"]
# "rte",
GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available:", GPU_avail)

# Task-specific max lengths
task_max_lengths = {
    "mrpc": 72,
    "rte": 150,
    "mnli": 72,
    "qqp": 56,
    "sst2": 32,
    "qnli": 80
}

# Tokenization function
def tokenize_function(examples, task):
    max_length = task_max_lengths[task]
    if task == "mnli":
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    elif task in ["mrpc", "rte"]:
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    elif task == "qnli":
        return tokenizer(
            examples["question"],
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    elif task == "qqp":
        return tokenizer(
            examples["question1"],
            examples["question2"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    elif task == "sst2":
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

# Custom batch size finder with safety margin (FP32 now, no GradScaler)
def find_max_batch_size(
    model,
    tokenized_train_dataset,
    initial_batch_size=8,
    max_memory=81559 * 1024 * 1024  # ~80GB in bytes
):
    device = torch.device("cuda")
    # No GradScaler for FP32
    optimizer = Adafactor(
        model.parameters(),
        lr=2e-5,
        scale_parameter=False,
        relative_step=False,
        weight_decay=0.01
    )

    batch_size = initial_batch_size
    step_succeeded = True

    while step_succeeded:
        print(f"Trying batch_size = {batch_size}")
        train_dataloader = DataLoader(
            tokenized_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        try:
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            batch = next(iter(train_dataloader))

            # Full precision forward pass
            input_ids = batch["input_ids"].to(device).to(torch.long)
            attention_mask = batch["attention_mask"].to(device).to(torch.long)
            labels = batch["label"].to(device).to(torch.long)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backprop in FP32
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            print(f"Peak memory usage: {peak_memory:.2f} MiB")

            if peak_memory > 0.8 * max_memory:  # 80% threshold for safety
                step_succeeded = False
                batch_size = batch_size // 2
            else:
                batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM detected at batch_size = {batch_size}")
                step_succeeded = False
                batch_size = batch_size // 2
            else:
                raise e

        torch.cuda.empty_cache()
        gc.collect()

    if batch_size < 1:
        batch_size = 1
    print(f"Selected train_batch_size = {batch_size}")
    return batch_size

# Training loop for each GLUE task
for task in GLUE_TASKS:
    print(f"\nTask: {task}")
    dataset = load_dataset("glue", task)
    print(f"Dataset Keys: {dataset.keys()}")

    # Split dataset
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=random_seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    if task == "mnli":
        test_dataset = dataset["validation_matched"]
    else:
        test_dataset = dataset["validation"]

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    for model_checkpoint in models:
        print("===========================================")
        print(f"Start test model -- {model_checkpoint} on task -- {task}")

        best_model_dir = f"{output_dir}/best_model_{model_checkpoint.replace('/', '_')}_{task}"
        gc.collect()
        torch.cuda.empty_cache()

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        num_labels = 3 if task.startswith("mnli") else 2

        # Load model in FP32
        print(f"Loading model {model_checkpoint} in float32 on GPU 0...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            torch_dtype=torch.float32,  # Keep model parameters in FP32
            use_auth_token=True
        ).to("cuda")

        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None

        # Tokenize datasets
        tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, task), batched=True)
        tokenized_val_dataset = val_dataset.map(lambda x: tokenize_function(x, task), batched=True)
        tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function(x, task), batched=True)

        tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # Find optimal batch size in FP32
        train_batch_size = find_max_batch_size(model, tokenized_train_dataset)
        # train_batch_size = 2
        gradient_accumulation_steps = max(1, 128 // train_batch_size)
        print(f"Adjusted gradient_accumulation_steps = {gradient_accumulation_steps}")

        # Create dataloaders
        train_dataloader = DataLoader(
            tokenized_train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0
        )
        # Use smaller eval/test batch size to reduce memory usage
        val_dataloader = DataLoader(
            tokenized_val_dataset,
            batch_size=32,  # <--- Reduced from 256 to 32
            shuffle=False,
            num_workers=0
        )
        test_dataloader = DataLoader(
            tokenized_test_dataset,
            batch_size=32,  # <--- Reduced from 256 to 32
            shuffle=False,
            num_workers=0
        )

        device = torch.device("cuda") if GPU_avail else torch.device("cpu")
        print("Device:", device)

        optimizer = Adafactor(
            model.parameters(),
            lr=2e-5,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=0.01
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=50,
            num_training_steps=(len(train_dataloader) // gradient_accumulation_steps) * 20
        )

        best_accuracy = 0.0
        early_stop_count = 0
        patience = 3
        training_stats = []
        detailed_training_stats = []

        time_s = time.time()

        # -------------------------------
        # Training loop in FP32
        # -------------------------------
        for epoch in range(20):
            print(f"\n======== Epoch {epoch + 1} / 20 ========")
            print("Training...")
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(tqdm(train_dataloader)):
                if step % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()

                # Full precision forward pass
                input_ids = batch["input_ids"].to(device).to(torch.long)
                attention_mask = batch["attention_mask"].to(device).to(torch.long)
                labels = batch["label"].to(device).to(torch.long)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                total_loss += loss.item() * gradient_accumulation_steps

                # Backprop in FP32
                loss.backward()

                if ((step + 1) % gradient_accumulation_steps == 0) or ((step + 1) == len(train_dataloader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                if step % 10 == 0:
                    detailed_training_stats.append({
                        "epoch": epoch + 1,
                        "step": step,
                        "Training Loss": loss.item() * gradient_accumulation_steps
                    })

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # Validation
            print("\nRunning Validation...")
            model.eval()
            val_loss = 0.0
            eval_metric = evaluate.load("accuracy")

            for batch in val_dataloader:
                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device).to(torch.long)
                    attention_mask = batch["attention_mask"].to(device).to(torch.long)
                    labels = batch["label"].to(device).to(torch.long)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    eval_metric.add_batch(predictions=predictions, references=labels)
                    val_loss += outputs.loss.item()

            val_loss /= len(val_dataloader)
            validation_accuracy = eval_metric.compute()["accuracy"]
            print(f"Validation Accuracy: {validation_accuracy:.4f}")

            training_stats.append({
                "epoch": epoch + 1,
                "Training Loss": avg_train_loss,
                "Validation Loss": val_loss,
                "Validation Accuracy": validation_accuracy
            })

            training_stats_filename = f"{output_dir}/training_stats_{model_checkpoint.replace('/', '_')}_{task}.json"
            with open(training_stats_filename, "w") as f:
                json.dump(training_stats, f)

            detailed_training_stats_filename = f"{output_dir}/detailed_training_stats_{model_checkpoint.replace('/', '_')}_{task}.json"
            with open(detailed_training_stats_filename, "w") as f:
                json.dump(detailed_training_stats, f)

            # Track best model
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                early_stop_count = 0
                print("Saving best model...")
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                print("Model and tokenizer saved.")
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print("Early stopping triggered")
                    break

            torch.cuda.empty_cache()
            gc.collect()

        time_e = time.time()

        # Free up memory from previous GPU operations
        del model
        del optimizer
        del scheduler
        del train_dataloader
        del val_dataloader
        gc.collect()
        torch.cuda.empty_cache()

        # Switch to CPU for testing to avoid GPU memory issues
        device = torch.device("cpu")
        print("Switched device to CPU for testing:", device)

        # Load the best model for testing, using FP32 and moving it to CPU
        model = AutoModelForSequenceClassification.from_pretrained(
            best_model_dir,
            torch_dtype=torch.float32  # Maintain FP32 consistency with training
        ).to(device)

        print("\nRunning Test (on dev set)...")
        model.eval()  # Set model to evaluation mode
        eval_metric = evaluate.load("accuracy")  # Load accuracy metric

        # Test loop: process batches from test_dataloader on CPU
        for batch in tqdm(test_dataloader):
            with torch.no_grad():  # Disable gradient computation for inference
                # Explicitly move inputs to CPU (ensures consistency even if DataLoader defaults to CPU)
                input_ids = batch["input_ids"].to(device).to(torch.long)
                attention_mask = batch["attention_mask"].to(device).to(torch.long)
                labels = batch["label"].to(device).to(torch.long)

                # Run model inference on CPU
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)  # Get predicted classes

                # Add batch results to accuracy metric
                eval_metric.add_batch(predictions=predictions, references=labels)

        # Compute and display test accuracy
        test_accuracy = eval_metric.compute()["accuracy"]
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Calculate and display total training time
        train_time = time_e - time_s
        actual_epochs = epoch + 1
        print(f"Total Training Time: {train_time:.2f} seconds")

        # Save training statistics to a JSON file
        final_stats_filename = (
            f"{output_dir}/final_stats_{model_checkpoint.replace('/', '_')}_{task}_"
            f"{train_time:.2f}s_{actual_epochs}epochs_{test_accuracy:.4f}.json"
        )
        with open(final_stats_filename, "w") as f:
            json.dump(training_stats, f)