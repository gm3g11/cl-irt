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
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
import json
import evaluate
from torch.amp import autocast, GradScaler

# Output directory
output_dir = "glue_baseline_Qwen2.5_7B_3"
os.makedirs(output_dir, exist_ok=True)

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Use float16 for mixed precision computations, model weights remain float32
torch.set_autocast_dtype("cuda", torch.float16)

# Set random seed for reproducibility
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)



# Model and tasks
models = ["Qwen/Qwen2.5-7B"]
GLUE_TASKS = [  "mrpc", "rte"]
# "rte","qqp", "sst2", "mnli",
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
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task in ["mrpc", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task == "qnli":
        return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task == "qqp":
        return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task == "sst2":
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)


# Custom batch size finder
def find_max_batch_size(model, tokenized_train_dataset, initial_batch_size=16,
                        max_memory=81559 * 1024 * 1024):  # 80GB in bytes
    device = torch.device("cuda")
    scaler = GradScaler('cuda')
    optimizer = Adafactor(model.parameters(), lr=2e-5, scale_parameter=False, relative_step=False, weight_decay=0.01)

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
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            batch = next(iter(train_dataloader))

            with autocast(device_type='cuda', dtype=torch.float16):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device).to(torch.long)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Check peak memory usage
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MiB
            print(f"Peak memory usage: {peak_memory:.2f} MiB")

            if peak_memory > 0.9 * max_memory:  # Stop if using >90% of GPU memory
                step_succeeded = False
                batch_size = batch_size // 2  # Back off to previous safe size
            else:
                batch_size *= 2  # Double the batch size and try again

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM detected at batch_size = {batch_size}")
                step_succeeded = False
                batch_size = batch_size // 2  # Back off to previous safe size
            else:
                raise e

        # Clean up
        torch.cuda.empty_cache()

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

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    for model_checkpoint in models:
        print("===========================================")
        print(f"Start test model -- {model_checkpoint} on task -- {task}")

        best_model_dir = f"{output_dir}/best_model_{model_checkpoint.replace('/', '_')}_{task}"
        gc.collect()
        torch.cuda.empty_cache()

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        num_labels = 3 if task.startswith("mnli") else 2
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            device_map="auto"  # Default torch_dtype is float32
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None

        # Tokenize datasets
        tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, task), batched=True)
        tokenized_val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, task), batched=True)
        tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, task), batched=True)

        tokenized_train_dataset.set_format("torch")
        tokenized_val_dataset.set_format("torch")
        tokenized_test_dataset.set_format("torch")

        # Find optimal batch size
        train_batch_size = find_max_batch_size(model, tokenized_train_dataset)
        gradient_accumulation_steps = max(1, 128 // train_batch_size)  # Target effective batch size of 128
        print(f"Adjusted gradient_accumulation_steps = {gradient_accumulation_steps}")

        # Create dataloaders
        train_dataloader = DataLoader(
            tokenized_train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0
        )
        val_dataloader = DataLoader(
            tokenized_val_dataset,
            batch_size=256,  # Fixed dev_batch_size
            shuffle=False,
            num_workers=0
        )
        test_dataloader = DataLoader(
            tokenized_test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0
        )

        device = torch.device("cuda" if GPU_avail else "cpu")
        print("Device:", device)

        # Optimizer and scheduler
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
            num_training_steps=(len(train_dataloader) // gradient_accumulation_steps) * 20  # 20 epochs
        )

        # Training setup
        best_accuracy = 0.0
        early_stop_count = 0
        patience = 3
        training_stats = []
        detailed_training_stats = []

        time_s = time.time()
        scaler = GradScaler('cuda')

        # Training loop
        for epoch in range(20):
            print(f"\n======== Epoch {epoch + 1} / 20 ========")
            print("Training...")
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(tqdm(train_dataloader)):
                if step % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()

                with autocast(device_type='cuda', dtype=torch.float16):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device).to(torch.long)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps
                    total_loss += loss.item() * gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
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
                    with autocast(device_type='cuda', dtype=torch.float16):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device).to(torch.long)
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        eval_metric.add_batch(predictions=predictions, references=labels)
                        val_loss += outputs.loss.item()

            val_loss /= len(val_dataloader)
            eval_score = eval_metric.compute()
            validation_accuracy = eval_score["accuracy"]
            print(f"Validation Accuracy: {validation_accuracy:.4f}")

            # Save stats
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

            # Early stopping and model saving
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                early_stop_count = 0
                print("Saving best model...")
                model.save_pretrained(best_model_dir)
                print("Model saved.")
                tokenizer.save_pretrained(best_model_dir)
                print("Tokenizer saved.")
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print("Early stopping triggered")
                    break
            gc.collect()
            torch.cuda.empty_cache()

        # Free memory and load best model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        time_e = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            best_model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Test phase
        print("\nRunning Test (on dev set)...")
        model.eval()
        eval_metric = evaluate.load("accuracy")

        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device).to(torch.long)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    eval_metric.add_batch(predictions=predictions, references=labels)

        eval_score = eval_metric.compute()
        test_accuracy = eval_score["accuracy"]
        print(f"Test Accuracy: {test_accuracy:.4f}")

        train_time = time_e - time_s
        actual_epochs = epoch + 1
        print(f"Total Training Time: {train_time:.2f} seconds")

        final_stats_filename = (
            f"{output_dir}/final_stats_{model_checkpoint.replace('/', '_')}_{task}_"
            f"{train_time:.2f}s_{actual_epochs}epochs_{test_accuracy:.4f}.json"
        )
        with open(final_stats_filename, "w") as f:
            json.dump(training_stats, f)