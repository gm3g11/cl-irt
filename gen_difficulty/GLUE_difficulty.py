import numpy as np
import time
import json
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from huggingface_hub import whoami

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Environment setup
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]
    print("Removed HF_TOKEN environment variable to use cached token.")

try:
    user_info = whoami()
    print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
except Exception as e:
    print(f"Error checking token: {e}")
    print("Continuing without token verification.")

# Define models, tasks, and parameters
models = [
    # "bert-base-uncased",
    # "roberta-base",
    # "albert-base-v2",
    # "distilbert-base-uncased",
    # "microsoft/deberta-base",
    # "gpt2",
    #
    #
    #
    # # "t5-base",
    # # "xlnet-base-cased",
    "electra-base-discriminator",
    # # "facebook/bart-base",
   "meta-llama/Meta-Llama-3.1-8B"
    # # "t5-3b"
]

GLUE_TASKS = ["mrpc", "qnli", "qqp","mnli","rte","sst2"]
task_max_lengths = {
    "mrpc": 72,
    # doesn't work fot t5-base
    "rte": 150,
    # "rte":64,
    "mnli": 72,
    "qqp": 56,
    "sst2": 32,
    "qnli": 80
}

model_batch_sizes = {
"t5": {"train": 8, "dev": 128},
    "t5-3b": {"train": 8, "dev": 16},
    "meta-llama/Meta-Llama-3.1-8B": {"train": 4, "dev": 128}
}

default_train_batch_size = 256
default_dev_batch_size = 256
epochs_list = [0, 1, 3, 5, 10]

# Mapping for model names in filenames
model_name_mapping = {
    "microsoft/deberta-base": "deberta",
    "bert-base-uncased": "bert",
    "roberta-base": "roberta",
    "albert-base-v2": "albert",
    "distilbert-base-uncased": "distilbert",
    "gpt2": "gpt2",
    "t5-base": "t5",
    "xlnet-base-cased": "xlnet",
    "electra-base-discriminator": "electra",
    "facebook/bart-base": "bart",
    "meta-llama/Meta-Llama-3.1-8B": "llama-3.1-8B",
    "t5-3b": "t5-3b"
}

# Tokenization function
def tokenize_function(examples, task, tokenizer):
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
    else:
        raise ValueError(f"Unsupported task: {task}")

# Model and tokenizer loading with padding token fix
def load_model_and_tokenizer(model_checkpoint, task):
    tokenizer_kwargs = {"use_auth_token": True} if "meta-llama" in model_checkpoint else {}
    model_kwargs = {"num_labels": 3 if task.startswith("mnli") else 2}
    model_kwargs.update({"use_auth_token": True} if "meta-llama" in model_checkpoint else {})

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, **tokenizer_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model_kwargs["pad_token_id"] = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, **model_kwargs)
    return model, tokenizer

# Main execution
for task in GLUE_TASKS:
    print(f"Task: {task}")
    dataset = load_dataset("glue", task)
    print(f"Dataset splits: {list(dataset.keys())}")

    for model_checkpoint in models:
        try:
            model, tokenizer = load_model_and_tokenizer(model_checkpoint, task)
        except Exception as e:
            print(f"Failed to load model {model_checkpoint}: {e}")
            continue

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, task, tokenizer),
            batched=True
        )
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        for num_epochs in epochs_list:
            subject_id = f"{model_checkpoint}_{num_epochs}"
            print(f"Processing subject: {subject_id}")

            # Set batch sizes
            train_batch_size = model_batch_sizes.get(model_checkpoint, {"train": default_train_batch_size})["train"]
            dev_batch_size = model_batch_sizes.get(model_checkpoint, {"dev": default_dev_batch_size})["dev"]

            # Create data loaders
            if task == "mnli":
                dev_split = "validation_matched"
                train_split = "train"
            else:
                dev_split = "validation"
                train_split = "train"

            dev_dataloader = DataLoader(
                tokenized_dataset[dev_split],
                batch_size=dev_batch_size,
                shuffle=True
            )
            train_dataloader = DataLoader(
                tokenized_dataset[train_split],
                batch_size=train_batch_size
            )

            # Move model to device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Fine-tuning
            if num_epochs > 0:
                # Use AdamW for all models
                optimizer = AdamW(model.parameters(), lr=5e-5)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=10,
                    num_training_steps=len(dev_dataloader) * num_epochs
                )
                time_s = time.time()
                for epoch in range(num_epochs):
                    model.train()
                    for batch in tqdm(dev_dataloader, desc=f"Fine-tuning Epoch {epoch + 1}/{num_epochs}"):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)
                        # Run in full precision (float32)
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                train_time = time.time() - time_s
            else:
                train_time = 0

            # Evaluation
            model.eval()
            logits_list = []
            responses = []
            for batch in tqdm(train_dataloader, desc="Evaluating on train set"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.no_grad():
                    # Run in full precision (float32)
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    logits = torch.nn.functional.softmax(logits, dim=-1)
                    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                    res = np.equal(pred, out_label_ids).astype(int)
                    logits_list.append(logits.detach().cpu().numpy())
                    responses.append(res)

            # Process results
            responses_arr = np.concatenate(responses)
            logits_arr = np.concatenate(logits_list, axis=0)
            accuracy = responses_arr.mean()
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Training time: {train_time:.2f} seconds")

            # Save results with mapped model name in specified directory
            short_model_name = model_name_mapping.get(model_checkpoint, model_checkpoint.replace("/", "_"))
            results_dir = f"./results/{task}"
            os.makedirs(results_dir, exist_ok=True)
            filename = os.path.join(results_dir, f"{short_model_name}_{task}_{train_time:.2f}_response_logits_Accuracy_{accuracy:.4f}_finetuning_{num_epochs}_epochs.json")
            data = {"logits": logits_arr.tolist(), "responses": responses_arr.tolist()}
            with open(filename, "w") as f:
                json.dump(data, f)
            print(f"Saved results to {filename}")