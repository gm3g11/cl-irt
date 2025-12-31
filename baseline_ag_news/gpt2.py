import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from transformers import EarlyStoppingCallback

# Set environment variables
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid parallelism warnings

# Ensure the directories exist
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Set random seed for reproducibility
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Load the contemmcm/ag_news dataset
dataset = load_dataset("contemmcm/ag_news", cache_dir=os.environ["HF_DATASETS_CACHE"])

# Split the 'complete' split into train (80%), validation (10%), test (10%)
complete_dataset = dataset['complete']
train_temp_split = complete_dataset.train_test_split(test_size=0.2, seed=random_seed)
train_dataset = train_temp_split['train']
temp_dataset = train_temp_split['test']
val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=random_seed)
validation_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})
print("Dataset splits:", dataset)

# Determine the number of labels dynamically
label_feature = dataset['train'].features['label']
num_labels = label_feature.num_classes
print(f"Number of labels: {num_labels}")

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)

# Set padding token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Rename 'label' to 'labels' for Trainer compatibility
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Load the GPT-2 model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels)

# Update model config with padding token ID
model.config.pad_token_id = tokenizer.pad_token_id

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_ag_news_2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=256,  # Adjusted for GPT-2's memory needs
    per_device_eval_batch_size=256,
    num_train_epochs=20,  # Reduced for faster experimentation
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    metric_for_best_model="eval_accuracy",
    load_best_model_at_end=True,
)

# Define metrics
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(p):
    predictions = p.predictions.argmax(axis=1)
    labels = p.label_ids
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Set up the trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test results:", test_results)