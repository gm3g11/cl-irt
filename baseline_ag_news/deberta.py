import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict  # Import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from transformers import EarlyStoppingCallback

# Set environment variables
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Added for safety, as per your preference

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

# Split the 'complete' split into 80% train, 10% validation, 10% test
complete_dataset = dataset['complete']
train_temp_split = complete_dataset.train_test_split(test_size=0.2, seed=random_seed)
train_dataset = train_temp_split['train']
temp_dataset = train_temp_split['test']
val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=random_seed)
validation_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Create a DatasetDict instead of a plain dictionary
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

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Rename 'label' to 'labels' for Trainer compatibility
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Load the model with the correct number of labels
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=num_labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./deberta_2",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=256,  # Adjust if GPU memory is insufficient
    per_device_eval_batch_size=256,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
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