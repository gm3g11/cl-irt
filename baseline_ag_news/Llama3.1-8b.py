import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from transformers import EarlyStoppingCallback
from transformers.optimization import Adafactor
from huggingface_hub import login, whoami

# Output directory
output_dir = "Llama3.1_8B_2"
os.makedirs(output_dir, exist_ok=True)

# Set environment variable for expandable memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set environment variables for Hugging Face cache
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid parallelism warnings

# Ensure cache directories exist
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

# Load the contemmcm/ag_news dataset
dataset = load_dataset("contemmcm/ag_news", cache_dir=os.environ["HF_DATASETS_CACHE"])

# Split the 'complete' split into train (80%) and temp (20%)
complete_dataset = dataset['complete']
train_temp_split = complete_dataset.train_test_split(test_size=0.2, seed=random_seed)
train_dataset = train_temp_split['train']
temp_dataset = train_temp_split['test']

# Split temp into validation (10%) and test (10%)
val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=random_seed)
validation_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Create a DatasetDict with the splits
dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

# Load the Llama3.1-8B tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Set padding token if not defined
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

# Rename 'label' to 'labels' for compatibility with Trainer
for split in tokenized_dataset:
    tokenized_dataset[split] = tokenized_dataset[split].rename_column("label", "labels")

# Determine the number of labels dynamically
label_feature = complete_dataset.features['label']
if hasattr(label_feature, 'names'):
    label_names = label_feature.names
    num_labels = len(label_names)
    print("Label names:", label_names)
else:
    raise ValueError("Label names not found in the dataset.")

# Load the Llama3.1-8B model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3.1-8B", num_labels=num_labels)

# Update model config with padding token ID
model.config.pad_token_id = tokenizer.pad_token_id

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=64,      # Adjust if memory issues occur
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=16,       # Effective batch size = 128 * 16
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    gradient_checkpointing=True,          # Memory optimization
    bf16=True,                           # Mixed precision for efficiency
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
)

# Define evaluation metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions = p.predictions
    labels = p.label_ids
    pred_labels = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=pred_labels, references=labels)

# Set up the trainer with Adafactor optimizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    optimizers=(Adafactor(model.parameters(), scale_parameter=False, relative_step=False, lr=2e-5), None),
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test results:", test_results)