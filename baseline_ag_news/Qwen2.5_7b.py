import os
import random
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig  # Added for consistency if needed, though not strictly used in your Qwen example for model loading
)
import evaluate
from transformers import EarlyStoppingCallback
from transformers.optimization import Adafactor  # For the Adafactor optimizer
from huggingface_hub import whoami  # To check login status if needed

# --- Environment Setup ---
# Set environment variable for expandable memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set environment variables for Hugging Face cache (as in your example)
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")  # More specific for hub downloads
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]  # Aligning with current practices
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure cache directories exist
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Output directory for this specific script
MODEL_SPECIFIC_OUTPUT_DIR = "./qwen2.5_7b_agnews_baseline"  # Changed to be more descriptive
os.makedirs(MODEL_SPECIFIC_OUTPUT_DIR, exist_ok=True)
LOGGING_DIR = os.path.join(MODEL_SPECIFIC_OUTPUT_DIR, "logs")
os.makedirs(LOGGING_DIR, exist_ok=True)

# Verify Hugging Face login (optional but good for gated models)
try:
    user_info = whoami()
    print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
except Exception as e:
    print(f"Hugging Face login check warning: {e}. Ensure CLI login if model is gated.")

# --- Reproducibility ---
RANDOM_SEED = 63
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Configuration ---
MODEL_CHECKPOINT = "Qwen/Qwen2.5-7B"
DATASET_ID = "contemmcm/ag_news"
MAX_LENGTH = 128

# Batch size considerations (from your Qwen script's comment and LLM practices)
# Your comment suggested physical BS 2 to get effective BS 32 with 16 accum steps.
# Let's make the physical batch size a configurable variable.
# On an 80GB GPU, you might fit slightly more.
PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Adjust based on your GPU VRAM (e.g., 1, 2, or 4)
PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE = 4  # Can usually be larger
GRADIENT_ACCUMULATION_STEPS = 16  # To reach an effective batch size

# Training Hyperparameters (from your Qwen script)
LEARNING_RATE_TRAINING_ARGS = 1e-5  # LR for TrainingArguments (scheduler might use this)
LEARNING_RATE_ADAFACTOR = 2e-5  # Specific LR for Adafactor
NUM_TRAIN_EPOCHS = 15
WEIGHT_DECAY = 0.01
USE_BF16 = True  # If your hardware supports it (Ampere/Hopper GPUs)
USE_GRADIENT_CHECKPOINTING = True

# --- Load Dataset ---
print(f"Loading dataset: {DATASET_ID}")
raw_dataset_full = load_dataset(DATASET_ID, cache_dir=os.environ["HF_DATASETS_CACHE"])

print("Splitting 'complete' dataset (80% train, 10% validation, 10% test)...")
if 'complete' not in raw_dataset_full:
    raise ValueError(f"Dataset {DATASET_ID} does not have a 'complete' split.")

complete_data = raw_dataset_full['complete']
# Shuffle before splitting is a good practice
complete_data = complete_data.shuffle(seed=RANDOM_SEED)

# Standard 80/10/10 split
train_temp_split = complete_data.train_test_split(test_size=0.2, seed=RANDOM_SEED)
train_dataset_split = train_temp_split['train']
temp_dataset_split = train_temp_split['test']
val_test_split = temp_dataset_split.train_test_split(test_size=0.5, seed=RANDOM_SEED)
validation_dataset_split = val_test_split['train']
test_dataset_split = val_test_split['test']

dataset_splits = DatasetDict({
    'train': train_dataset_split,
    'validation': validation_dataset_split,
    'test': test_dataset_split
})
print("Dataset splits created:")
for split_name, ds in dataset_splits.items():
    print(f"  {split_name}: {len(ds)} samples")

# Determine num_labels dynamically (will be 10 for unfiltered contemmcm/ag_news)
label_feature_info = dataset_splits['train'].features['label']
if hasattr(label_feature_info, 'num_classes'):
    num_labels_for_model = label_feature_info.num_classes
    print(f"Number of labels dynamically determined: {num_labels_for_model}")
    if hasattr(label_feature_info, 'names'):
        print(f"Label names: {label_feature_info.names}")
else:
    raise ValueError("Could not determine number of labels from dataset.")

# --- Tokenizer ---
print(f"Loading tokenizer for: {MODEL_CHECKPOINT}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer: pad_token set to eos_token ('{tokenizer.eos_token}')")
    else:  # Should not happen for most models like Qwen
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Tokenizer: Added a new [PAD] token.")


# tokenizer.pad_token_id is now implicitly set if pad_token was changed.

# --- Tokenization Function ---
def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing datasets...")
# Remove text columns that might be present in contemmcm/ag_news
columns_to_remove_on_tokenize = ["text"]
for col in ["html", "title", "content", "description"]:
    if col in dataset_splits['train'].column_names:
        columns_to_remove_on_tokenize.append(col)
columns_to_remove_on_tokenize = list(set(columns_to_remove_on_tokenize))  # Unique

tokenized_datasets = dataset_splits.map(
    tokenize_data,
    batched=True,
    remove_columns=[col for col in columns_to_remove_on_tokenize if col in dataset_splits['train'].column_names]
    # Ensure columns exist
).rename_column("label", "labels")  # For Trainer compatibility
print("Tokenization complete.")

# --- Model ---
print(f"Loading model: {MODEL_CHECKPOINT} for {num_labels_for_model}-class classification")
model_config = AutoConfig.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels_for_model)

# Ensure model_config.pad_token_id is aligned with tokenizer
if tokenizer.pad_token_id is not None:
    if model_config.pad_token_id != tokenizer.pad_token_id:
        model_config.pad_token_id = tokenizer.pad_token_id
        print(f"Model config: pad_token_id updated to {tokenizer.pad_token_id}")
else:
    print("Warning: Tokenizer pad_token_id is None. Model might not behave as expected if padding is used.")

model_load_kwargs = {"config": model_config}
if USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    model_load_kwargs["torch_dtype"] = torch.bfloat16
    print("Model will be loaded with torch_dtype=torch.bfloat16")
elif USE_BF16:  # User wants BF16 but it's not supported
    print(
        "Warning: BF16 set to True but not supported by hardware/PyTorch. Model will load in FP32 or FP16 if applicable.")
    USE_BF16 = False  # Turn it off for TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, **model_load_kwargs)

# Resize embeddings if a new pad token '[PAD]' was programmatically added
if tokenizer.pad_token == '[PAD]' and hasattr(model, 'resize_token_embeddings') and \
        len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
    print("Resizing model token embeddings due to newly added [PAD] token.")
    model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing on the model instance if desired
if USE_GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
    print("Explicitly enabling gradient checkpointing on the model.")
    gc_enable_kwargs = {}
    if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
        try:
            import inspect

            sig = inspect.signature(model.gradient_checkpointing_enable)
            if 'gradient_checkpointing_kwargs' in sig.parameters:
                gc_enable_kwargs['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
        except (AttributeError, ValueError):
            pass
    model.gradient_checkpointing_enable(**gc_enable_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # Necessary for models using grad checkpointing
        print("Set model.config.use_cache = False")
elif USE_GRADIENT_CHECKPOINTING:
    print(f"Warning: Model {MODEL_CHECKPOINT} does not have 'gradient_checkpointing_enable' or it's misconfigured.")

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=MODEL_SPECIFIC_OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",  # To enable load_best_model_at_end
    learning_rate=LEARNING_RATE_TRAINING_ARGS,  # Will be overridden by Adafactor's LR if Adafactor is used
    per_device_train_batch_size=PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PHYSICAL_PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=LOGGING_DIR,
    logging_steps=100,  # As in your Qwen script
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,  # Argument for Trainer
    gradient_checkpointing_kwargs={"use_reentrant": False} if USE_GRADIENT_CHECKPOINTING and packaging.version.parse(
        torch.__version__) >= packaging.version.parse("2.0.0") else None,
    bf16=USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=(not (
                USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())) and torch.cuda.is_available(),
    # Use FP16 if BF16 is not used/available but CUDA is
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Your Qwen script used eval_accuracy
    greater_is_better=True,
    save_total_limit=2,  # Save fewer checkpoints
    report_to="none",  # Disable external reporting like wandb
    dataloader_num_workers=min(4, os.cpu_count() if os.cpu_count() else 1)  # Add some workers for dataloader
)
print(f"Effective training batch size per device: {PHYSICAL_PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

# --- Metrics ---
accuracy_metric = evaluate.load("accuracy")


def compute_metrics_callback(p):  # Renamed to avoid conflict if script is imported
    predictions_logits = p.predictions
    # Handle case where predictions might be a tuple (e.g., logits, past_key_values)
    if isinstance(predictions_logits, tuple):
        predictions_logits = predictions_logits[0]

    pred_labels = np.argmax(predictions_logits, axis=1)
    return accuracy_metric.compute(predictions=pred_labels, references=p.label_ids)


# --- Trainer ---
# Setup Adafactor optimizer
adafactor_optimizer = Adafactor(
    model.parameters(),
    lr=LEARNING_RATE_ADAFACTOR,  # Use the specific LR for Adafactor
    scale_parameter=False,  # As in your Qwen script
    relative_step=False  # As in your Qwen script
)
# Trainer will create its own scheduler compatible with Adafactor based on TrainingArguments
optimizers_tuple = (adafactor_optimizer, None)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics_callback,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # As in your Qwen script
    tokenizer=tokenizer,  # Good practice to pass tokenizer
    optimizers=optimizers_tuple
)

# --- Train ---
print("Starting training...")
try:
    train_output = trainer.train()
    print("Training finished.")
    print(f"Train output: {train_output}")
    trainer.save_model(os.path.join(MODEL_SPECIFIC_OUTPUT_DIR, "best_model_after_train"))  # Save the best model
    print(f"Best model saved to {MODEL_SPECIFIC_OUTPUT_DIR}/best_model_after_train")

    # --- Evaluate on Test Set ---
    print("Evaluating on the test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test set evaluation results:", test_results)

    # Save test results
    results_file_path = os.path.join(MODEL_SPECIFIC_OUTPUT_DIR, "test_set_results.json")
    with open(results_file_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {results_file_path}")

except Exception as e_train:
    print(f"An error occurred during training or evaluation: {e_train}")
    import traceback

    traceback.print_exc()
finally:
    print("Script finished.")
    # Optional: Clean up GPU memory if needed, though script end usually handles this.
    # del model, trainer, tokenized_datasets, dataset_splits
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()