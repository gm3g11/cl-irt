import os
import datetime
import random
import traceback
import sys
import json

import torch
import numpy as np

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import evaluate

# Environment Debug
print("--- Environment Debug ---")
print(f"Python Executable: {sys.executable}")
print(f"Torch Version: {torch.__version__}")
if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
import transformers

print(f"Transformers Version: {transformers.__version__}")
print("-------------------------\n")

# Configuration
model_id = "Qwen/Qwen2.5-7B"
dataset_id = "GBaker/MedQA-USMLE-4-options"
current_datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
output_dir = f"./qwen2.5_7b_instruct_medqa_qlora_optimized_{current_datetime_str}"
max_seq_length = 512 + 10
max_prompt_len_config = 512
per_device_train_bs = 2
per_device_eval_bs = 4
grad_accum_steps = 16
num_train_epochs = 5
learning_rate = 1e-4
weight_decay_train = 0.01
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
random_seed = 63

# Setup
os.makedirs(output_dir, exist_ok=True)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
print(f"Output directory: {output_dir}")
print(f"Using random seed: {random_seed}")

# Load Dataset
print(f"Loading dataset: {dataset_id}")
raw_dataset = load_dataset(dataset_id)
if "validation" not in raw_dataset:
    print("Validation split not found. Splitting train (90/10)...")
    if 'train' not in raw_dataset:
        exit("Error: Train split not found!")
    split = raw_dataset["train"].train_test_split(test_size=0.1, seed=random_seed, shuffle=True)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": raw_dataset["test"],
    })
else:
    dataset = raw_dataset
print("Dataset splits:", dataset)

# Load Tokenizer
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        print("Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        new_pad_token = "␂"
        if new_pad_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": new_pad_token})
            print(f"Added new pad_token: {new_pad_token}")
        else:
            tokenizer.pad_token = new_pad_token
            print(f"Set pad_token: {new_pad_token}")
print(
    f"Tokenizer loaded. Vocab size: {len(tokenizer)}. Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

# QLoRA Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load Model
print(f"Loading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
if len(tokenizer) > model.config.vocab_size:
    print(f"Resizing embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
if model.config.pad_token_id != tokenizer.pad_token_id:
    print(f"Updating model pad_token_id to {tokenizer.pad_token_id}")
    model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    target_modules=lora_target_modules, bias="none"
)
model = get_peft_model(model, peft_config)
print("QLoRA model prepared.")
model.print_trainable_parameters()

# Preprocessing
answer_map_keys = ["A", "B", "C", "D"]


def create_prompt_and_target_letter(example):
    question = example["question"].strip()
    options_dict = example["options"]
    answer_idx_key = example["answer_idx"]
    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in answer_map_keys:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option text not found]')}")
    else:
        for key_char in answer_map_keys:
            prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    prompt_text = "\n".join(prompt_parts)
    target_letter = answer_idx_key if answer_idx_key in answer_map_keys else ""
    return {"prompt": prompt_text, "target_letter": target_letter}


def preprocess_sft_format(examples):
    inputs = []
    labels_list = []
    for i in range(len(examples["question"])):
        processed = create_prompt_and_target_letter({
            "question": examples["question"][i],
            "options": examples["options"][i],
            "answer_idx": examples["answer_idx"][i]
        })
        prompt_text = processed["prompt"]
        target_letter = processed["target_letter"]

        tokenized_prompt = tokenizer(prompt_text, truncation=False, padding=False, add_special_tokens=False)
        tokenized_target = tokenizer(target_letter, truncation=False, padding=False, add_special_tokens=False)

        prompt_input_ids = tokenized_prompt.input_ids
        target_input_ids = tokenized_target.input_ids

        input_ids_concat = []
        if tokenizer.bos_token_id is not None and getattr(tokenizer, 'add_bos_token', True):
            input_ids_concat.append(tokenizer.bos_token_id)
        input_ids_concat.extend(prompt_input_ids)
        input_ids_concat.extend(target_input_ids)
        if tokenizer.eos_token_id is not None:
            input_ids_concat.append(tokenizer.eos_token_id)

        len_prompt_and_bos = (1 if tokenizer.bos_token_id is not None and getattr(tokenizer, 'add_bos_token',
                                                                                  True) else 0) + len(prompt_input_ids)
        labels_concat = ([-100] * len_prompt_and_bos) + target_input_ids
        if tokenizer.eos_token_id is not None:
            labels_concat.append(tokenizer.eos_token_id)

        final_input_ids = input_ids_concat[:max_seq_length]
        final_labels = labels_concat[:max_seq_length]
        if len(final_labels) < len(final_input_ids):
            final_labels.extend([-100] * (len(final_input_ids) - len(final_labels)))

        inputs.append(final_input_ids)
        labels_list.append(final_labels)
    return {"input_ids": inputs, "labels": labels_list}


print(f"Tokenizing dataset. Max length: {max_seq_length}")
tokenized_dataset = dataset.map(preprocess_sft_format, batched=True, num_proc=max(1, os.cpu_count() // 2))
keep_cols = ["input_ids", "labels"]
tokenized_dataset = tokenized_dataset.remove_columns(
    [c for c in tokenized_dataset["train"].column_names if c not in keep_cols])
tokenized_dataset.set_format(type="torch", columns=keep_cols)
print(f"Dataset formatted. Columns: {tokenized_dataset['train'].column_names}")

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding="longest"
)


# Metrics (Unused by Trainer due to prediction_loss_only)
def compute_metrics(eval_pred):
    print("WARNING: compute_metrics limited by prediction_loss_only=True")
    eval_loss = eval_pred.metrics.get("eval_loss", -1.0) if eval_pred.metrics else -1.0
    return {"accuracy": 0.0, "eval_loss": eval_loss}


# Training Arguments
print(f"Setting up Training Args. Train BS: {per_device_train_bs}, Eval BS: {per_device_eval_bs}")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=per_device_train_bs,
    per_device_eval_batch_size=per_device_eval_bs,
    gradient_accumulation_steps=grad_accum_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay_train,
    fp16=False,
    bf16=True,
    logging_strategy="epoch",
    logging_steps=max(1, 50 // grad_accum_steps),
    eval_strategy="epoch",
    save_strategy="epoch",
    prediction_loss_only=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to=[],
    remove_unused_columns=False,
    seed=random_seed,
)

# Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Training
print(f"Starting training for {model_id}...")
try:
    trainer.train()
    final_adapter_path = os.path.join(output_dir, "final_qlora_adapter")
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"✅ Training complete. Adapter saved to {final_adapter_path}")
except Exception:
    print("💥 Training failed.")
    traceback.print_exc()
    sys.exit(1)

# Manual Evaluation with Batch Processing
print("\nPerforming manual evaluation on test set...")
model.eval()
test_examples = raw_dataset["test"]
num_test_examples = len(test_examples)
eval_batch_size = 16  # Adjust based on your GPU memory
num_batches = (num_test_examples + eval_batch_size - 1) // eval_batch_size

all_predicted_letters = []
all_true_answers = []

device = model.device

# Clear memory before starting evaluation
torch.cuda.empty_cache()

for batch_idx in range(num_batches):
    start_idx = batch_idx * eval_batch_size
    end_idx = min(start_idx + eval_batch_size, num_test_examples)
    batch_examples = [test_examples[i] for i in range(start_idx, end_idx)]

    prompts = [create_prompt_and_target_letter(ex)["prompt"] for ex in batch_examples]
    true_answers = [create_prompt_and_target_letter(ex)["target_letter"] for ex in batch_examples]

    tokenized_prompts = tokenizer(prompts, padding=True, truncation=True, max_length=max_prompt_len_config,
                                  return_tensors="pt")
    input_ids = tokenized_prompts["input_ids"].to(device)
    attention_mask = tokenized_prompts["attention_mask"].to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1,
                                       pad_token_id=tokenizer.pad_token_id)

    generated_token_ids = generated_ids[:, -1]
    letter_token_ids = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_map_keys}
    predicted_letters = [next((letter for letter, tid in letter_token_ids.items() if tid == token_id.item()), None) for
                         token_id in generated_token_ids]

    all_predicted_letters.extend(predicted_letters)
    all_true_answers.extend(true_answers)

    # Clear memory after each batch
    del input_ids, attention_mask, generated_ids, generated_token_ids
    torch.cuda.empty_cache()

# Compute accuracy
correct = sum(1 for pred, true in zip(all_predicted_letters, all_true_answers) if pred == true and pred is not None)
invalid = sum(1 for pred in all_predicted_letters if pred is None)
total = len(all_true_answers)
accuracy = correct / total if total > 0 else 0.0

print(f"Manual evaluation results:")
print(f"Total examples: {total}")
print(f"Correct predictions: {correct}")
print(f"Invalid predictions: {invalid}")
print(f"Accuracy: {accuracy:.4f}")

detailed_results = [
    {
        "prompt": create_prompt_and_target_letter(test_examples[i])["prompt"],
        "predicted_letter": all_predicted_letters[i] if all_predicted_letters[i] is not None else "INVALID",
        "true_letter": all_true_answers[i],
        "is_correct": all_predicted_letters[i] == all_true_answers[i] and all_predicted_letters[i] is not None
    }
    for i in range(total)
]
eval_results_file = os.path.join(output_dir, "manual_evaluation_results.json")
with open(eval_results_file, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {eval_results_file}")

print(f"\nScript finished at {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
print(f"Output and results in: {output_dir}")