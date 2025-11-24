# ----- Imports -----
import sys
import os
import datetime
import random
import traceback
import json
from tqdm import tqdm
import re

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
    PeftModel,
)
import evaluate

# ----- Configuration -----
model_id = "meta-llama/Meta-Llama-3.1-8B"
dataset_id = "GBaker/MedQA-USMLE-4-options"
output_dir = "./qlora_medqa_letter_target_llama3_8b_2"

# Sequence lengths
max_seq_length = 512 + 10
max_prompt_len_config = 512
max_target_len_config = 5

# Training hyperparameters
per_device_train_bs = 8
per_device_eval_bs_trainer = 16
grad_accum_steps = 2
num_train_epochs = 5
early_stopping_patience_train = 3
learning_rate = 2e-4
weight_decay_train = 0.01

# LoRA parameters
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05

random_seed = 63
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation configuration
EVAL_SPLIT_CUSTOM_EVAL = "test"
MAX_NEW_TOKENS_GEN_CUSTOM = 1  # Generate only one token for evaluation
TEMPERATURE_CUSTOM_EVAL = 0.1
TOP_P_CUSTOM_EVAL = 0.9
DO_SAMPLE_CUSTOM_EVAL = False
EVAL_BATCH_SIZE_CUSTOM = 16

ANSWER_MAP_KEYS = ["A", "B", "C", "D"]


# ----- Preprocessing function for Training -----
def preprocess_sft_letter_target(examples, tokenizer_ref):
    """
    Preprocesses examples for supervised fine-tuning by tokenizing prompts and single-letter targets.

    Args:
        examples: Dataset examples containing 'question', 'options', and 'answer_idx'.
        tokenizer_ref: The tokenizer instance to use for encoding.

    Returns:
        Dictionary with 'input_ids' and 'labels' for training.
    """
    inputs_tokenized_batch = []
    labels_tokenized_batch = []

    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        options_dict = examples["options"][i]
        answer_idx_key = examples["answer_idx"][i]

        prompt_parts = [f"Question: {question}\n\nOptions:"]
        if isinstance(options_dict, dict):
            for key_char in ANSWER_MAP_KEYS:
                prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option text not found]')}")
        else:
            for key_char in ANSWER_MAP_KEYS:
                prompt_parts.append(f"{key_char}) [Invalid options format]")
        prompt_parts.append("\nAnswer:")
        prompt_text = "\n".join(prompt_parts)

        target_letter = answer_idx_key if answer_idx_key in ANSWER_MAP_KEYS else ""
        if not target_letter:
            print(
                f"Warning: Invalid answer_idx_key '{answer_idx_key}' for question '{question[:30]}...'. Target will be empty.")

        tokenized_prompt = tokenizer_ref(prompt_text, truncation=True, max_length=max_prompt_len_config, padding=False,
                                         add_special_tokens=True)
        tokenized_target = tokenizer_ref(target_letter, truncation=True, max_length=max_target_len_config,
                                         padding=False, add_special_tokens=False)

        prompt_input_ids = tokenized_prompt.input_ids
        target_input_ids = tokenized_target.input_ids

        input_ids = prompt_input_ids + target_input_ids
        if tokenizer_ref.eos_token_id is not None:
            input_ids.append(tokenizer_ref.eos_token_id)

        labels = ([-100] * len(prompt_input_ids)) + target_input_ids
        if tokenizer_ref.eos_token_id is not None:
            labels.append(tokenizer_ref.eos_token_id)

        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        elif len(input_ids) < len(labels):
            labels = labels[:len(input_ids)]

        inputs_tokenized_batch.append(input_ids)
        labels_tokenized_batch.append(labels)

    return {"input_ids": inputs_tokenized_batch, "labels": labels_tokenized_batch}


# ----- Helper: Prompt Creation for Custom Evaluation -----
def create_custom_evaluation_prompt(example):
    """
    Creates a prompt for evaluation from a dataset example.

    Args:
        example: A single example with 'question' and 'options'.

    Returns:
        Formatted prompt string.
    """
    question = example["question"].strip()
    options_dict = example["options"]
    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option text not found]')}")
    else:
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


# ----- Main Script Execution -----
if __name__ == "__main__":
    # Set up output directory and random seeds
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"Output directory: {output_dir}")
    print(f"Using device: {DEVICE}")
    print(f"Random seed: {random_seed}")

    # Load and filter dataset
    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id)

    # Filter to keep only examples with valid answer_idx (A, B, C, D)
    valid_answers = ['A', 'B', 'C', 'D']
    dataset = dataset.filter(lambda ex: ex["answer_idx"].strip().upper() in valid_answers)

    # Ensure dataset has train, validation, and test splits
    if "validation" not in dataset or "test" not in dataset:
        print("Validation or test split not found. Splitting train -> 80% train, 10% val, 10% test...")
        if 'train' not in dataset:
            sys.exit("Error: Original 'train' split not found in dataset!")
        train_temp_split = dataset['train'].train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
        val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=random_seed, shuffle=True)
        dataset = DatasetDict({
            "train": train_temp_split["train"],
            "validation": val_test_split["train"],
            "test": val_test_split["test"],
        })
    else:
        print("Using predefined train, validation, and test splits.")
    print("Final dataset splits:", dataset)

    # Load and configure tokenizer
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="left", trust_remote_code=True,
    )
    original_vocab_size = len(tokenizer)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        new_pad_token_str = "␂"
        if new_pad_token_str not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"pad_token": new_pad_token_str})
            print(f"Added NEW special pad_token: '{new_pad_token_str}', ID: {tokenizer.pad_token_id}")
        else:
            tokenizer.pad_token = new_pad_token_str
            print(f"Set pad_token to EXISTING token: '{new_pad_token_str}', ID: {tokenizer.pad_token_id}")
    elif tokenizer.pad_token != "␂":
        print(f"Tokenizer already has a pad_token: '{tokenizer.pad_token}'. Using it.")

    print(
        f"Tokenizer: Original vocab size: {original_vocab_size}, Current: {len(tokenizer)}. Pad: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

    # Define letter tokens for evaluation mapping
    letter_tokens = {letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ANSWER_MAP_KEYS}

    # =================================================================================
    #                                TRAINING PHASE
    # =================================================================================
    print("\n--- STARTING TRAINING PHASE ---")
    print("Configuring BitsAndBytes for QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {model_id} for training...")
    model_train = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    # Sync tokenizer and model configurations
    if len(tokenizer) > model_train.config.vocab_size:
        print(f"Resizing model token embeddings from {model_train.config.vocab_size} to {len(tokenizer)}")
        model_train.resize_token_embeddings(len(tokenizer))
    if model_train.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Syncing model pad_token_id to tokenizer's: {tokenizer.pad_token_id}")
        model_train.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for QLoRA training
    model_train = prepare_model_for_kbit_training(model_train)
    lora_target_modules_train = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config_train = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=lora_target_modules_train, bias="none"
    )
    model_train = get_peft_model(model_train, peft_config_train)
    print("QLoRA Causal LM model prepared for training.")
    model_train.print_trainable_parameters()

    # Tokenize dataset for training
    print(f"Tokenizing dataset for training (target: single letter). Max sequence length: {max_seq_length}")
    tokenized_dataset = dataset.map(
        lambda ex: preprocess_sft_letter_target(ex, tokenizer),
        batched=True, num_proc=max(1, os.cpu_count() // 2)
    )
    print("Training tokenization complete.")

    # Format dataset for training
    keep_cols_train = ["input_ids", "labels"]
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset["train"].column_names if c not in keep_cols_train]
    )
    tokenized_dataset.set_format(type="torch", columns=keep_cols_train)
    print(f"Training dataset formatted. Columns: {tokenized_dataset['train'].column_names}")

    # Define data collator
    data_collator_train = DataCollatorForSeq2Seq(
        tokenizer, model=model_train, label_pad_token_id=-100, padding="longest"
    )


    # Define metrics computation for trainer
    def compute_metrics_for_trainer(eval_pred):
        eval_loss = eval_pred.metrics.get("eval_loss", -1.0) if hasattr(eval_pred,
                                                                        'metrics') and eval_pred.metrics else -1.0
        return {"loss": eval_loss}


    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir, overwrite_output_dir=True, do_train=True, do_eval=True,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=per_device_eval_bs_trainer,
        gradient_accumulation_steps=grad_accum_steps, num_train_epochs=num_train_epochs,
        learning_rate=learning_rate, weight_decay=weight_decay_train, bf16=True,
        logging_strategy="steps", logging_steps=25,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        prediction_loss_only=True,
        load_best_model_at_end=True, metric_for_best_model="loss",
        greater_is_better=False, report_to=[], seed=random_seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model_train, args=training_args,
        train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer, data_collator=data_collator_train,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_train)],
    )

    # Train the model
    print(f"Starting QLoRA Causal LM fine-tuning for {model_id}...")
    final_adapter_path = os.path.join(output_dir, "final_qlora_adapter")
    try:
        train_result = trainer.train()
        trainer.save_model(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        print(f"✅ Training complete. QLoRA adapter saved to {final_adapter_path}")
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    except Exception as e:
        print(f"💥 Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Clean up training resources
    del model_train, trainer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Training model resources released.")

    # =================================================================================
    #                                CUSTOM EVALUATION PHASE
    # =================================================================================
    print("\n\n--- STARTING CUSTOM EVALUATION PHASE ---")
    adapter_path_for_eval = final_adapter_path
    eval_results_file = os.path.join(output_dir, f"evaluation_results_{EVAL_SPLIT_CUSTOM_EVAL}.json")

    # Verify adapter path
    if not os.path.exists(adapter_path_for_eval) or not os.path.isdir(adapter_path_for_eval):
        print(f"ERROR: Trained adapter path '{adapter_path_for_eval}' not found. Cannot run custom evaluation.")
        sys.exit(1)

    # Load base model for evaluation
    print(f"\nLoading base model '{model_id}' with QLoRA config for custom evaluation...")
    base_model_eval = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )

    # Sync model and tokenizer for evaluation
    if len(tokenizer) > base_model_eval.config.vocab_size:
        print(
            f"Resizing base model (eval) token embeddings from {base_model_eval.config.vocab_size} to {len(tokenizer)}")
        base_model_eval.resize_token_embeddings(len(tokenizer))
    if base_model_eval.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Syncing base model (eval) pad_token_id to tokenizer's: {tokenizer.pad_token_id}")
        base_model_eval.config.pad_token_id = tokenizer.pad_token_id

    # Load PEFT adapter
    print(f"Loading and applying LoRA adapter from: {adapter_path_for_eval}...")
    try:
        model_eval = PeftModel.from_pretrained(base_model_eval, adapter_path_for_eval)
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}. Ensure path is correct and contains adapter files.")
        traceback.print_exc()
        sys.exit(1)
    model_eval.eval()
    print("PEFT model (eval) loaded and set to eval mode.")

    # Prepare evaluation dataset
    custom_eval_dataset_raw = dataset[EVAL_SPLIT_CUSTOM_EVAL]
    print(f"Loaded {len(custom_eval_dataset_raw)} examples from {EVAL_SPLIT_CUSTOM_EVAL} split for custom evaluation.")

    all_prompts_custom_eval = [create_custom_evaluation_prompt(ex) for ex in custom_eval_dataset_raw]
    all_true_letters = [ex["answer_idx"].strip().upper() for ex in custom_eval_dataset_raw]

    # Generate predictions
    print(
        f"\nGenerating predictions for custom evaluation (batch size {EVAL_BATCH_SIZE_CUSTOM}, max new tokens {MAX_NEW_TOKENS_GEN_CUSTOM})...")
    all_predicted_letters_custom_eval = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(all_prompts_custom_eval), EVAL_BATCH_SIZE_CUSTOM)):
            batch_prompts = all_prompts_custom_eval[i:i + EVAL_BATCH_SIZE_CUSTOM]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_prompt_len_config
            ).to(DEVICE)
            generated_ids = model_eval.generate(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                max_new_tokens=MAX_NEW_TOKENS_GEN_CUSTOM,
                pad_token_id=tokenizer.pad_token_id
            )
            for j, gen_ids_sample in enumerate(generated_ids):
                input_len = inputs['input_ids'].shape[1]
                predicted_token_id = gen_ids_sample[input_len].item()  # The new token
                predicted_letter = next((letter for letter, tid in letter_tokens.items() if tid == predicted_token_id),
                                        None)
                all_predicted_letters_custom_eval.append(predicted_letter)

    # Compute accuracy
    correct = sum(1 for pred, true in zip(all_predicted_letters_custom_eval, all_true_letters) if pred == true)
    total = len(all_true_letters)
    accuracy_custom = correct / total
    num_invalid = sum(1 for pred in all_predicted_letters_custom_eval if pred is None)
    print(f"Number of invalid predictions: {num_invalid}")
    print(f"Accuracy on {EVAL_SPLIT_CUSTOM_EVAL} split: {accuracy_custom:.4f}")

    # Save detailed results
    detailed_results_custom_eval = []
    for i in range(len(all_prompts_custom_eval)):
        predicted_letter = all_predicted_letters_custom_eval[i]
        true_letter = all_true_letters[i]
        is_correct = predicted_letter == true_letter and predicted_letter is not None
        detailed_results_custom_eval.append({
            "id": i,
            "prompt": all_prompts_custom_eval[i],
            "predicted_letter": predicted_letter if predicted_letter is not None else "INVALID",
            "true_letter": true_letter,
            "is_correct": is_correct
        })

    # Prepare configuration summary
    config_summary = {
        "model_id": model_id, "dataset_id": dataset_id, "output_dir": output_dir,
        "max_seq_length": max_seq_length, "max_prompt_len_config": max_prompt_len_config,
        "max_target_len_config": max_target_len_config,
        "per_device_train_bs": per_device_train_bs, "grad_accum_steps": grad_accum_steps,
        "num_train_epochs": num_train_epochs, "learning_rate": learning_rate, "weight_decay_train": weight_decay_train,
        "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
        "random_seed": random_seed,
        "MAX_NEW_TOKENS_GEN_CUSTOM": MAX_NEW_TOKENS_GEN_CUSTOM,
        "final_accuracy": accuracy_custom
    }

    # Save results to file
    try:
        with open(eval_results_file, "w") as f:
            json.dump({"config_summary": config_summary, "detailed_results": detailed_results_custom_eval}, f, indent=2)
        print(f"Detailed custom evaluation results saved to: {eval_results_file}")
    except Exception as e_save:
        print(f"Error saving detailed custom evaluation results: {e_save}")

    print(f"\nScript finished at {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Output, logs, adapter, and results are in: {output_dir}")