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
    DataCollatorForSeq2Seq,  # We might use this or a simpler one if pre-padding
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,  # Needed for loading adapters
    TaskType,
)
# import evaluate # Not used in the new evaluation logic

from tqdm import tqdm  # For progress bars in custom evaluation

# --- Environment Debug (Optional, can be kept or removed) ---
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

# --- Configuration ---
model_id = "Qwen/Qwen2.5-7B"  # Using Qwen2.5-7B as per baseline
dataset_id = "GBaker/MedQA-USMLE-4-options"

# Output Directories
current_datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
base_output_dir = f"./qwen2.5_7b_medqa_sft_run_{current_datetime_str}"
RESULTS_OUTPUT_DIR = os.path.join(base_output_dir, "Qwen_MedQA_results")  # Specific results folder
training_checkpoints_dir = os.path.join(base_output_dir, "training_checkpoints")

# Sequence Lengths
max_prompt_len_for_eval = 512  # Max length for prompt part during custom evaluation
max_seq_length_sft = 512 + 10  # Max sequence length for SFT data (prompt + answer letter + special tokens)

# Training Hyperparameters
per_device_train_bs = 2  # As in baseline
per_device_eval_bs_custom = 16  # Batch size for our custom evaluation loop
grad_accum_steps = 16  # As in baseline
num_train_epochs_total = 10  # CHANGED: Train for 10 epochs to get all checkpoints
learning_rate = 1e-4  # As in baseline
weight_decay_train = 0.01  # As in baseline

# LoRA Configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

random_seed = 63
ANSWER_MAP_KEYS = ["A", "B", "C", "D"]  # Defined globally

# --- Setup ---
os.makedirs(base_output_dir, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(training_checkpoints_dir, exist_ok=True)

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

print(f"Base output directory: {base_output_dir}")
print(f"Results JSON directory: {RESULTS_OUTPUT_DIR}")
print(f"Training checkpoints directory: {training_checkpoints_dir}")
print(f"Using random seed: {random_seed}")

# --- Load Dataset ---
print(f"Loading dataset: {dataset_id}")
raw_dataset_full = load_dataset(dataset_id)


# We need 'train' and 'test' splits. Filter out invalid examples early.
# The Llama script filtered after loading, this is also fine.
def filter_valid_examples(example):
    return example["answer_idx"] is not None and example["answer_idx"].strip().upper() in ANSWER_MAP_KEYS


raw_dataset_full = raw_dataset_full.filter(filter_valid_examples)

if "train" not in raw_dataset_full or "test" not in raw_dataset_full:
    raise ValueError(f"Dataset {dataset_id} must contain 'train' and 'test' splits after filtering.")

# We will use raw_dataset_full["test"] for training and raw_dataset_full["train"] for evaluation.
# No need to create a validation split for the Trainer's internal eval.
dataset_for_sft_training = raw_dataset_full["test"]
dataset_for_custom_evaluation = raw_dataset_full["train"]

print(f"Using {len(dataset_for_sft_training)} examples from 'test' split for SFT.")
print(f"Using {len(dataset_for_custom_evaluation)} examples from 'train' split for custom evaluation.")

# --- Load Tokenizer ---
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",  # Important for taking logits[:, -1, :]
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        print("Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Adding a new pad token might require resizing model embeddings if not already present
        new_pad_token = "<|pad|>"  # A common choice for Qwen if eos is not suitable
        tokenizer.add_special_tokens({"pad_token": new_pad_token})
        print(f"Added new pad_token: {new_pad_token} with ID {tokenizer.pad_token_id}")
print(
    f"Tokenizer loaded. Vocab size: {len(tokenizer)}. Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

# --- QLoRA Config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# --- Helper Function: Create Prompt for SFT and Evaluation ---
def create_medqa_prompt(example, for_evaluation=False):
    question = example["question"].strip()
    options_dict = example["options"]

    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:  # Ensure consistent A, B, C, D order
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option text not found]')}")
    else:  # Fallback if options format is unexpected
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")  # Prompt ends here for evaluation to predict next token

    prompt_text = "\n".join(prompt_parts)

    if for_evaluation:
        return prompt_text
    else:  # For SFT, also return target letter
        answer_idx_key = example["answer_idx"].strip().upper()
        target_letter = answer_idx_key if answer_idx_key in ANSWER_MAP_KEYS else ""  # Should be valid due to pre-filtering
        return {"prompt": prompt_text, "target_letter": target_letter}


# --- Preprocessing for SFT ---
def preprocess_sft_for_qwen(examples):
    prompts_and_targets = [create_medqa_prompt(
        {"question": q, "options": o, "answer_idx": aid}, for_evaluation=False
    ) for q, o, aid in zip(examples["question"], examples["options"], examples["answer_idx"])]

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in prompts_and_targets:
        prompt_text = item["prompt"]
        target_letter = item["target_letter"]

        # Tokenize prompt and target
        # For Qwen, let tokenizer handle BOS/EOS if trust_remote_code=True implies its conventions
        # Set add_special_tokens=False for prompt and target, then add manually for precise label masking
        tokenized_prompt = tokenizer(prompt_text, truncation=False, padding=False, add_special_tokens=False)
        tokenized_target = tokenizer(target_letter, truncation=False, padding=False, add_special_tokens=False)

        prompt_input_ids = tokenized_prompt.input_ids
        target_input_ids = tokenized_target.input_ids

        # Construct input_ids: Typically BOS + prompt + target_letter + EOS
        full_input_ids = []
        # Qwen tokenizers often add special tokens like <|im_start|> or <|im_end|> if used in chat format.
        # For simple SFT prompt + completion, BOS + prompt + target + EOS is standard.
        # We rely on tokenizer.encode() or tokenizer() with add_special_tokens=True for full sequence with special tokens.
        # Let's form the text sequence first then tokenize once.
        sft_sequence_text = prompt_text + " " + target_letter  # Add space for clarity, tokenizer handles it.

        # Tokenize the full SFT sequence including the answer letter
        # Qwen often uses add_special_tokens=True by default in call.
        # max_seq_length_sft is for the whole sequence including special tokens
        tokenized_full_sequence = tokenizer(
            sft_sequence_text,
            truncation=True,
            max_length=max_seq_length_sft,  # Max length for the entire SFT example
            padding="max_length",  # Pad to max_seq_length_sft
            add_special_tokens=True  # Let tokenizer add BOS/EOS etc.
        )

        input_ids = tokenized_full_sequence.input_ids
        attention_mask = tokenized_full_sequence.attention_mask

        # Create labels: -100 for prompt tokens, actual token_id for target tokens
        # Find where the target_letter's tokens start in the `input_ids`
        # This requires tokenizing prompt_text alone with same special token handling
        # to find its length.
        tokenized_prompt_only_for_len = tokenizer(
            prompt_text + " ",  # include the space to match sft_sequence_text tokenization
            truncation=False,  # Dont truncate prompt here, just get its length
            add_special_tokens=True
        )
        # Assuming target is short (single letter, single token usually)
        # The labels should mask out tokens up to the start of the target letter.
        # And also the EOS token if the target letter itself is not what we want to predict (but it is here).

        labels = list(input_ids)  # Start with a copy of input_ids

        # Determine the start of the target response within the tokenized_full_sequence
        # This is tricky if prompt + target is truncated.
        # A simpler SFT is usually: model_inputs = tokenizer(prompt + target + eos, ...), labels = model_inputs.input_ids.copy()
        # then mask prompt_only_tokens in labels.

        # Re-evaluate label creation for robustness:
        # Tokenize prompt and target separately, then combine.
        # This matches the Llama script more closely.

        prompt_part_tokens = tokenizer(prompt_text, add_special_tokens=True).input_ids  # Includes BOS
        target_part_tokens = tokenizer(target_letter,
                                       add_special_tokens=False).input_ids  # No special tokens for target itself

        # BOS + prompt_tokens_no_bos + target_tokens + EOS
        # Qwen tokenizer might handle BOS/EOS differently, let's assume standard for now or test tokenizer behavior.
        # For simplicity, let's assume tokenizer adds BOS/EOS correctly with add_special_tokens=True

        temp_prompt_tokenized = tokenizer(prompt_text, add_special_tokens=True, truncation=True,
                                          max_length=max_prompt_len_for_eval)

        # We need to identify where the target starts.
        # Let's use the target_letter provided.
        input_text = prompt_text + " " + target_letter  # Text for model input
        output_text = target_letter  # Text for model output (that we want to have non -100 labels for)

        # Tokenize input_text that includes the target for input_ids and attention_mask
        model_inputs = tokenizer(
            input_text,
            max_length=max_seq_length_sft,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Get lists first
            add_special_tokens=True
        )

        # For labels, we need to mask the prompt part.
        # Tokenize the prompt_text part to find its length when fully tokenized (with special tokens)
        prompt_tokenized_for_labels = tokenizer(
            prompt_text + " ",  # Add space to ensure tokenization up to target is distinct
            max_length=max_seq_length_sft,  # Can be long
            truncation=True,  # Truncate if prompt is too long
            add_special_tokens=True  # To get accurate length including BOS
        )
        len_prompt_tokens = len(prompt_tokenized_for_labels.input_ids)

        current_labels = list(model_inputs["input_ids"])  # Start with a copy

        # Mask prompt tokens
        for k in range(len_prompt_tokens):
            if k < len(current_labels):  # Check bounds
                current_labels[k] = -100

        # Mask padding tokens in labels if tokenizer didn't use -100 for them
        # (DataCollatorForSeq2Seq usually handles this, but explicit is safer if not using it for SFT)
        for k in range(len(model_inputs["input_ids"])):
            if model_inputs["attention_mask"][k] == 0:  # If it's a padding token
                if k < len(current_labels):
                    current_labels[k] = -100

        input_ids_batch.append(model_inputs["input_ids"])
        labels_batch.append(current_labels)
        attention_mask_batch.append(model_inputs["attention_mask"])

    return {"input_ids": input_ids_batch, "labels": labels_batch, "attention_mask": attention_mask_batch}


print(f"Tokenizing SFT dataset ('test' split of raw data). Max SFT sequence length: {max_seq_length_sft}")
tokenized_sft_dataset = dataset_for_sft_training.map(
    preprocess_sft_for_qwen,
    batched=True,
    num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
    desc="Running SFT tokenizer on dataset"
)


# `remove_unused_columns=True` in TrainingArguments will handle original columns.


# --- Evaluation Function (New, similar to Llama's) ---
def evaluate_qwen_model(model_to_eval, current_tokenizer, eval_dataset_raw, epoch, fine_tuning_duration):
    print(f"\nStarting custom evaluation for epoch {epoch}...")
    model_to_eval.eval()  # Ensure model is in evaluation mode

    all_prompts_for_eval = [create_medqa_prompt(ex, for_evaluation=True) for ex in
                            tqdm(eval_dataset_raw, desc="Creating prompts for custom eval")]
    all_true_letters = [ex["answer_idx"].strip().upper() for ex in eval_dataset_raw]

    # Get token IDs for A, B, C, D
    letter_token_ids_map = {}
    valid_letter_tokenization = True
    for letter in ANSWER_MAP_KEYS:
        # For Qwen, ensure add_special_tokens=False is respected. Some tokenizers might still add prefix space.
        # It's usually robust for single characters.
        token_ids = current_tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) == 1:
            letter_token_ids_map[letter] = token_ids[0]
        else:
            print(
                f"Warning: Letter '{letter}' tokenized into {len(token_ids)} tokens: {token_ids}. Using first if available.")
            if token_ids:
                letter_token_ids_map[letter] = token_ids[0]
            else:
                letter_token_ids_map[letter] = -1  # Invalid
                valid_letter_tokenization = False

    if not valid_letter_tokenization:
        print("CRITICAL WARNING: Some A,B,C,D choice letters could not be tokenized properly!")

    # This list of token IDs will be used to gather logits
    token_ids_for_abcd_options = [letter_token_ids_map.get(letter, -1) for letter in ANSWER_MAP_KEYS]

    all_choice_probs_list = []
    all_correctness_list = []
    device = model_to_eval.device  # Get device from model

    for i in tqdm(range(0, len(all_prompts_for_eval), per_device_eval_bs_custom),
                  desc=f"Evaluating epoch {epoch} batches"):
        batch_prompts = all_prompts_for_eval[i: i + per_device_eval_bs_custom]

        # Tokenize prompts for the model
        # Padding side for tokenizer must be 'left' if taking logits for next token at last position.
        inputs = current_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",  # Pad to longest in batch
            truncation=True,
            max_length=max_prompt_len_for_eval,  # Max length for the prompt part
            add_special_tokens=True  # Let tokenizer handle BOS/EOS for prompts
        ).to(device)

        with torch.no_grad():
            outputs = model_to_eval(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            # Logits for the *next* token prediction are at the last sequence position for each item in batch
            # This assumes tokenizer.padding_side == "left"
            next_token_logits_batch = []
            if current_tokenizer.padding_side == "right":  # Should be left, but defensive
                sequence_lengths = inputs.attention_mask.sum(dim=1)
                for batch_idx in range(outputs.logits.shape[0]):
                    seq_len = sequence_lengths[batch_idx].item()
                    next_token_logits_batch.append(outputs.logits[batch_idx, seq_len - 1 if seq_len > 0 else 0, :])
                logits_for_next_token = torch.stack(next_token_logits_batch)
            else:  # padding_side == "left"
                logits_for_next_token = outputs.logits[:, -1, :]

        for j in range(logits_for_next_token.shape[0]):  # Iterate over samples in the batch
            single_sample_all_vocab_logits = logits_for_next_token[j]  # Shape: (vocab_size,)

            # Extract logits only for the A, B, C, D tokens that were validly tokenized
            valid_option_token_ids = [tid for tid in token_ids_for_abcd_options if tid != -1]

            current_choice_probs_full = np.zeros(len(ANSWER_MAP_KEYS), dtype=np.float32)

            if not valid_option_token_ids:  # If no valid A,B,C,D tokens found
                current_choice_probs_full[:] = 1.0 / len(ANSWER_MAP_KEYS)  # Uniform
            else:
                logits_for_valid_options = single_sample_all_vocab_logits[valid_option_token_ids]
                probs_for_valid_options = torch.nn.functional.softmax(logits_for_valid_options, dim=0).to(
                    torch.float32).cpu().numpy()

                # Place probabilities into the full A,B,C,D structure
                prob_idx = 0
                for k, original_letter in enumerate(ANSWER_MAP_KEYS):
                    if letter_token_ids_map.get(original_letter, -1) != -1:  # If this letter had a valid token
                        if prob_idx < len(probs_for_valid_options):
                            current_choice_probs_full[k] = probs_for_valid_options[prob_idx]
                            prob_idx += 1

                # Normalize if sum isn't 1.0 (e.g. if some letters were untokenizable)
                sum_probs = np.sum(current_choice_probs_full)
                if not np.isclose(sum_probs, 1.0) and sum_probs > 1e-6:
                    current_choice_probs_full /= sum_probs
                elif sum_probs < 1e-6:  # All zero, fallback to uniform
                    current_choice_probs_full[:] = 1.0 / len(ANSWER_MAP_KEYS)

            predicted_option_idx = np.argmax(current_choice_probs_full)
            predicted_letter = ANSWER_MAP_KEYS[predicted_option_idx]

            true_letter_for_sample = all_true_letters[i + j]
            is_correct = int(predicted_letter == true_letter_for_sample)

            all_choice_probs_list.append(current_choice_probs_full.tolist())
            all_correctness_list.append(is_correct)

    accuracy = np.mean(all_correctness_list) if all_correctness_list else 0.0

    # Filename and Output Data Formatting
    model_name_part = model_id.split('/')[-1].replace('.', '-')  # Qwen/Qwen2.5-7B -> Qwen2-5-7B
    dataset_name_part = dataset_id.split('/')[0] if '/' in dataset_id else dataset_id  # GBaker
    dataset_name_part_short = dataset_id.split('/')[-1].split('-')[0]  # MedQA

    # Format: qwen2-5-7B_MedQA_qwen_0.00_Acc_0.5099_epochs_0.json
    task_name_for_file = f"{dataset_name_part_short}_qwen"
    output_filename = f"{model_name_part}_{task_name_for_file}_{fine_tuning_duration:.2f}_Acc_{accuracy:.4f}_epochs_{epoch}.json"
    output_filepath = os.path.join(RESULTS_OUTPUT_DIR, output_filename)

    data_to_save = {
        "logits": all_choice_probs_list,  # List of [P(A), P(B), P(C), P(D)]
        "responses": all_correctness_list  # List of 0 or 1
    }
    with open(output_filepath, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved custom evaluation results for epoch {epoch} to {output_filepath}")
    print(f"Epoch {epoch} custom evaluation accuracy: {accuracy:.4f}")
    return accuracy


# --- Main Script Flow ---
if __name__ == "__main__":
    # 1. Evaluate Pre-trained Model (Epoch 0)
    print("\n--- Evaluating Pre-trained Model (Epoch 0) ---")
    base_model_for_epoch0 = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,  # Apply quantization to base model for fair comparison if desired
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    if len(tokenizer) > base_model_for_epoch0.config.vocab_size:
        print(f"Resizing base model embeddings from {base_model_for_epoch0.config.vocab_size} to {len(tokenizer)}")
        base_model_for_epoch0.resize_token_embeddings(len(tokenizer))
    if base_model_for_epoch0.config.pad_token_id != tokenizer.pad_token_id:
        print(
            f"Updating base model pad_token_id from {base_model_for_epoch0.config.pad_token_id} to {tokenizer.pad_token_id}")
        base_model_for_epoch0.config.pad_token_id = tokenizer.pad_token_id

    evaluate_qwen_model(base_model_for_epoch0, tokenizer, dataset_for_custom_evaluation, epoch=0,
                        fine_tuning_duration=0.0)

    del base_model_for_epoch0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Training
    print("\n--- Starting SFT Training (on 'test' split) for Qwen Model ---")

    # Load model for training
    model_for_training = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    if len(tokenizer) > model_for_training.config.vocab_size:
        print(f"Resizing training model embeddings from {model_for_training.config.vocab_size} to {len(tokenizer)}")
        model_for_training.resize_token_embeddings(len(tokenizer))
    if model_for_training.config.pad_token_id != tokenizer.pad_token_id:
        print(
            f"Updating training model pad_token_id from {model_for_training.config.pad_token_id} to {tokenizer.pad_token_id}")
        model_for_training.config.pad_token_id = tokenizer.pad_token_id

    model_for_training = prepare_model_for_kbit_training(model_for_training)
    # model_for_training.gradient_checkpointing_enable() # Already called in baseline, good.

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=lora_target_modules, bias="none"
    )
    peft_model_for_training = get_peft_model(model_for_training, peft_config)
    print("QLoRA model prepared for training.")
    peft_model_for_training.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=training_checkpoints_dir,  # Save checkpoints here
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,  # No Trainer internal evaluation
        per_device_train_batch_size=per_device_train_bs,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_train_epochs_total,  # Train for 10 epochs
        learning_rate=learning_rate,
        weight_decay=weight_decay_train,
        fp16=False,  # Qwen baseline uses bf16
        bf16=True,  # Ensure GPU supports bf16
        logging_strategy="steps",  # Log more frequently than epoch
        logging_steps=max(1, 50 // grad_accum_steps),  # Log every 50 effective steps
        eval_strategy="no",  # No evaluation during training by Trainer
        save_strategy="epoch",  # Save a checkpoint at the end of each epoch
        save_total_limit=None,  # Save all epoch checkpoints (or set a number like 3-5)
        report_to=[],  # Disable reporting to wandb/tensorboard unless configured
        remove_unused_columns=True,  # Important for avoiding issues with extra string columns
        seed=random_seed,
        # load_best_model_at_end=False, # Not needed, no internal eval
        # metric_for_best_model=None,   # Not needed
        # prediction_loss_only=True, # Not relevant if do_eval=False
    )

    # Data Collator for SFT (padding already handled in preprocess, but this ensures tensor conversion)
    # DataCollatorForSeq2Seq can be used if labels are present and need padding (which they are, with -100)
    data_collator_sft = DataCollatorForSeq2Seq(
        tokenizer,
        model=peft_model_for_training,  # Model is optional but can help with some settings
        label_pad_token_id=-100,  # Standard for LM labels
        padding="longest"  # This will mostly convert lists to tensors if already padded
    )

    trainer = Trainer(
        model=peft_model_for_training,
        args=training_args,
        train_dataset=tokenized_sft_dataset,  # This is dataset_for_sft_training after tokenization
        eval_dataset=None,  # No validation set for Trainer
        tokenizer=tokenizer,
        data_collator=data_collator_sft,
        # compute_metrics=None, # Not needed for trainer internal eval
        # callbacks=None, # No EarlyStopping
    )

    print(f"Starting SFT training for {num_train_epochs_total} epochs...")
    train_result = trainer.train()
    total_training_time = train_result.metrics.get("train_runtime", 0.0)
    print(f"SFT Training completed. Total training time: {total_training_time:.2f} seconds.")

    final_adapter_save_path = os.path.join(base_output_dir, f"final_qlora_adapter_{num_train_epochs_total}_epochs")
    peft_model_for_training.save_pretrained(final_adapter_save_path)  # Save final PEFT adapter
    tokenizer.save_pretrained(final_adapter_save_path)  # Save tokenizer with adapter
    print(f"Final adapter for {num_train_epochs_total} epochs saved to {final_adapter_save_path}")

    del model_for_training, peft_model_for_training, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Evaluate Saved Checkpoints
    print("\n--- Evaluating Specific Epoch Checkpoints (on 'train' split) ---")
    desired_epochs_to_eval = [1, 3, 5, 10]

    for desired_epoch in desired_epochs_to_eval:
        # Construct checkpoint path: output_dir/checkpoint-<steps_for_epoch>
        # This requires knowing how many steps correspond to an epoch.
        # Trainer saves checkpoints in folders like "checkpoint-500", "checkpoint-1000".
        # We need to find the checkpoint folder that corresponds to the end of `desired_epoch`.

        found_checkpoint_for_epoch = False
        potential_ckpt_path = None

        if os.path.exists(training_checkpoints_dir):
            # Iterate through checkpoint folders, assuming they are somewhat ordered by step number
            # and look into trainer_state.json
            ckpt_folders = sorted([
                os.path.join(training_checkpoints_dir, d)
                for d in os.listdir(training_checkpoints_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_checkpoints_dir, d))
            ])

            for ckpt_path_candidate in ckpt_folders:
                trainer_state_file = os.path.join(ckpt_path_candidate, "trainer_state.json")
                if os.path.exists(trainer_state_file):
                    with open(trainer_state_file, "r") as f:
                        state = json.load(f)
                    if round(state.get("epoch", 0.0)) == desired_epoch:
                        potential_ckpt_path = ckpt_path_candidate
                        found_checkpoint_for_epoch = True
                        break

        # Fallback for epoch 10 to use the final explicitly saved adapter if checkpoint not found
        if desired_epoch == num_train_epochs_total and not found_checkpoint_for_epoch:
            if os.path.exists(final_adapter_save_path):
                print(
                    f"Checkpoint for epoch {desired_epoch} not found via trainer_state, using final saved adapter: {final_adapter_save_path}")
                potential_ckpt_path = final_adapter_save_path
                found_checkpoint_for_epoch = True

        if found_checkpoint_for_epoch and potential_ckpt_path:
            print(f"\nLoading checkpoint for epoch {desired_epoch} from: {potential_ckpt_path}")
            # Load base model (quantized)
            base_model_for_eval = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            if len(tokenizer) > base_model_for_eval.config.vocab_size:
                base_model_for_eval.resize_token_embeddings(len(tokenizer))
            if base_model_for_eval.config.pad_token_id != tokenizer.pad_token_id:
                base_model_for_eval.config.pad_token_id = tokenizer.pad_token_id

            # Load PEFT adapter onto the base model
            model_eval_peft = PeftModel.from_pretrained(base_model_for_eval, potential_ckpt_path, is_trainable=False)
            model_eval_peft.eval()

            evaluate_qwen_model(
                model_eval_peft, tokenizer, dataset_for_custom_evaluation,
                epoch=desired_epoch, fine_tuning_duration=total_training_time  # total_training_time is for the full run
            )
            del base_model_for_eval, model_eval_peft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(
                f"Warning: Could not find a suitable checkpoint for epoch {desired_epoch} in {training_checkpoints_dir}")
            if desired_epoch == num_train_epochs_total:
                print(f"  Also checked final adapter path: {final_adapter_save_path}")

    print(f"\nScript finished at {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"All outputs in: {base_output_dir}")