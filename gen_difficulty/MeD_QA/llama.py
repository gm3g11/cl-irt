import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

# --- Environment Setup ---
# Try to use the user's specified path, but fall back if it doesn't exist.
# This is a common setup for clusters; adjust if running locally without this path.
HF_HOME_SPECIFIED = HF_HOME
if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(HF_HOME_SPECIFIED):
    HF_HOME = HF_HOME_SPECIFIED
else:
    print(f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' does not exist or is not a directory.")
    print("Using default Hugging Face cache directory.")
    HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Good for memory management

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Reproducibility
RANDOM_SEED = 63
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCH_SIZE_CUSTOM = 16
ANSWER_MAP_KEYS = ["A", "B", "C", "D"]
model_id = "meta-llama/Meta-Llama-3.1-8B"
dataset_id = "GBaker/MedQA-USMLE-4-options"

# Main output directory for checkpoints etc.
base_output_dir = "MedQA_llama31_8b_run_output"  # Changed to be more descriptive of the run
os.makedirs(base_output_dir, exist_ok=True)

# Specific directory for JSON evaluation results as per new request
RESULTS_OUTPUT_DIR = os.path.join(base_output_dir, "MedQA_results")
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

tokenizer = None  # Global tokenizer variable


# --- Helper Functions ---
def preprocess_sft_letter_target(examples):
    global tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer has not been initialized.")

    questions = examples['question']
    options_dicts = examples['options']
    answer_idxs = examples['answer_idx']
    prompts = []

    for i in range(len(questions)):
        question_text = questions[i]
        options_text_parts = []
        if isinstance(options_dicts[i], dict):
            for key in ANSWER_MAP_KEYS:
                if key in options_dicts[i]:
                    options_text_parts.append(f"{key}) {options_dicts[i][key]}")
                else:
                    options_text_parts.append(f"{key}) [Option not available]")
        else:
            options_text_parts.append("[Options format error]")
        options_str = "\n".join(options_text_parts)
        prompts.append(f"Question: {question_text}\n\nOptions:\n{options_str}\n\nAnswer:")

    # Tokenize prompts
    prompt_tokens = tokenizer(prompts, truncation=True, max_length=500, add_special_tokens=True)
    # Tokenize targets (answer letters)
    targets = [a.strip().upper() for a in answer_idxs]
    target_tokens = tokenizer(targets, add_special_tokens=False, max_length=3)  # Max 3 for safety, should be 1

    input_ids_list = []
    labels_list = []
    attention_mask_list = []  # ADDED for attention mask

    current_max_length = 512  # Define max sequence length for model

    for i in range(len(prompt_tokens['input_ids'])):
        prompt_tok_ids = prompt_tokens['input_ids'][i]
        # Ensure target_tokens['input_ids'][i] is not empty and is a list of token IDs for the letter
        target_tok_ids = target_tokens['input_ids'][i]
        if not target_tok_ids:
            print(
                f"Warning: Empty target token for prompt: {prompts[i][:50]}... Original target: {targets[i]}. Skipping this example.")
            continue  # Skip this example

        # Concatenate prompt, target, and EOS token
        input_ids_unpadded = prompt_tok_ids + target_tok_ids + [tokenizer.eos_token_id]
        labels_unpadded = ([-100] * len(prompt_tok_ids)) + target_tok_ids + [tokenizer.eos_token_id]

        # Pad or truncate
        padding_length = current_max_length - len(input_ids_unpadded)

        if padding_length >= 0:  # Pad
            input_ids = input_ids_unpadded + [tokenizer.pad_token_id] * padding_length
            labels = labels_unpadded + [-100] * padding_length
            attention_mask = [1] * len(input_ids_unpadded) + [0] * padding_length
        else:  # Truncate
            input_ids = input_ids_unpadded[:current_max_length]
            labels = labels_unpadded[:current_max_length]
            attention_mask = [1] * current_max_length

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_mask_list}


def create_custom_evaluation_prompt(example):
    global tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer has not been initialized for create_custom_evaluation_prompt.")
    question_text = example['question']
    options_dict = example['options']
    options_text_parts = []
    if isinstance(options_dict, dict):
        for key in ANSWER_MAP_KEYS:
            if key in options_dict:
                options_text_parts.append(f"{key}) {options_dict[key]}")
            else:
                options_text_parts.append(f"{key}) [Option not available]")
    else:
        options_text_parts.append("[Options format error]")
    options_str = "\n".join(options_text_parts)
    return f"Question: {question_text}\n\nOptions:\n{options_str}\n\nAnswer:"


def evaluate_model(model, current_tokenizer, dataset_raw, epoch, fine_tuning_duration):
    """Evaluate model on dataset and save results to RESULTS_OUTPUT_DIR."""
    model.eval()
    all_prompts = [create_custom_evaluation_prompt(ex) for ex in tqdm(dataset_raw, desc="Creating prompts for eval")]
    all_true_letters = [ex["answer_idx"].strip().upper() for ex in dataset_raw]

    letter_tokens = {}
    valid_letter_tokenization = True
    for letter in ANSWER_MAP_KEYS:
        token_id_list = current_tokenizer.encode(letter, add_special_tokens=False)
        if len(token_id_list) != 1:
            print(
                f"Warning: Letter '{letter}' tokenized into {len(token_id_list)} tokens: {token_id_list}. This may affect logit extraction.")
            if token_id_list:
                letter_tokens[letter] = token_id_list[0]  # Use first token if split
            else:
                print(f"CRITICAL: Letter '{letter}' could not be tokenized by {current_tokenizer}.")
                letter_tokens[letter] = -1  # Invalid placeholder
                valid_letter_tokenization = False
        else:
            letter_tokens[letter] = token_id_list[0]

    if not valid_letter_tokenization:
        print(
            "Warning: One or more answer choice letters (A,B,C,D) could not be reliably tokenized. Accuracy may be impacted.")

    token_ids_for_abcd = [letter_tokens[letter] for letter in ANSWER_MAP_KEYS]  # List of token IDs, can include -1

    all_extracted_logits_probs = []
    all_responses_correctness = []

    for i in tqdm(range(0, len(all_prompts), EVAL_BATCH_SIZE_CUSTOM), desc=f"Evaluating epoch {epoch}"):
        batch_prompts = all_prompts[i:i + EVAL_BATCH_SIZE_CUSTOM]
        inputs = current_tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=510,  # Max prompt length, leaves space for model to generate
            add_special_tokens=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            # Logits for the *next* token prediction
            logits_for_next_token_batch = []
            if current_tokenizer.padding_side == "right":
                sequence_lengths = inputs.attention_mask.sum(dim=1)
                for batch_idx in range(outputs.logits.shape[0]):
                    seq_len = sequence_lengths[batch_idx].item()
                    # Ensure valid index for logits
                    logits_for_next_token_batch.append(outputs.logits[batch_idx, seq_len - 1 if seq_len > 0 else 0, :])
            else:  # padding_side == "left" (default for Llama generation)
                logits_for_next_token_batch = outputs.logits[:, -1, :]

            # Ensure it's a tensor
            if isinstance(logits_for_next_token_batch, list):
                logits_for_next_token = torch.stack(logits_for_next_token_batch)
            else:
                logits_for_next_token = logits_for_next_token_batch

        for j in range(logits_for_next_token.shape[0]):  # Iterate over samples in the batch
            single_sample_all_vocab_logits = logits_for_next_token[j]  # Shape: (vocab_size,)

            # Filter out invalid token_ids (-1) if any letter failed to tokenize
            current_valid_token_ids_for_abcd = [tid for tid in token_ids_for_abcd if tid != -1]

            if not current_valid_token_ids_for_abcd:  # All A,B,C,D tokens are invalid
                # This case should be rare if tokenizer includes A,B,C,D
                probs_for_abcd_options_numpy = np.array([0.25] * len(ANSWER_MAP_KEYS))  # Uniform probability
                predicted_letter = random.choice(ANSWER_MAP_KEYS)  # Random guess
            else:
                logits_for_abcd_options = single_sample_all_vocab_logits[current_valid_token_ids_for_abcd]
                probs_for_abcd_options = torch.nn.functional.softmax(logits_for_abcd_options, dim=0)
                # Convert to numpy for consistent processing
                probs_for_abcd_options_numpy_part = probs_for_abcd_options.to(torch.float32).cpu().numpy()

                # Reconstruct the full probability array for A,B,C,D, inserting 0 for untokenizable letters
                probs_for_abcd_options_numpy = np.zeros(len(ANSWER_MAP_KEYS), dtype=np.float32)
                valid_token_idx = 0
                for original_idx, letter_key in enumerate(ANSWER_MAP_KEYS):
                    if letter_tokens[letter_key] != -1:  # if this letter (A,B,C,D) had a valid token
                        if valid_token_idx < len(probs_for_abcd_options_numpy_part):
                            probs_for_abcd_options_numpy[original_idx] = probs_for_abcd_options_numpy_part[
                                valid_token_idx]
                            valid_token_idx += 1
                    # else, it remains 0.0 for that option (e.g. if 'A' was not tokenizable)

                # Normalize if sum is not 1 (e.g. due to missing options or numerical precision)
                # and sum is not zero (to avoid division by zero)
                if not np.isclose(np.sum(probs_for_abcd_options_numpy), 1.0) and np.sum(
                        probs_for_abcd_options_numpy) > 1e-6:
                    probs_for_abcd_options_numpy = probs_for_abcd_options_numpy / np.sum(probs_for_abcd_options_numpy)
                elif np.sum(probs_for_abcd_options_numpy) < 1e-6:  # if all valid options had zero prob
                    # Fallback to uniform if all probabilities are zero for some reason
                    probs_for_abcd_options_numpy = np.array([1.0 / len(ANSWER_MAP_KEYS)] * len(ANSWER_MAP_KEYS))

                predicted_option_idx = np.argmax(probs_for_abcd_options_numpy)
                predicted_letter = ANSWER_MAP_KEYS[predicted_option_idx]

            true_letter_for_sample = all_true_letters[i + j]
            is_correct = int(predicted_letter == true_letter_for_sample)

            all_extracted_logits_probs.append(probs_for_abcd_options_numpy.tolist())
            all_responses_correctness.append(is_correct)

    accuracy = np.mean(all_responses_correctness) if all_responses_correctness else 0.0

    # --- Filename and Output Data Formatting (as per user request) ---
    model_name_part = model_id.split('/')[-1]  # e.g., "Meta-Llama-3.1-8B"
    filename_model_part = model_name_part.replace("Meta-", "")  # e.g., "Llama-3.1-8B"
    # Try to match "llama-3.1-8B" format more closely
    if filename_model_part.startswith("Llama-"):
        filename_model_part = "llama-" + filename_model_part.split("Llama-", 1)[1]
    else:
        filename_model_part = filename_model_part.lower()  # General fallback

    dataset_name_part = dataset_id.split('/')[-1].split('-')[0]  # e.g., "MedQA" from "GBaker/MedQA-USMLE-4-options"
    task_name_for_file = f"{dataset_name_part}_llama"  # e.g., "MedQA_llama"

    # Filename example: llama-3.1-8B_MedQA_llama_18438.39_Acc_0.6725_epochs_3.json
    output_filename = f"{filename_model_part}_{task_name_for_file}_{fine_tuning_duration:.2f}_Acc_{accuracy:.4f}_epochs_{epoch}.json"
    output_filepath = os.path.join(RESULTS_OUTPUT_DIR, output_filename)

    data_to_save = {
        "logits": all_extracted_logits_probs,  # These are softmax probabilities for A,B,C,D choices
        "responses": all_responses_correctness  # List of 0s (incorrect) and 1s (correct)
    }
    with open(output_filepath, "w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"Saved evaluation results for epoch {epoch} to {output_filepath}")
    return accuracy


# --- Main Execution ---
if __name__ == "__main__":
    dataset = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])
    # Filter for valid answers first
    dataset = dataset.filter(
        lambda x: x["answer_idx"] is not None and x["answer_idx"].strip().upper() in ANSWER_MAP_KEYS, num_proc=2)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            # This case is less ideal as it adds a new token the base model wasn't trained on.
            # Model resizing will be needed.
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added new [PAD] token. Model embeddings will be resized.")
    # Crucial for Llama if padding_side is not set by default for generation-style inputs
    if tokenizer.padding_side is None or tokenizer.padding_side == "right":  # Common for training, but left for generation
        tokenizer.padding_side = "left"  # Set to left for consistency with how logits[:, -1, :] is used
        print(f"Set tokenizer.padding_side to 'left'")

    # Ensure dataset has 'train' and 'test' splits as expected by the script
    if "test" not in dataset:
        raise ValueError(
            f"Dataset {dataset_id} does not contain a 'test' split, which is required for training in this script.")
    if "train" not in dataset:
        raise ValueError(
            f"Dataset {dataset_id} does not contain a 'train' split, which is required for evaluation in this script.")

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_sft_letter_target,
        batched=True,
        num_proc=2,  # Adjust num_proc based on your CPU cores
        desc="Running tokenizer on dataset"
    )
    print("Dataset tokenization complete.")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("\n--- Evaluating Pre-trained Model (Epoch 0) ---")
    base_model_epoch0 = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    if len(tokenizer) > base_model_epoch0.config.vocab_size:  # If tokenizer had new tokens added (e.g. [PAD])
        print(
            f"Resizing token embeddings for base model from {base_model_epoch0.config.vocab_size} to {len(tokenizer)}")
        base_model_epoch0.resize_token_embeddings(len(tokenizer))
    # Ensure pad token ID in model config matches tokenizer
    if base_model_epoch0.config.pad_token_id is None or base_model_epoch0.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Setting base model pad_token_id to {tokenizer.pad_token_id}")
        base_model_epoch0.config.pad_token_id = tokenizer.pad_token_id

    # Evaluate pre-trained model on the 'train' split of the dataset
    evaluate_model(base_model_epoch0, tokenizer, dataset["train"], epoch=0, fine_tuning_duration=0.0)
    del base_model_epoch0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Training on Test Dataset (for 10 epochs total) ---
    print("\n--- Training on Test Dataset (for 10 epochs total) ---")
    print("Disabling Trainer's internal evaluation (no validation dataset/early stopping as per request).")

    training_checkpoints_dir = os.path.join(base_output_dir, "training_checkpoints")
    training_args = TrainingArguments(
        output_dir=training_checkpoints_dir,
        num_train_epochs=10,  # Train for 10 epochs to generate checkpoints up to epoch 10
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 * num_gpus
        eval_strategy="no",  # No evaluation during Trainer.train()
        per_device_eval_batch_size=None,  # Not used as eval_strategy is "no"
        save_strategy="epoch",  # Save checkpoint at the end of each epoch
        save_total_limit=None,  # Save all checkpoints, or set a number to limit
        logging_steps=50,
        learning_rate=2e-4,
        fp16=True if DEVICE == "cuda" else False,  # Use fp16 only if on CUDA and supported
        bf16=False,  # Can set to True if Ampere or newer GPU and want to use bfloat16
        warmup_steps=50,
        weight_decay=0.01,
        report_to="none",  # ["tensorboard", "wandb"] or "none"
        remove_unused_columns=True,  # IMPORTANT: Set to True to avoid issues with extra string columns
        seed=RANDOM_SEED,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,  # Often lora_alpha = 2*r
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Common for Llama
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model_to_train = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    if len(tokenizer) > model_to_train.config.vocab_size:
        print(
            f"Resizing token embeddings for training model from {model_to_train.config.vocab_size} to {len(tokenizer)}")
        model_to_train.resize_token_embeddings(len(tokenizer))
    if model_to_train.config.pad_token_id is None or model_to_train.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Setting training model pad_token_id to {tokenizer.pad_token_id}")
        model_to_train.config.pad_token_id = tokenizer.pad_token_id

    model_to_train = prepare_model_for_kbit_training(model_to_train)
    peft_model_for_training = get_peft_model(model_to_train, lora_config)
    print("PEFT model prepared for training:")
    peft_model_for_training.print_trainable_parameters()

    # Using tokenized_dataset["test"] for training as requested
    if "test" not in tokenized_dataset:
        raise ValueError("Tokenized dataset does not have 'test' split for training.")
    train_data_for_trainer = tokenized_dataset["test"]  # This should now contain input_ids, labels, attention_mask

    trainer = Trainer(
        model=peft_model_for_training,
        args=training_args,
        train_dataset=train_data_for_trainer,
        eval_dataset=None,  # No evaluation dataset for Trainer
        tokenizer=tokenizer,  # Providing tokenizer enables default DataCollatorWithPadding
        # data_collator=None, # Default collator will be used.
        # If issues persist, a DataCollatorForLanguageModeling(tokenizer, mlm=False) could be explicit.
    )

    print("Starting training...")
    train_result = trainer.train()
    total_training_time = train_result.metrics.get("train_runtime", 0.0)
    print(f"Training completed. Total training time: {total_training_time:.2f} seconds.")

    final_adapter_path = os.path.join(base_output_dir, "final_qlora_adapter_10_epochs")
    trainer.save_model(final_adapter_path)  # Save the final LoRA adapter
    print(f"Final adapter (10 epochs) saved to {final_adapter_path}")

    del model_to_train, peft_model_for_training, trainer  # Clear VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Evaluate Checkpoints on the 'train' split ---
    print("\n--- Evaluating Specific Epoch Checkpoints on dataset['train'] ---")
    desired_epochs_to_eval = [1, 3, 5, 10]  # Epoch 0 was pre-trained

    if "train" not in dataset:  # Should have been caught earlier, but good check
        raise ValueError("Dataset 'train' split not found for evaluation of checkpoints.")
    evaluation_dataset_raw = dataset["train"]  # Use the raw 'train' part of the original dataset for eval prompts

    for desired_epoch in desired_epochs_to_eval:
        found_checkpoint_for_epoch = False
        potential_ckpt_path = None

        # Strategy 1: Check for standard checkpoint folders from save_strategy="epoch"
        if os.path.exists(training_checkpoints_dir):
            for ckpt_folder_name in sorted(os.listdir(training_checkpoints_dir)):  # Sort to process in order if needed
                if ckpt_folder_name.startswith("checkpoint-"):
                    current_ckpt_path = os.path.join(training_checkpoints_dir, ckpt_folder_name)
                    trainer_state_file = os.path.join(current_ckpt_path, "trainer_state.json")
                    if os.path.exists(trainer_state_file):
                        with open(trainer_state_file, "r") as f:
                            state = json.load(f)
                        # state["epoch"] is float, round it for comparison
                        if round(state.get("epoch", 0.0)) == desired_epoch:
                            potential_ckpt_path = current_ckpt_path
                            found_checkpoint_for_epoch = True
                            break  # Found checkpoint for this desired_epoch

        # Strategy 2: If desired_epoch is 10, also consider the final saved adapter
        # This is a fallback if the checkpoint folder for epoch 10 isn't found via trainer_state.json
        if desired_epoch == 10 and not found_checkpoint_for_epoch:
            if os.path.exists(final_adapter_path) and os.path.isdir(final_adapter_path):  # Check if it's a directory
                print(
                    f"Checkpoint for epoch 10 not found via trainer_state.json, trying final saved adapter: {final_adapter_path}")
                potential_ckpt_path = final_adapter_path
                found_checkpoint_for_epoch = True

        if found_checkpoint_for_epoch and potential_ckpt_path:
            print(f"\nEvaluating model state for epoch {desired_epoch} from: {potential_ckpt_path}")

            base_model_for_eval = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=os.environ["TRANSFORMERS_CACHE"]
            )
            if len(tokenizer) > base_model_for_eval.config.vocab_size:
                print(
                    f"Resizing token embeddings for eval model from {base_model_for_eval.config.vocab_size} to {len(tokenizer)}")
                base_model_for_eval.resize_token_embeddings(len(tokenizer))
            if base_model_for_eval.config.pad_token_id is None or base_model_for_eval.config.pad_token_id != tokenizer.pad_token_id:
                print(f"Setting eval model pad_token_id to {tokenizer.pad_token_id}")
                base_model_for_eval.config.pad_token_id = tokenizer.pad_token_id

            # Load the LoRA adapter
            # is_trainable=False is good practice for evaluation
            model_eval_peft = PeftModel.from_pretrained(base_model_for_eval, potential_ckpt_path, is_trainable=False)
            model_eval_peft.eval()  # Ensure model is in eval mode

            evaluate_model(model_eval_peft, tokenizer, evaluation_dataset_raw,
                           epoch=desired_epoch, fine_tuning_duration=total_training_time)

            del model_eval_peft, base_model_for_eval  # Clear VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"Warning: Could not find a checkpoint or final adapter path for epoch {desired_epoch}.")
            if desired_epoch == 10:  # Specific help for epoch 10
                print(
                    f"  Checked standard checkpoints in '{training_checkpoints_dir}' and final adapter path '{final_adapter_path}'")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n===== MEDQA_LLaMA Experiment Completed =====")