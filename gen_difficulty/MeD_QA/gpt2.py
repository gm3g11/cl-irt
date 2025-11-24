import os
import datetime
import random
import numpy as np
import torch
import json
from tqdm import tqdm
import shutil
import traceback
import sys
import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig,
)
from huggingface_hub import whoami

# ----- Environment Setup -----
HF_HOME_SPECIFIED = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(HF_HOME_SPECIFIED):
    HF_HOME = HF_HOME_SPECIFIED
else:
    print(f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' does not exist. Using default.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ----- Hugging Face Token Check -----
if "HF_TOKEN" not in os.environ:
    print("HF_TOKEN environment variable not set. Some private models may not be accessible.")
try:
    user_info = whoami()
    print(f"Logged in to Hugging Face as: {user_info.get('name', 'Unknown User')}")
except Exception as e:
    print(f"Hugging Face login check error: {e}. Ensure CLI login or set HF_TOKEN for private models.")

# ----- Reproducibility -----
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
print(f"Using random seed: {random_seed}")

# ----- Configuration -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

MODEL_CHECKPOINT_ID = "gpt2"
DATASET_ID_MC = "GBaker/MedQA-USMLE-4-options"
TASK_NAME_FOR_RESULTS_FILE_MC = "MedQA_MC_GPT2"

MAX_SFT_SEQ_LENGTH = 512 + 10
MAX_EVAL_PROMPT_LENGTH = 512

TRAIN_BATCH_SIZE_GPT2 = 4
EVAL_BATCH_SIZE_CUSTOM_GPT2 = 8
EFFECTIVE_BATCH_SIZE_GPT2 = 32

NUM_EPOCHS_TOTAL_MC = 10
EPOCHS_TO_EVALUATE_MC = [0, 1, 3, 5, 10]
LEARNING_RATE_GPT2 = 5e-5

ANSWER_MAP_KEYS_MC = ['A', 'B', 'C', 'D']
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS_MC)

# --- Output Directories ---
current_datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
model_short_name = MODEL_CHECKPOINT_ID.replace('/', '_')
base_output_dir_model = f"./{model_short_name}_MedQA_run_{current_datetime_str}"
RESULTS_OUTPUT_DIR_MODEL = os.path.join(base_output_dir_model, "GPT2_MedQA_Results_JSON")
TRAINING_CHECKPOINTS_DIR_MODEL = os.path.join(base_output_dir_model, "Training_Checkpoints")

os.makedirs(base_output_dir_model, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR_MODEL, exist_ok=True)
os.makedirs(TRAINING_CHECKPOINTS_DIR_MODEL, exist_ok=True)

print(f"Base output directory for {MODEL_CHECKPOINT_ID}: {base_output_dir_model}")
print(f"Results JSON directory: {RESULTS_OUTPUT_DIR_MODEL}")
print(f"Training checkpoints directory: {TRAINING_CHECKPOINTS_DIR_MODEL}")

# --- Load MedQA Dataset ---
print(f"Loading MedQA dataset: {DATASET_ID_MC}")
try:
    raw_dataset_medqa_full = load_dataset(DATASET_ID_MC, cache_dir=os.environ["HF_DATASETS_CACHE"])
except Exception as e:
    print(f"Error loading MedQA dataset: {e}");
    traceback.print_exc();
    sys.exit(1)


def filter_medqa(example):
    return example["answer_idx"] is not None and example["answer_idx"].strip().upper() in ANSWER_MAP_KEYS_MC


raw_dataset_medqa_full = raw_dataset_medqa_full.filter(filter_medqa)

if 'train' not in raw_dataset_medqa_full or 'test' not in raw_dataset_medqa_full or \
        len(raw_dataset_medqa_full["train"]) == 0 or len(raw_dataset_medqa_full["test"]) == 0:
    print(f"Error: MedQA Dataset {DATASET_ID_MC} must contain non-empty 'train' and 'test' splits after filtering.")
    sys.exit(1)

medqa_dataset_for_training_raw = raw_dataset_medqa_full["test"]
medqa_dataset_for_evaluation_raw = raw_dataset_medqa_full["train"]
print(f"Using {len(medqa_dataset_for_training_raw)} MedQA 'test' examples for fine-tuning.")
print(f"Using {len(medqa_dataset_for_evaluation_raw)} MedQA 'train' examples for evaluation.")

# --- Load Tokenizer for GPT-2 ---
print(f"Loading tokenizer for {MODEL_CHECKPOINT_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_ID, cache_dir=os.environ["TRANSFORMERS_CACHE"])
added_new_pad_token_flag_gpt2 = False
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set GPT-2 pad_token to eos_token ('{tokenizer.eos_token}') ID: {tokenizer.eos_token_id}")
    else:
        new_pad_token_val = "[PAD]"
        tokenizer.add_special_tokens({'pad_token': new_pad_token_val})
        added_new_pad_token_flag_gpt2 = True
        print(f"Added new pad_token '{new_pad_token_val}' for GPT-2.")
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
print(f"GPT-2 Tokenizer: Pad Token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")


# --- Preprocessing Functions for GPT-2 ---
def create_gpt2_sft_text(example_dict):  # Expects a single example dictionary
    question_text = example_dict["question"]
    options_dict = example_dict["options"]
    answer_key = example_dict["answer_idx"].strip().upper()
    prompt_parts = [f"Question: {question_text}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS_MC:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option unavailable]')}")
    else:
        for key_char in ANSWER_MAP_KEYS_MC: prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    prompt_text_for_sft = "\n".join(prompt_parts)
    return prompt_text_for_sft + " " + answer_key


def preprocess_gpt2_sft(examples_batch):  # examples_batch is a dict of lists
    texts_for_sft = []
    batch_size = len(examples_batch["question"])  # Determine batch size

    for i in range(batch_size):
        single_example_dict = {key: examples_batch[key][i] for key in examples_batch.keys()}
        texts_for_sft.append(create_gpt2_sft_text(single_example_dict))

    tokenized_sequences = tokenizer(
        texts_for_sft,
        max_length=MAX_SFT_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        add_special_tokens=True
    )

    input_ids_batch_processed = tokenized_sequences["input_ids"]
    attention_mask_batch_processed = tokenized_sequences["attention_mask"]
    labels_batch_processed = []

    for i in range(len(texts_for_sft)):
        original_prompt_part = texts_for_sft[i].rsplit(' ', 1)[0] + " "
        prompt_part_tokenized = tokenizer(
            original_prompt_part,
            max_length=MAX_SFT_SEQ_LENGTH,
            truncation=True,
            add_special_tokens=True
        )
        len_prompt_tokens = len(prompt_part_tokenized.input_ids)

        current_input_ids = input_ids_batch_processed[i]
        current_labels = list(current_input_ids)

        for k in range(len_prompt_tokens):
            if k < len(current_labels):
                current_labels[k] = -100

        current_attention_mask = attention_mask_batch_processed[i]
        for k in range(len(current_input_ids)):
            if current_attention_mask[k] == 0 and k < len(current_labels):
                current_labels[k] = -100
        labels_batch_processed.append(current_labels)

    return {"input_ids": input_ids_batch_processed, "attention_mask": attention_mask_batch_processed,
            "labels": labels_batch_processed}


def create_gpt2_eval_prompt(example_dict):
    question_text = example_dict["question"]
    options_dict = example_dict["options"]
    prompt_parts = [f"Question: {question_text}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS_MC:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option unavailable]')}")
    else:
        for key_char in ANSWER_MAP_KEYS_MC: prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


# --- Evaluation Function for GPT-2 ---
def evaluate_gpt2_mc_model(model_to_eval, current_tokenizer, eval_dataset_raw,
                           epoch, fine_tuning_duration, model_ckpt_id, task_name_str, results_dir):
    print(f"\nCustom evaluation for GPT-2 MC ({model_ckpt_id}), Epoch {epoch}...")
    model_to_eval.eval()
    model_to_eval.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    original_padding_side = current_tokenizer.padding_side
    current_tokenizer.padding_side = "left"

    eval_prompts = [create_gpt2_eval_prompt(eval_dataset_raw[i]) for i in
                    tqdm(range(len(eval_dataset_raw)), desc="Creating GPT-2 eval prompts")]
    true_answer_letters = [eval_dataset_raw[i]["answer_idx"].strip().upper() for i in range(len(eval_dataset_raw))]

    letter_token_ids = {}
    for letter in ANSWER_MAP_KEYS_MC:
        tokens = current_tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) == 1:
            letter_token_ids[letter] = tokens[0]
        else:
            print(f"Warning: GPT-2 - Letter '{letter}' tokenized to {tokens}.")
            letter_token_ids[letter] = -1

    all_choice_probs_list, all_correctness_list = [], []
    device = model_to_eval.device

    for i in tqdm(range(0, len(eval_prompts), EVAL_BATCH_SIZE_CUSTOM_GPT2), desc=f"Predicting GPT-2 MC Epoch {epoch}"):
        batch_prompts_text = eval_prompts[i: i + EVAL_BATCH_SIZE_CUSTOM_GPT2]
        inputs = current_tokenizer(
            batch_prompts_text, return_tensors="pt", padding="longest", truncation=True,
            max_length=MAX_EVAL_PROMPT_LENGTH, add_special_tokens=True
        ).to(device)
        with torch.no_grad():
            outputs = model_to_eval(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
        next_token_probs_softmax = torch.softmax(next_token_logits, dim=-1).cpu()

        for j_idx_in_batch, single_sample_probs in enumerate(next_token_probs_softmax):
            current_probs = np.zeros(NUM_CHOICES_MC, dtype=float)
            for choice_idx, key in enumerate(ANSWER_MAP_KEYS_MC):
                tid = letter_token_ids.get(key, -1)
                if tid != -1:
                    current_probs[choice_idx] = single_sample_probs[tid].item()
                else:
                    current_probs[choice_idx] = 0.0

            sum_p = np.sum(current_probs)
            if sum_p > 1e-6:
                current_probs /= sum_p
            else:
                current_probs[:] = 1.0 / NUM_CHOICES_MC
            all_choice_probs_list.append(current_probs.tolist())

            pred_idx = np.argmax(current_probs)
            pred_l = ANSWER_MAP_KEYS_MC[pred_idx]
            true_l = true_answer_letters[i + j_idx_in_batch]
            all_correctness_list.append(int(pred_l == true_l))

    current_tokenizer.padding_side = original_padding_side
    accuracy = np.mean(all_correctness_list) if all_correctness_list else 0.0

    model_name_part = model_ckpt_id.replace('/', '_')
    fn = f"{model_name_part}_{task_name_str}_{fine_tuning_duration:.2f}_Acc_{accuracy:.4f}_epochs_{epoch}.json"
    fp = os.path.join(results_dir, fn)
    with open(fp, "w") as f:
        json.dump({"logits": all_choice_probs_list, "responses": all_correctness_list}, f, indent=2)
    print(f"Saved GPT-2 MC results for epoch {epoch} to {fp} (Acc: {accuracy:.4f})")
    return accuracy


# --- Main Script Flow for GPT-2 ---
if __name__ == "__main__":
    print(f"\n===== Processing Model: {MODEL_CHECKPOINT_ID} for MedQA MC =====")
    eval_epoch0_start_time = time.time()
    config_epoch0 = AutoConfig.from_pretrained(MODEL_CHECKPOINT_ID, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if tokenizer.pad_token_id is not None and hasattr(config_epoch0,
                                                      "pad_token_id") and config_epoch0.pad_token_id != tokenizer.pad_token_id:
        config_epoch0.pad_token_id = tokenizer.pad_token_id
    if hasattr(config_epoch0,
               "eos_token_id") and config_epoch0.eos_token_id is None and tokenizer.eos_token_id is not None: config_epoch0.eos_token_id = tokenizer.eos_token_id
    if hasattr(config_epoch0,
               "bos_token_id") and config_epoch0.bos_token_id is None and tokenizer.bos_token_id is not None: config_epoch0.bos_token_id = tokenizer.bos_token_id

    model_epoch0 = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT_ID, config=config_epoch0,
                                                        cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if added_new_pad_token_flag_gpt2 and hasattr(model_epoch0, 'resize_token_embeddings') and len(
            tokenizer) > model_epoch0.config.vocab_size:
        print(f"Resizing Epoch 0 GPT-2 model embeddings: {model_epoch0.config.vocab_size} -> {len(tokenizer)}")
        model_epoch0.resize_token_embeddings(len(tokenizer))
    if model_epoch0.config.pad_token_id != tokenizer.pad_token_id and tokenizer.pad_token_id is not None: model_epoch0.config.pad_token_id = tokenizer.pad_token_id

    evaluate_gpt2_mc_model(model_epoch0, tokenizer, medqa_dataset_for_evaluation_raw, 0, 0.0, MODEL_CHECKPOINT_ID,
                           TASK_NAME_FOR_RESULTS_FILE_MC, RESULTS_OUTPUT_DIR_MODEL)
    del model_epoch0
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"Epoch 0 evaluation took {time.time() - eval_epoch0_start_time:.2f}s")

    print(f"\nTokenizing MedQA 'test' split for GPT-2 SFT...")
    num_proc_sft_tok = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
    tokenized_medqa_for_sft = medqa_dataset_for_training_raw.map(
        preprocess_gpt2_sft, batched=True, num_proc=num_proc_sft_tok,
        remove_columns=medqa_dataset_for_training_raw.column_names, desc="Tokenizing MedQA for GPT-2 SFT"
    )

    print(f"\n--- Training {MODEL_CHECKPOINT_ID} for {NUM_EPOCHS_TOTAL_MC} epochs ---")
    config_train = AutoConfig.from_pretrained(MODEL_CHECKPOINT_ID, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if tokenizer.pad_token_id is not None and hasattr(config_train,
                                                      "pad_token_id") and config_train.pad_token_id != tokenizer.pad_token_id: config_train.pad_token_id = tokenizer.pad_token_id
    if hasattr(config_train,
               "eos_token_id") and config_train.eos_token_id is None and tokenizer.eos_token_id is not None: config_train.eos_token_id = tokenizer.eos_token_id
    if hasattr(config_train,
               "bos_token_id") and config_train.bos_token_id is None and tokenizer.bos_token_id is not None: config_train.bos_token_id = tokenizer.bos_token_id

    model_for_training = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINT_ID, config=config_train,
                                                              cache_dir=os.environ["TRANSFORMERS_CACHE"])
    if added_new_pad_token_flag_gpt2 and hasattr(model_for_training, 'resize_token_embeddings') and len(
            tokenizer) > model_for_training.config.vocab_size:
        print(f"Resizing Training GPT-2 model embeddings: {model_for_training.config.vocab_size} -> {len(tokenizer)}")
        model_for_training.resize_token_embeddings(len(tokenizer))
    if model_for_training.config.pad_token_id != tokenizer.pad_token_id and tokenizer.pad_token_id is not None: model_for_training.config.pad_token_id = tokenizer.pad_token_id

    grad_accum_gpt2 = max(1, EFFECTIVE_BATCH_SIZE_GPT2 // TRAIN_BATCH_SIZE_GPT2) if TRAIN_BATCH_SIZE_GPT2 > 0 else 1
    training_args = TrainingArguments(
        output_dir=TRAINING_CHECKPOINTS_DIR_MODEL, overwrite_output_dir=True, num_train_epochs=NUM_EPOCHS_TOTAL_MC,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_GPT2, gradient_accumulation_steps=grad_accum_gpt2,
        learning_rate=LEARNING_RATE_GPT2, weight_decay=0.01, save_strategy="epoch", eval_strategy="no",
        logging_strategy="steps", logging_steps=max(1, len(tokenized_medqa_for_sft) // (
                    TRAIN_BATCH_SIZE_GPT2 * grad_accum_gpt2 * (
                torch.cuda.device_count() if torch.cuda.is_available() else 1) * 10) + 1),
        fp16=torch.cuda.is_available(), report_to="none", remove_unused_columns=True, seed=random_seed,
        save_total_limit=None, dataloader_num_workers=min(2, os.cpu_count() if os.cpu_count() else 1),
    )
    data_collator_sft_gpt2 = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model_for_training, args=training_args, train_dataset=tokenized_medqa_for_sft,
        tokenizer=tokenizer, data_collator=data_collator_sft_gpt2,
    )

    print(f"Starting SFT training for {MODEL_CHECKPOINT_ID}...")
    train_start_time = time.time()
    train_result = trainer.train()  # Corrected variable name
    total_training_time_for_run = time.time() - train_start_time
    print(f"Training for {MODEL_CHECKPOINT_ID} completed in {total_training_time_for_run:.2f}s.")
    del model_for_training, trainer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n--- Checkpoint Evaluation for {MODEL_CHECKPOINT_ID} ---")
    for desired_epoch in EPOCHS_TO_EVALUATE_MC:
        if desired_epoch == 0: continue
        found_ckpt, ckpt_path_to_load = False, None
        if os.path.exists(TRAINING_CHECKPOINTS_DIR_MODEL):
            ckpt_folders = sorted(
                [os.path.join(TRAINING_CHECKPOINTS_DIR_MODEL, d) for d in os.listdir(TRAINING_CHECKPOINTS_DIR_MODEL) if
                 d.startswith("checkpoint-") and os.path.isdir(os.path.join(TRAINING_CHECKPOINTS_DIR_MODEL, d))])
            for ckpt_candidate in ckpt_folders:
                state_file = os.path.join(ckpt_candidate, "trainer_state.json")
                if os.path.exists(state_file):
                    with open(state_file, "r") as f:
                        state = json.load(f)
                    if round(state.get("epoch", 0.0)) == desired_epoch:
                        ckpt_path_to_load = ckpt_candidate;
                        found_ckpt = True;
                        break

        if found_ckpt and ckpt_path_to_load:
            print(f"Loading checkpoint for GPT-2 epoch {desired_epoch} from {ckpt_path_to_load}")
            config_ckpt = AutoConfig.from_pretrained(ckpt_path_to_load)
            if tokenizer.pad_token_id is not None and hasattr(config_ckpt,
                                                              "pad_token_id") and config_ckpt.pad_token_id != tokenizer.pad_token_id: config_ckpt.pad_token_id = tokenizer.pad_token_id
            if hasattr(config_ckpt,
                       "eos_token_id") and config_ckpt.eos_token_id is None and tokenizer.eos_token_id is not None: config_ckpt.eos_token_id = tokenizer.eos_token_id
            if hasattr(config_ckpt,
                       "bos_token_id") and config_ckpt.bos_token_id is None and tokenizer.bos_token_id is not None: config_ckpt.bos_token_id = tokenizer.bos_token_id
            model_checkpoint = AutoModelForCausalLM.from_pretrained(ckpt_path_to_load, config=config_ckpt)
            if model_checkpoint.config.pad_token_id != tokenizer.pad_token_id and tokenizer.pad_token_id is not None: model_checkpoint.config.pad_token_id = tokenizer.pad_token_id
            evaluate_gpt2_mc_model(model_checkpoint, tokenizer, medqa_dataset_for_evaluation_raw, desired_epoch,
                                   total_training_time_for_run, MODEL_CHECKPOINT_ID, TASK_NAME_FOR_RESULTS_FILE_MC,
                                   RESULTS_OUTPUT_DIR_MODEL)
            del model_checkpoint
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        else:
            print(f"Warning: Checkpoint for GPT-2 epoch {desired_epoch} not found in {TRAINING_CHECKPOINTS_DIR_MODEL}")

    print(f"\n===== GPT-2 {TASK_NAME_FOR_RESULTS_FILE_MC} Experiment Completed =====")
    print(f"All outputs for GPT-2 in: {base_output_dir_model}")