# ----- Imports -----
import sys
import os
import datetime
import random
import traceback
import json
from tqdm import tqdm
import re
import math  # For ceil
import time  # For heuristic script
import gc  # For heuristic script
from typing import Sized, Iterator, Dict, List, Any, Optional, Tuple  # For Sampler type hints and others

import torch
import numpy as np

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset  # Added Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,  # Will be replaced by CustomHeuristicTrainer in the main loop
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    AutoConfig,
)
from torch.utils.data import DataLoader, Sampler  # Added Sampler (Dataset, Subset already imported via datasets)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
# import evaluate # Not strictly needed if compute_metrics only returns loss from trainer
import transformers  # For logging verbosity from heuristic script

# ----- Baseline Configuration (MedQA) -----
model_id = "meta-llama/Meta-Llama-3.1-8B"
dataset_id = "GBaker/MedQA-USMLE-4-options"
# output_dir will be set per run

# Sequence lengths
max_seq_length = 512 + 10
max_prompt_len_config = 512
max_target_len_config = 5

# Training hyperparameters from MedQA script
per_device_train_bs = 8  # Corresponds to PER_DEVICE_TRAIN_BATCH_SIZE in heuristic
per_device_eval_bs_trainer = 16  # Corresponds to PER_DEVICE_EVAL_BATCH_SIZE
grad_accum_steps = 2  # Corresponds to GRADIENT_ACCUMULATION_STEPS
num_train_epochs = 5  # Corresponds to NUM_TRAIN_EPOCHS for max_steps calculation
early_stopping_patience_train = 3  # Corresponds to EARLY_STOPPING_PATIENCE
learning_rate = 2e-4  # Corresponds to LEARNING_RATE
weight_decay_train = 0.01  # Corresponds to WEIGHT_DECAY

# LoRA parameters
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05

random_seed = 63
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----- Heuristic Configuration -----
# Heuristic Params (from Llama heuristic script)
ORDERING = 'easiest'
COMPETENCY_PARAM = 5
MIN_TRAIN_PERCENT = 0.05
C_INIT = 0.01

LOGGING_STEPS_HEURISTIC = 100  # Used by sampler for logging its state

# --- Root Output Directory ---
BASE_OUTPUT_DIR_ROOT = "./medqa_llama3_8b_heuristic_runs"
os.makedirs(BASE_OUTPUT_DIR_ROOT, exist_ok=True)

# --- Loop Definitions (from heuristic script) ---
difficulty_measures_to_run = ['sentence_length', 'word_rarity']
schedulers_to_run = ['linear', 'root']

# Evaluation configuration (MedQA)
EVAL_SPLIT_CUSTOM_EVAL = "test"
MAX_NEW_TOKENS_GEN_CUSTOM = 1
TEMPERATURE_CUSTOM_EVAL = 0.1
TOP_P_CUSTOM_EVAL = 0.9
DO_SAMPLE_CUSTOM_EVAL = False
EVAL_BATCH_SIZE_CUSTOM = 16

ANSWER_MAP_KEYS = ["A", "B", "C", "D"]

# --- Environment Setup (from heuristic script) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Uncommented for OOM
print(f"Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

# HF_HOME = "/path_to_your_hf_home" # Example, customize if needed
# if HF_HOME and os.path.exists(os.path.dirname(HF_HOME)):
# os.environ["HF_HOME"] = HF_HOME
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
# os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
# else:
# print("HF_HOME not set or path invalid, using default Hugging Face cache locations.")
# pass # Use default HF cache locations

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ensure default cache directories exist if specific ones are not set or invalid
default_hf_home = os.path.expanduser("~/.cache/huggingface")
transformers_cache_dir = os.environ.get("TRANSFORMERS_CACHE", os.path.join(default_hf_home, "hub"))
datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE", os.path.join(default_hf_home, "datasets"))
os.makedirs(transformers_cache_dir, exist_ok=True)
os.makedirs(datasets_cache_dir, exist_ok=True)

# --- Global Random Seed (from heuristic script) ---
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed);
np.random.seed(random_seed);
random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
transformers.logging.set_verbosity_warning()


# ----- Helper Functions for Heuristic Difficulty -----
def simple_tokenize(sent: str) -> List[str]:
    if not isinstance(sent, str): return []
    sent = re.sub(r'\s+', ' ', sent)
    tokens = [x.strip() for x in re.findall(r"[\w']+|[^\w\s]", sent) if x.strip()]
    return tokens


def get_example_rarities(texts: List[str]) -> List[float]:
    if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts): return [0.0] * len(texts)
    tokenized_corpus = [simple_tokenize(text) for text in texts];
    vocab = set();
    counts: Dict[str, int] = dict();
    N_tokens = 0
    for tokens_in_doc in tokenized_corpus:
        valid_tokens = [t for t in tokens_in_doc if t];
        vocab.update(valid_tokens);
        N_tokens += len(valid_tokens)
        for tok in valid_tokens: counts.setdefault(tok, 0); counts[tok] += 1
    if N_tokens == 0: return [0.0] * len(texts)
    result: List[float] = [];
    epsilon = 1e-9
    for tokens_in_doc in tokenized_corpus:
        valid_tokens = [t for t in tokens_in_doc if t]
        if not valid_tokens:
            p_hat = 0.0
        else:
            log_probs = [np.log(counts.get(tok, 0) / N_tokens + epsilon) for tok in valid_tokens]
            p_hat = -np.mean(log_probs) if log_probs else 0.0
        result.append(p_hat)
    return result


def calculate_difficulty_scores(dataset_split: Dataset, difficulty_measurer: str, text_column: str = 'text') -> tuple[
    List[float], List[int]]:
    print(f"Calculating '{difficulty_measurer}' difficulty scores on column '{text_column}'...")
    texts = dataset_split[text_column];
    original_indices = list(range(len(texts)))

    processed_texts_with_indices = []
    for i, text_item in enumerate(texts):
        if isinstance(text_item, str) and text_item.strip():
            processed_texts_with_indices.append((text_item, i))

    valid_texts = [item[0] for item in processed_texts_with_indices]
    valid_indices = [item[1] for item in processed_texts_with_indices]

    if len(valid_indices) < len(original_indices):
        print(
            f"Warning: Kept {len(valid_indices)} out of {len(original_indices)} original examples after filtering invalid text for difficulty scoring.")

    if not valid_texts: return [], []

    if difficulty_measurer == 'sentence_length':
        difficulty_scores = [len(text) for text in valid_texts];
        print("Calculated sentence length difficulty.")
    elif difficulty_measurer == 'word_rarity':
        difficulty_scores = get_example_rarities(valid_texts);
        print("Calculated word rarity difficulty.")
    else:
        raise ValueError(f"Unsupported difficulty_measurer: {difficulty_measurer}")

    return difficulty_scores, valid_indices


def combine_text_for_difficulty(example: Dict[str, Any]) -> Dict[str, str]:
    question = example["question"].strip() if isinstance(example["question"], str) else ""
    options_dict = example["options"]
    options_text_parts = []
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            option_val = options_dict.get(key_char)
            options_text_parts.append(str(option_val) if option_val is not None else "")
    options_combined = " ".join(filter(None, options_text_parts))
    full_text = f"{question} {options_combined}".strip()
    if not full_text:
        return {"full_text_for_difficulty": " "}
    return {"full_text_for_difficulty": full_text}


# --- Custom Heuristic Sampler Definition ---
class HeuristicSampler(Sampler[int]):
    def __init__(self, num_samples_total: int, batch_size: int, sorted_indices: list[int], heuristic_config: dict,
                 num_replicas: int = 1, rank: int = 0, seed: int = 42):
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas: raise ValueError("Invalid num_replicas or rank.")
        if not isinstance(batch_size, int) or batch_size <= 0: raise ValueError("batch_size should be positive.")
        self.num_replicas = num_replicas;
        self.rank = rank;
        self.epoch = 0;
        self.seed = seed;
        self.batch_size = batch_size
        self._full_data_len = num_samples_total;
        self._sorted_indices = sorted_indices;
        self.heuristic_config = heuristic_config

        min_data_from_percent = int(self.heuristic_config['min_train_percent'] * self._full_data_len)
        min_data_from_batch = batch_size * num_replicas
        min_len_abs_perc = max(1, min_data_from_batch, min_data_from_percent)

        self.abs_min_train_length = self.heuristic_config.get('abs_min_train_length', 0)
        self._min_train_length = max(min_len_abs_perc, self.abs_min_train_length)
        if self._min_train_length > self._full_data_len: self._min_train_length = self._full_data_len

        self.indices_for_epoch: List[int] = [];
        self.num_samples_epoch_replica = 0
        self.set_epoch(0)

    def _get_num_samples_for_epoch(self, epoch: int) -> int:
        scheduler_type = self.heuristic_config['scheduler'];
        num_total = self._full_data_len
        competency_epoch_config = max(1, self.heuristic_config['competency_param'])
        c_init_val = self.heuristic_config['c_init']
        current_epoch_float = float(epoch)

        if scheduler_type == 'linear':
            epoch_competency = c_init_val + (1.0 - c_init_val) * (
                        current_epoch_float / competency_epoch_config) if current_epoch_float < competency_epoch_config else 1.0
        elif scheduler_type == 'root':
            epoch_competency = c_init_val + (1.0 - c_init_val) * np.sqrt(
                current_epoch_float / competency_epoch_config) if current_epoch_float < competency_epoch_config else 1.0
        else:
            raise NotImplementedError(f"Scheduler '{scheduler_type}' unknown.")

        num_train = int(epoch_competency * num_total)
        num_train = max(self._min_train_length, num_train);
        num_train = min(num_train, num_total)
        return num_train

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch;
        num_samples_for_epoch = self._get_num_samples_for_epoch(epoch)
        new_indices = self._sorted_indices[:num_samples_for_epoch]

        if epoch == 0 or len(new_indices) != len(self.indices_for_epoch) or epoch % LOGGING_STEPS_HEURISTIC == 0:
            print(f"[Sampler] Epoch {epoch}: Selecting {len(new_indices)} samples out of {self._full_data_len} total.")

        self.indices_for_epoch = new_indices

        if self.num_replicas > 1:
            num_samples_this_epoch_total = len(self.indices_for_epoch)
            if num_samples_this_epoch_total % self.num_replicas != 0:
                self.indices_for_epoch = self.indices_for_epoch[:num_samples_this_epoch_total - (
                            num_samples_this_epoch_total % self.num_replicas)]
            self.num_samples_epoch_replica = len(self.indices_for_epoch) // self.num_replicas
        else:
            self.num_samples_epoch_replica = len(self.indices_for_epoch)

    def __iter__(self) -> Iterator[int]:
        if not self.indices_for_epoch: return iter([])
        g = torch.Generator();
        g.manual_seed(self.seed + self.epoch)
        indices_shuffled = [self.indices_for_epoch[i] for i in
                            torch.randperm(len(self.indices_for_epoch), generator=g).tolist()]
        if self.num_replicas > 1:
            return iter(indices_shuffled[self.rank: len(indices_shuffled): self.num_replicas])
        else:
            return iter(indices_shuffled)

    def __len__(self) -> int:
        return self.num_samples_epoch_replica


# --- Custom Trainer Definition ---
class CustomHeuristicTrainer(Trainer):
    def __init__(self, *args: Any, sorted_indices: Optional[List[int]] = None,
                 heuristic_config: Optional[Dict[str, Any]] = None, num_samples_total: Optional[int] = None,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        if sorted_indices is None or heuristic_config is None or num_samples_total is None:
            raise ValueError("CustomHeuristicTrainer requires sorted_indices, heuristic_config, and num_samples_total.")
        self.sorted_indices = sorted_indices;
        self.heuristic_config = heuristic_config;
        self.num_samples_total = num_samples_total
        print("CustomHeuristicTrainer initialized.")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None: raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        heuristic_sampler = HeuristicSampler(
            num_samples_total=self.num_samples_total,
            batch_size=self._train_batch_size,
            sorted_indices=self.sorted_indices,
            heuristic_config=self.heuristic_config,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=self.args.seed
        )
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=heuristic_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ----- Preprocessing function for Training (MedQA) -----
def preprocess_sft_letter_target(examples, tokenizer_ref, max_seq_len, max_prompt_len, max_target_len):
    inputs_tokenized_batch = []
    labels_tokenized_batch = []

    for i in range(len(examples["question"])):
        question = str(examples["question"][i] if examples["question"][i] is not None else "").strip()
        options_dict = examples["options"][i]
        answer_idx_key = str(examples["answer_idx"][i] if examples["answer_idx"][i] is not None else "").strip().upper()

        prompt_parts = [f"Question: {question}\n\nOptions:"]
        if isinstance(options_dict, dict):
            for key_char in ANSWER_MAP_KEYS:
                option_text = options_dict.get(key_char)
                prompt_parts.append(
                    f"{key_char}) {str(option_text) if option_text is not None else '[Option text not found]'}")
        else:
            for key_char in ANSWER_MAP_KEYS:
                prompt_parts.append(f"{key_char}) [Invalid options format]")
        prompt_parts.append("\nAnswer:")
        prompt_text = "\n".join(prompt_parts)

        target_letter = answer_idx_key if answer_idx_key in ANSWER_MAP_KEYS else ""

        tokenized_prompt = tokenizer_ref(prompt_text, truncation=True, max_length=max_prompt_len, padding=False,
                                         add_special_tokens=True)
        tokenized_target = tokenizer_ref(target_letter, truncation=True, max_length=max_target_len, padding=False,
                                         add_special_tokens=False)

        prompt_input_ids = tokenized_prompt.input_ids
        target_input_ids = tokenized_target.input_ids

        input_ids = prompt_input_ids + target_input_ids
        if tokenizer_ref.eos_token_id is not None:
            input_ids.append(tokenizer_ref.eos_token_id)

        labels = ([-100] * len(prompt_input_ids)) + target_input_ids
        if tokenizer_ref.eos_token_id is not None:
            labels.append(tokenizer_ref.eos_token_id)

        current_len = len(input_ids)
        if current_len > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        inputs_tokenized_batch.append(input_ids)
        labels_tokenized_batch.append(labels)

    return {"input_ids": inputs_tokenized_batch, "labels": labels_tokenized_batch}


# ----- Helper: Prompt Creation for Custom Evaluation (MedQA) -----
def create_custom_evaluation_prompt(example):
    question = str(example["question"] if example["question"] is not None else "").strip()
    options_dict = example["options"]
    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            option_text = options_dict.get(key_char)
            prompt_parts.append(
                f"{key_char}) {str(option_text) if option_text is not None else '[Option text not found]'}")
    else:
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


# ----- Main Training Function (incorporating MedQA logic) -----
def run_training_heuristic_medqa(
        difficulty_measurer: str,
        training_scheduler: str,
        current_output_dir: str,
        global_tokenizer
):
    print("\n" + "=" * 80);
    print(f"Starting Run: Difficulty='{difficulty_measurer}', Scheduler='{training_scheduler}'");
    print(f"Output Dir: {current_output_dir}");
    print("=" * 80 + "\n")
    os.makedirs(current_output_dir, exist_ok=True)

    model_train_run, trainer_run = None, None
    final_accuracy_custom = 0.0
    run_status = "failed"

    try:
        # 1. Load and filter dataset
        print(f"Loading dataset: {dataset_id}")
        dataset_full = load_dataset(dataset_id, cache_dir=datasets_cache_dir)

        valid_answers = ['A', 'B', 'C', 'D']
        dataset_full = dataset_full.filter(
            lambda ex: ex["answer_idx"] is not None and ex["answer_idx"].strip().upper() in valid_answers)
        print(f"Filtered dataset to {len(dataset_full['train'])} train examples with valid answer_idx.")

        if "validation" not in dataset_full or "test" not in dataset_full:
            print("Validation or test split not found. Splitting train -> 80% train, 10% val, 10% test...")
            if 'train' not in dataset_full or len(dataset_full['train']) == 0:
                sys.exit("Error: Original 'train' split not found or empty after filtering!")

            stratify_column = 'answer_idx' if 'answer_idx' in dataset_full['train'].column_names else None
            try:
                # Check if stratification is possible
                if stratify_column:
                    unique_labels = dataset_full['train'].unique(stratify_column)
                    label_counts = {
                        label: dataset_full['train'].filter(lambda ex: ex[stratify_column] == label).num_rows for label
                        in unique_labels}
                    min_samples = min(label_counts.values()) if label_counts else 0
                    # If any class has fewer than 2 samples, stratification for train/test split is problematic
                    if min_samples < 2:  # Stricter for train/test + test/val later
                        print(
                            f"Warning: Stratification by {stratify_column} might fail due to low sample counts for some classes (min_samples: {min_samples}). Attempting anyway.")

                train_temp_split = dataset_full['train'].train_test_split(test_size=0.2, seed=random_seed, shuffle=True,
                                                                          stratify_by_column=stratify_column)

                # For val_test_split, ensure its 'test' part can also be stratified
                stratify_column_val_test = stratify_column if stratify_column and stratify_column in train_temp_split[
                    'test'].column_names else None
                if stratify_column_val_test:
                    unique_labels_val_test = train_temp_split['test'].unique(stratify_column_val_test)
                    label_counts_val_test = {label: train_temp_split['test'].filter(
                        lambda ex: ex[stratify_column_val_test] == label).num_rows for label in unique_labels_val_test}
                    min_samples_val_test = min(label_counts_val_test.values()) if label_counts_val_test else 0
                    if min_samples_val_test < 2:
                        print(
                            f"Warning: Stratification for val/test split might fail for {stratify_column_val_test} (min_samples: {min_samples_val_test}).")

                val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=random_seed,
                                                                           shuffle=True,
                                                                           stratify_by_column=stratify_column_val_test)

            except Exception as e_stratify:  # Fallback if stratification fails
                print(f"Stratified split failed ({e_stratify}), falling back to random split.")
                train_temp_split = dataset_full['train'].train_test_split(test_size=0.2, seed=random_seed, shuffle=True)
                val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, seed=random_seed,
                                                                           shuffle=True)

            current_run_dataset = DatasetDict({
                "train": train_temp_split["train"],
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            })
        else:
            print("Using predefined train, validation, and test splits.")
            current_run_dataset = dataset_full

        print("Final dataset splits for this run:", current_run_dataset)
        if len(current_run_dataset['train']) == 0 or len(current_run_dataset['validation']) == 0:
            raise ValueError("Training or Validation dataset is empty for this run!")

        # 2. Calculate Difficulty Scores & Get Sorted Indices
        train_dataset_for_difficulty_calc = current_run_dataset['train'].map(
            combine_text_for_difficulty,
            num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        )

        difficulty_scores, valid_original_indices = calculate_difficulty_scores(
            train_dataset_for_difficulty_calc, difficulty_measurer, text_column='full_text_for_difficulty'
        )
        train_dataset_for_sorting = current_run_dataset['train'].select(valid_original_indices)

        if len(difficulty_scores) != len(train_dataset_for_sorting):
            raise RuntimeError(
                f"Mismatch difficulty scores ({len(difficulty_scores)}) and final data size ({len(train_dataset_for_sorting)}).")
        print(f"Selected {len(train_dataset_for_sorting)} training examples with valid difficulty scores.")

        print(f"Sorting training data indices by difficulty ({ORDERING})...")
        start_time = time.time()
        difficulties_final = np.array(difficulty_scores)
        if ORDERING == 'easiest':
            sorted_sampler_indices = np.argsort(difficulties_final).tolist()
        elif ORDERING == 'hardest':
            sorted_sampler_indices = np.argsort(difficulties_final)[::-1].tolist()
        else:
            raise NotImplementedError(f"Ordering '{ORDERING}' not implemented.")
        print(f"Sorting completed in {time.time() - start_time:.2f}s.")
        num_samples_total_for_sampler = len(train_dataset_for_sorting)

        # 3. Tokenize dataset for training
        print(f"Tokenizing dataset for training (target: single letter). Max sequence length: {max_seq_length}")
        tokenized_train_dataset = train_dataset_for_sorting.map(
            lambda ex: preprocess_sft_letter_target(ex, global_tokenizer, max_seq_length, max_prompt_len_config,
                                                    max_target_len_config),
            batched=True, num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
            remove_columns=train_dataset_for_sorting.column_names
        )
        tokenized_validation_dataset = current_run_dataset['validation'].map(
            lambda ex: preprocess_sft_letter_target(ex, global_tokenizer, max_seq_length, max_prompt_len_config,
                                                    max_target_len_config),
            batched=True, num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
            remove_columns=current_run_dataset['validation'].column_names
        )
        tokenized_test_dataset = current_run_dataset['test'].map(  # For custom eval later
            lambda ex: preprocess_sft_letter_target(ex, global_tokenizer, max_seq_length, max_prompt_len_config,
                                                    max_target_len_config),
            batched=True, num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
            remove_columns=current_run_dataset['test'].column_names
        )
        print("Dataset tokenization complete.")
        tokenized_train_dataset.set_format(type="torch")
        tokenized_validation_dataset.set_format(type="torch")
        print(f"Training dataset formatted. Columns: {tokenized_train_dataset.column_names}")

        # 4. Model Loading & QLoRA Setup
        print("Configuring BitsAndBytes for QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        print(f"Loading base model {model_id} for training...")
        model_train_run = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=transformers_cache_dir
        )
        if len(global_tokenizer) > model_train_run.config.vocab_size:
            print(
                f"Resizing model token embeddings from {model_train_run.config.vocab_size} to {len(global_tokenizer)}")
            model_train_run.resize_token_embeddings(len(global_tokenizer))
        if model_train_run.config.pad_token_id != global_tokenizer.pad_token_id:
            print(f"Syncing model pad_token_id to tokenizer's: {global_tokenizer.pad_token_id}")
            model_train_run.config.pad_token_id = global_tokenizer.pad_token_id

        model_train_run = prepare_model_for_kbit_training(model_train_run, use_gradient_checkpointing=True)
        lora_target_modules_train = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config_train = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=lora_target_modules_train, bias="none"
        )
        model_train_run = get_peft_model(model_train_run, peft_config_train)
        print("QLoRA Causal LM model prepared for training.")
        model_train_run.print_trainable_parameters()

        # 5. Data Collator
        data_collator_train = DataCollatorForSeq2Seq(
            global_tokenizer, model=model_train_run, label_pad_token_id=-100, padding="longest"
        )

        # 6. Metrics for Trainer (only loss)
        def compute_metrics_for_trainer(eval_pred):
            if hasattr(eval_pred, 'metrics') and eval_pred.metrics is not None:
                eval_loss = eval_pred.metrics.get("eval_loss", -1.0)
                return {"loss": eval_loss}
            print("Warning: eval_pred.metrics not found, cannot extract eval_loss for compute_metrics_for_trainer.")
            return {"loss": -1.0}

        # 7. Calculate max_steps
        try:
            world_size_train = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception:
            world_size_train = 1
        total_train_batch_size_effective = per_device_train_bs * world_size_train * grad_accum_steps
        if total_train_batch_size_effective == 0: raise ValueError("Total effective train batch size is zero.")
        if num_samples_total_for_sampler == 0:
            print("Warning: num_samples_total_for_sampler is zero. Max steps might be small or zero.")
            if len(current_run_dataset['train']) == 0:
                raise ValueError("The 'train' split is empty before difficulty calculation.")
            steps_per_epoch_full = 1
        else:
            steps_per_epoch_full = math.ceil(num_samples_total_for_sampler / total_train_batch_size_effective)
        calculated_max_steps = math.ceil(num_train_epochs * steps_per_epoch_full)
        print(
            f"Effective Batch Size: {total_train_batch_size_effective}, Steps/Epoch (Full Dataset): {steps_per_epoch_full}")
        print(f"Calculated max_steps for {num_train_epochs} nominal epochs: {calculated_max_steps}")
        if calculated_max_steps == 0 and num_samples_total_for_sampler > 0:
            calculated_max_steps = 1
            print(f"Warning: max_steps was 0 with data, setting to 1.")
        if num_samples_total_for_sampler == 0:
            calculated_max_steps = 0
            print("No samples for sampler, max_steps set to 0.")

        # 8. Training Arguments
        training_args = TrainingArguments(
            output_dir=current_output_dir, overwrite_output_dir=True, do_train=True, do_eval=True,
            per_device_train_batch_size=per_device_train_bs,
            per_device_eval_batch_size=per_device_eval_bs_trainer,
            gradient_accumulation_steps=grad_accum_steps,
            max_steps=calculated_max_steps if calculated_max_steps > 0 else 1,
            learning_rate=learning_rate,
            weight_decay=weight_decay_train,
            bf16=True,
            logging_strategy="steps", logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            prediction_loss_only=True,  # *** IMPORTANT FIX FOR OOM ***
            load_best_model_at_end=True, metric_for_best_model="loss",
            greater_is_better=False,
            report_to=[],
            seed=random_seed,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        heuristic_config_dict = {
            "scheduler": training_scheduler, "ordering": ORDERING,
            "competency_param": COMPETENCY_PARAM, "min_train_percent": MIN_TRAIN_PERCENT,
            "c_init": C_INIT,
        }

        # 9. Initialize CustomHeuristicTrainer
        print("Initializing CustomHeuristicTrainer...")
        trainer_run = CustomHeuristicTrainer(
            model=model_train_run, args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_validation_dataset,
            tokenizer=global_tokenizer, data_collator=data_collator_train,
            compute_metrics=compute_metrics_for_trainer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_train)],
            sorted_indices=sorted_sampler_indices,
            heuristic_config=heuristic_config_dict,
            num_samples_total=num_samples_total_for_sampler
        )

        # 10. Train
        final_adapter_path = os.path.join(current_output_dir, "final_qlora_adapter")
        if calculated_max_steps > 0 and num_samples_total_for_sampler > 0:
            print(f"Starting QLoRA Causal LM fine-tuning ({model_id}, max_steps={calculated_max_steps})...")
            train_result = trainer_run.train()
            trainer_run.save_model(final_adapter_path)
            global_tokenizer.save_pretrained(final_adapter_path)
            print(f"✅ Training complete. QLoRA adapter saved to {final_adapter_path}")
            trainer_run.log_metrics("train", train_result.metrics)
            trainer_run.save_metrics("train", train_result.metrics)
            trainer_run.save_state()
        else:
            print("Skipping training as max_steps or num_samples_total_for_sampler is zero.")
            os.makedirs(final_adapter_path, exist_ok=True)
            dummy_adapter_config = {"base_model_name_or_path": model_id, "peft_type": "LORA"}  # Basic config
            with open(os.path.join(final_adapter_path, "adapter_config.json"), "w") as f:
                json.dump(dummy_adapter_config, f)
            # Create an empty dummy adapter model file if PeftModel.from_pretrained strictly requires it
            # For some versions, just adapter_config.json might be enough to avoid crashing the load attempt.
            # with open(os.path.join(final_adapter_path, "adapter_model.bin"), "w") as f: pass # Empty file
            print("Created dummy adapter directory for evaluation phase due to no training.")

        # 11. Custom Evaluation Phase (MedQA)
        print("\n\n--- STARTING CUSTOM EVALUATION PHASE (for this run) ---")
        eval_results_file = os.path.join(current_output_dir, f"evaluation_results_{EVAL_SPLIT_CUSTOM_EVAL}.json")

        if not os.path.exists(final_adapter_path) or not os.path.isdir(final_adapter_path):
            print(
                f"ERROR: Trained adapter path '{final_adapter_path}' not found. Cannot run custom evaluation for this run.")
        else:
            base_model_eval = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True, cache_dir=transformers_cache_dir
            )
            if len(global_tokenizer) > base_model_eval.config.vocab_size:
                base_model_eval.resize_token_embeddings(len(global_tokenizer))
            if base_model_eval.config.pad_token_id != global_tokenizer.pad_token_id:
                base_model_eval.config.pad_token_id = global_tokenizer.pad_token_id

            try:
                if calculated_max_steps > 0 and num_samples_total_for_sampler > 0:
                    model_eval = PeftModel.from_pretrained(base_model_eval, final_adapter_path)
                    model_eval.eval()
                    print("PEFT model (eval) loaded and set to eval mode.")
                else:
                    model_eval = base_model_eval  # Use base model if no training
                    model_eval.eval()
                    print("No training performed, using base model for evaluation.")

                # Use the original (untokenized) test split from current_run_dataset for custom eval
                custom_eval_dataset_raw = current_run_dataset[EVAL_SPLIT_CUSTOM_EVAL]
                all_prompts_custom_eval = [create_custom_evaluation_prompt(ex) for ex in custom_eval_dataset_raw]
                all_true_letters = [ex["answer_idx"].strip().upper() for ex in custom_eval_dataset_raw]
                letter_tokens_eval = {letter: global_tokenizer.encode(letter, add_special_tokens=False)[0] for letter in
                                      ANSWER_MAP_KEYS}

                all_predicted_letters_custom_eval = []
                with torch.inference_mode():
                    for i in tqdm(range(0, len(all_prompts_custom_eval), EVAL_BATCH_SIZE_CUSTOM), desc="Custom Eval"):
                        batch_prompts = all_prompts_custom_eval[i:i + EVAL_BATCH_SIZE_CUSTOM]
                        inputs = global_tokenizer(
                            batch_prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_prompt_len_config
                        ).to(DEVICE)
                        gen_pad_token_id = global_tokenizer.pad_token_id if global_tokenizer.pad_token_id is not None else global_tokenizer.eos_token_id
                        generated_ids = model_eval.generate(
                            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                            max_new_tokens=MAX_NEW_TOKENS_GEN_CUSTOM,
                            pad_token_id=gen_pad_token_id,
                            temperature=TEMPERATURE_CUSTOM_EVAL,
                            top_p=TOP_P_CUSTOM_EVAL,
                            do_sample=DO_SAMPLE_CUSTOM_EVAL
                        )
                        for j, gen_ids_sample in enumerate(generated_ids):
                            input_len = inputs['input_ids'].shape[1]
                            if len(gen_ids_sample) > input_len:
                                predicted_token_id = gen_ids_sample[input_len].item()
                                predicted_letter = next(
                                    (letter for letter, tid in letter_tokens_eval.items() if tid == predicted_token_id),
                                    None)
                            else:
                                predicted_letter = None
                            all_predicted_letters_custom_eval.append(predicted_letter)

                if not all_true_letters:
                    final_accuracy_custom = 0.0
                    print("Test set is empty, accuracy is 0.")
                else:
                    correct = sum(1 for pred, true in zip(all_predicted_letters_custom_eval, all_true_letters) if
                                  pred == true and pred is not None)
                    total = len(all_true_letters)
                    final_accuracy_custom = correct / total if total > 0 else 0.0

                num_invalid = sum(1 for pred in all_predicted_letters_custom_eval if pred is None)
                print(f"Number of invalid predictions (custom eval): {num_invalid}")
                print(f"Accuracy on {EVAL_SPLIT_CUSTOM_EVAL} split (custom eval): {final_accuracy_custom:.4f}")

                # Save detailed results
                detailed_results_custom_eval = []
                for i_res in range(len(all_prompts_custom_eval)):
                    predicted_letter_res = all_predicted_letters_custom_eval[i_res]
                    true_letter_res = all_true_letters[i_res]
                    is_correct_res = predicted_letter_res == true_letter_res and predicted_letter_res is not None
                    detailed_results_custom_eval.append({
                        "id": i_res,
                        "prompt": all_prompts_custom_eval[i_res],
                        "predicted_letter": predicted_letter_res if predicted_letter_res is not None else "INVALID",
                        "true_letter": true_letter_res,
                        "is_correct": is_correct_res
                    })
                config_summary_run = {"final_accuracy_custom_eval": final_accuracy_custom}  # Basic summary for this run
                with open(eval_results_file, "w") as f_res:
                    json.dump(
                        {"config_summary_run": config_summary_run, "detailed_results": detailed_results_custom_eval},
                        f_res, indent=2)
                print(f"Custom evaluation results for this run saved to: {eval_results_file}")


            except Exception as e_eval:
                print(f"💥 Custom evaluation failed for this run: {e_eval}")
                traceback.print_exc()
                final_accuracy_custom = 0.0
            del base_model_eval, model_eval
        run_status = "completed"

    except Exception as e_run:
        print(f"\n!!! ERROR during MedQA Heuristic run: Diff='{difficulty_measurer}', Sched='{training_scheduler}' !!!")
        print(f"Output Dir: {current_output_dir}");
        traceback.print_exc()
        run_status = f"error: {str(e_run)}"
    finally:
        print(f"Cleaning up resources for run: {current_output_dir}")
        del model_train_run, trainer_run
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Cleanup complete for run {current_output_dir}. Status: {run_status}. Final Accuracy: {final_accuracy_custom:.4f}")
        print("=" * 80 + "\n")

    return {"accuracy": final_accuracy_custom, "status": run_status, "output_dir": current_output_dir}


# ----- Main Script Execution Orchestrator -----
if __name__ == "__main__":
    print(f"Output base directory: {BASE_OUTPUT_DIR_ROOT}")
    print(f"Using device: {DEVICE}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. QLoRA training will fail. Exiting.")
        sys.exit(1)

    # Global Tokenizer Initialization
    print(f"Loading global tokenizer for {model_id}...")
    tokenizer_main = AutoTokenizer.from_pretrained(
        model_id, padding_side="left", trust_remote_code=True, cache_dir=transformers_cache_dir
    )
    if tokenizer_main.pad_token is None:
        tokenizer_main.pad_token = tokenizer_main.eos_token
        print(f"Set pad_token to EOS token: '{tokenizer_main.eos_token}', ID: {tokenizer_main.eos_token_id}")
    print(
        f"Global Tokenizer: Vocab size: {len(tokenizer_main)}. Pad: '{tokenizer_main.pad_token}', ID: {tokenizer_main.pad_token_id}")

    overall_results = {}
    print("\n===== Starting MedQA Llama (Custom Sampler) Experiment Loops =====")
    for diff_measure in difficulty_measures_to_run:
        for scheduler in schedulers_to_run:
            run_id = f"{diff_measure}_{scheduler}_{ORDERING}"
            current_run_output_dir = os.path.join(BASE_OUTPUT_DIR_ROOT, run_id)

            run_summary = run_training_heuristic_medqa(
                diff_measure,
                scheduler,
                current_run_output_dir,
                global_tokenizer=tokenizer_main
            )
            overall_results[run_id] = run_summary
            time.sleep(5)

    print("\n\n===== Overall MedQA Llama (Custom Sampler) Experiment Summary =====")
    for run_id_summary, results_summary in overall_results.items():
        print(f"\n--- Results for Run: {run_id_summary} ---")
        print(f"  Status: {results_summary.get('status', 'unknown')}")
        acc_str = f"{results_summary.get('accuracy', 0.0):.4f}"
        print(f"  Test Accuracy (Custom Eval): {acc_str}")
        print(f"  Output Dir: {results_summary.get('output_dir', 'N/A')}")

    print("======================================")
    overall_summary_file = os.path.join(BASE_OUTPUT_DIR_ROOT, "overall_heuristic_summary.json")
    try:
        serializable_results = {}
        for k_sum, v_sum in overall_results.items():
            serializable_results[k_sum] = {
                key_item: (float(val_item) if isinstance(val_item, (np.float32, np.float64, np.float16)) else val_item)
                for key_item, val_item in v_sum.items()
            }
        with open(overall_summary_file, "w") as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Overall summary saved to: {overall_summary_file}")
    except Exception as e_summary_save:
        print(f"Warning: Failed to save overall summary: {e_summary_save}")

    print("\nAll MedQA Llama (Custom Sampler) experiment runs completed.")
    print(f"Script finished at {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")