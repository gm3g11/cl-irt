import os
import datetime
import random
import traceback
import sys
import json
import re  # For simple_tokenize
import math  # For ceil
import time  # For heuristic script
import gc  # For heuristic script
from typing import List, Dict, Any, Optional, Tuple, Iterator  # For type hints
from tqdm import tqdm
import torch
import numpy as np

from datasets import load_dataset, DatasetDict, Dataset  # Added Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,  # Will be replaced by CustomHeuristicTrainer
    DataCollatorForSeq2Seq,  # Used by Qwen script
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    AutoConfig,  # Potentially useful
)
from torch.utils.data import DataLoader, Sampler  # Added for HeuristicSampler
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,  # For loading adapter if needed, though Qwen saves full model
)
# import evaluate # Not used by Qwen script's trainer compute_metrics
import transformers as hf_transformers  # For logging verbosity

# Environment Debug (from Qwen script)
print("--- Environment Debug ---")
print(f"Python Executable: {sys.executable}")
print(f"Torch Version: {torch.__version__}")
if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Transformers Version: {hf_transformers.__version__}")
print("-------------------------\n")

# ----- Baseline Configuration (Qwen) -----
model_id = "Qwen/Qwen2.5-7B"  # Qwen model
dataset_id = "GBaker/MedQA-USMLE-4-options"
# output_dir will be set per run

# Qwen script's original sequence and training parameters
max_seq_length = 512 + 10
max_prompt_len_config = 512  # Used in manual eval
QWEN_PER_DEVICE_TRAIN_BS = 2
QWEN_PER_DEVICE_EVAL_BS = 4
QWEN_GRAD_ACCUM_STEPS = 16
QWEN_NUM_TRAIN_EPOCHS = 5
QWEN_LEARNING_RATE = 1e-4
QWEN_WEIGHT_DECAY = 0.01
QWEN_LORA_R = 16
QWEN_LORA_ALPHA = 32
QWEN_LORA_DROPOUT = 0.05
QWEN_EARLY_STOPPING_PATIENCE = 2
QWEN_METRIC_FOR_BEST = "loss"  # Already "loss"

random_seed = 63

# ----- Heuristic Configuration -----
BASE_OUTPUT_DIR_ROOT = "./qwen2.5_7b_medqa_heuristic_runs"
os.makedirs(BASE_OUTPUT_DIR_ROOT, exist_ok=True)

difficulty_measures_to_run = ['sentence_length', 'word_rarity']
schedulers_to_run = ['linear', 'root']

ORDERING = 'easiest'
COMPETENCY_PARAM = 5
MIN_TRAIN_PERCENT = 0.05
C_INIT = 0.01
HEURISTIC_LOGGING_STEPS = 50  # For sampler's own logging

ANSWER_MAP_KEYS = ["A", "B", "C", "D"]  # From Qwen script

# ----- Global Setup -----
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
hf_transformers.logging.set_verbosity_warning()
print(f"Global random seed set to: {random_seed}")
# Default HF cache setup
default_hf_home = os.path.expanduser("~/.cache/huggingface")
transformers_cache_dir = os.environ.get("TRANSFORMERS_CACHE", os.path.join(default_hf_home, "hub"))
datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE", os.path.join(default_hf_home, "datasets"))
os.makedirs(transformers_cache_dir, exist_ok=True)
os.makedirs(datasets_cache_dir, exist_ok=True)


# ----- Helper Functions for Heuristic Difficulty (Copied) -----
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


def combine_text_for_difficulty(example: Dict[str, Any]) -> Dict[str, str]:  # Reused
    question = example["question"].strip() if isinstance(example["question"], str) else ""
    options_dict = example["options"]
    options_text_parts = []
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            option_val = options_dict.get(key_char)
            options_text_parts.append(str(option_val) if option_val is not None else "")
    options_combined = " ".join(filter(None, options_text_parts))
    full_text = f"{question} {options_combined}".strip()
    if not full_text: return {"full_text_for_difficulty": " "}
    return {"full_text_for_difficulty": full_text}


# ----- Custom Heuristic Sampler Definition (Copied) -----
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
        if epoch == 0 or len(new_indices) != len(self.indices_for_epoch) or epoch % HEURISTIC_LOGGING_STEPS == 0:
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


# ----- Custom Trainer Definition (Copied) -----
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
            num_samples_total=self.num_samples_total, batch_size=self._train_batch_size,
            sorted_indices=self.sorted_indices, heuristic_config=self.heuristic_config,
            num_replicas=self.args.world_size, rank=self.args.process_index, seed=self.args.seed
        )
        return DataLoader(
            train_dataset, batch_size=self._train_batch_size, sampler=heuristic_sampler,
            collate_fn=self.data_collator, drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
        )


# ----- Qwen Specific Preprocessing (from original script) -----
def create_prompt_and_target_letter_qwen(example):  # Renamed for clarity
    question = example["question"].strip()
    options_dict = example["options"]
    answer_idx_key = example["answer_idx"]  # Should be a string like 'A', 'B'
    prompt_parts = [f"Question: {question}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option text not found]')}")
    else:
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    prompt_text = "\n".join(prompt_parts)
    target_letter = answer_idx_key if isinstance(answer_idx_key,
                                                 str) and answer_idx_key.strip().upper() in ANSWER_MAP_KEYS else ""
    return {"prompt": prompt_text, "target_letter": target_letter.strip().upper()}


def preprocess_sft_format_qwen(examples, tokenizer_ref, max_seq_len_ref):  # Added tokenizer_ref and max_seq_len_ref
    inputs = []
    labels_list = []
    # Determine batch size from the input structure
    first_key = next(iter(examples))
    batch_size_dynamic = len(examples[first_key])

    for i in range(batch_size_dynamic):
        single_example_dict = {key: examples[key][i] for key in examples.keys()}
        processed = create_prompt_and_target_letter_qwen(single_example_dict)
        prompt_text = processed["prompt"]
        target_letter = processed["target_letter"]

        # Qwen tokenizer might handle BOS/EOS differently, check its specific usage
        # The original Qwen script tokenized prompt and target separately without explicit BOS/EOS then combined.
        tokenized_prompt = tokenizer_ref(prompt_text, truncation=False, padding=False, add_special_tokens=False)
        tokenized_target = tokenizer_ref(target_letter, truncation=False, padding=False, add_special_tokens=False)

        prompt_input_ids = tokenized_prompt.input_ids
        target_input_ids = tokenized_target.input_ids

        input_ids_concat = []
        # Add BOS if tokenizer usually adds it or if model expects it
        # Qwen models might handle this internally or expect specific chat templates if not doing raw SFT
        # For this SFT format, let's replicate the original script's behavior.
        # Original script added BOS if tokenizer.bos_token_id and getattr(tokenizer, 'add_bos_token', True)
        # Qwen tokenizers might not always have add_bos_token attribute in the same way.
        # Safest to check if bos_token_id exists.
        if tokenizer_ref.bos_token_id is not None:  # Simplified check
            input_ids_concat.append(tokenizer_ref.bos_token_id)

        input_ids_concat.extend(prompt_input_ids)
        input_ids_concat.extend(target_input_ids)  # Target letter tokens

        if tokenizer_ref.eos_token_id is not None:
            input_ids_concat.append(tokenizer_ref.eos_token_id)

        # Create labels: -100 for prompt, actual tokens for target
        len_prompt_and_bos = (1 if tokenizer_ref.bos_token_id is not None else 0) + len(prompt_input_ids)
        labels_concat = ([-100] * len_prompt_and_bos) + target_input_ids
        if tokenizer_ref.eos_token_id is not None:
            labels_concat.append(tokenizer_ref.eos_token_id)  # Also label EOS token if present

        # Truncate to max_seq_length
        final_input_ids = input_ids_concat[:max_seq_len_ref]
        final_labels = labels_concat[:max_seq_len_ref]

        # Ensure labels are same length as input_ids after truncation (pad with -100 if labels shorter)
        if len(final_labels) < len(final_input_ids):
            final_labels.extend([-100] * (len(final_input_ids) - len(final_labels)))

        inputs.append(final_input_ids)
        labels_list.append(final_labels)

    return {"input_ids": inputs, "labels": labels_list}


# ----- Main Training Function for Qwen with Heuristics -----
def run_training_heuristic_qwen(
        difficulty_measurer: str,
        training_scheduler: str,
        current_output_dir: str,
        global_tokenizer: AutoTokenizer  # Pass the globally initialized tokenizer
):
    print("\n" + "=" * 80);
    print(f"Starting Qwen Run: Diff='{difficulty_measurer}', Scheduler='{training_scheduler}'");
    print(f"Output Dir: {current_output_dir}");
    print("=" * 80 + "\n")
    os.makedirs(current_output_dir, exist_ok=True)

    model_run, trainer_run = None, None
    manual_eval_accuracy = 0.0
    run_status = "failed"

    try:
        # 1. Load and Prepare Dataset (using logic from original Qwen script)
        print(f"Loading dataset for run: {dataset_id}")
        raw_dataset_run = load_dataset(dataset_id, cache_dir=datasets_cache_dir)

        # Filter for valid answer_idx before splitting, as in Llama/DeBERTa heuristic scripts
        raw_dataset_run = raw_dataset_run.filter(
            lambda ex: ex["answer_idx"] is not None and ex["answer_idx"].strip().upper() in ANSWER_MAP_KEYS)

        if "validation" not in raw_dataset_run:
            print("Validation split not found for run. Splitting train (90/10)...")  # Qwen used 90/10
            if 'train' not in raw_dataset_run or len(raw_dataset_run['train']) == 0:
                raise ValueError("Original train split not found or empty after filtering!")

            stratify_column = 'answer_idx' if 'answer_idx' in raw_dataset_run['train'].column_names else None
            try:  # Robust stratification
                if stratify_column:
                    unique_labels = raw_dataset_run['train'].unique(stratify_column)
                    if not unique_labels or any(label is None for label in unique_labels):
                        stratify_column = None
                    else:
                        label_counts = {
                            label: raw_dataset_run['train'].filter(lambda ex: ex[stratify_column] == label).num_rows for
                            label in unique_labels}
                        if any(count < 2 for count in
                               label_counts.values()): stratify_column = None  # Needs at least 2 for split
                split_run = raw_dataset_run["train"].train_test_split(test_size=0.1, seed=random_seed, shuffle=True,
                                                                      stratify_by_column=stratify_column)
            except Exception as e_stratify:
                print(f"Stratification attempt failed ({e_stratify}), using random split.")
                split_run = raw_dataset_run["train"].train_test_split(test_size=0.1, seed=random_seed, shuffle=True)

            current_run_dataset = DatasetDict({
                "train": split_run["train"], "validation": split_run["test"], "test": raw_dataset_run["test"],
            })
        else:
            current_run_dataset = raw_dataset_run  # Use as is if validation exists

        print("Final dataset splits for this run:", current_run_dataset)
        if any(len(current_run_dataset[split_name]) == 0 for split_name in ['train', 'validation', 'test']):
            raise ValueError("One or more dataset splits are empty for this run!")

        # 2. Calculate Difficulty Scores
        train_dataset_for_difficulty_calc = current_run_dataset['train'].map(
            combine_text_for_difficulty, num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        )
        difficulty_scores, valid_original_indices = calculate_difficulty_scores(
            train_dataset_for_difficulty_calc, difficulty_measurer, text_column='full_text_for_difficulty'
        )
        train_dataset_for_sorting = current_run_dataset['train'].select(valid_original_indices)
        if len(difficulty_scores) != len(train_dataset_for_sorting):
            raise RuntimeError("Mismatch difficulty scores and final data size.")
        print(f"Selected {len(train_dataset_for_sorting)} training examples with valid difficulty scores.")

        print(f"Sorting training data indices by difficulty ({ORDERING})...")
        start_time_sort = time.time()
        if ORDERING == 'easiest':
            sorted_sampler_indices = np.argsort(np.array(difficulty_scores)).tolist()
        elif ORDERING == 'hardest':
            sorted_sampler_indices = np.argsort(np.array(difficulty_scores))[::-1].tolist()
        else:
            raise NotImplementedError(f"Ordering '{ORDERING}' not implemented.")
        print(f"Sorting completed in {time.time() - start_time_sort:.2f}s.")
        num_samples_total_for_sampler = len(train_dataset_for_sorting)

        # 3. QLoRA Config & Model Loading (inside run for fresh model)
        bnb_config_run = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        print(f"Loading model {model_id} for run...")
        model_run = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config_run, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=transformers_cache_dir
        )
        if len(global_tokenizer) > model_run.config.vocab_size:  # Check if tokenizer vocab grew (e.g. new pad token)
            print(f"Resizing model token embeddings: {model_run.config.vocab_size} -> {len(global_tokenizer)}")
            model_run.resize_token_embeddings(len(global_tokenizer))
        if model_run.config.pad_token_id != global_tokenizer.pad_token_id:
            print(f"Updating model_run.config.pad_token_id to {global_tokenizer.pad_token_id}")
            model_run.config.pad_token_id = global_tokenizer.pad_token_id

        model_run = prepare_model_for_kbit_training(model_run)
        model_run.gradient_checkpointing_enable()  # Explicitly enable, though prepare_model might do it.

        # PEFT Config (Qwen specific target modules if different, else use common ones)
        lora_target_modules_qwen = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                    "down_proj"]  # Common for many LLMs
        # For Qwen2 specifically, common target modules often include 'c_attn', 'attn.c_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
        # Let's use the original script's common list for now, but this might need Qwen-specific adjustment for best results.

        peft_config_run = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=QWEN_LORA_R, lora_alpha=QWEN_LORA_ALPHA, lora_dropout=QWEN_LORA_DROPOUT,
            target_modules=lora_target_modules_qwen, bias="none"
        )
        model_run = get_peft_model(model_run, peft_config_run)
        print("QLoRA model prepared for this run.")
        model_run.print_trainable_parameters()

        # 4. Tokenize Datasets (SFT specific)
        print(f"Tokenizing dataset for SFT. Max length: {max_seq_length}")
        tokenized_train = train_dataset_for_sorting.map(
            lambda ex: preprocess_sft_format_qwen(ex, global_tokenizer, max_seq_length), batched=True,
            num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
            remove_columns=train_dataset_for_sorting.column_names
        )
        tokenized_validation = current_run_dataset['validation'].map(
            lambda ex: preprocess_sft_format_qwen(ex, global_tokenizer, max_seq_length), batched=True,
            num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
            remove_columns=current_run_dataset['validation'].column_names
        )
        # Test set for manual eval will use raw data, no SFT tokenization needed here for it.

        keep_cols_run = ["input_ids", "labels"]  # From Qwen script
        tokenized_train = tokenized_train.remove_columns(
            [c for c in tokenized_train.column_names if c not in keep_cols_run])
        tokenized_validation = tokenized_validation.remove_columns(
            [c for c in tokenized_validation.column_names if c not in keep_cols_run])

        tokenized_train.set_format(type="torch", columns=keep_cols_run)
        tokenized_validation.set_format(type="torch", columns=keep_cols_run)
        print(f"Dataset formatted for Trainer. Train columns: {tokenized_train.column_names}")

        # 5. Data Collator
        data_collator_run = DataCollatorForSeq2Seq(  # Qwen script used this
            global_tokenizer, model=model_run, label_pad_token_id=-100, padding="longest"
        )

        # 6. Metrics for Trainer (only loss, as per prediction_loss_only=True)
        def compute_metrics_trainer_qwen(eval_pred):  # Renamed
            # Original Qwen script had a warning here, this is fine if only loss is needed.
            if hasattr(eval_pred, 'metrics') and eval_pred.metrics is not None:
                eval_loss = eval_pred.metrics.get("eval_loss", -1.0)
                return {"loss": eval_loss}  # Trainer uses "loss" if metric_for_best_model="loss"
            return {"loss": -1.0}

        # 7. Calculate max_steps
        try:
            world_size_train = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception:
            world_size_train = 1
        total_train_batch_size_effective = QWEN_PER_DEVICE_TRAIN_BS * world_size_train * QWEN_GRAD_ACCUM_STEPS
        if total_train_batch_size_effective == 0: raise ValueError("Total effective train batch size is zero.")
        if num_samples_total_for_sampler == 0:
            steps_per_epoch_full = 1
        else:
            steps_per_epoch_full = math.ceil(num_samples_total_for_sampler / total_train_batch_size_effective)
        calculated_max_steps = math.ceil(QWEN_NUM_TRAIN_EPOCHS * steps_per_epoch_full)
        print(
            f"Effective Batch Size: {total_train_batch_size_effective}, Steps/Epoch (Full Dataset): {steps_per_epoch_full}")
        print(f"Calculated max_steps for {QWEN_NUM_TRAIN_EPOCHS} nominal epochs: {calculated_max_steps}")
        if calculated_max_steps == 0 and num_samples_total_for_sampler > 0: calculated_max_steps = 1
        if num_samples_total_for_sampler == 0: calculated_max_steps = 0

        # Logging steps from original script
        logging_steps_qwen = max(1, 50 // QWEN_GRAD_ACCUM_STEPS)  # Qwen's logging steps logic

        # 8. TrainingArguments
        training_args_run = TrainingArguments(
            output_dir=os.path.join(current_output_dir, "training_checkpoints"),
            overwrite_output_dir=True, do_train=True, do_eval=True,
            per_device_train_batch_size=QWEN_PER_DEVICE_TRAIN_BS,
            per_device_eval_batch_size=QWEN_PER_DEVICE_EVAL_BS,
            gradient_accumulation_steps=QWEN_GRAD_ACCUM_STEPS,
            # num_train_epochs=QWEN_NUM_TRAIN_EPOCHS, # Using max_steps
            max_steps=calculated_max_steps if calculated_max_steps > 0 else 1,
            learning_rate=QWEN_LEARNING_RATE,
            weight_decay=QWEN_WEIGHT_DECAY,
            fp16=False, bf16=True,  # As per Qwen script
            logging_strategy="epoch",  # Qwen script used epoch
            # logging_steps=logging_steps_qwen, # If epoch, this might not be needed or used differently
            eval_strategy="epoch", save_strategy="epoch",
            prediction_loss_only=True,  # Qwen script used this
            save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model=QWEN_METRIC_FOR_BEST,  # "loss"
            greater_is_better=False,
            report_to=[], remove_unused_columns=False, seed=random_seed,
        )

        heuristic_config_dict = {
            "scheduler": training_scheduler, "ordering": ORDERING,
            "competency_param": COMPETENCY_PARAM, "min_train_percent": MIN_TRAIN_PERCENT, "c_init": C_INIT,
        }

        # 9. Initialize CustomHeuristicTrainer
        trainer_run = CustomHeuristicTrainer(
            model=model_run, args=training_args_run,
            train_dataset=tokenized_train, eval_dataset=tokenized_validation,
            tokenizer=global_tokenizer, data_collator=data_collator_run,
            compute_metrics=compute_metrics_trainer_qwen,  # Uses only loss
            callbacks=[EarlyStoppingCallback(early_stopping_patience=QWEN_EARLY_STOPPING_PATIENCE)],
            sorted_indices=sorted_sampler_indices,
            heuristic_config=heuristic_config_dict,
            num_samples_total=num_samples_total_for_sampler
        )

        # 10. Train
        final_adapter_path_run = os.path.join(current_output_dir, "final_qlora_adapter")
        training_successful_run = False
        if calculated_max_steps > 0 and num_samples_total_for_sampler > 0:
            print(f"Starting QLoRA training for {model_id} (max_steps={calculated_max_steps})...")
            trainer_run.train()
            # Qwen script saves adapter using model.save_pretrained
            model_run.save_pretrained(final_adapter_path_run)  # Save PEFT adapter
            global_tokenizer.save_pretrained(final_adapter_path_run)
            training_successful_run = True
            print(f"✅ Training complete for run. Adapter saved to {final_adapter_path_run}")
            # Trainer's log_metrics and save_state can be added if desired
            # trainer_run.log_metrics("train", train_result.metrics)
            # trainer_run.save_state()
        else:
            print("Skipping training for this run as max_steps or num_samples_total_for_sampler is zero.")
            os.makedirs(final_adapter_path_run, exist_ok=True)
            # Create dummy adapter config for manual eval phase if needed
            dummy_adapter_config = {"base_model_name_or_path": model_id, "peft_type": "LORA"}
            with open(os.path.join(final_adapter_path_run, "adapter_config.json"), "w") as f_cfg:
                json.dump(dummy_adapter_config, f_cfg)

        # 11. Manual Evaluation (from Qwen script)
        print("\nPerforming manual evaluation on test set for this run...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # For manual eval, reload the base model and apply the trained adapter for this run
        # This ensures a clean state for evaluation.
        # Model used for manual eval needs to be on the correct device.

        if training_successful_run and os.path.exists(final_adapter_path_run):
            print(f"Loading base model and adapter from {final_adapter_path_run} for manual evaluation...")
            base_model_eval = AutoModelForCausalLM.from_pretrained(
                model_id,
                # quantization_config=bnb_config_run, # For full model eval, often load without bnb for speed if memory allows, or with bnb if needed
                torch_dtype=torch.bfloat16,  # Or float16
                device_map="auto",  # Or specific device
                trust_remote_code=True,
                cache_dir=transformers_cache_dir
            )
            if len(global_tokenizer) > base_model_eval.config.vocab_size:
                base_model_eval.resize_token_embeddings(len(global_tokenizer))
            if base_model_eval.config.pad_token_id != global_tokenizer.pad_token_id:
                base_model_eval.config.pad_token_id = global_tokenizer.pad_token_id

            model_for_eval = PeftModel.from_pretrained(base_model_eval, final_adapter_path_run)
            model_for_eval.eval()
            # model_for_eval.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # device_map="auto" should handle this
            eval_device = model_for_eval.device  # Get device from model
        elif not training_successful_run and calculated_max_steps == 0:  # Evaluate base model if no training happened
            print(f"No training performed, evaluating base model {model_id}...")
            model_for_eval = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
                cache_dir=transformers_cache_dir
            )
            if len(global_tokenizer) > model_for_eval.config.vocab_size: model_for_eval.resize_token_embeddings(
                len(global_tokenizer))
            if model_for_eval.config.pad_token_id != global_tokenizer.pad_token_id: model_for_eval.config.pad_token_id = global_tokenizer.pad_token_id
            model_for_eval.eval()
            eval_device = model_for_eval.device
        else:
            print("Skipping manual evaluation as training was not successful or adapter not found.")
            model_for_eval = None
            eval_device = None

        if model_for_eval and eval_device:
            test_examples_run = current_run_dataset["test"]  # Use the test set for this run
            num_test_examples_run = len(test_examples_run)
            eval_batch_size_manual = QWEN_PER_DEVICE_EVAL_BS  # Or a specific manual eval batch size like 16 in original
            num_batches_run = (num_test_examples_run + eval_batch_size_manual - 1) // eval_batch_size_manual

            all_predicted_letters_run = []
            all_true_answers_run = []

            for batch_idx_run in tqdm(range(num_batches_run), desc="Manual Eval Batches"):
                start_idx_run = batch_idx_run * eval_batch_size_manual
                end_idx_run = min(start_idx_run + eval_batch_size_manual, num_test_examples_run)
                batch_examples_run = [test_examples_run[i] for i in range(start_idx_run, end_idx_run)]

                prompts_run = [create_prompt_and_target_letter_qwen(ex)["prompt"] for ex in batch_examples_run]
                true_answers_run = [create_prompt_and_target_letter_qwen(ex)["target_letter"] for ex in
                                    batch_examples_run]

                tokenized_prompts_run = global_tokenizer(
                    prompts_run, padding=True, truncation=True,
                    max_length=max_prompt_len_config, return_tensors="pt"
                )
                input_ids_run = tokenized_prompts_run["input_ids"].to(eval_device)
                attention_mask_run = tokenized_prompts_run["attention_mask"].to(eval_device)

                gen_pad_token_id_run = global_tokenizer.pad_token_id if global_tokenizer.pad_token_id is not None else global_tokenizer.eos_token_id

                with torch.no_grad():  # Removed autocast, as model_for_eval is already bf16
                    generated_ids_run = model_for_eval.generate(
                        input_ids=input_ids_run, attention_mask=attention_mask_run,
                        max_new_tokens=1, pad_token_id=gen_pad_token_id_run
                    )

                # Get the generated token (the one after the input sequence)
                generated_token_ids_run = generated_ids_run[:,
                                          input_ids_run.shape[1]]  # Correct slicing for single new token

                letter_token_ids_run = {letter: global_tokenizer.encode(letter, add_special_tokens=False)[0] for letter
                                        in ANSWER_MAP_KEYS}
                predicted_letters_run = [
                    next((letter for letter, tid in letter_token_ids_run.items() if tid == token_id.item()), None)
                    for token_id in generated_token_ids_run
                ]
                all_predicted_letters_run.extend(predicted_letters_run)
                all_true_answers_run.extend(true_answers_run)
                del input_ids_run, attention_mask_run, generated_ids_run, generated_token_ids_run
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            correct_run = sum(1 for pred, true in zip(all_predicted_letters_run, all_true_answers_run) if
                              pred == true and pred is not None)
            invalid_run = sum(1 for pred in all_predicted_letters_run if pred is None)
            total_run = len(all_true_answers_run)
            manual_eval_accuracy = correct_run / total_run if total_run > 0 else 0.0
            print(
                f"Manual evaluation results for run: Total={total_run}, Correct={correct_run}, Invalid={invalid_run}, Accuracy={manual_eval_accuracy:.4f}")

            detailed_results_run = [
                {"prompt": create_prompt_and_target_letter_qwen(test_examples_run[i])["prompt"],
                 "predicted_letter": all_predicted_letters_run[i] if all_predicted_letters_run[
                                                                         i] is not None else "INVALID",
                 "true_letter": all_true_answers_run[i],
                 "is_correct": all_predicted_letters_run[i] == all_true_answers_run[i] and all_predicted_letters_run[
                     i] is not None}
                for i in range(total_run)
            ]
            eval_results_file_run = os.path.join(current_output_dir, "manual_evaluation_results.json")
            with open(eval_results_file_run, "w") as f_eval:
                json.dump(detailed_results_run, f_eval, indent=2)
            print(f"Detailed manual evaluation results saved to: {eval_results_file_run}")
            del model_for_eval, base_model_eval  # Clean up eval models

        run_status = "completed"

    except Exception as e_main_run:
        print(f"\n!!! ERROR during Qwen Heuristic run: Diff='{difficulty_measurer}', Sched='{training_scheduler}' !!!")
        print(f"Output Dir: {current_output_dir}");
        traceback.print_exc()
        run_status = f"error: {str(e_main_run)}"
    finally:
        print(f"Cleaning up resources for run: {current_output_dir}")
        del model_run, trainer_run  # Ensure these are deleted
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Cleanup complete for run {current_output_dir}. Status: {run_status}. Manual Eval Accuracy: {manual_eval_accuracy:.4f}")
        print("=" * 80 + "\n")

    return {"accuracy": manual_eval_accuracy, "status": run_status, "output_dir": current_output_dir}


# ----- Main Script Execution Orchestrator -----
if __name__ == "__main__":
    print(f"Qwen Heuristic Script: Output base directory: {BASE_OUTPUT_DIR_ROOT}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. QLoRA training will likely fail or be extremely slow. Exiting.")
        sys.exit(1)

    # Global Tokenizer Initialization (from Qwen script)
    print(f"Loading global tokenizer for {model_id}...")
    try:
        main_tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", trust_remote_code=True, cache_dir=transformers_cache_dir
        )
        if main_tokenizer.pad_token is None:
            if main_tokenizer.eos_token:
                print("Setting global tokenizer.pad_token = eos_token.")
                main_tokenizer.pad_token = main_tokenizer.eos_token
            else:  # Fallback if no EOS, though uncommon for generative models
                new_pad_token_main = "␂"  # Arbitrary new pad token
                if new_pad_token_main not in main_tokenizer.get_vocab():
                    main_tokenizer.add_special_tokens({"pad_token": new_pad_token_main})
                    print(f"Added new pad_token to global tokenizer: {new_pad_token_main}")
                else:
                    main_tokenizer.pad_token = new_pad_token_main  # Use existing if somehow present
                    print(f"Set global tokenizer.pad_token to existing token: {new_pad_token_main}")

        # Ensure pad_token_id is set if pad_token exists
        if main_tokenizer.pad_token is not None and main_tokenizer.pad_token_id is None:
            main_tokenizer.pad_token_id = main_tokenizer.convert_tokens_to_ids(main_tokenizer.pad_token)

        print(
            f"Global Tokenizer loaded. Vocab size: {len(main_tokenizer)}. Pad token: '{main_tokenizer.pad_token}', ID: {main_tokenizer.pad_token_id}")
    except Exception as e_tok_load:
        print(f"Fatal error loading global Qwen tokenizer: {e_tok_load}");
        traceback.print_exc();
        sys.exit(1)

    overall_results = {}
    print("\n===== Starting Qwen (Custom Sampler) Experiment Loops =====")
    for diff_measure_loop_qwen in difficulty_measures_to_run:
        for scheduler_loop_qwen in schedulers_to_run:
            run_id_qwen = f"{diff_measure_loop_qwen}_{scheduler_loop_qwen}_{ORDERING}"
            current_run_output_dir_qwen = os.path.join(BASE_OUTPUT_DIR_ROOT, run_id_qwen)

            run_summary_qwen = run_training_heuristic_qwen(
                diff_measure_loop_qwen, scheduler_loop_qwen, current_run_output_dir_qwen,
                global_tokenizer=main_tokenizer
            )
            overall_results[run_id_qwen] = run_summary_qwen
            time.sleep(5)  # Brief pause

    # Overall Summary
    print("\n\n===== Overall Qwen (Custom Sampler) Experiment Summary =====")
    for run_id_summary_qwen, results_summary_qwen in overall_results.items():
        print(f"\n--- Results for Run: {run_id_summary_qwen} ---")
        print(f"  Status: {results_summary_qwen.get('status', 'unknown')}")
        acc_str_qwen = f"{results_summary_qwen.get('accuracy', 0.0):.4f}"  # Custom test accuracy
        print(f"  Manual Eval Accuracy: {acc_str_qwen}")
        print(f"  Output Dir: {results_summary_qwen.get('output_dir', 'N/A')}")

    print("======================================")
    overall_summary_file_qwen = os.path.join(BASE_OUTPUT_DIR_ROOT, "overall_qwen_heuristic_summary.json")
    try:
        serializable_results_qwen = {}
        for k_sum_qwen, v_sum_qwen in overall_results.items():
            serializable_results_qwen[k_sum_qwen] = {
                key_item_qwen: (float(val_item_qwen) if isinstance(val_item_qwen, (
                np.float32, np.float64, np.float16)) else val_item_qwen)
                for key_item_qwen, val_item_qwen in v_sum_qwen.items()
            }
        with open(overall_summary_file_qwen, "w") as f_qwen:
            json.dump(serializable_results_qwen, f_qwen, indent=4)
        print(f"Overall summary saved to: {overall_summary_file_qwen}")
    except Exception as e_summary_save_qwen:
        print(f"Warning: Failed to save overall Qwen summary: {e_summary_save_qwen}")

    print("\nAll Qwen (Custom Sampler) experiment runs completed.")
    print(f"Script finished at {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    