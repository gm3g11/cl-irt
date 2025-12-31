import os
import datetime
import random
import numpy as np
import torch
import json
import time
from tqdm import tqdm
import shutil  # For cleanup if needed, though less critical with run-specific dirs
import traceback
import sys
import re  # For simple_tokenize
import math  # For ceil
import gc  # For heuristic script
from typing import List, Dict, Any, Optional, Tuple, Iterator  # For type hints

from datasets import load_dataset, DatasetDict, Dataset  # Added Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,  # Will be replaced by CustomHeuristicTrainer
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    AutoConfig,
)
from torch.utils.data import DataLoader, Sampler  # Added for HeuristicSampler
from evaluate import load as load_metric  # For perplexity, though custom eval is primary
from huggingface_hub import whoami
import transformers as hf_transformers  # For logging verbosity

# ----- Environment Setup -----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_HOME_SPECIFIED = HF_HOME  # User specific
if os.path.exists(HF_HOME_SPECIFIED) and os.path.isdir(HF_HOME_SPECIFIED):
    HF_HOME = HF_HOME_SPECIFIED
else:
    print(f"Warning: Specified HF_HOME path '{HF_HOME_SPECIFIED}' does not exist. Using default.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ----- Random Seed -----
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
hf_transformers.logging.set_verbosity_warning()
print(f"Using random seed: {random_seed}")

# ----- Baseline Config (GPT-2) -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_name = "gpt2"
dataset_id = "GBaker/MedQA-USMLE-4-options"

MAX_SFT_SEQ_LENGTH = 512
MAX_EVAL_PROMPT_LENGTH = 512  # Used in custom eval

# GPT-2 script's original training parameters
GPT2_PER_DEVICE_TRAIN_BATCH_SIZE = 4
GPT2_PER_DEVICE_EVAL_BATCH_SIZE = 8  # For Trainer's internal eval
GPT2_EFFECTIVE_BATCH_SIZE_TARGET = 32  # For grad accum calculation
GPT2_NUM_TRAIN_EPOCHS = 20
GPT2_LEARNING_RATE = 5e-5
GPT2_EARLY_STOPPING_PATIENCE = 3  # if NUM_TRAIN_EPOCHS > 1 else 100
GPT2_METRIC_FOR_BEST = "loss"  # Already "loss"

VALIDATION_SUBSET_SIZE = 200  # For Trainer's internal eval

ANSWER_MAP_KEYS = ['A', 'B', 'C', 'D']
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS)  # Used in custom eval

# ----- Heuristic Configuration -----
BASE_OUTPUT_DIR_ROOT = "./gpt2_medqa_heuristic_runs"
os.makedirs(BASE_OUTPUT_DIR_ROOT, exist_ok=True)

difficulty_measures_to_run = ['sentence_length', 'word_rarity']
schedulers_to_run = ['linear', 'root']

ORDERING = 'easiest'
COMPETENCY_PARAM = 5  # Epochs to reach full dataset in curriculum
MIN_TRAIN_PERCENT = 0.05
C_INIT = 0.01
HEURISTIC_LOGGING_STEPS = 50  # For sampler's own logging


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
        for key_char in ANSWER_MAP_KEYS:  # Consistent with SFT prompt structure
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
        if epoch == 0 or len(new_indices) != len(
                self.indices_for_epoch) or epoch % HEURISTIC_LOGGING_STEPS == 0:  # Using HEURISTIC_LOGGING_STEPS
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


# ----- GPT-2 Specific Preprocessing and Model Loading (from original script) -----
def create_sft_text_gpt2(example_dict):
    question_text = example_dict["question"]
    options_dict = example_dict["options"]
    answer_key = example_dict["answer_idx"].strip().upper()
    prompt_parts = [f"Question: {question_text}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option unavailable]')}")
    else:
        for key_char in ANSWER_MAP_KEYS: prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    prompt_text = "\n".join(prompt_parts)
    return prompt_text + " " + answer_key  # Appends the target answer letter


def preprocess_sft_function_gpt2(examples_batch, tokenizer_ref):  # Added tokenizer_ref
    texts_for_sft = []
    # Determine batch size from the input structure
    # examples_batch is a dict of lists. Pick one key to get length.
    first_key = next(iter(examples_batch))
    batch_size_dynamic = len(examples_batch[first_key])

    for i in range(batch_size_dynamic):
        single_example_dict = {key: examples_batch[key][i] for key in examples_batch.keys()}
        texts_for_sft.append(create_sft_text_gpt2(single_example_dict))

    tokenized_output = tokenizer_ref(  # Use passed tokenizer_ref
        texts_for_sft, max_length=MAX_SFT_SEQ_LENGTH, padding=False, truncation=True, add_special_tokens=True
    )
    # For CLM, labels are typically input_ids shifted or copied.
    # DataCollatorForLanguageModeling will handle label creation.
    return tokenized_output


def create_gpt2_eval_prompt(example_dict):  # For custom eval
    question_text = example_dict["question"]
    options_dict = example_dict["options"]
    prompt_parts = [f"Question: {question_text}\n\nOptions:"]
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS:
            prompt_parts.append(f"{key_char}) {options_dict.get(key_char, '[Option unavailable]')}")
    else:
        for key_char in ANSWER_MAP_KEYS: prompt_parts.append(f"{key_char}) [Invalid options format]")
    prompt_parts.append("\nAnswer:")
    return "\n".join(prompt_parts)


def load_model_for_training_or_eval(model_checkpoint_name_or_path, tokenizer_instance, new_pad_token_added_flag,
                                    is_training=True):
    purpose = "training" if is_training else "evaluation"
    print(f"Loading model {model_checkpoint_name_or_path} for {purpose}...")
    config = AutoConfig.from_pretrained(model_checkpoint_name_or_path, cache_dir=os.environ["TRANSFORMERS_CACHE"])

    if tokenizer_instance.pad_token_id is not None: config.pad_token_id = tokenizer_instance.pad_token_id
    if hasattr(config,
               "eos_token_id") and tokenizer_instance.eos_token_id is not None: config.eos_token_id = tokenizer_instance.eos_token_id
    if hasattr(config,
               "bos_token_id") and tokenizer_instance.bos_token_id is not None: config.bos_token_id = tokenizer_instance.bos_token_id

    model_instance = GPT2LMHeadModel.from_pretrained(model_checkpoint_name_or_path, config=config,
                                                     cache_dir=os.environ["TRANSFORMERS_CACHE"])

    # Resize embeddings only if a new pad token was truly *added* to tokenizer, not just assigned
    if new_pad_token_added_flag:  # This flag should be set when tokenizer.add_special_tokens was called
        if hasattr(model_instance.config, "vocab_size") and len(tokenizer_instance) > model_instance.config.vocab_size:
            print(
                f"Resizing model token embeddings for {purpose}: {model_instance.config.vocab_size} -> {len(tokenizer_instance)}")
            model_instance.resize_token_embeddings(len(tokenizer_instance))
            # After resizing, ensure model's pad_token_id in config is synced if it was part of the resize.
            # This is often handled by resize_token_embeddings or should be re-checked.
            if model_instance.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
                model_instance.config.pad_token_id = tokenizer_instance.pad_token_id

    # Final check/sync for pad_token_id, even if no resize happened but tokenizer.pad_token was set to an existing token like EOS
    if model_instance.config.pad_token_id != tokenizer_instance.pad_token_id and tokenizer_instance.pad_token_id is not None:
        print(
            f"Syncing model's pad_token_id ({model_instance.config.pad_token_id}) to tokenizer's ({tokenizer_instance.pad_token_id}) for {purpose}.")
        model_instance.config.pad_token_id = tokenizer_instance.pad_token_id
    return model_instance


# ----- Main Training Function for GPT-2 with Heuristics -----
def run_training_heuristic_gpt2(
        difficulty_measurer: str,
        training_scheduler: str,
        current_output_dir: str,
        global_tokenizer: GPT2Tokenizer,  # Pass the globally initialized tokenizer
        global_added_pad_token_flag: bool  # Pass the flag
):
    print("\n" + "=" * 80);
    print(f"Starting GPT-2 Run: Diff='{difficulty_measurer}', Scheduler='{training_scheduler}'");
    print(f"Output Dir: {current_output_dir}");
    print("=" * 80 + "\n")
    os.makedirs(current_output_dir, exist_ok=True)

    model_run, trainer_run = None, None
    custom_test_accuracy = 0.0
    run_status = "failed"

    try:
        # 1. Load and Prepare Dataset (using logic from original GPT-2 script)
        print(f"Loading dataset for run: {dataset_id}")
        dataset_full_run = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])
        dataset_full_run = dataset_full_run.filter(
            lambda ex: ex["answer_idx"] is not None and ex["answer_idx"].strip().upper() in ANSWER_MAP_KEYS)

        if 'train' not in dataset_full_run or 'test' not in dataset_full_run or \
                len(dataset_full_run["train"]) == 0 or len(dataset_full_run["test"]) == 0:
            raise ValueError("Dataset must contain non-empty 'train' and 'test' splits after filtering for this run.")

        if 'validation' not in dataset_full_run:
            print("Validation split not found for run. Splitting 'train' (80% train / 20% validation)...")
            train_val_split_run = dataset_full_run['train'].train_test_split(test_size=0.2, seed=random_seed,
                                                                             shuffle=True)
            run_dataset_dict_args = {'train': train_val_split_run['train'], 'validation': train_val_split_run['test'],
                                     'test': dataset_full_run['test']}
        else:
            run_dataset_dict_args = {'train': dataset_full_run['train'], 'validation': dataset_full_run['validation'],
                                     'test': dataset_full_run['test']}

        current_processed_dataset = DatasetDict(run_dataset_dict_args)
        print("Final dataset splits for this run:", current_processed_dataset)
        if any(len(current_processed_dataset[split]) == 0 for split in ['train', 'validation', 'test']):
            raise ValueError("One or more dataset splits are empty for this run!")

        # 2. Calculate Difficulty Scores
        train_dataset_for_difficulty_calc = current_processed_dataset['train'].map(
            combine_text_for_difficulty, num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
        )
        difficulty_scores, valid_original_indices = calculate_difficulty_scores(
            train_dataset_for_difficulty_calc, difficulty_measurer, text_column='full_text_for_difficulty'
        )
        train_dataset_for_sorting = current_processed_dataset['train'].select(valid_original_indices)
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

        # 3. Tokenize Datasets (SFT specific)
        print("Tokenizing datasets for SFT for this run...")
        num_proc_map_run = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)

        # Apply SFT preprocessing to the difficulty-sorted training data
        tokenized_train = train_dataset_for_sorting.map(
            lambda ex: preprocess_sft_function_gpt2(ex, global_tokenizer), batched=True, num_proc=num_proc_map_run,
            remove_columns=train_dataset_for_sorting.column_names
        )
        # And to the original validation and test splits
        tokenized_validation = current_processed_dataset['validation'].map(
            lambda ex: preprocess_sft_function_gpt2(ex, global_tokenizer), batched=True, num_proc=num_proc_map_run,
            remove_columns=current_processed_dataset['validation'].column_names
        )
        # Test set is not tokenized here as custom eval uses raw prompts
        print("SFT Tokenization complete for train/validation.")

        # 4. Data Collator
        data_collator_run = DataCollatorForLanguageModeling(tokenizer=global_tokenizer, mlm=False)

        # 5. Metric for Trainer's internal eval (loss/perplexity)
        def compute_metrics_for_trainer_run(eval_pred):  # Renamed for run
            eval_loss = eval_pred.metrics.get("eval_loss", None)
            metrics_to_return = {}
            if eval_loss is not None:
                metrics_to_return["eval_loss"] = eval_loss
                try:
                    perplexity = np.exp(eval_loss); metrics_to_return["perplexity"] = perplexity
                except OverflowError:
                    metrics_to_return["perplexity"] = float('inf')
            else:
                metrics_to_return["eval_loss"] = -1.0; metrics_to_return["perplexity"] = -1.0
            return metrics_to_return

        # 6. Load Model for this run
        model_run = load_model_for_training_or_eval(model_name, global_tokenizer, global_added_pad_token_flag,
                                                    is_training=True)

        # 7. Calculate max_steps
        gradient_accumulation_steps_run = max(1, GPT2_EFFECTIVE_BATCH_SIZE_TARGET // GPT2_PER_DEVICE_TRAIN_BATCH_SIZE)
        try:
            world_size_train = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception:
            world_size_train = 1

        # Total effective batch size considering devices and accumulation
        total_train_batch_size_effective = GPT2_PER_DEVICE_TRAIN_BATCH_SIZE * world_size_train * gradient_accumulation_steps_run
        if total_train_batch_size_effective == 0: raise ValueError("Total effective train batch size is zero.")

        if num_samples_total_for_sampler == 0:
            print("Warning: num_samples_total_for_sampler is zero. Max steps might be small or zero.")
            steps_per_epoch_full = 1  # Avoid division by zero
        else:
            steps_per_epoch_full = math.ceil(num_samples_total_for_sampler / total_train_batch_size_effective)

        calculated_max_steps = math.ceil(GPT2_NUM_TRAIN_EPOCHS * steps_per_epoch_full)
        print(
            f"Effective Batch Size (grad_accum={gradient_accumulation_steps_run}): {total_train_batch_size_effective}, Steps/Epoch (Full Dataset): {steps_per_epoch_full}")
        print(f"Calculated max_steps for {GPT2_NUM_TRAIN_EPOCHS} nominal epochs: {calculated_max_steps}")
        if calculated_max_steps == 0 and num_samples_total_for_sampler > 0: calculated_max_steps = 1
        if num_samples_total_for_sampler == 0: calculated_max_steps = 0

        # Logging steps calculation from original script
        num_devices_for_logging = torch.cuda.device_count() if torch.cuda.is_available() else 1
        train_dataset_len_for_logging = len(tokenized_train) if tokenized_train else 1  # Use tokenized train length
        logging_steps_run = max(1, (train_dataset_len_for_logging // (
                    GPT2_PER_DEVICE_TRAIN_BATCH_SIZE * gradient_accumulation_steps_run * num_devices_for_logging)) // 20 + 1)

        # 8. TrainingArguments
        training_args_run = TrainingArguments(
            output_dir=os.path.join(current_output_dir, "training_checkpoints"),
            overwrite_output_dir=True,
            # num_train_epochs=GPT2_NUM_TRAIN_EPOCHS, # Using max_steps
            max_steps=calculated_max_steps if calculated_max_steps > 0 else 1,
            per_device_train_batch_size=GPT2_PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=GPT2_PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=gradient_accumulation_steps_run,
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
            logging_steps=logging_steps_run,
            learning_rate=GPT2_LEARNING_RATE, weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type="linear",
            save_total_limit=2, load_best_model_at_end=True,
            metric_for_best_model=GPT2_METRIC_FOR_BEST, greater_is_better=False,
            fp16=torch.cuda.is_available(),
            report_to="none", seed=random_seed,
            dataloader_num_workers=min(2, os.cpu_count() if os.cpu_count() else 1),
            prediction_loss_only=True,  # From original GPT-2 script, good for memory
        )

        heuristic_config_dict = {
            "scheduler": training_scheduler, "ordering": ORDERING,
            "competency_param": COMPETENCY_PARAM, "min_train_percent": MIN_TRAIN_PERCENT, "c_init": C_INIT,
        }

        # Prepare validation subset for Trainer's internal eval
        validation_dataset_for_trainer_run = tokenized_validation
        if len(validation_dataset_for_trainer_run) > VALIDATION_SUBSET_SIZE:
            print(f"Using a subset of {VALIDATION_SUBSET_SIZE} samples from validation for Trainer's internal eval.")
            validation_dataset_for_trainer_run = validation_dataset_for_trainer_run.select(
                range(VALIDATION_SUBSET_SIZE))

        # 9. Initialize CustomHeuristicTrainer
        trainer_run = CustomHeuristicTrainer(
            model=model_run, args=training_args_run,
            train_dataset=tokenized_train,
            eval_dataset=validation_dataset_for_trainer_run,
            tokenizer=global_tokenizer, data_collator=data_collator_run,
            compute_metrics=compute_metrics_for_trainer_run,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=GPT2_EARLY_STOPPING_PATIENCE if GPT2_NUM_TRAIN_EPOCHS > 1 else 100)],
            sorted_indices=sorted_sampler_indices,
            heuristic_config=heuristic_config_dict,
            num_samples_total=num_samples_total_for_sampler
        )

        # 10. Train
        training_successful_run = False
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        run_best_model_dir = os.path.join(current_output_dir, "best_model_run")  # For this specific run

        if calculated_max_steps > 0 and num_samples_total_for_sampler > 0:
            print(f"Starting GPT-2 SFT training (max_steps={calculated_max_steps})...")
            train_result_run = trainer_run.train()
            training_successful_run = True
            print("Training for this run finished.")
            trainer_run.log_metrics("train", train_result_run.metrics)
            trainer_run.save_state()
            if training_successful_run:
                trainer_run.save_model(run_best_model_dir)  # Save best model from this run
                print(f"Best model for this run saved to {run_best_model_dir}")
        else:
            print("Skipping training for this run as max_steps or num_samples_total_for_sampler is zero.")
            os.makedirs(run_best_model_dir, exist_ok=True)  # Create dir for consistency

        # 11. Custom Evaluation on Full Test Set (using logic from original script)
        print("\nEvaluating on the full test set with the best model from this run (custom accuracy calculation)...")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if training_successful_run and os.path.exists(run_best_model_dir):
            try:
                model_for_test_run = load_model_for_training_or_eval(run_best_model_dir, global_tokenizer,
                                                                     global_added_pad_token_flag, is_training=False)
                model_for_test_run.eval()
                model_for_test_run.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                original_padding_side_eval = global_tokenizer.padding_side
                global_tokenizer.padding_side = "left"  # For generation

                # Use the untokenized test set from current_processed_dataset
                test_set_raw_run = current_processed_dataset["test"]
                test_prompts_run = [create_gpt2_eval_prompt(test_set_raw_run[i]) for i in
                                    tqdm(range(len(test_set_raw_run)), desc="Creating Test Prompts for Run")]
                test_true_letters_run = [test_set_raw_run[i]["answer_idx"].strip().upper() for i in
                                         range(len(test_set_raw_run))]

                letter_token_ids_run = {}
                for letter_char_eval in ANSWER_MAP_KEYS:
                    tokens_eval = global_tokenizer.encode(letter_char_eval, add_special_tokens=False)
                    if len(tokens_eval) == 1:
                        letter_token_ids_run[letter_char_eval] = tokens_eval[0]
                    else:
                        print(f"Warning: Test Eval Run - Letter '{letter_char_eval}' tokenized to {tokens_eval}.");
                        letter_token_ids_run[letter_char_eval] = -1

                test_all_correctness_run = []
                custom_eval_batch_size = GPT2_PER_DEVICE_EVAL_BATCH_SIZE  # Reuse eval batch size

                for i_batch_eval in tqdm(range(0, len(test_prompts_run), custom_eval_batch_size),
                                         desc="Predicting on Test Set for Run"):
                    batch_prompts_eval = test_prompts_run[i_batch_eval: i_batch_eval + custom_eval_batch_size]
                    inputs_eval = global_tokenizer(
                        batch_prompts_eval, return_tensors="pt", padding="longest", truncation=True,
                        max_length=MAX_EVAL_PROMPT_LENGTH, add_special_tokens=True
                    ).to(model_for_test_run.device)

                    with torch.no_grad():
                        outputs_eval = model_for_test_run(**inputs_eval)
                        next_token_logits_eval = outputs_eval.logits[:, -1, :]  # Logits for the token after prompt

                    next_token_probs_softmax_eval = torch.softmax(next_token_logits_eval, dim=-1).cpu()

                    for j_batch_eval, s_probs_eval in enumerate(next_token_probs_softmax_eval):
                        choice_p_eval = np.zeros(NUM_CHOICES_MC, dtype=float)
                        for choice_i_eval, key_l_eval in enumerate(ANSWER_MAP_KEYS):
                            tid_eval = letter_token_ids_run.get(key_l_eval, -1)
                            if tid_eval != -1: choice_p_eval[choice_i_eval] = s_probs_eval[tid_eval].item()

                        # Normalize if needed (though argmax doesn't require it)
                        # sum_p_test_eval = np.sum(choice_p_eval)
                        # if sum_p_test_eval > 1e-6: choice_p_eval /= sum_p_test_eval
                        # else: choice_p_eval[:] = 1.0 / NUM_CHOICES_MC

                        pred_l_idx_eval = np.argmax(choice_p_eval)
                        pred_l_char_eval = ANSWER_MAP_KEYS[pred_l_idx_eval]
                        true_l_char_eval = test_true_letters_run[i_batch_eval + j_batch_eval]
                        test_all_correctness_run.append(int(pred_l_char_eval == true_l_char_eval))

                global_tokenizer.padding_side = original_padding_side_eval  # Restore padding side
                custom_test_accuracy = np.mean(test_all_correctness_run) if test_all_correctness_run else 0.0
                print(f"Custom Test Set Accuracy for this run: {custom_test_accuracy:.4f}")

                with open(os.path.join(current_output_dir, "custom_test_accuracy.json"), "w") as f_acc:
                    json.dump({"custom_test_accuracy": custom_test_accuracy, "num_test_samples": len(test_prompts_run)},
                              f_acc, indent=4)

                del model_for_test_run
            except Exception as e_cust_eval:
                print(f"Error during custom evaluation for this run: {e_cust_eval}");
                traceback.print_exc()
        else:
            print("Skipping custom test evaluation: training not successful or best model not found for this run.")

        run_status = "completed"

    except Exception as e_main_run:
        print(f"\n!!! ERROR during GPT-2 Heuristic run: Diff='{difficulty_measurer}', Sched='{training_scheduler}' !!!")
        print(f"Output Dir: {current_output_dir}");
        traceback.print_exc()
        run_status = f"error: {str(e_main_run)}"
    finally:
        print(f"Cleaning up resources for run: {current_output_dir}")
        del model_run, trainer_run
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Cleanup complete for run {current_output_dir}. Status: {run_status}. Custom Test Accuracy: {custom_test_accuracy:.4f}")
        print("=" * 80 + "\n")

    return {"accuracy": custom_test_accuracy, "status": run_status, "output_dir": current_output_dir}


# ----- Main Script Execution Orchestrator -----
if __name__ == "__main__":
    print(f"GPT-2 Heuristic Script: Output base directory: {BASE_OUTPUT_DIR_ROOT}")

    # Global Tokenizer Initialization (from original script)
    print(f"Loading global {model_name} tokenizer...")
    try:
        main_tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
    except Exception as e_tok_load:
        print(f"Error loading global tokenizer: {e_tok_load}");
        traceback.print_exc();
        sys.exit(1)

    main_added_pad_token_flag = False
    if main_tokenizer.pad_token is None:
        if main_tokenizer.eos_token:
            main_tokenizer.pad_token = main_tokenizer.eos_token
            print(
                f"Set global tokenizer.pad_token to tokenizer.eos_token: {main_tokenizer.eos_token} (ID: {main_tokenizer.eos_token_id})")
        else:  # Should not happen for GPT2, but as a fallback
            new_pad_token_val_main = '[PAD]'  # Must be a new token if eos_token doesn't exist
            main_tokenizer.add_special_tokens({'pad_token': new_pad_token_val_main})
            main_added_pad_token_flag = True  # A new token was actually added to vocab
            print(
                f"Added new pad_token '{new_pad_token_val_main}' to global tokenizer (ID: {main_tokenizer.pad_token_id})")

    # Ensure pad_token_id is set if pad_token exists
    if main_tokenizer.pad_token is not None and main_tokenizer.pad_token_id is None:
        main_tokenizer.pad_token_id = main_tokenizer.convert_tokens_to_ids(main_tokenizer.pad_token)

    main_tokenizer.padding_side = "right"  # Default for GPT2 SFT, but custom eval might change it locally
    print(
        f"Global Tokenizer padding side: {main_tokenizer.padding_side}, Pad token ID: {main_tokenizer.pad_token_id}, Added new pad: {main_added_pad_token_flag}")

    overall_results = {}
    print("\n===== Starting GPT-2 (Custom Sampler) Experiment Loops =====")
    for diff_measure_loop_gpt2 in difficulty_measures_to_run:
        for scheduler_loop_gpt2 in schedulers_to_run:
            run_id_gpt2 = f"{diff_measure_loop_gpt2}_{scheduler_loop_gpt2}_{ORDERING}"
            current_run_output_dir_gpt2 = os.path.join(BASE_OUTPUT_DIR_ROOT, run_id_gpt2)

            run_summary_gpt2 = run_training_heuristic_gpt2(
                diff_measure_loop_gpt2, scheduler_loop_gpt2, current_run_output_dir_gpt2,
                global_tokenizer=main_tokenizer,
                global_added_pad_token_flag=main_added_pad_token_flag
            )
            overall_results[run_id_gpt2] = run_summary_gpt2
            time.sleep(5)  # Brief pause between runs

    # Overall Summary
    print("\n\n===== Overall GPT-2 (Custom Sampler) Experiment Summary =====")
    for run_id_summary_gpt2, results_summary_gpt2 in overall_results.items():
        print(f"\n--- Results for Run: {run_id_summary_gpt2} ---")
        print(f"  Status: {results_summary_gpt2.get('status', 'unknown')}")
        acc_str_gpt2 = f"{results_summary_gpt2.get('accuracy', 0.0):.4f}"  # Custom test accuracy
        print(f"  Custom Test Accuracy: {acc_str_gpt2}")
        print(f"  Output Dir: {results_summary_gpt2.get('output_dir', 'N/A')}")

    print("======================================")
    overall_summary_file_gpt2 = os.path.join(BASE_OUTPUT_DIR_ROOT, "overall_gpt2_heuristic_summary.json")
    try:
        serializable_results_gpt2 = {}
        for k_sum_gpt2, v_sum_gpt2 in overall_results.items():
            serializable_results_gpt2[k_sum_gpt2] = {
                key_item_gpt2: (float(val_item_gpt2) if isinstance(val_item_gpt2, (
                np.float32, np.float64, np.float16)) else val_item_gpt2)
                for key_item_gpt2, val_item_gpt2 in v_sum_gpt2.items()
            }
        with open(overall_summary_file_gpt2, "w") as f_gpt2:
            json.dump(serializable_results_gpt2, f_gpt2, indent=4)
        print(f"Overall summary saved to: {overall_summary_file_gpt2}")
    except Exception as e_summary_save_gpt2:
        print(f"Warning: Failed to save overall summary: {e_summary_save_gpt2}")

    print("\nAll GPT-2 (Custom Sampler) experiment runs completed.")
    print(f"Script finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M%S')}")