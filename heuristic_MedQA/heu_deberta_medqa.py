import os
import datetime
import random
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset  # Added Dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForMultipleChoice,
    Trainer,  # Will be replaced by CustomHeuristicTrainer
    TrainingArguments,
    DataCollatorForMultipleChoice,
    EarlyStoppingCallback
)
from torch.utils.data import DataLoader, Sampler  # Added for HeuristicSampler
from evaluate import load as load_metric
import shutil  # Not strictly needed with new save_total_limit=1 and explicit best_model save
import glob  # Not strictly needed
import traceback
import json  # For overall summary
import re  # For simple_tokenize
import math  # For ceil
import time  # For heuristic script
import gc  # For heuristic script
from typing import List, Dict, Any, Optional, Tuple, Iterator  # For type hints

# ----- Environment Setup -----
# Attempt to use a shared lab cache, fall back to user's default if not accessible
HF_HOME_original = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
if os.path.isdir(HF_HOME_original):  # Check if the directory itself exists
    HF_HOME = HF_HOME_original
    print(f"Using shared Hugging Face cache directory: {HF_HOME}")
else:
    print(
        f"Warning: Shared path {HF_HOME_original} not found or not a directory. Using default Hugging Face cache directory.")
    HF_HOME = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
import transformers  # For logging verbosity

# ----- Random Seed -----
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
transformers.logging.set_verbosity_warning()
print(f"Using random seed: {random_seed}")

# ----- Timestamp and Baseline Config -----
print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model_name = "microsoft/deberta-v3-base"  # DeBERTa model
dataset_id = "GBaker/MedQA-USMLE-4-options"
max_length = 512  # DeBERTa script's max_length

# ----- Heuristic Configuration -----
BASE_OUTPUT_DIR_ROOT = "./deberta_medqa_heuristic_runs_v2"  # Changed suffix to avoid overwriting
os.makedirs(BASE_OUTPUT_DIR_ROOT, exist_ok=True)

difficulty_measures_to_run = ['sentence_length', 'word_rarity']
schedulers_to_run = ['linear', 'root']

ORDERING = 'easiest'
COMPETENCY_PARAM = 5
MIN_TRAIN_PERCENT = 0.05
C_INIT = 0.01

# DeBERTa script training parameters
DEBERTA_LEARNING_RATE = 2e-5
DEBERTA_PER_DEVICE_TRAIN_BATCH_SIZE = 16
DEBERTA_PER_DEVICE_EVAL_BATCH_SIZE = DEBERTA_PER_DEVICE_TRAIN_BATCH_SIZE * 2
DEBERTA_NUM_EPOCHS = 20  # Used for max_steps calculation
DEBERTA_WEIGHT_DECAY = 0.01
DEBERTA_LOGGING_STEPS = 50
DEBERTA_EARLY_STOPPING_PATIENCE = 2  # User's original setting, can be experimented with
DEBERTA_METRIC_FOR_BEST = "accuracy"


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


ANSWER_MAP_KEYS_DIFFICULTY = ['A', 'B', 'C', 'D']


def combine_text_for_difficulty(example: Dict[str, Any]) -> Dict[str, str]:
    question = example["question"].strip() if isinstance(example["question"], str) else ""
    options_dict = example["options"]
    options_text_parts = []
    if isinstance(options_dict, dict):
        for key_char in ANSWER_MAP_KEYS_DIFFICULTY:
            option_val = options_dict.get(key_char)
            options_text_parts.append(str(option_val) if option_val is not None else "")
    options_combined = " ".join(filter(None, options_text_parts))
    full_text = f"{question} {options_combined}".strip()
    if not full_text: return {"full_text_for_difficulty": " "}
    return {"full_text_for_difficulty": full_text}


# ----- Custom Heuristic Sampler Definition -----
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
        if epoch == 0 or len(new_indices) != len(self.indices_for_epoch) or epoch % DEBERTA_LOGGING_STEPS == 0:
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


# ----- Custom Trainer Definition -----
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


# ----- Main Training Function for DeBERTa with Heuristics -----
def run_training_heuristic_deberta(
        difficulty_measurer: str,
        training_scheduler: str,
        current_output_dir: str,
        global_tokenizer: DebertaV2Tokenizer
):
    print("\n" + "=" * 80);
    print(f"Starting DeBERTa Run: Diff='{difficulty_measurer}', Scheduler='{training_scheduler}'");
    print(f"Output Dir: {current_output_dir}");
    print("=" * 80 + "\n")
    os.makedirs(current_output_dir, exist_ok=True)

    model_run, trainer_run = None, None
    test_accuracy = 0.0
    run_status = "failed"

    try:
        # 1. Load Dataset
        print(f"Loading dataset: {dataset_id}")
        dataset_full = load_dataset(dataset_id, cache_dir=os.environ["HF_DATASETS_CACHE"])

        if 'validation' not in dataset_full:
            print("Validation split not found. Splitting train set into 80% train / 20% validation...")
            if 'train' not in dataset_full or len(dataset_full['train']) == 0:
                raise ValueError("Original train split not found or empty!")

            stratify_column = 'answer_idx' if 'answer_idx' in dataset_full['train'].column_names else None
            try:
                if stratify_column:
                    unique_labels = dataset_full['train'].unique(stratify_column)
                    if not unique_labels or any(label is None for label in unique_labels):  # handle None labels if any
                        print(
                            f"Warning: Column {stratify_column} contains None values or is empty, cannot stratify. Using random split.")
                        stratify_column = None
                    else:
                        label_counts = {
                            label: dataset_full['train'].filter(lambda ex: ex[stratify_column] == label).num_rows for
                            label in unique_labels}
                        min_samples = min(label_counts.values()) if label_counts else 0
                        if min_samples < 2:
                            print(
                                f"Warning: Stratification by {stratify_column} might fail (min_samples: {min_samples}). Attempting random split.")
                            stratify_column = None  # Fallback to random if stratification seems problematic

                split = dataset_full['train'].train_test_split(test_size=0.2, seed=random_seed, shuffle=True,
                                                               stratify_by_column=stratify_column)
            except Exception as e_stratify_outer:
                print(f"Warning: Stratification by {stratify_column} failed ({e_stratify_outer}). Using random split.")
                split = dataset_full['train'].train_test_split(test_size=0.2, seed=random_seed,
                                                               shuffle=True)  # Fallback

            # Further split the 'test' part of the above into validation and test for our use
            stratify_column_val_test = stratify_column if stratify_column and stratify_column in split[
                'test'].column_names else None
            try:
                if stratify_column_val_test:  # Check again for the new subset
                    unique_labels_vt = split['test'].unique(stratify_column_val_test)
                    if not unique_labels_vt or any(label is None for label in unique_labels_vt):
                        stratify_column_val_test = None
                    else:
                        label_counts_vt = {
                            label: split['test'].filter(lambda ex: ex[stratify_column_val_test] == label).num_rows for
                            label in unique_labels_vt}
                        min_samples_vt = min(label_counts_vt.values()) if label_counts_vt else 0
                        if min_samples_vt < 2: stratify_column_val_test = None

                val_test_split = split['test'].train_test_split(test_size=0.5, seed=random_seed, shuffle=True,
                                                                stratify_by_column=stratify_column_val_test)
            except Exception as e_stratify_inner:
                print(
                    f"Warning: Stratification for val/test split failed ({e_stratify_inner}). Using random split for val/test.")
                val_test_split = split['test'].train_test_split(test_size=0.5, seed=random_seed, shuffle=True)

            current_run_dataset = DatasetDict({
                'train': split['train'],
                'validation': val_test_split['train'],  # This is our new validation set
                'test': val_test_split['test']  # This is our new test set
            })
        else:  # If 'validation' exists, assume 'test' also exists in the desired form or is from dataset_full['test']
            current_run_dataset = DatasetDict({
                'train': dataset_full['train'],
                'validation': dataset_full['validation'],
                'test': dataset_full['test']  # Use the original test split
            })
        print("Final dataset splits for this run:", current_run_dataset)
        if len(current_run_dataset['train']) == 0 or len(current_run_dataset['validation']) == 0 or len(
                current_run_dataset['test']) == 0:
            raise ValueError("Training, Validation or Test dataset is empty for this run!")

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
        start_time = time.time()
        if ORDERING == 'easiest':
            sorted_sampler_indices = np.argsort(np.array(difficulty_scores)).tolist()
        elif ORDERING == 'hardest':
            sorted_sampler_indices = np.argsort(np.array(difficulty_scores))[::-1].tolist()
        else:
            raise NotImplementedError(f"Ordering '{ORDERING}' not implemented.")
        print(f"Sorting completed in {time.time() - start_time:.2f}s.")
        num_samples_total_for_sampler = len(train_dataset_for_sorting)

        # 3. Load Model
        print(f"Loading model: {model_name}")
        model_run = DebertaV2ForMultipleChoice.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])

        # 4. Preprocessing (DeBERTa specific)
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        option_keys_mc = ['A', 'B', 'C', 'D']
        num_choices_mc = len(option_keys_mc)

        def preprocess_mc_function(examples):
            num_examples = len(examples["question"])
            first_sentences = [
                [str(examples["question"][i] if examples["question"][i] is not None else "")] * num_choices_mc for i in
                range(num_examples)]
            second_sentences_options = []
            labels_mc = []
            for i in range(num_examples):
                options_dict = examples["options"][i]
                answer_idx_key = str(
                    examples["answer_idx"][i] if examples["answer_idx"][i] is not None else "").strip().upper()
                if not isinstance(options_dict, dict):
                    option_texts = ["[placeholder option]"] * num_choices_mc
                else:
                    option_texts = [str(options_dict.get(key, "[option unavailable]") if options_dict.get(
                        key) is not None else "[option unavailable]") for key in option_keys_mc]
                second_sentences_options.append(option_texts)
                if answer_idx_key in answer_map:
                    labels_mc.append(answer_map[answer_idx_key])
                else:
                    labels_mc.append(-100)

            first_sentences_flat = [s for sublist in first_sentences for s in sublist]
            second_sentences_flat = [s for sublist in second_sentences_options for s in sublist]

            tokenized_examples = global_tokenizer(
                first_sentences_flat, second_sentences_flat,
                max_length=max_length, truncation=True, padding=False
            )
            unflattened = {}
            for k, v in tokenized_examples.items():
                unflattened[k] = [v[j: j + num_choices_mc] for j in range(0, len(v), num_choices_mc)]
            unflattened["labels"] = labels_mc
            return unflattened

        print("Tokenizing dataset for DeBERTa...")
        num_proc_tok = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)

        tokenized_train = train_dataset_for_sorting.map(
            preprocess_mc_function, batched=True, num_proc=num_proc_tok,
            remove_columns=train_dataset_for_sorting.column_names
        )
        tokenized_validation = current_run_dataset['validation'].map(
            preprocess_mc_function, batched=True, num_proc=num_proc_tok,
            remove_columns=current_run_dataset['validation'].column_names
        )
        # Tokenize the test set from current_run_dataset for evaluation
        tokenized_test_eval = current_run_dataset['test'].map(
            preprocess_mc_function, batched=True, num_proc=num_proc_tok,
            remove_columns=current_run_dataset['test'].column_names
        )
        print("Tokenization complete.")

        # 5. Metric
        accuracy_metric_eval = load_metric("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])

        def compute_metrics_mc(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            valid_labels = np.array(labels).astype(int)
            valid_indices = (valid_labels != -100)
            valid_preds_final = predictions[valid_indices]
            valid_labels_final = valid_labels[valid_indices]
            if len(valid_labels_final) == 0: return {"accuracy": 0.0}
            acc = accuracy_metric_eval.compute(predictions=valid_preds_final, references=valid_labels_final)
            return {"accuracy": acc["accuracy"]}

        # 6. Data Collator
        data_collator_mc = DataCollatorForMultipleChoice(tokenizer=global_tokenizer, padding=True)

        # 7. Calculate max_steps
        try:
            world_size_train = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception:
            world_size_train = 1
        total_train_batch_size_effective = DEBERTA_PER_DEVICE_TRAIN_BATCH_SIZE * world_size_train
        if total_train_batch_size_effective == 0: raise ValueError("Total effective train batch size is zero.")
        if num_samples_total_for_sampler == 0:
            print("Warning: num_samples_total_for_sampler is zero. Max steps might be small or zero.")
            steps_per_epoch_full = 1
        else:
            steps_per_epoch_full = math.ceil(num_samples_total_for_sampler / total_train_batch_size_effective)
        calculated_max_steps = math.ceil(DEBERTA_NUM_EPOCHS * steps_per_epoch_full)
        print(
            f"Effective Batch Size: {total_train_batch_size_effective}, Steps/Epoch (Full Dataset): {steps_per_epoch_full}")
        print(f"Calculated max_steps for {DEBERTA_NUM_EPOCHS} nominal epochs: {calculated_max_steps}")
        if calculated_max_steps == 0 and num_samples_total_for_sampler > 0: calculated_max_steps = 1
        if num_samples_total_for_sampler == 0: calculated_max_steps = 0

        # 8. TrainingArguments
        training_args_run = TrainingArguments(
            output_dir=f"{current_output_dir}/checkpoints",
            eval_strategy="epoch", save_strategy="epoch",
            learning_rate=DEBERTA_LEARNING_RATE,
            per_device_train_batch_size=DEBERTA_PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=DEBERTA_PER_DEVICE_EVAL_BATCH_SIZE,
            max_steps=calculated_max_steps if calculated_max_steps > 0 else 1,
            weight_decay=DEBERTA_WEIGHT_DECAY,
            logging_dir=f"{current_output_dir}/logs",
            logging_strategy="steps", logging_steps=DEBERTA_LOGGING_STEPS,
            load_best_model_at_end=True, metric_for_best_model=DEBERTA_METRIC_FOR_BEST,
            save_total_limit=1,
            fp16=torch.cuda.is_available(), report_to="none", seed=random_seed,
            remove_unused_columns=False,
        )

        heuristic_config_dict = {
            "scheduler": training_scheduler, "ordering": ORDERING,
            "competency_param": COMPETENCY_PARAM, "min_train_percent": MIN_TRAIN_PERCENT, "c_init": C_INIT,
        }

        # 9. Initialize CustomHeuristicTrainer
        trainer_run = CustomHeuristicTrainer(
            model=model_run, args=training_args_run,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,
            tokenizer=global_tokenizer, data_collator=data_collator_mc,
            compute_metrics=compute_metrics_mc,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=DEBERTA_EARLY_STOPPING_PATIENCE)],
            sorted_indices=sorted_sampler_indices,
            heuristic_config=heuristic_config_dict,
            num_samples_total=num_samples_total_for_sampler
        )

        # 10. Train
        if calculated_max_steps > 0 and num_samples_total_for_sampler > 0:
            print("Starting DeBERTa training with heuristic sampler...")
            train_result = trainer_run.train()
            trainer_run.log_metrics("train", train_result.metrics)
            trainer_run.save_metrics("train", train_result.metrics)
            trainer_run.save_state()
            print("Training finished. Best model loaded into trainer.")
        else:
            print("Skipping training as max_steps or num_samples_total_for_sampler is zero.")

        # 11. Evaluation on Test Set
        print("Evaluating on the test set...")
        # *** CORRECTED CONDITION FOR TEST EVALUATION ***
        if tokenized_test_eval and len(tokenized_test_eval) > 0 and \
                (calculated_max_steps > 0 and num_samples_total_for_sampler > 0):
            print(f"Test set has {len(tokenized_test_eval)} examples. Proceeding with evaluation.")
            test_results = trainer_run.evaluate(eval_dataset=tokenized_test_eval)  # Use tokenized_test_eval
            test_accuracy = test_results.get("eval_accuracy", 0.0)
            print(f"Test results for this run: Accuracy = {test_accuracy:.4f}, Full metrics: {test_results}")
            trainer_run.log_metrics("test", test_results)
            with open(os.path.join(current_output_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=4)
        elif not (calculated_max_steps > 0 and num_samples_total_for_sampler > 0):
            print("Skipping test evaluation as no training was performed.")
            test_accuracy = 0.0
        else:
            print(
                f"Test split not found or is empty in tokenized data (size: {len(tokenized_test_eval) if tokenized_test_eval else 'None'}). Skipping final test evaluation.")
            test_accuracy = 0.0

        # 12. Save Final Best Model
        final_best_model_path = os.path.join(current_output_dir, "best_model")
        if calculated_max_steps > 0 and num_samples_total_for_sampler > 0:
            trainer_run.save_model(final_best_model_path)
            global_tokenizer.save_pretrained(final_best_model_path)
            print(f"Best model from training saved to {final_best_model_path}")
        else:
            os.makedirs(final_best_model_path, exist_ok=True)
            print(
                f"No training performed, so no specific 'best_model' saved to {final_best_model_path}, but directory created.")

        run_status = "completed"

    except Exception as e_run:
        print(
            f"\n!!! ERROR during DeBERTa Heuristic run: Diff='{difficulty_measurer}', Sched='{training_scheduler}' !!!")
        print(f"Output Dir: {current_output_dir}");
        traceback.print_exc()
        run_status = f"error: {str(e_run)}"
    finally:
        print(f"Cleaning up resources for run: {current_output_dir}")
        del model_run, trainer_run
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Cleanup complete for run {current_output_dir}. Status: {run_status}. Test Accuracy: {test_accuracy:.4f}")
        print("=" * 80 + "\n")

    return {"accuracy": test_accuracy, "status": run_status, "output_dir": current_output_dir}


# ----- Main Script Execution Orchestrator -----
if __name__ == "__main__":
    print(f"DeBERTa Heuristic Script: Output base directory: {BASE_OUTPUT_DIR_ROOT}")

    # Global Tokenizer Initialization
    print(f"Loading global DeBERTa tokenizer: {model_name}")
    try:
        main_tokenizer = DebertaV2Tokenizer.from_pretrained(model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"])
        print("Global DeBERTa Tokenizer loaded.")
    except Exception as e:
        print(f"Fatal error loading global DeBERTa tokenizer: {e}")
        traceback.print_exc()
        sys.exit(1)  # Exit if global tokenizer fails

    overall_results = {}
    print("\n===== Starting DeBERTa (Custom Sampler) Experiment Loops =====")
    for diff_measure_loop in difficulty_measures_to_run:
        for scheduler_loop in schedulers_to_run:
            run_id = f"{diff_measure_loop}_{scheduler_loop}_{ORDERING}"
            current_run_output_dir_loop = os.path.join(BASE_OUTPUT_DIR_ROOT, run_id)

            run_summary = run_training_heuristic_deberta(
                diff_measure_loop, scheduler_loop, current_run_output_dir_loop, global_tokenizer=main_tokenizer
            )
            overall_results[run_id] = run_summary
            time.sleep(5)

    # Overall Summary
    print("\n\n===== Overall DeBERTa (Custom Sampler) Experiment Summary =====")
    for run_id_summary, results_summary in overall_results.items():
        print(f"\n--- Results for Run: {run_id_summary} ---")
        print(f"  Status: {results_summary.get('status', 'unknown')}")
        acc_str = f"{results_summary.get('accuracy', 0.0):.4f}"
        print(f"  Test Accuracy: {acc_str}")
        print(f"  Output Dir: {results_summary.get('output_dir', 'N/A')}")

    print("======================================")
    overall_summary_file = os.path.join(BASE_OUTPUT_DIR_ROOT, "overall_deberta_heuristic_summary.json")
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

    print("\nAll DeBERTa (Custom Sampler) experiment runs completed.")
    print(f"Script finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")