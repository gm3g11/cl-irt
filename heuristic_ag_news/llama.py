# -*- coding: utf-8 -*-
# Llama Heuristic Script with Custom Sampler Fix + Label Filter

import numpy as np
import random
import os
import time
import json
import torch
import traceback # Import traceback for detailed error logging
import math      # For ceil
from typing import Sized, Iterator # For Sampler type hints
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback, AutoConfig # Added AutoConfig just in case needed later
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, SequentialSampler, Sampler # Added Sampler
from datasets import load_dataset, DatasetDict, concatenate_datasets
import evaluate
# No BeautifulSoup needed
import warnings
import packaging.version
import gc
import re
from tqdm.auto import tqdm
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint
from transformers.optimization import Adafactor # Import Adafactor
from huggingface_hub import login, whoami # For Llama access check

# --- Static Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
DATASET_NAME = "contemmcm/ag_news"
MAX_LEN = 128
random_seed = 63
NUM_LABELS = 4 # *** Explicitly 4 for standard AG News interpretation ***
# Batch sizes and accumulation
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 16
DATALOADER_WORKERS = 4
NUM_PREPROCESSING_PROCS = max(1, os.cpu_count() // 2)
# Heuristic Params
ORDERING = 'easiest'
COMPETENCY_PARAM = 5
MIN_TRAIN_PERCENT = 0.05
C_INIT = 0.01
BALANCED_SORTING = False
# Training Params
LEARNING_RATE = 1e-5
ADAFACOR_LR = 2e-5
NUM_TRAIN_EPOCHS = 15 # Used for max_steps calculation
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
GRADIENT_CHECKPOINTING = True
USE_BF16 = True
EARLY_STOPPING_PATIENCE = 3
METRIC_FOR_BEST = "eval_accuracy"

# --- Root Output Directory ---
BASE_OUTPUT_DIR_ROOT = "./Llama3.1_8B_agnews_heuristic_runs_customsampler"
os.makedirs(BASE_OUTPUT_DIR_ROOT, exist_ok=True)

# --- Loop Definitions ---
difficulty_measures_to_run = ['sentence_length', 'word_rarity']
schedulers_to_run = ['linear', 'root']

# --- Environment Setup ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(f"Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# --- Hugging Face Login Check ---
try: user_info = whoami(); print(f"Hugging Face Login Check: Logged in as: {user_info.get('name', 'Unknown User')}")
except Exception as e: print(f"HF Login Check Error: {e}. WARNING: May fail on gated models.")

# --- Global Random Seed ---
print(f"Setting global random seed to: {random_seed}")
torch.manual_seed(random_seed); np.random.seed(random_seed); random.seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
transformers.logging.set_verbosity_warning()

# --- BF16 Check ---
bf16_ready = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
amp_dtype = torch.bfloat16 if bf16_ready and USE_BF16 else torch.float32
print(f"AMP Dtype selected: {amp_dtype}")

# --- Helper Functions ---
def simple_tokenize(sent):
    # ... (implementation unchanged) ...
    if not isinstance(sent, str): return []
    sent = re.sub(r'\s+', ' ', sent)
    tokens = [x.strip() for x in re.findall(r"[\w']+|[^\w\s]", sent) if x.strip()]
    return tokens

def get_example_rarities(texts):
    # ... (implementation unchanged) ...
    if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts): return [0.0] * len(texts)
    tokenized_corpus = [simple_tokenize(text) for text in texts]; vocab = set(); counts = dict(); N = 0
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]; vocab.update(valid_tokens); N += len(valid_tokens)
        for tok in valid_tokens: counts.setdefault(tok, 0); counts[tok] += 1
    if N == 0: return [0.0] * len(texts)
    result = []; epsilon = 1e-9
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]
        if not valid_tokens: p_hat = 0.0
        else: log_probs = [np.log(counts.get(tok, 0) / N + epsilon) for tok in valid_tokens]; p_hat = -np.mean(log_probs) if log_probs else 0.0
        result.append(p_hat)
    return result

def calculate_difficulty_scores(dataset_split, difficulty_measurer, text_column='text'):
    # ... (implementation unchanged) ...
    print(f"Calculating '{difficulty_measurer}' difficulty scores...")
    texts = dataset_split[text_column]; original_indices = list(range(len(texts)))
    valid_indices = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
    valid_texts = [text for i, text in enumerate(texts) if i in valid_indices]
    if len(valid_indices) < len(original_indices): print(f"Warning: Removed {len(original_indices) - len(valid_indices)} invalid examples.")
    if not valid_texts: return [], []
    if difficulty_measurer == 'sentence_length':
        difficulty_scores = [len(text) for text in valid_texts]; print("Calculated sentence length difficulty.")
    elif difficulty_measurer == 'word_rarity':
        difficulty_scores = get_example_rarities(valid_texts); print("Calculated word rarity difficulty.")
    else: raise ValueError(f"Unsupported difficulty_measurer: {difficulty_measurer}")
    return difficulty_scores, valid_indices

# --- Tokenization Function ---
tokenizer = None
def tokenize_function(examples):
    # ... (implementation unchanged - keeps label, text) ...
    if tokenizer is None: raise RuntimeError("Tokenizer not set")
    text_input = examples.get("text"); label_input = examples.get("label")
    if not isinstance(text_input, list): text_input, single_example = [str(text_input) if text_input is not None else ""], True
    else: text_input, single_example = [str(t) if t is not None else "" for t in text_input], False
    tokenized_output = tokenizer(text_input, padding="max_length", truncation=True, max_length=MAX_LEN)
    if label_input is not None:
        if single_example and isinstance(label_input, list): tokenized_output['label'] = label_input[0] if label_input else -1
        elif not single_example and not isinstance(label_input, list): tokenized_output['label'] = [label_input] * len(text_input)
        else: tokenized_output['label'] = label_input
    else: tokenized_output['label'] = [-1] * len(text_input) if not single_example else -1
    if 'text' in examples: tokenized_output['text'] = examples['text'] # Keep text
    return tokenized_output

# --- Custom Heuristic Sampler Definition ---
class HeuristicSampler(Sampler[int]):
    # ... (implementation unchanged from previous fixes) ...
    def __init__(self, num_samples_total: int, batch_size: int, sorted_indices: list[int], heuristic_config: dict, num_replicas: int = 1, rank: int = 0, seed: int = 42):
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas: raise ValueError("Invalid num_replicas or rank.")
        if not isinstance(batch_size, int) or batch_size <= 0: raise ValueError("batch_size should be positive.")
        self.num_replicas = num_replicas; self.rank = rank; self.epoch = 0; self.seed = seed; self.batch_size = batch_size
        self._full_data_len = num_samples_total; self._sorted_indices = sorted_indices; self.heuristic_config = heuristic_config
        min_len_abs_perc = max(1, batch_size * num_replicas, int(self.heuristic_config['min_train_percent'] * self._full_data_len))
        self.abs_min_train_length = self.heuristic_config.get('abs_min_train_length', 0)
        self._min_train_length = max(min_len_abs_perc, self.abs_min_train_length)
        if self._min_train_length > self._full_data_len: self._min_train_length = self._full_data_len
        self.indices_for_epoch = []; self.num_samples_epoch_replica = 0
        self.set_epoch(0)
    def _get_num_samples_for_epoch(self, epoch: int) -> int:
        scheduler_type = self.heuristic_config['scheduler']; num_total = self._full_data_len
        competency_epoch = max(1, self.heuristic_config['competency_param']); c_init = self.heuristic_config['c_init']
        current_epoch_float = float(epoch)
        if scheduler_type == 'linear': epoch_competency = c_init + (1.0 - c_init) * (current_epoch_float / competency_epoch) if current_epoch_float < competency_epoch else 1.0
        elif scheduler_type == 'root': epoch_competency = c_init + (1.0 - c_init) * np.sqrt(current_epoch_float / competency_epoch) if current_epoch_float < competency_epoch else 1.0
        else: raise NotImplementedError(f"Scheduler '{scheduler_type}' unknown.")
        num_train = int(epoch_competency * num_total)
        num_train = max(self._min_train_length, num_train); num_train = min(num_train, num_total)
        return num_train
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch; num_samples_for_epoch = self._get_num_samples_for_epoch(epoch)
        new_indices = self._sorted_indices[:num_samples_for_epoch]
        if epoch == 0 or len(new_indices) != len(self.indices_for_epoch): print(f"[Sampler] Epoch {epoch}: Selecting {len(new_indices)} samples.")
        self.indices_for_epoch = new_indices
        if self.num_replicas > 1:
            num_samples_per_replica = len(self.indices_for_epoch) // self.num_replicas; self.num_samples_epoch_replica = num_samples_per_replica
            total_size = self.num_samples_epoch_replica * self.num_replicas; self.indices_for_epoch = self.indices_for_epoch[:total_size]
        else: self.num_samples_epoch_replica = len(self.indices_for_epoch)
    def __iter__(self) -> Iterator[int]:
        if not self.indices_for_epoch: return iter([])
        g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
        indices_shuffled = [self.indices_for_epoch[i] for i in torch.randperm(len(self.indices_for_epoch), generator=g).tolist()]
        if self.num_replicas > 1: return iter(indices_shuffled[self.rank : len(indices_shuffled) : self.num_replicas])
        else: return iter(indices_shuffled)
    def __len__(self) -> int: return self.num_samples_epoch_replica

# --- Custom Trainer Definition (Accepts Optimizers for Llama/Adafactor) ---
class CustomHeuristicTrainer(Trainer):
    def __init__(self, *args, sorted_indices=None, heuristic_config=None, num_samples_total=None, optimizers=None, **kwargs):
        super().__init__(*args, optimizers=optimizers, **kwargs) # Pass optimizers
        if sorted_indices is None or heuristic_config is None or num_samples_total is None:
             raise ValueError("CustomHeuristicTrainer requires sorted_indices, heuristic_config, and num_samples_total.")
        self.sorted_indices = sorted_indices; self.heuristic_config = heuristic_config; self.num_samples_total = num_samples_total
        print("CustomHeuristicTrainer initialized.")

    def get_train_dataloader(self) -> DataLoader:
        # ... (implementation unchanged from previous fixes) ...
        if self.train_dataset is None: raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset; data_collator = self.data_collator
        heuristic_sampler = HeuristicSampler(
            num_samples_total=self.num_samples_total, batch_size=self._train_batch_size,
            sorted_indices=self.sorted_indices, heuristic_config=self.heuristic_config,
            num_replicas=self.args.world_size, rank=self.args.process_index, seed=self.args.seed
        )
        return DataLoader(
            train_dataset, batch_size=self._train_batch_size, sampler=heuristic_sampler,
            collate_fn=data_collator, drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False
        )

# --- Metrics Definition ---
accuracy_metric = evaluate.load("accuracy", cache_dir=os.environ["HF_DATASETS_CACHE"])
def compute_metrics(p):
    # ... (implementation unchanged - calculates accuracy) ...
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    labels = p.label_ids
    if not isinstance(labels, np.ndarray): labels = np.array(labels)
    if labels.shape != preds.shape: print(f"Warning: Label shape {labels.shape} != Pred shape {preds.shape}"); return {"accuracy": 0.0}
    valid_idx = labels != -100
    if not np.any(valid_idx): return {"accuracy": 0.0}
    preds = preds[valid_idx]; labels = labels[valid_idx]
    return accuracy_metric.compute(predictions=preds, references=labels)

# --- Main Training Function ---
def run_training_heuristic(difficulty_measurer, training_scheduler, output_dir_run):
    """
    Runs a single heuristic training loop for Llama 3.1 on AG News using Custom Sampler.
    Includes label filtering.
    """
    global tokenizer

    print("\n" + "="*80); print(f"Starting Run: Difficulty='{difficulty_measurer}', Scheduler='{training_scheduler}'"); print(f"Output Dir: {output_dir_run}"); print("="*80 + "\n")
    os.makedirs(output_dir_run, exist_ok=True)
    model, trainer = None, None
    adafactor_optimizer, optimizers = None, None
    dataset_raw, dataset_filtered = None, None
    sorted_indices_for_sampler = None; num_samples_total_for_sampler = 0

    try:
        # 1. Load and Split Dataset
        print(f"Loading dataset: {DATASET_NAME}")
        raw_dataset = load_dataset(DATASET_NAME, cache_dir=os.environ["HF_DATASETS_CACHE"])
        complete_dataset = raw_dataset['complete']
        train_temp_split = complete_dataset.train_test_split(test_size=0.2, seed=random_seed)
        train_dataset_raw = train_temp_split['train']; temp_dataset = train_temp_split['test']
        val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=random_seed)
        validation_dataset_raw = val_test_split['train']; test_dataset_raw = val_test_split['test']
        dataset_raw = DatasetDict({'train': train_dataset_raw, 'validation': validation_dataset_raw, 'test': test_dataset_raw})
        del complete_dataset, train_temp_split, temp_dataset, val_test_split, raw_dataset; gc.collect()

        # 2. *** ADD Explicit Label Filtering Step ***
        print(f"Filtering dataset splits to keep labels 0-{NUM_LABELS-1}...")
        dataset_filtered = dataset_raw.filter(
            lambda x: 'label' in x and 0 <= x['label'] < NUM_LABELS,
            num_proc=NUM_PREPROCESSING_PROCS
        )
        train_dataset_filtered = dataset_filtered['train']
        validation_dataset_filtered = dataset_filtered['validation']
        test_dataset_filtered = dataset_filtered['test']
        print(f"Filtered sizes: Train={len(train_dataset_filtered)}, Val={len(validation_dataset_filtered)}, Test={len(test_dataset_filtered)}")
        if len(train_dataset_filtered) == 0 or len(validation_dataset_filtered) == 0:
            raise ValueError("Training or Validation dataset is empty after label filtering!")
        del dataset_raw; gc.collect()
        # Use the constant NUM_LABELS for the rest of the script
        num_labels = NUM_LABELS # Ensure this is set correctly

        # 3. Calculate Difficulty Scores & Get Sorted Indices
        difficulty_scores, valid_indices = calculate_difficulty_scores(train_dataset_filtered, difficulty_measurer, text_column='text')
        train_dataset_final_raw = train_dataset_filtered.select(valid_indices)
        if len(difficulty_scores) != len(train_dataset_final_raw): raise RuntimeError("Mismatch difficulty scores and final data size.")
        print(f"Selected {len(train_dataset_final_raw)} training examples with valid difficulty scores.")
        train_dataset_final_raw = train_dataset_final_raw.add_column('difficulty', difficulty_scores)

        print(f"Sorting training data indices by difficulty ({ORDERING})...")
        start_time = time.time()
        difficulties_final = np.array(train_dataset_final_raw['difficulty'])
        if ORDERING == 'easiest': sorted_indices_for_sampler = np.argsort(difficulties_final).tolist()
        elif ORDERING == 'hardest': sorted_indices_for_sampler = np.argsort(difficulties_final)[::-1].tolist()
        else: raise NotImplementedError(f"Ordering '{ORDERING}' not implemented.")
        print(f"Sorting completed in {time.time() - start_time:.2f}s.")
        num_samples_total_for_sampler = len(train_dataset_final_raw)
        del train_dataset_filtered; gc.collect() # Use train_dataset_final_raw now

        # 4. Load Tokenizer & Set Padding
        print(f"Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            print("Setting Llama padding token to EOS token.")
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is None: raise ValueError("Tokenizer pad_token_id is None.")

        # 5. Tokenize Datasets
        print(f"Tokenizing datasets...")
        # Remove difficulty and text after tokenization
        tokenized_train_full = train_dataset_final_raw.map(tokenize_function, batched=True, num_proc=NUM_PREPROCESSING_PROCS, keep_in_memory=False, remove_columns=['difficulty', 'text'])
        tokenized_validation = validation_dataset_filtered.map(tokenize_function, batched=True, num_proc=NUM_PREPROCESSING_PROCS, keep_in_memory=False, remove_columns=['text'])
        tokenized_test = test_dataset_filtered.map(tokenize_function, batched=True, num_proc=NUM_PREPROCESSING_PROCS, keep_in_memory=False, remove_columns=['text'])
        del train_dataset_final_raw, validation_dataset_filtered, test_dataset_filtered; gc.collect()
        print(f"Columns after tokenization (train): {tokenized_train_full.column_names}")

        # 6. Prepare Final Datasets for Trainer (Cleaned)
        print("Preparing cleaned datasets for Trainer...")
        tokenized_train_full = tokenized_train_full.rename_column("label", "labels")
        tokenized_validation = tokenized_validation.rename_column("label", "labels")
        tokenized_test = tokenized_test.rename_column("label", "labels")
        columns_for_model = ['input_ids', 'attention_mask', 'labels']
        train_dataset_for_trainer = tokenized_train_full.select_columns(columns_for_model)
        eval_dataset_for_trainer = tokenized_validation.select_columns(columns_for_model)
        test_dataset_for_eval = tokenized_test.select_columns(columns_for_model)
        print(f"Columns passed to Trainer: {train_dataset_for_trainer.column_names}")
        del tokenized_train_full, tokenized_validation, tokenized_test; gc.collect()

        # 7. Load Model & Set Config
        print(f"Loading model: {MODEL_NAME} (Num Labels: {num_labels})") # Use filtered num_labels
        model_dtype = torch.bfloat16 if USE_BF16 and torch.cuda.is_bf16_supported() else torch.float32
        print(f"Loading model with dtype: {model_dtype}")
        # Load config first to ensure pad token ID is set if needed
        model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
        model_config.pad_token_id = tokenizer.pad_token_id
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config, torch_dtype=model_dtype, # Pass config
            cache_dir=os.environ["TRANSFORMERS_CACHE"]
        )
        # Verify pad token ID in the loaded model's config
        if model.config.pad_token_id != tokenizer.pad_token_id:
            print(f"Warning: Model config pad token ID {model.config.pad_token_id} differs from tokenizer {tokenizer.pad_token_id}. Setting model config.")
            model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model.config.pad_token_id to: {model.config.pad_token_id}")


        # 8. Calculate max_steps
        try: world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        except Exception: world_size = 1
        total_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS
        if total_train_batch_size == 0: raise ValueError("Total train batch size is zero.")
        if num_samples_total_for_sampler == 0: raise ValueError("Total number of samples is zero.")
        steps_per_epoch_full = math.ceil(num_samples_total_for_sampler / total_train_batch_size)
        calculated_max_steps = math.ceil(NUM_TRAIN_EPOCHS * steps_per_epoch_full)
        print(f"Effective Batch Size: {total_train_batch_size}, Steps/Epoch (Full): {steps_per_epoch_full}")
        print(f"Calculated max_steps for {NUM_TRAIN_EPOCHS} epochs: {calculated_max_steps}")


        # 9. Set Training Arguments
        bf16_arg = USE_BF16 and bf16_ready
        fp16_arg = not bf16_arg and torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=output_dir_run,
            eval_strategy="epoch", save_strategy="epoch", weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True, metric_for_best_model=METRIC_FOR_BEST, greater_is_better=True,
            dataloader_num_workers=DATALOADER_WORKERS, report_to="none", seed=random_seed,
            # remove_unused_columns=True, # Set True explicitly
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_steps=calculated_max_steps, # *** SET MAX STEPS ***
            logging_dir=os.path.join(output_dir_run, "logs"),
            logging_steps=LOGGING_STEPS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs={"use_reentrant": False} if GRADIENT_CHECKPOINTING and packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0") else {},
            bf16=bf16_arg, fp16=fp16_arg,
        )

        # 10. Define Optimizer (Adafactor)
        print("Using Adafactor optimizer.")
        adafactor_optimizer = Adafactor(
            model.parameters(), scale_parameter=False, relative_step=False, lr=ADAFACOR_LR, warmup_init=False,
            weight_decay=WEIGHT_DECAY # Pass weight decay
        )
        optimizers = (adafactor_optimizer, None)

        # Prepare heuristic config dict
        heuristic_config_dict = {
            "scheduler": training_scheduler, "ordering": ORDERING,
            "competency_param": COMPETENCY_PARAM, "min_train_percent": MIN_TRAIN_PERCENT,
            "c_init": C_INIT,
        }

        # 11. Initialize CustomHeuristicTrainer
        print("Initializing CustomHeuristicTrainer...")
        trainer = CustomHeuristicTrainer(
            model=model, args=training_args,
            train_dataset=train_dataset_for_trainer, # Cleaned
            eval_dataset=eval_dataset_for_trainer,  # Cleaned
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
            tokenizer=tokenizer,
            optimizers=optimizers, # Pass Adafactor tuple
            # Pass custom args
            sorted_indices=sorted_indices_for_sampler,
            heuristic_config=heuristic_config_dict,
            num_samples_total=num_samples_total_for_sampler
        )

        # 12. Enable Gradient Checkpointing AFTER trainer init
        if GRADIENT_CHECKPOINTING:
             if hasattr(trainer.model, 'gradient_checkpointing_enable'):
                 print("Enabling gradient checkpointing...")
                 trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
                 if hasattr(trainer.model.config, "use_cache"): trainer.model.config.use_cache = False
                 print("Gradient checkpointing enabled. use_cache set to False.")
             else: print("Warning: Model does not support gradient_checkpointing_enable method.")


        # 13. Train
        print(f"Starting training for run: {output_dir_run} (max_steps={calculated_max_steps})...")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        train_resume_path = last_checkpoint if last_checkpoint else None
        print(f"Attempting to resume from checkpoint: {train_resume_path}")
        train_result = trainer.train(resume_from_checkpoint=train_resume_path)

        # 14. Save Metrics & State
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics); trainer.save_metrics("train", metrics); trainer.save_state()
        print("Training finished.")

        # 15. Evaluate on Test Set
        test_results = {}
        if test_dataset_for_eval is not None and len(test_dataset_for_eval) > 0:
            print("Evaluating on the test set using the best model...")
            test_metrics = trainer.evaluate(eval_dataset=test_dataset_for_eval)
            trainer.log_metrics("test", test_metrics)
            test_metrics["difficulty_measurer"] = difficulty_measurer
            test_metrics["training_scheduler"] = training_scheduler; test_metrics["ordering"] = ORDERING
            trainer.save_metrics("test", test_metrics); test_results = test_metrics
            print("Test results:", test_results)
        elif test_dataset_for_eval is not None:
             print("Test set is empty. Skipping evaluation."); test_results = {"eval_accuracy": 0.0, "status": "empty"}
        else: print("No 'test' split found."); test_results = {"eval_accuracy": 0.0, "status": "no_test_split"}

        # 16. Save Final Model
        print(f"Saving the best model to {output_dir_run}/best_model ...")
        trainer.save_model(f"{output_dir_run}/best_model"); tokenizer.save_pretrained(f"{output_dir_run}/best_model")
        print("Model and tokenizer saved.")

        return test_results

    except Exception as e:
        print(f"\n!!! ERROR during Llama run: Diff='{difficulty_measurer}', Sched='{training_scheduler}' !!!"); print(f"Output Dir: {output_dir_run}"); traceback.print_exc()
        return {"error": str(e)}
    finally:
        # 17. Cleanup
        print(f"Cleaning up resources for run: {output_dir_run}")
        del model, trainer, adafactor_optimizer, optimizers, dataset_filtered
        try: del train_dataset_for_trainer, eval_dataset_for_trainer, test_dataset_for_eval
        except NameError: pass
        tokenizer = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup complete for run."); print("="*80 + "\n")

# --- Experiment Loop ---
overall_results = {}
print("\n===== Starting Llama (Custom Sampler) Experiment Loops =====")
for diff_measure in difficulty_measures_to_run:
    for scheduler in schedulers_to_run:
        run_id = f"{diff_measure}_{scheduler}_{ORDERING}"
        current_output_dir = os.path.join(BASE_OUTPUT_DIR_ROOT, run_id)
        run_results = run_training_heuristic(diff_measure, scheduler, current_output_dir)
        overall_results[run_id] = run_results
        time.sleep(2) # Shorter sleep

# --- Overall Summary ---
print("\n\n===== Overall Llama (Custom Sampler) Experiment Summary =====")
for run_id, results in overall_results.items():
     print(f"\n--- Results for Run: {run_id} ---")
     if "error" in results: print(f"  ERROR: {results['error']}")
     elif "eval_accuracy" in results:
         acc = results['eval_accuracy']
         loss = results.get('eval_loss', 'N/A')
         acc_str = f"{acc:.4f}" if isinstance(acc, (float, np.number)) else acc
         loss_str = f"{loss:.4f}" if isinstance(loss, (float, np.number)) else loss
         print(f"  Test Accuracy: {acc_str}")
         print(f"  Test Loss: {loss_str}")
         if results.get("status") == "empty": print("  (Test set was empty)")
     else: print(f"  Test results format unknown or incomplete: {results}")
print("======================================")
overall_summary_file = os.path.join(BASE_OUTPUT_DIR_ROOT, "overall_summary.json")
try:
    serializable_results = {}
    for k, v in overall_results.items():
        serializable_results[k] = { key: (float(val) if isinstance(val, (np.float32, np.float64)) else val) for key, val in v.items() }
    with open(overall_summary_file, "w") as f: json.dump(serializable_results, f, indent=4)
    print(f"Overall summary saved to: {overall_summary_file}")
except Exception as e: print(f"Warning: Failed to save overall summary: {e}")
print("\nAll Llama (Custom Sampler) experiment runs completed.")