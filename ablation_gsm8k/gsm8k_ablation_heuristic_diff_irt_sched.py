#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSM8K Ablation Study: Heuristic Difficulty + IRT Scheduler
Uses sentence_length/word_rarity difficulty with adaptive IRT-based curriculum (theta estimation)
"""

# ============================================================================
# 🔥 MEMORY FIXES - MUST BE FIRST, BEFORE ANY IMPORTS!
# ============================================================================
import os

# Fix 1: Prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Fix 2: Set cache directories
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Imports
# ============================================================================
import sys
import datetime
import random
import traceback
import json
import argparse
from tqdm import tqdm
import re
import warnings
import gc
import time

import torch
import numpy as np

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler as TorchAmpGradScaler, autocast as torch_amp_autocast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)

# IRT Scoring imports
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import expit

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Creating a tensor from a list of numpy.ndarrays is extremely slow')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*TRANSFORMERS_CACHE.*')

print("✅ Imports ready")

# ============================================================================
# Create cache directories
# ============================================================================
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# ============================================================================
# IRT Scoring Functions (from PUDF code)
# ============================================================================

def theta_fn_irt(difficulties, student_prior, response_pattern):
    """Objective function for IRT theta estimation"""
    def fn(theta_val):
        theta_val = theta_val[0]
        probabilities = expit(theta_val - difficulties)
        log_likelihood = student_prior.logpdf(theta_val)
        probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
        for i, rp_val in enumerate(response_pattern):
            if rp_val == 1:
                log_likelihood += np.log(probabilities[i])
            elif rp_val == -1:
                log_likelihood += np.log(1 - probabilities[i])
        return -log_likelihood
    return fn


def calculate_theta_irt(difficulties, response_pattern, num_obs=-1, initial_theta_val=0.0):
    """Calculate student capacity (theta) using IRT"""
    start_time_irt = time.time()
    difficulties_np = np.array(difficulties, dtype=float)
    response_pattern_np = np.array(response_pattern, dtype=float)
    
    if len(difficulties_np) == 0 or len(difficulties_np) != len(response_pattern_np):
        return initial_theta_val, time.time() - start_time_irt
    
    valid_indices = ~np.isnan(difficulties_np)
    difficulties_filt = difficulties_np[valid_indices]
    response_pattern_filt = response_pattern_np[valid_indices]
    
    if len(difficulties_filt) == 0:
        print("  calculate_theta_irt: No valid difficulties. Returning initial_theta_val.")
        return initial_theta_val, time.time() - start_time_irt
    
    student_prior = norm(loc=0., scale=1.)
    
    if num_obs > 0 and len(difficulties_filt) > num_obs:
        idx = np.random.choice(len(difficulties_filt), num_obs, replace=False)
        difficulties_sample = difficulties_filt[idx]
        response_pattern_sample = response_pattern_filt[idx]
    else:
        difficulties_sample = difficulties_filt
        response_pattern_sample = response_pattern_filt
    
    if len(difficulties_sample) == 0:
        return initial_theta_val, time.time() - start_time_irt
    
    fn_min = theta_fn_irt(difficulties_sample, student_prior, response_pattern_sample)
    res = minimize(fn_min, [initial_theta_val], method='Nelder-Mead', 
                   options={'xatol': 1e-4, 'fatol': 1e-4})
    
    est_theta = res['x'][0]
    if np.isnan(est_theta) or np.isinf(est_theta):
        print(f"  Estimated theta is NaN/Inf. Returning initial_theta_val.")
        est_theta = initial_theta_val
    
    return est_theta, time.time() - start_time_irt


# ============================================================================
# PUDF Data Selection (from PUDF code)
# ============================================================================

def select_data_for_pudf_epoch(
        full_train_dataset, capacity_theta, difficulty_col='difficulty',
        pudf_ordering='easiest', lower_offset=-float('inf'), upper_offset=0.0,
        min_samples_per_epoch=100):
    """Select training data based on capacity and difficulty window"""
    print(f"  Selecting data: capacity_theta={capacity_theta:.4f}, "
          f"window=[{capacity_theta + lower_offset:.4f}, {capacity_theta + upper_offset:.4f}), "
          f"min_samples={min_samples_per_epoch}")
    
    if difficulty_col not in full_train_dataset.column_names:
        print(f"  Error: Difficulty column '{difficulty_col}' not found. Returning empty selection.")
        return full_train_dataset.select([])
    
    min_diff_target = capacity_theta + lower_offset
    max_diff_target = capacity_theta + upper_offset
    
    selected_dataset = full_train_dataset.filter(
        lambda x: x[difficulty_col] is not None and 
                  min_diff_target <= x[difficulty_col] < max_diff_target,
        load_from_cache_file=False
    )
    
    print(f"  Initially selected {len(selected_dataset)} samples by difficulty window.")
    
    if len(selected_dataset) < min_samples_per_epoch:
        print(f"  Selected data ({len(selected_dataset)}) < min_samples ({min_samples_per_epoch}).")
        
        if len(full_train_dataset) == 0:
            return selected_dataset
        
        num_to_take = min(min_samples_per_epoch, len(full_train_dataset))
        print(f"  Taking {num_to_take} samples via '{pudf_ordering}' ordering.")
        
        dataset_to_sort = full_train_dataset.filter(
            lambda x: x[difficulty_col] is not None,
            load_from_cache_file=False
        )
        
        if len(dataset_to_sort) == 0:
            print(f"  Warning: All difficulties were None. Cannot fulfill min_samples.")
            return selected_dataset
        
        reverse_sort = True if pudf_ordering == 'hardest' else False
        
        try:
            sorted_ds = dataset_to_sort.sort(difficulty_col, reverse=reverse_sort, 
                                            load_from_cache_file=False)
            actual_take = min(num_to_take, len(sorted_ds))
            selected_dataset = sorted_ds.select(range(actual_take))
        except Exception as e:
            print(f"  Error sorting for min_samples: {e}")
    
    print(f"  Final samples for epoch: {len(selected_dataset)}")
    return selected_dataset


# ============================================================================
# Heuristic Difficulty Calculation (from heuristic CL code)
# ============================================================================

def simple_tokenize(sent: str) -> list:
    """Simple tokenization for word rarity calculation"""
    if not isinstance(sent, str):
        return []
    sent = re.sub(r'\s+', ' ', sent)
    tokens = [x.strip() for x in re.findall(r"[\w']+|[^\w\s]", sent) if x.strip()]
    return tokens


def get_example_rarities(texts: list) -> list:
    """Calculate word rarity (perplexity-based) for each text"""
    if not texts or not all(isinstance(t, str) for t in texts):
        return [0.0] * len(texts)
    
    tokenized_corpus = [simple_tokenize(text) for text in texts]
    vocab = set()
    counts = {}
    N_tokens = 0
    
    for tokens_in_doc in tokenized_corpus:
        valid_tokens = [t for t in tokens_in_doc if t]
        vocab.update(valid_tokens)
        N_tokens += len(valid_tokens)
        for tok in valid_tokens:
            counts[tok] = counts.get(tok, 0) + 1
    
    if N_tokens == 0:
        return [0.0] * len(texts)
    
    result = []
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


def calculate_difficulty_scores(
    dataset_split,
    difficulty_measurer: str,
    question_column: str = 'question'
) -> tuple:
    """Calculate difficulty scores for GSM8K dataset"""
    print(f"Calculating '{difficulty_measurer}' difficulty scores...")
    
    questions = dataset_split[question_column]
    original_indices = list(range(len(questions)))
    
    # Filter valid examples
    valid_indices = []
    valid_texts = []
    for i, q in enumerate(questions):
        if isinstance(q, str) and q.strip():
            valid_indices.append(i)
            valid_texts.append(q)
    
    if len(valid_indices) < len(original_indices):
        print(f"Warning: Kept {len(valid_indices)} out of {len(original_indices)} examples")
    
    if not valid_texts:
        return [], []
    
    # Calculate difficulty based on measure
    if difficulty_measurer == 'sentence_length':
        difficulty_scores = [len(text) for text in valid_texts]
        print(f"Calculated sentence length (min={min(difficulty_scores)}, max={max(difficulty_scores)})")
        
    elif difficulty_measurer == 'word_rarity':
        difficulty_scores = get_example_rarities(valid_texts)
        print(f"Calculated word rarity (min={min(difficulty_scores):.3f}, max={max(difficulty_scores):.3f})")
        
    else:
        raise ValueError(f"Unsupported difficulty_measurer: {difficulty_measurer}")
    
    return difficulty_scores, valid_indices


# ============================================================================
# Helper Functions
# ============================================================================

def extract_answer(text):
    """Extract numerical answer from text"""
    if '####' in text:
        match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '').strip()
    
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace(',', ''))
    return numbers[-1].strip() if numbers else None


def normalize_answer(answer_str):
    """Normalize answer for comparison"""
    if answer_str is None:
        return None
    try:
        num = float(answer_str.replace(',', ''))
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        return answer_str


def preprocess_gsm8k_sft(examples, tokenizer_ref, max_seq_len, max_prompt_len):
    """Preprocess GSM8K for supervised fine-tuning (Qwen style)"""
    inputs_batch = []
    labels_batch = []
    difficulties_batch = []

    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        answer = examples["answer"][i].strip()

        prompt_text = (
            f"Solve this math problem step by step. "
            f"Show your work and write the final answer after '####'.\n\n"
            f"Question: {question}\n\n"
            f"Answer: Let's solve this step by step.\n"
        )
        
        full_text = prompt_text + answer

        tokenized_prompt = tokenizer_ref(
            prompt_text,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )
        
        tokenized_full = tokenizer_ref(
            full_text,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        prompt_ids = tokenized_prompt.input_ids
        full_ids = tokenized_full.input_ids

        input_ids = []
        if tokenizer_ref.bos_token_id is not None and getattr(tokenizer_ref, 'add_bos_token', True):
            input_ids.append(tokenizer_ref.bos_token_id)
        
        input_ids.extend(full_ids)
        
        if tokenizer_ref.eos_token_id is not None:
            input_ids.append(tokenizer_ref.eos_token_id)

        len_prompt_with_bos = len(prompt_ids)
        if tokenizer_ref.bos_token_id is not None and getattr(tokenizer_ref, 'add_bos_token', True):
            len_prompt_with_bos += 1

        labels = [-100] * len_prompt_with_bos + full_ids[len(prompt_ids):]
        
        if tokenizer_ref.eos_token_id is not None:
            labels.append(tokenizer_ref.eos_token_id)

        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))

        inputs_batch.append(np.array(input_ids, dtype=np.int64))
        labels_batch.append(np.array(labels, dtype=np.int64))
        
        if "difficulty" in examples:
            difficulties_batch.append(examples["difficulty"][i])

    output_dict = {
        "input_ids": inputs_batch,
        "labels": labels_batch
    }
    
    if difficulties_batch and "difficulty" in examples:
        output_dict["difficulty"] = difficulties_batch
    
    return output_dict


def create_evaluation_prompt(question):
    """Create evaluation prompt"""
    return (
        f"Solve this math problem step by step. "
        f"Show your work and write the final answer after '####'.\n\n"
        f"Question: {question}\n\n"
        f"Answer: Let's solve this step by step.\n"
    )


# ============================================================================
# Evaluation and Theta Estimation (from PUDF code)
# ============================================================================

def evaluate_and_estimate_theta(
        peft_model_eval, validation_tokenized, validation_raw,
        tokenizer_eval, device_eval, epoch_num, theta_init,
        max_seq_len, max_gen_tokens, eval_batch_size,
        data_collator, num_obs_theta=-1,
        theta_max_new_tokens=512, theta_batch_size=32):
    """Evaluate model on validation set and estimate capacity (theta)"""
    
    peft_model_eval.eval()
    
    # 1. Calculate validation loss
    print(f"  E{epoch_num} Val: Calculating validation loss...")
    
    val_cols = ['input_ids', 'attention_mask', 'labels']
    current_val_cols = list(validation_tokenized.column_names)
    cols_to_set = [col for col in val_cols if col in current_val_cols]
    validation_tokenized.set_format(type='torch', columns=cols_to_set)
    
    val_dataloader = DataLoader(
        validation_tokenized,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    total_val_loss = 0
    num_val_batches = 0
    
    for val_batch in tqdm(val_dataloader, desc=f"  E{epoch_num} Val Loss", leave=False):
        val_batch = {k: v.to(device_eval) for k, v in val_batch.items() if k in cols_to_set}
        
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=True, dtype=torch.bfloat16):
            outputs = peft_model_eval(**val_batch)
            total_val_loss += outputs.loss.item()
            num_val_batches += 1
        
        del val_batch
        if device_eval.type == 'cuda':
            torch.cuda.empty_cache()
    
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
    perplexity = np.exp(avg_val_loss) if not np.isnan(avg_val_loss) else float('inf')
    
    print(f"  E{epoch_num} Val: Avg Loss = {avg_val_loss:.4f}, Perplexity = {perplexity:.4f}")
    
    # 2. Generate answers and estimate theta
    num_theta_batches = 6
    num_theta_samples = num_theta_batches * theta_batch_size
    
    print(f"  E{epoch_num} Val: Generating answers for theta estimation ({num_theta_samples} samples)...")
    
    has_difficulty = "difficulty" in validation_raw.column_names
    if not has_difficulty:
        print(f"  E{epoch_num} Val: 'difficulty' missing in validation. Theta defaults to initial.")
        return avg_val_loss, perplexity, theta_init, 0.0
    
    if len(validation_raw) > num_theta_samples:
        validation_raw_sampled = validation_raw.select(range(num_theta_samples))
    else:
        validation_raw_sampled = validation_raw
    
    item_diffs_theta = []
    resp_pattern_theta = []
    
    time_theta_start = time.time()
    
    prompts = [create_evaluation_prompt(ex["question"]) for ex in validation_raw_sampled]
    true_answers = [ex["answer"] for ex in validation_raw_sampled]
    difficulties = validation_raw_sampled["difficulty"] if has_difficulty else [np.nan] * len(validation_raw_sampled)
    
    original_padding_side = tokenizer_eval.padding_side
    tokenizer_eval.padding_side = "left"
    
    for i in tqdm(range(0, len(prompts), theta_batch_size), 
                  desc=f"  E{epoch_num} Theta Est", leave=False):
        batch_prompts = prompts[i:i + theta_batch_size]
        batch_true_answers = true_answers[i:i + theta_batch_size]
        batch_difficulties = difficulties[i:i + theta_batch_size]
        
        inputs = tokenizer_eval(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        ).to(device_eval)
        
        with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=True, dtype=torch.bfloat16):
            generated_ids = peft_model_eval.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=theta_max_new_tokens,
                pad_token_id=tokenizer_eval.pad_token_id,
                eos_token_id=tokenizer_eval.eos_token_id,
                do_sample=False
            )
        
        for j in range(len(batch_prompts)):
            if np.isnan(batch_difficulties[j]):
                continue
            
            input_len = inputs['input_ids'][j].shape[0]
            generated_text = tokenizer_eval.decode(
                generated_ids[j][input_len:],
                skip_special_tokens=True
            )
            
            pred_answer = normalize_answer(extract_answer(generated_text))
            true_answer = normalize_answer(extract_answer(batch_true_answers[j]))
            
            is_correct = 1 if (pred_answer == true_answer and pred_answer is not None) else -1
            
            resp_pattern_theta.append(is_correct)
            item_diffs_theta.append(batch_difficulties[j])
        
        del inputs, generated_ids
        if device_eval.type == 'cuda':
            torch.cuda.empty_cache()
    
    tokenizer_eval.padding_side = original_padding_side
    theta_est_time = time.time() - time_theta_start
    
    # Calculate theta
    final_theta_est = theta_init
    if item_diffs_theta and resp_pattern_theta:
        final_theta_est, _ = calculate_theta_irt(
            item_diffs_theta, resp_pattern_theta, num_obs_theta, theta_init
        )
    else:
        print(f"  E{epoch_num} Val: No valid data for IRT theta. Using initial.")
    
    return avg_val_loss, perplexity, final_theta_est, theta_est_time


# ============================================================================
# Parse Command-Line Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study: Heuristic Difficulty + IRT Scheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Difficulty measure
    parser.add_argument(
        "--difficulty-measure",
        type=str,
        choices=["sentence_length", "word_rarity"],
        default="sentence_length",
        help="Heuristic difficulty measure"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--pudf-epochs",
        type=int,
        default=8,
        help="Number of PUDF outer epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Per-device training batch size"
    )
    
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Per-device evaluation batch size"
    )
    
    parser.add_argument(
        "--theta-batch-size",
        type=int,
        default=32,
        help="Batch size for theta estimation"
    )
    
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps"
    )
    
    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )
    
    # PUDF parameters
    parser.add_argument(
        "--initial-theta",
        type=float,
        default=0.0,
        help="Initial capacity (theta) value"
    )
    
    parser.add_argument(
        "--lower-offset",
        type=float,
        default=-1000.0,
        help="Lower difficulty offset from capacity"
    )
    
    parser.add_argument(
        "--upper-offset",
        type=float,
        default=0.0,
        help="Upper difficulty offset from capacity"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples per PUDF epoch"
    )
    
    parser.add_argument(
        "--pudf-ordering",
        type=str,
        choices=["easiest", "hardest"],
        default="easiest",
        help="Ordering strategy when min samples not met"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--theta-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for theta estimation"
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Only evaluate on a subset of test data"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        args.output_dir = f"./gsm8k_ablation_{args.difficulty_measure}_irt_sched_{timestamp}"
    
    return args


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    model_id = "Qwen/Qwen2.5-7B"
    dataset_id = "openai/gsm8k"
    dataset_config = "main"
    
    max_seq_length = 640
    max_prompt_len = 512
    max_gen_tokens = 512
    
    lora_dropout = 0.05
    weight_decay = 0.01
    warmup_ratio = 0.05
    max_grad_norm = 1.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_adapter_path = os.path.join(args.output_dir, "best_pudf_adapter")
    os.makedirs(best_adapter_path, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"{'=' * 70}")
    print(f"ABLATION: {args.difficulty_measure.upper()} DIFFICULTY + IRT SCHEDULER")
    print(f"{'=' * 70}\n")
    
    print(f"🗂️  Cache: {HF_HOME}\n")
    
    print(f"📋 Configuration:")
    print(f"   Model:            {model_id}")
    print(f"   Difficulty:       {args.difficulty_measure}")
    print(f"   Scheduler:        IRT (adaptive theta)")
    print(f"   Output dir:       {args.output_dir}\n")
    
    print(f"🔧 PUDF Parameters:")
    print(f"   PUDF epochs:          {args.pudf_epochs}")
    print(f"   Initial theta:        {args.initial_theta}")
    print(f"   Lower offset:         {args.lower_offset}")
    print(f"   Upper offset:         {args.upper_offset}")
    print(f"   Min samples/epoch:    {args.min_samples}")
    print(f"   PUDF ordering:        {args.pudf_ordering}")
    print(f"   Final 2 epochs:       Train on ALL data\n")
    
    # Load dataset
    print("📚 Loading GSM8K...")
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    
    print(f"✅ Dataset splits:")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Test:  {len(dataset['test'])} examples\n")
    
    # Calculate heuristic difficulty scores
    print(f"Calculating {args.difficulty_measure} difficulty scores...")
    difficulty_scores, valid_indices = calculate_difficulty_scores(
        dataset['train'],
        args.difficulty_measure
    )
    
    if not difficulty_scores:
        raise ValueError("No valid difficulty scores calculated!")
    
    # Select valid training data
    train_dataset_sorted = dataset['train'].select(valid_indices)
    
    # Normalize difficulty scores to make them comparable across measures
    difficulty_scores = np.array(difficulty_scores)
    difficulty_scores = (difficulty_scores - np.mean(difficulty_scores)) / (np.std(difficulty_scores) + 1e-9)
    
    print(f"✅ Normalized difficulty scores:")
    print(f"   Min: {np.min(difficulty_scores):.4f}")
    print(f"   Max: {np.max(difficulty_scores):.4f}")
    print(f"   Mean: {np.mean(difficulty_scores):.4f}")
    print(f"   Std: {np.std(difficulty_scores):.4f}\n")
    
    # Add difficulty to training set
    train_with_difficulty = train_dataset_sorted.add_column("difficulty", difficulty_scores.tolist())
    print(f"✅ Added 'difficulty' column to training set\n")
    
    # Create train/validation split
    print("Splitting train into 80/20 train/validation...")
    train_val_split = train_with_difficulty.train_test_split(
        test_size=0.2, seed=args.seed, shuffle=False
    )
    train_split = train_val_split['train']
    val_split = train_val_split['test']
    
    dataset_with_diff = DatasetDict({
        'train': train_split,
        'validation': val_split,
        'test': dataset['test']
    })
    
    print(f"\nDataset splits:")
    for split_name, split_ds in dataset_with_diff.items():
        print(f"  {split_name}: {len(split_ds)} examples, cols: {split_ds.column_names}")
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            print("   Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            new_pad_token = "␂"
            if new_pad_token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"pad_token": new_pad_token})
                print(f"   Added new pad_token: {new_pad_token}")
            else:
                tokenizer.pad_token = new_pad_token
    
    print(f"✅ Tokenizer loaded (vocab: {len(tokenizer)})\n")
    
    # Tokenize dataset
    print(f"📝 Tokenizing dataset...")
    tokenized_dataset = dataset_with_diff.map(
        lambda ex: preprocess_gsm8k_sft(ex, tokenizer, max_seq_length, max_prompt_len),
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
        remove_columns=[c for c in dataset_with_diff["train"].column_names if c not in ['difficulty']],
        desc="Tokenizing"
    )
    
    print("✅ Tokenization complete\n")
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=-100,
        padding="longest"
    )
    
    # Load model
    print(f"\n{'=' * 70}")
    print("MODEL SETUP")
    print(f"{'=' * 70}\n")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"📥 Loading base model {model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16 if bf16_enabled else torch.float16,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=True,
    )
    
    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
    
    if base_model.config.pad_token_id != tokenizer.pad_token_id:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = False
    
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        use_rslora=True,
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    if hasattr(peft_model, 'enable_input_require_grads'):
        peft_model.enable_input_require_grads()
    if hasattr(peft_model, 'gradient_checkpointing_enable'):
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    print("\n✅ QLoRA PEFT model prepared:")
    peft_model.print_trainable_parameters()
    
    # Setup optimizer and scheduler
    optimizer = AdamW(peft_model.parameters(), lr=args.lr, weight_decay=weight_decay)
    
    num_total_train_items = len(tokenized_dataset['train'])
    steps_per_epoch_approx = (num_total_train_items * 0.75) // (args.train_batch_size * args.grad_accum_steps)
    total_training_steps_approx = int(steps_per_epoch_approx * args.pudf_epochs)
    if total_training_steps_approx == 0:
        total_training_steps_approx = args.pudf_epochs
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_training_steps_approx),
        num_training_steps=max(1, total_training_steps_approx)
    )
    
    scaler = TorchAmpGradScaler(enabled=bf16_enabled)
    
    # PUDF Training Loop
    print(f"\n{'=' * 70}")
    print("PUDF TRAINING LOOP (IRT SCHEDULER)")
    print(f"{'=' * 70}\n")
    
    current_capacity_theta = args.initial_theta
    best_val_loss = float('inf')
    early_stop_counter = 0
    all_pudf_stats = []
    
    full_tokenized_train = tokenized_dataset['train']
    
    for pudf_epoch in range(args.pudf_epochs):
        epoch_start_time = time.time()
        
        print(f"\n===== PUDF Epoch {pudf_epoch + 1}/{args.pudf_epochs} =====")
        print(f"  Current capacity (theta): {current_capacity_theta:.4f}")
        
        is_final_epoch = (pudf_epoch >= args.pudf_epochs - 2)
        
        if is_final_epoch:
            print(f"  ⚡ FINAL EPOCH MODE: Training on ALL data")
        
        # Evaluate and estimate theta
        val_loss, val_ppl, new_theta, theta_time = evaluate_and_estimate_theta(
            peft_model,
            tokenized_dataset['validation'],
            dataset_with_diff['validation'],
            tokenizer,
            DEVICE,
            pudf_epoch + 1,
            current_capacity_theta,
            max_prompt_len,
            max_gen_tokens,
            args.eval_batch_size,
            data_collator,
            num_obs_theta=-1,
            theta_max_new_tokens=args.theta_max_new_tokens,
            theta_batch_size=args.theta_batch_size
        )
        
        # Update capacity with nudge if needed
        if not np.isnan(new_theta) and new_theta <= current_capacity_theta and pudf_epoch > 0:
            current_capacity_theta += 1.0
            print(f"  Theta nudge (+1.0). New capacity: {current_capacity_theta:.4f}")
        elif not np.isnan(new_theta):
            current_capacity_theta = new_theta
        else:
            print(f"  Theta est NaN. Keeping previous: {current_capacity_theta:.4f}")
        
        print(f"  E{pudf_epoch + 1}: Updated capacity = {current_capacity_theta:.4f}")
        print(f"  E{pudf_epoch + 1}: Val Loss (pre-train) = {val_loss:.4f}, PPL = {val_ppl:.4f}")
        
        # Select training data
        if is_final_epoch:
            combined_dataset = concatenate_datasets([
                tokenized_dataset['train'],
                tokenized_dataset['validation']
            ])
            epoch_train_data = combined_dataset
            print(f"  E{pudf_epoch + 1}: Total samples = {len(epoch_train_data)}")
        else:
            print(f"  E{pudf_epoch + 1}: Selecting samples with difficulty < {current_capacity_theta:.4f}")
            epoch_train_data = select_data_for_pudf_epoch(
                full_tokenized_train,
                current_capacity_theta,
                'difficulty',
                args.pudf_ordering,
                args.lower_offset,
                args.upper_offset,
                args.min_samples
            )
        
        num_selected_samples = len(epoch_train_data)
        avg_train_loss = float('nan')
        
        if num_selected_samples > 0:
            train_cols = ['input_ids', 'attention_mask', 'labels']
            current_train_cols = list(epoch_train_data.column_names)
            cols_to_set = [col for col in train_cols if col in current_train_cols]
            epoch_train_data.set_format(type='torch', columns=cols_to_set)
            
            train_dataloader = DataLoader(
                epoch_train_data,
                batch_size=args.train_batch_size,
                collate_fn=data_collator,
                shuffle=True,
                num_workers=min(2, os.cpu_count() if os.cpu_count() else 1)
            )
            
            peft_model.train()
            total_train_loss = 0
            grad_steps = 0
            
            for step, train_batch in enumerate(
                tqdm(train_dataloader, desc=f"  Training E{pudf_epoch + 1}", leave=False)
            ):
                optimizer.zero_grad()
                
                train_batch = {k: v.to(DEVICE) for k, v in train_batch.items() if k in cols_to_set}
                
                with torch_amp_autocast(device_type=DEVICE.type, enabled=bf16_enabled, 
                                       dtype=torch.bfloat16 if bf16_enabled else torch.float16):
                    outputs = peft_model(**train_batch)
                    loss = outputs.loss / args.grad_accum_steps
                
                scaler.scale(loss).backward()
                total_train_loss += loss.item() * args.grad_accum_steps
                
                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    grad_steps += 1
                
                if step % 25 == 0 and DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            
            avg_train_loss = total_train_loss / grad_steps if grad_steps > 0 else float('nan')
            print(f"  E{pudf_epoch + 1}: Avg train loss = {avg_train_loss:.4f}")
        else:
            print(f"  E{pudf_epoch + 1}: No data selected. Skipping training.")
        
        # Post-training evaluation
        val_loss_post, val_ppl_post, theta_post, _ = evaluate_and_estimate_theta(
            peft_model,
            tokenized_dataset['validation'],
            dataset_with_diff['validation'],
            tokenizer,
            DEVICE,
            pudf_epoch + 1,
            current_capacity_theta,
            max_prompt_len,
            max_gen_tokens,
            args.eval_batch_size,
            data_collator,
            num_obs_theta=-1,
            theta_max_new_tokens=args.theta_max_new_tokens,
            theta_batch_size=args.theta_batch_size
        )
        
        print(f"  E{pudf_epoch + 1}: Val Loss (post-train) = {val_loss_post:.4f}, PPL = {val_ppl_post:.4f}")
        
        epoch_duration = time.time() - epoch_start_time
        
        all_pudf_stats.append({
            "pudf_epoch": pudf_epoch + 1,
            "capacity_theta": current_capacity_theta,
            "num_selected_samples": num_selected_samples,
            "avg_train_loss": avg_train_loss,
            "val_loss": val_loss_post,
            "val_perplexity": val_ppl_post,
            "theta_post_train": theta_post,
            "duration_s": epoch_duration,
            "theta_est_time_s": theta_time
        })
        
        # Check for improvement
        if not np.isnan(val_loss_post) and val_loss_post < best_val_loss:
            print(f"  New best val loss: {val_loss_post:.4f}. Saving adapter.")
            best_val_loss = val_loss_post
            early_stop_counter = 0
            
            peft_model.save_pretrained(best_adapter_path)
            tokenizer.save_pretrained(best_adapter_path)
        elif not np.isnan(val_loss_post):
            early_stop_counter += 1
            print(f"  Val loss not improved. EarlyStop: {early_stop_counter}/{args.early_stopping_patience}")
        else:
            early_stop_counter += 1
            print(f"  Val loss is NaN. EarlyStop: {early_stop_counter}/{args.early_stopping_patience}")
        
        if early_stop_counter >= args.early_stopping_patience:
            print(f"  Early stopping at E{pudf_epoch + 1}.")
            break
        
        print(f"  PUDF E{pudf_epoch + 1} ended. Duration: {epoch_duration:.2f}s")
    
    print(f"\n✅ PUDF Training finished. Best val loss: {best_val_loss:.4f}")
    
    # Save training statistics
    stats_file = os.path.join(args.output_dir, "pudf_training_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(all_pudf_stats, f, indent=4)
    print(f"📊 Training stats saved: {stats_file}")
    
    # FINAL TEST EVALUATION
    print(f"\n{'=' * 70}")
    print("FINAL TEST EVALUATION")
    print(f"{'=' * 70}\n")
    
    if os.path.exists(os.path.join(best_adapter_path, "adapter_model.safetensors")) or \
       os.path.exists(os.path.join(best_adapter_path, "adapter_model.bin")):
        
        print(f"📥 Loading best adapter from {best_adapter_path}...")
        
        base_model_eval = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            dtype=torch.bfloat16 if bf16_enabled else torch.float16,
            device_map="auto",
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            trust_remote_code=True,
        )
        
        if len(tokenizer) > base_model_eval.config.vocab_size:
            base_model_eval.resize_token_embeddings(len(tokenizer))
        
        if base_model_eval.config.pad_token_id != tokenizer.pad_token_id:
            base_model_eval.config.pad_token_id = tokenizer.pad_token_id
        
        final_model = PeftModel.from_pretrained(base_model_eval, best_adapter_path)
        final_model.eval()
        print("✅ Model loaded for evaluation\n")
        
        test_dataset = dataset_with_diff["test"]
        if args.test_subset:
            print(f"⚠️  Using subset of {args.test_subset} test examples")
            test_dataset = test_dataset.select(range(min(args.test_subset, len(test_dataset))))
        
        print(f"🧮 Evaluating on {len(test_dataset)} examples...")
        
        results = []
        correct = 0
        total = 0
        
        with tqdm(total=len(test_dataset), desc="Evaluating") as pbar:
            for idx in range(0, len(test_dataset), args.eval_batch_size):
                batch_end = min(idx + args.eval_batch_size, len(test_dataset))
                batch = test_dataset[idx:batch_end]
                
                prompts = [create_evaluation_prompt(q) for q in batch["question"]]
                
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_len
                ).to(DEVICE)
                
                with torch.inference_mode():
                    generated_ids = final_model.generate(
                        **inputs,
                        max_new_tokens=max_gen_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                for i in range(len(prompts)):
                    input_len = inputs['input_ids'][i].shape[0]
                    generated_text = tokenizer.decode(
                        generated_ids[i][input_len:],
                        skip_special_tokens=True
                    )
                    
                    pred_answer = normalize_answer(extract_answer(generated_text))
                    true_answer = normalize_answer(extract_answer(batch["answer"][i]))
                    
                    is_correct = (pred_answer == true_answer) and (pred_answer is not None)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    results.append({
                        "id": idx + i,
                        "question": batch["question"][i][:100] + "...",
                        "predicted": pred_answer,
                        "true": true_answer,
                        "correct": is_correct
                    })
                
                pbar.update(len(prompts))
                pbar.set_postfix({"accuracy": f"{correct / total:.2%}"})
                
                del inputs, generated_ids
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
        
        accuracy = correct / total
        
        print(f"\n{'=' * 70}")
        print(f"🎉 FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Difficulty:  {args.difficulty_measure}")
        print(f"Scheduler:   IRT (adaptive theta)")
        print(f"Total:       {total} examples")
        print(f"Correct:     {correct}")
        print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"{'=' * 70}\n")
        
        # Save results
        results_file = os.path.join(args.output_dir, "test_evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "model_id": model_id,
                "ablation_type": f"{args.difficulty_measure}_difficulty_irt_scheduler",
                "pudf_config": {
                    "difficulty_measure": args.difficulty_measure,
                    "scheduler": "IRT (adaptive theta)",
                    "pudf_epochs": args.pudf_epochs,
                    "initial_theta": args.initial_theta,
                    "lower_offset": args.lower_offset,
                    "upper_offset": args.upper_offset,
                    "min_samples": args.min_samples,
                    "pudf_ordering": args.pudf_ordering,
                },
                "hyperparameters": {
                    "learning_rate": args.lr,
                    "train_batch_size": args.train_batch_size,
                    "effective_batch_size": args.train_batch_size * args.grad_accum_steps,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": lora_dropout,
                    "max_seq_length": max_seq_length,
                },
                "metrics": {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "best_val_loss": best_val_loss,
                },
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    else:
        print(f"Best adapter not found. Skipping final test.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n✨ Complete! Output directory: {args.output_dir}")
