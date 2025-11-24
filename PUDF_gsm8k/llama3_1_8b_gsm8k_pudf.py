#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================================
# GSM8K PUDF Training with QLoRA
# Progressive Uncertainty-aware Difficulty Filtering
# Updated: 8 epochs (1-5 PUDF, 6-8 full data), faster evaluation
# ============================================================================

# ============================================================================
# 🔥 MEMORY FIXES - MUST BE FIRST, BEFORE ANY IMPORTS!
# ============================================================================
import os

# Fix 1: Prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Fix 2: Set cache directories
HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Now safe to import everything else
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

print("✅ Ready to train (using cached authentication)")


# ============================================================================
# IRT Scoring Functions
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
# PUDF Data Selection
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
    """Preprocess GSM8K for supervised fine-tuning"""
    inputs_batch = []
    labels_batch = []
    difficulties_batch = []
    
    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        answer = examples["answer"][i].strip()
        
        # Improved prompt format
        prompt_text = (
            f"Solve this math problem step by step. "
            f"Show your work and write the final answer after '####'.\n\n"
            f"Question: {question}\n\n"
            f"Answer: Let's solve this step by step.\n"
        )
        
        full_text = prompt_text + answer
        
        # Tokenize
        tokenized_full = tokenizer_ref(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            add_special_tokens=True
        )
        
        tokenized_prompt = tokenizer_ref(
            prompt_text,
            truncation=True,
            max_length=max_prompt_len,
            padding=False,
            add_special_tokens=True
        )
        
        input_ids = tokenized_full.input_ids
        prompt_len = len(tokenized_prompt.input_ids)
        
        # Mask prompt in labels
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        
        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        
        inputs_batch.append(np.array(input_ids, dtype=np.int64))
        labels_batch.append(np.array(labels, dtype=np.int64))
        
        # Add difficulty if present
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


def load_difficulties_from_file(difficulty_file_path, difficulty_key='diff'):
    """Load difficulty scores from JSON file"""
    print(f"Loading difficulty scores from: {difficulty_file_path}")
    try:
        with open(difficulty_file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if difficulty_key in data:
            scores = data[difficulty_key]
        elif isinstance(data, list):
            scores = data
        else:
            raise KeyError(f"Key '{difficulty_key}' not found and data is not a list.")
        
        return np.array(scores, dtype=float)
    except Exception as e:
        print(f"Error loading difficulty scores: {e}")
        traceback.print_exc()
        raise


# ============================================================================
# Evaluation and Theta Estimation (FASTER VERSION)
# ============================================================================

def evaluate_and_estimate_theta(
        peft_model_eval, validation_tokenized, validation_raw,
        tokenizer_eval, device_eval, epoch_num, theta_init,
        max_seq_len, max_gen_tokens, eval_batch_size,
        data_collator, num_obs_theta=-1, theta_eval_batches=15,
        theta_max_new_tokens=1024, theta_batch_size=64):
    """Evaluate model on validation set and estimate capacity (theta) - FASTER VERSION"""
    
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
    
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
    perplexity = np.exp(avg_val_loss) if not np.isnan(avg_val_loss) else float('inf')
    
    print(f"  E{epoch_num} Val: Avg Loss = {avg_val_loss:.4f}, Perplexity = {perplexity:.4f}")
    
    # 2. Generate answers and estimate theta (FASTER: use batches instead of samples)
    print(f"  E{epoch_num} Val: Generating answers for theta estimation...")
    print(f"  E{epoch_num} Val: Using {theta_eval_batches} batches of size {theta_batch_size} "
          f"(~{theta_eval_batches * theta_batch_size} samples)")
    
    has_difficulty = "difficulty" in validation_raw.column_names
    if not has_difficulty:
        print(f"  E{epoch_num} Val: 'difficulty' missing in validation. Theta defaults to initial.")
        return avg_val_loss, perplexity, theta_init, 0.0
    
    # Sample by batches instead of individual samples for faster evaluation
    total_samples_needed = theta_eval_batches * theta_batch_size
    if len(validation_raw) > total_samples_needed:
        print(f"  E{epoch_num} Val: Sampling ~{total_samples_needed} examples for theta estimation")
        sample_indices = np.random.choice(len(validation_raw), total_samples_needed, replace=False)
        validation_raw_sampled = validation_raw.select(sample_indices.tolist())
    else:
        print(f"  E{epoch_num} Val: Using all {len(validation_raw)} examples for theta estimation")
        validation_raw_sampled = validation_raw
    
    item_diffs_theta = []
    resp_pattern_theta = []
    
    time_theta_start = time.time()
    
    # Prepare prompts
    prompts = [create_evaluation_prompt(ex["question"]) for ex in validation_raw_sampled]
    true_answers = [ex["answer"] for ex in validation_raw_sampled]
    difficulties = validation_raw_sampled["difficulty"] if has_difficulty else [np.nan] * len(validation_raw_sampled)
    
    original_padding_side = tokenizer_eval.padding_side
    tokenizer_eval.padding_side = "left"
    
    # Process exactly theta_eval_batches batches
    num_batches_to_process = min(theta_eval_batches, (len(prompts) + theta_batch_size - 1) // theta_batch_size)
    
    for i in tqdm(range(num_batches_to_process), 
                  desc=f"  E{epoch_num} Theta Est", leave=False):
        start_idx = i * theta_batch_size
        end_idx = min(start_idx + theta_batch_size, len(prompts))
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_true_answers = true_answers[start_idx:end_idx]
        batch_difficulties = difficulties[start_idx:end_idx]
        
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
        description="Fine-tune Llama 3.1 8B on GSM8K with PUDF and QLoRA (8 epochs: 1-5 PUDF, 6-8 full data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "instruct"],
        default="base",
        help="Which model to fine-tune: 'base' or 'instruct'"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=8,
        help="Total number of epochs (1-5 PUDF, 6-8 full data)"
    )
    
    parser.add_argument(
        "--pudf-epochs",
        type=int,
        default=5,
        help="Number of PUDF curriculum epochs (remaining epochs use full data)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
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
        default=128,
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
        default=16,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
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
        default=-float('inf'),
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
    
    parser.add_argument(
        "--difficulty-file",
        type=str,
        default="../gen_diff_gsm8k/test-1pl/best_parameters.json",
        help="Path to difficulty scores JSON file"
    )
    
    parser.add_argument(
        "--difficulty-key",
        type=str,
        default="diff",
        help="Key in JSON file containing difficulty scores"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--theta-eval-batches",
        type=int,
        default=10,
        help="Number of batches to use for theta estimation (faster than sample-based)"
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
        help="Only evaluate on a subset of test data (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        args.output_dir = f"./gsm8k_pudf_llama3_1_8b_{args.model}_{timestamp}"
    
    return args


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set model ID
    if args.model == "base":
        model_id = "meta-llama/Meta-Llama-3.1-8B"
    else:
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    dataset_id = "openai/gsm8k"
    dataset_config = "main"
    max_seq_length = 640
    max_prompt_len = 512
    max_gen_tokens = 1024
    
    lora_dropout = 0.05
    weight_decay = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    best_adapter_path = os.path.join(args.output_dir, "best_pudf_adapter")
    os.makedirs(best_adapter_path, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Track total training time
    training_start_time = time.time()
    
    # Print configuration
    print(f"{'=' * 70}")
    print(f"GSM8K PUDF FINE-TUNING (8 Epochs: 1-5 PUDF → 6-8 Full Data)")
    print(f"{'=' * 70}\n")
    
    print(f"🗂️  Cache: {HF_HOME}\n")
    
    print(f"📋 Configuration:")
    print(f"   Model:                {model_id}")
    print(f"   Model type:           {args.model}")
    print(f"   Output dir:           {args.output_dir}")
    print(f"   Device:               {DEVICE}")
    print(f"   BF16 enabled:         {bf16_enabled}\n")
    
    print(f"🔧 Training Strategy:")
    print(f"   Total epochs:         {args.total_epochs}")
    print(f"   PUDF epochs (1-{args.pudf_epochs}):    Curriculum learning with difficulty filtering")
    print(f"   Full data epochs ({args.pudf_epochs+1}-{args.total_epochs}): Train on ALL 7473 samples (train+val combined)\n")
    
    print(f"🔧 PUDF Parameters:")
    print(f"   Initial theta:        {args.initial_theta}")
    print(f"   Lower offset:         {args.lower_offset}")
    print(f"   Upper offset:         {args.upper_offset}")
    print(f"   Min samples/epoch:    {args.min_samples}")
    print(f"   PUDF ordering:        {args.pudf_ordering}")
    print(f"   Theta eval batches:   {args.theta_eval_batches} (FASTER: ~{args.theta_eval_batches * args.theta_batch_size} samples)")
    print(f"   Theta batch size:     {args.theta_batch_size}")
    print(f"   Theta max tokens:     {args.theta_max_new_tokens}")
    print(f"   Theta nudge:          +1.0 if no improvement\n")
    
    print(f"🔧 Training Hyperparameters:")
    print(f"   Learning rate:        {args.lr}")
    print(f"   Train batch size:     {args.train_batch_size}")
    print(f"   Eval batch size:      {args.eval_batch_size}")
    print(f"   Grad accum steps:     {args.grad_accum_steps}")
    print(f"   Effective batch size: {args.train_batch_size * args.grad_accum_steps}")
    print(f"   LoRA rank:            {args.lora_r}")
    print(f"   LoRA alpha:           {args.lora_alpha}")
    print(f"   LoRA dropout:         {lora_dropout}")
    print(f"   Random seed:          {args.seed}\n")
    
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
    
    # Load difficulty scores
    difficulty_scores = load_difficulties_from_file(args.difficulty_file, args.difficulty_key)
    
    if len(difficulty_scores) != len(dataset['train']):
        raise ValueError(
            f"❌ CRITICAL: Mismatch! Difficulty scores ({len(difficulty_scores)}) != "
            f"train samples ({len(dataset['train'])}).\n"
            f"   Difficulty file must have exactly one score per training sample in same order!"
        )
    
    # Add difficulty to training set (preserves original order - DO NOT SHUFFLE!)
    train_with_difficulty = dataset['train'].add_column("difficulty", difficulty_scores)
    print(f"✅ Added 'difficulty' column to training set")
    print(f"   ⚠️  Dataset order preserved to maintain difficulty alignment!\n")
    
    # Create train/validation split (shuffle=False to preserve alignment!)
    print("Splitting train into 80/20 train/validation...")
    train_val_split = train_with_difficulty.train_test_split(
        test_size=0.2, seed=args.seed, shuffle=False
    )
    print(f"   ⚠️  Split without shuffling to maintain difficulty alignment!")
    train_split = train_val_split['train']
    val_split = train_val_split['test']
    
    # Create combined dataset for full data epochs
    print("Creating combined train+val dataset for epochs 6-8...")
    full_train_val = concatenate_datasets([train_split, val_split])
    print(f"   Combined dataset: {len(full_train_val)} examples\n")
    
    dataset_with_diff = DatasetDict({
        'train': train_split,
        'validation': val_split,
        'full_train_val': full_train_val,  # NEW: for epochs 6-8
        'test': dataset['test']
    })
    
    print(f"Dataset splits:")
    for split_name, split_ds in dataset_with_diff.items():
        print(f"  {split_name}: {len(split_ds)} examples, cols: {split_ds.column_names}")
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        cache_dir=HF_HOME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer ready (vocab: {len(tokenizer)})\n")
    
    # Tokenize dataset
    print(f"📝 Tokenizing dataset (max_seq_length={max_seq_length})...")
    tokenized_dataset = dataset_with_diff.map(
        lambda ex: preprocess_gsm8k_sft(ex, tokenizer, max_seq_length, max_prompt_len),
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2 if os.cpu_count() else 1),
        remove_columns=[c for c in dataset_with_diff["train"].column_names if c not in ['difficulty']],
        desc="Tokenizing"
    )
    
    print("✅ Tokenization complete")
    for split_name in ['train', 'validation', 'full_train_val', 'test']:
        split_ds = tokenized_dataset[split_name]
        print(f"   {split_name}: {len(split_ds)} examples, cols = {split_ds.column_names}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=-100,
        padding="longest"
    )
    
    # ========================================================================
    # Load Model for PUDF Training
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("MODEL SETUP")
    print(f"{'=' * 70}\n")
    
    print("⚙️  Configuring 4-bit quantization...")
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
        torch_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
        cache_dir=HF_HOME,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
    )
    
    # Sync tokenizer and model
    if len(tokenizer) > base_model.config.vocab_size:
        print(f"Resizing model embeddings: {base_model.config.vocab_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    
    if base_model.config.pad_token_id != tokenizer.pad_token_id:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare for QLoRA
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.config.use_cache = False
    
    print("⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    # Configure gradient checkpointing
    if hasattr(peft_model, 'enable_input_require_grads'):
        peft_model.enable_input_require_grads()
    if hasattr(peft_model, 'gradient_checkpointing_enable'):
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    print("\n✅ QLoRA PEFT model prepared:")
    peft_model.print_trainable_parameters()
    
    # Setup optimizer and scheduler
    optimizer = AdamW(peft_model.parameters(), lr=args.lr, weight_decay=weight_decay)
    
    num_total_train_items = len(tokenized_dataset['full_train_val'])
    steps_per_epoch_approx = (num_total_train_items * 0.75) // (args.train_batch_size * args.grad_accum_steps)
    total_training_steps_approx = int(steps_per_epoch_approx * args.total_epochs)
    if total_training_steps_approx == 0:
        total_training_steps_approx = args.total_epochs
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_training_steps_approx),
        num_training_steps=max(1, total_training_steps_approx)
    )
    
    scaler = TorchAmpGradScaler(enabled=bf16_enabled)
    
    # ========================================================================
    # TRAINING LOOP (8 Epochs: 1-5 PUDF, 6-8 Full Data)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("TRAINING LOOP")
    print(f"{'=' * 70}\n")
    
    current_capacity_theta = args.initial_theta
    best_val_loss = float('inf')
    early_stop_counter = 0
    all_epoch_stats = []
    
    full_tokenized_train = tokenized_dataset['train']
    full_tokenized_train_val = tokenized_dataset['full_train_val']
    
    for epoch in range(args.total_epochs):
        epoch_start_time = time.time()
        epoch_num = epoch + 1
        
        print(f"\n{'=' * 70}")
        if epoch_num <= args.pudf_epochs:
            print(f"EPOCH {epoch_num}/{args.total_epochs} - PUDF CURRICULUM")
            print(f"{'=' * 70}")
            print(f"  Current capacity (theta): {current_capacity_theta:.4f}")
        else:
            print(f"EPOCH {epoch_num}/{args.total_epochs} - FULL DATA TRAINING")
            print(f"{'=' * 70}")
            print(f"  Training on ALL {len(full_tokenized_train_val)} samples (train + validation)")
        
        # Evaluate and estimate theta (always use validation split)
        val_loss, val_ppl, new_theta, theta_time = evaluate_and_estimate_theta(
            peft_model,
            tokenized_dataset['validation'],
            dataset_with_diff['validation'],
            tokenizer,
            DEVICE,
            epoch_num,
            current_capacity_theta,
            max_prompt_len,
            max_gen_tokens,
            args.eval_batch_size,
            data_collator,
            num_obs_theta=-1,
            theta_eval_batches=args.theta_eval_batches,
            theta_max_new_tokens=args.theta_max_new_tokens,
            theta_batch_size=args.theta_batch_size
        )
        
        # Update capacity (only relevant for PUDF epochs)
        if epoch_num <= args.pudf_epochs:
            if not np.isnan(new_theta) and new_theta <= current_capacity_theta and epoch_num > 1:
                current_capacity_theta += 1.0
                print(f"  Theta nudge (+1.0). New capacity: {current_capacity_theta:.4f}")
            elif not np.isnan(new_theta):
                current_capacity_theta = new_theta
            else:
                print(f"  Theta est NaN. Keeping previous: {current_capacity_theta:.4f}")
            
            print(f"  E{epoch_num}: Updated capacity = {current_capacity_theta:.4f}")
        
        print(f"  E{epoch_num}: Val Loss (pre-train) = {val_loss:.4f}, PPL = {val_ppl:.4f}")
        
        # Select training data based on epoch
        if epoch_num <= args.pudf_epochs:
            # PUDF epochs: curriculum learning
            print(f"  E{epoch_num}: PUDF - Selecting samples with difficulty < {current_capacity_theta:.4f}")
            epoch_train_data = select_data_for_pudf_epoch(
                full_tokenized_train,
                current_capacity_theta,
                'difficulty',
                args.pudf_ordering,
                args.lower_offset,
                args.upper_offset,
                args.min_samples
            )
        else:
            # Full data epochs: use all training data
            print(f"  E{epoch_num}: FULL DATA - Using all {len(full_tokenized_train_val)} samples")
            epoch_train_data = full_tokenized_train_val
        
        num_selected_samples = len(epoch_train_data)
        avg_train_loss = float('nan')
        
        if num_selected_samples > 0:
            # Prepare training dataloader
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
            
            # Training loop
            peft_model.train()
            total_train_loss = 0
            grad_steps = 0
            
            for step, train_batch in enumerate(
                tqdm(train_dataloader, desc=f"  Training E{epoch_num}", leave=False)
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
                    torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    grad_steps += 1
            
            avg_train_loss = total_train_loss / grad_steps if grad_steps > 0 else float('nan')
            print(f"  E{epoch_num}: Avg train loss = {avg_train_loss:.4f}")
        else:
            print(f"  E{epoch_num}: No data selected. Skipping training.")
        
        # Post-training evaluation
        val_loss_post, val_ppl_post, theta_post, _ = evaluate_and_estimate_theta(
            peft_model,
            tokenized_dataset['validation'],
            dataset_with_diff['validation'],
            tokenizer,
            DEVICE,
            epoch_num,
            current_capacity_theta,
            max_prompt_len,
            max_gen_tokens,
            args.eval_batch_size,
            data_collator,
            num_obs_theta=-1,
            theta_eval_batches=args.theta_eval_batches,
            theta_max_new_tokens=args.theta_max_new_tokens,
            theta_batch_size=args.theta_batch_size
        )
        
        print(f"  E{epoch_num}: Val Loss (post-train) = {val_loss_post:.4f}, PPL = {val_ppl_post:.4f}")
        
        epoch_duration = time.time() - epoch_start_time
        
        # Save statistics
        all_epoch_stats.append({
            "epoch": epoch_num,
            "phase": "PUDF" if epoch_num <= args.pudf_epochs else "FULL_DATA",
            "capacity_theta": current_capacity_theta if epoch_num <= args.pudf_epochs else None,
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
            print(f"  ✨ New best val loss: {val_loss_post:.4f} (prev {best_val_loss:.4f}). Saving adapter.")
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
        
        # Early stopping
        if early_stop_counter >= args.early_stopping_patience:
            print(f"  Early stopping at E{epoch_num}.")
            break
        
        print(f"  Epoch {epoch_num} ended. Duration: {epoch_duration:.2f}s")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    print(f"\n✅ Training finished. Best val loss: {best_val_loss:.4f}")
    print(f"⏱️  Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    
    # Save training statistics
    stats_file = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_file, 'w') as f:
        json.dump({
            "total_training_time_s": total_training_time,
            "total_training_time_minutes": total_training_time / 60,
            "best_val_loss": best_val_loss,
            "epochs": all_epoch_stats
        }, f, indent=4)
    print(f"📊 Training stats saved: {stats_file}")
    
    # ========================================================================
    # FINAL TEST EVALUATION
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("FINAL TEST EVALUATION")
    print(f"{'=' * 70}\n")
    
    if os.path.exists(os.path.join(best_adapter_path, "adapter_model.safetensors")) or \
       os.path.exists(os.path.join(best_adapter_path, "adapter_model.bin")):
        
        print(f"📥 Loading best adapter from {best_adapter_path}...")
        
        # Load base model
        base_model_eval = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
            device_map="auto",
            cache_dir=HF_HOME,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        
        if len(tokenizer) > base_model_eval.config.vocab_size:
            base_model_eval.resize_token_embeddings(len(tokenizer))
        
        if base_model_eval.config.pad_token_id != tokenizer.pad_token_id:
            base_model_eval.config.pad_token_id = tokenizer.pad_token_id
        
        # Load adapter
        final_model = PeftModel.from_pretrained(base_model_eval, best_adapter_path)
        final_model.eval()
        print("✅ Model loaded for evaluation\n")
        
        # Get test dataset
        test_dataset = dataset_with_diff["test"]
        if args.test_subset:
            print(f"⚠️  Using subset of {args.test_subset} test examples")
            test_dataset = test_dataset.select(range(min(args.test_subset, len(test_dataset))))
        
        print(f"🧮 Evaluating on {len(test_dataset)} examples...")
        
        eval_start_time = time.time()
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
        
        eval_time = time.time() - eval_start_time
        accuracy = correct / total
        
        print(f"\n{'=' * 70}")
        print(f"🎉 FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Model:            {model_id}")
        print(f"Model type:       {args.model}")
        print(f"Training time:    {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
        print(f"Evaluation time:  {eval_time:.2f}s ({eval_time/60:.2f} min)")
        print(f"Total:            {total} examples")
        print(f"Correct:          {correct}")
        print(f"Incorrect:        {total - correct}")
        print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Best val loss:    {best_val_loss:.4f}")
        print(f"{'=' * 70}\n")
        
        # Show sample predictions
        print("📝 Sample predictions:")
        for i in range(min(5, len(results))):
            r = results[i]
            status = "✅" if r['correct'] else "❌"
            print(f"\n{status} Example {i + 1}:")
            print(f"   Q: {r['question']}")
            print(f"   Predicted: {r['predicted']}")
            print(f"   True: {r['true']}")
        
        # Save results
        results_file = os.path.join(args.output_dir, "test_evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "model_id": model_id,
                "model_type": args.model,
                "training_strategy": {
                    "total_epochs": args.total_epochs,
                    "pudf_epochs": args.pudf_epochs,
                    "full_data_epochs": args.total_epochs - args.pudf_epochs,
                    "description": f"Epochs 1-{args.pudf_epochs}: PUDF curriculum, Epochs {args.pudf_epochs+1}-{args.total_epochs}: Full data"
                },
                "pudf_config": {
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
                "timing": {
                    "total_training_time_s": total_training_time,
                    "total_training_time_minutes": total_training_time / 60,
                    "evaluation_time_s": eval_time,
                    "evaluation_time_minutes": eval_time / 60,
                },
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save summary
        summary_file = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"GSM8K PUDF Fine-Tuning Results\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Model type: {args.model}\n\n")
            f.write(f"Training Strategy:\n")
            f.write(f"  Total epochs: {args.total_epochs}\n")
            f.write(f"  Epochs 1-{args.pudf_epochs}: PUDF curriculum learning\n")
            f.write(f"  Epochs {args.pudf_epochs+1}-{args.total_epochs}: Full data (train + val = 7473 samples)\n\n")
            f.write(f"PUDF Configuration:\n")
            f.write(f"  Initial theta: {args.initial_theta}\n")
            f.write(f"  Lower offset: {args.lower_offset}\n")
            f.write(f"  Upper offset: {args.upper_offset}\n")
            f.write(f"  Min samples: {args.min_samples}\n")
            f.write(f"  PUDF ordering: {args.pudf_ordering}\n\n")
            f.write(f"Training Hyperparameters:\n")
            f.write(f"  Learning rate: {args.lr}\n")
            f.write(f"  LoRA rank: {args.lora_r}\n")
            f.write(f"  LoRA alpha: {args.lora_alpha}\n")
            f.write(f"  LoRA dropout: {lora_dropout}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Final Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
            f.write(f"  Correct: {correct}/{total}\n")
            f.write(f"  Best Validation Loss: {best_val_loss:.4f}\n\n")
            f.write(f"Timing:\n")
            f.write(f"  Training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)\n")
            f.write(f"  Evaluation time: {eval_time:.2f}s ({eval_time/60:.2f} min)\n")
        
        print(f"📄 Summary saved to: {summary_file}")
    
    else:
        print(f"Best adapter not found at {best_adapter_path}. Skipping final test.")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n✨ Complete! Output directory: {args.output_dir}")
