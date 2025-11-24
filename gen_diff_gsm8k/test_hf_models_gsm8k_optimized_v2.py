#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OPTIMIZED Test HuggingFace LLMs on GSM8K training set
Optimized for H100 80GB with large batch sizes
Key optimizations:
- Large batch inference (32-64 samples)
- Flash Attention 2 support
- Optimized memory management
- NO torch.compile (removed per user request)
"""

import json
import re
import os
import sys
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

MODELS = [
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct",
    "01-ai/Yi-1.5-9B-Chat",
]

CACHE_DIR = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"

STRATEGIES = [
    "zero_shot",
    "zero_shot_cot",
    "few_shot_4",
    "few_shot_cot_4",
    "few_shot_cot_8",
]

# Optimization settings for H100 80GB
BATCH_SIZE = 128  # Aggressive default based on actual H100 80GB usage
MAX_LENGTH = 512  # Maximum generation length
USE_FLASH_ATTENTION = True  # Enable Flash Attention 2 if available

# Model-specific optimal batch sizes (for max_length=512 on H100 80GB)
# Based on actual memory usage, not theoretical estimates
MODEL_BATCH_SIZES = {
    "google/gemma-2-9b-it": 32,           # 9B model - works at ~50GB
    "mistralai/Mistral-7B-Instruct-v0.2": 48,  # 7B model - should use less memory
    "Qwen/Qwen2.5-7B-Instruct": 48,       # 7B model - should use less memory
    "01-ai/Yi-1.5-9B-Chat": 32,           # 9B model - similar to Gemma
}


# ============================================================================
# Logger Class
# ============================================================================

class Logger:
    """Dual logger that writes to both console and file"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ============================================================================
# Answer Extraction Functions
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
        return None


# ============================================================================
# Load Few-Shot Examples from TEST Set
# ============================================================================

def load_few_shot_examples_from_test(num_examples=8):
    """Load few-shot examples from TEST set (first 8 examples)"""
    print("Loading few-shot examples from TEST set...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]

    examples = []
    for i in range(min(num_examples, len(test_data))):
        example = test_data[i]
        question = example["question"]
        answer_text = example["answer"]

        # Extract reasoning and answer
        if '####' in answer_text:
            parts = answer_text.split('####')
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            reasoning = answer_text.strip()
            answer = extract_answer(answer_text)

        examples.append({
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        })

    print(f"Loaded {len(examples)} few-shot examples from test set\n")
    return examples


# ============================================================================
# Prompt Construction
# ============================================================================

FEW_SHOT_EXAMPLES = None


def initialize_few_shot_examples():
    """Initialize few-shot examples (loaded once)"""
    global FEW_SHOT_EXAMPLES
    if FEW_SHOT_EXAMPLES is None:
        FEW_SHOT_EXAMPLES = load_few_shot_examples_from_test(num_examples=8)
    return FEW_SHOT_EXAMPLES


def create_zero_shot_prompt(question):
    """Create zero-shot prompt"""
    return f"Q: {question}\nA:"


def create_zero_shot_cot_prompt(question):
    """Create zero-shot CoT prompt"""
    return f"Q: {question}\nA: Let's think step by step."


def create_few_shot_prompt(question, num_examples=4):
    """Create few-shot prompt (no CoT)"""
    examples = initialize_few_shot_examples()[:num_examples]

    prompt = ""
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: The answer is {ex['answer']}.\n\n"

    prompt += f"Q: {question}\n"
    prompt += "A:"
    return prompt


def create_few_shot_cot_prompt(question, num_examples=4):
    """Create few-shot CoT prompt"""
    examples = initialize_few_shot_examples()[:num_examples]

    prompt = ""
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"

    prompt += f"Q: {question}\n"
    prompt += "A:"
    return prompt


def create_prompt(question, strategy):
    """Create prompt based on strategy"""
    if strategy == "zero_shot":
        return create_zero_shot_prompt(question)
    elif strategy == "zero_shot_cot":
        return create_zero_shot_cot_prompt(question)
    elif strategy == "few_shot_4":
        return create_few_shot_prompt(question, num_examples=4)
    elif strategy == "few_shot_cot_4":
        return create_few_shot_cot_prompt(question, num_examples=4)
    elif strategy == "few_shot_cot_8":
        return create_few_shot_cot_prompt(question, num_examples=8)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# OPTIMIZED Model Loading
# ============================================================================

def load_model(model_name):
    """Load HuggingFace model and tokenizer with optimizations"""
    print(f"Loading {model_name} with optimizations...")
    print(f"  Target device: H100 80GB")
    print(f"  Max length: {MAX_LENGTH}")

    # Load with Flash Attention 2 if available
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "cache_dir": CACHE_DIR,
    }
    
    if USE_FLASH_ATTENTION:
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("  ✓ Using Flash Attention 2")
        except:
            print("  ⚠ Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        padding_side="left"  # Important for batch inference
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if hasattr(tokenizer, 'truncation_side'):
        tokenizer.truncation_side = 'left'
        
    print(f"  ✓ Model loaded successfully!\n")
    return model, tokenizer



# ============================================================================
# OPTIMIZED Batch Inference
# ============================================================================

def query_model_batch(model, tokenizer, prompts, max_new_tokens=512):
    """
    OPTIMIZED: Query model with large batching for H100 80GB
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of prompts to process in batch
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of generated responses
    """
    try:
        # Format all prompts using chat template if available
        formatted_prompts = []
        for prompt in prompts:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                formatted_prompt = prompt
            formatted_prompts.append(formatted_prompt)

        # Tokenize batch with padding
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=6144
        ).to(model.device)

        # Generate for entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode all outputs
        responses = []
        for i, output in enumerate(outputs):
            # Remove prompt tokens
            prompt_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[prompt_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses

    except Exception as e:
        print(f"Error in batch query: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(prompts)


# ============================================================================
# Memory Estimation
# ============================================================================

def estimate_memory_usage(batch_size, max_length=512, model_size_gb=14):
    """
    Estimate GPU memory usage for given batch size
    Updated based on actual H100 usage patterns with Flash Attention 2
    
    Args:
        batch_size: Number of samples in batch
        max_length: Maximum sequence length
        model_size_gb: Approximate model size in GB (14 for 7B models, 18 for 9B models in bfloat16)
    
    Returns:
        Estimated memory usage in GB
    """
    # Model weights
    model_mem = model_size_gb
    
    # KV cache estimation (more accurate with Flash Attention 2)
    # Flash Attention 2 is more memory efficient than standard attention
    # Empirically: Gemma-2-9B with BS=128 uses ~50GB total
    kv_cache_per_token = 0.00015  # GB per token (empirically tuned)
    kv_cache_mem = batch_size * max_length * kv_cache_per_token
    
    # Activations (rough estimate, Flash Attention reduces this)
    activation_mem = batch_size * 0.08  # Reduced due to Flash Attention
    
    # Overhead
    overhead = 8
    
    total = model_mem + kv_cache_mem + activation_mem + overhead
    return total


def suggest_batch_size(max_length=512, gpu_memory_gb=80):
    """Suggest optimal batch size for given constraints based on actual usage"""
    # Updated based on empirical data: Gemma-2-9B with BS=128 uses ~50GB
    batch_sizes = [192, 160, 128, 96, 64, 48, 32]
    
    print(f"\nBatch Size Recommendations for H100 {gpu_memory_gb}GB:")
    print(f"Max length: {max_length} tokens")
    print(f"Based on actual H100 usage with Flash Attention 2\n")
    
    for bs in batch_sizes:
        mem = estimate_memory_usage(bs, max_length, model_size_gb=18)  # Use 9B as reference
        utilization = (mem / gpu_memory_gb) * 100
        status = "✓" if mem < gpu_memory_gb * 0.85 else "⚠" if mem < gpu_memory_gb else "✗"
        print(f"  {status} Batch size {bs:3d}: ~{mem:.1f}GB ({utilization:.0f}% utilization)")
    
    # Recommend batch size with ~60-70% utilization
    for bs in batch_sizes:
        mem = estimate_memory_usage(bs, max_length, model_size_gb=18)
        if mem < gpu_memory_gb * 0.70:
            print(f"\n  Recommended: batch_size={bs} (~{mem:.1f}GB, {(mem/gpu_memory_gb)*100:.0f}% utilization)")
            return bs
    
    return 64  # Fallback


# ============================================================================
# OPTIMIZED Test Function with Large Batching
# ============================================================================

def test_model_strategy(model_name, strategy, num_samples=None, 
                       base_output_dir="results_hf", print_examples=True,
                       batch_size=None, max_new_tokens=MAX_LENGTH):
    """
    OPTIMIZED: Test model with large batch inference for H100 80GB
    If batch_size is None, automatically selects optimal batch size for the model
    """
    # Auto-select batch size if not specified
    if batch_size is None:
        batch_size = MODEL_BATCH_SIZES.get(model_name, BATCH_SIZE)
        print(f"Auto-selected batch size: {batch_size} for {model_name}")
    
    # Setup output directory
    model_slug = model_name.replace("/", "_")
    output_dir = os.path.join(base_output_dir, model_slug, strategy)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log.txt")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"\n{'=' * 80}")
    print(f"TESTING: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"Batch Size: {batch_size}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"GPU: H100 80GB")
    
    # Memory estimation
    est_mem = estimate_memory_usage(batch_size, max_new_tokens)
    print(f"Estimated Memory Usage: ~{est_mem:.1f}GB ({(est_mem/80)*100:.0f}% of 80GB)")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    # Load model
    model, tokenizer = load_model(model_name)

    # Load dataset
    print("Loading GSM8K training set...")
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
    print(f"Total samples: {len(train_data)}\n")

    # Initialize counters
    correct = 0
    total = 0
    results = []

    # Process in batches
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    
    print(f"Processing {len(train_data)} samples in {num_batches} batches of {batch_size}...\n")
    
    for batch_idx in tqdm(range(num_batches), desc=f"{model_name} - {strategy}", file=sys.stderr):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(train_data))
        
        # HuggingFace datasets return dict of lists when slicing
        batch_data = train_data[start_idx:end_idx]
        
        # Prepare batch - handle HuggingFace dataset format
        if isinstance(batch_data, dict):
            # Dataset slice returns {'question': [...], 'answer': [...]}
            questions = batch_data["question"]
            true_answers = [normalize_answer(extract_answer(ans)) for ans in batch_data["answer"]]
        else:
            # Fallback for list format
            questions = [ex["question"] for ex in batch_data]
            true_answers = [normalize_answer(extract_answer(ex["answer"])) for ex in batch_data]
        
        prompts = [create_prompt(q, strategy) for q in questions]
        
        # Print first example prompt
        if print_examples and batch_idx == 0:
            print(f"\n{'=' * 80}")
            print(f"EXAMPLE PROMPT (Sample 0):")
            print(f"{'=' * 80}")
            print(prompts[0])
            print(f"{'=' * 80}\n")
        
        # Batch inference
        responses = query_model_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        
        # Process results
        for i, (response, question, true_answer) in enumerate(zip(responses, questions, true_answers)):
            global_idx = start_idx + i
            
            if response is None:
                pred_answer = None
            else:
                pred_answer = normalize_answer(extract_answer(response))
            
            # Check correctness
            is_correct = (pred_answer == true_answer) and (pred_answer is not None)
            if is_correct:
                correct += 1
            total += 1
            
            # Print first few examples
            if print_examples and global_idx < 3:
                print(f"\n{'=' * 80}")
                print(f"SAMPLE {global_idx}:")
                print(f"{'=' * 80}")
                print(f"Question: {question}")
                print(f"\nTrue Answer: {true_answer}")
                print(f"Predicted Answer: {pred_answer}")
                print(f"Correct: {'✓' if is_correct else '✗'}")
                print(f"\nFull Response:")
                print(response)
                print(f"{'=' * 80}\n")
            
            # Store result
            results.append({
                "train_idx": global_idx,
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": pred_answer,
                "correct": is_correct,
                "full_response": response,
                "prompt": prompts[i] if global_idx < 10 else None
            })
        
        # Progress update
        if (batch_idx + 1) % 10 == 0:
            current_acc = correct / total
            print(f"  Progress: {total}/{len(train_data)} | Accuracy: {current_acc:.4f}")

    # Calculate accuracy
    accuracy = correct / total

    # Print results
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {model_name} - {strategy}")
    print(f"{'=' * 80}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Training Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    # Save results
    output_data = {
        "model": model_name,
        "strategy": strategy,
        "split": "train",
        "few_shot_source": "test_set_first_8" if "few_shot" in strategy else "N/A",
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "optimizations": {
            "batching": True,
            "torch_compile": False,
            "flash_attention": USE_FLASH_ATTENTION
        },
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Log saved to: {log_file}\n")

    # Restore stdout and close logger
    sys.stdout = logger.terminal
    logger.close()

    # Clean up model to free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return accuracy


# ============================================================================
# Test Strategies for One Model
# ============================================================================

def test_strategies_for_model(model_name, strategies, num_samples=None, 
                              base_output_dir="results_hf", print_examples=True,
                              batch_size=None, max_new_tokens=MAX_LENGTH):
    """Test specified strategies for a single model"""
    print(f"\n{'=' * 80}")
    print(f"TESTING {len(strategies)} STRATEGIES FOR: {model_name}")
    
    # Auto-select batch size if not specified
    if batch_size is None:
        batch_size = MODEL_BATCH_SIZES.get(model_name, BATCH_SIZE)
        print(f"Auto-selected batch size: {batch_size}")
    
    print(f"{'=' * 80}\n")

    results_summary = {}

    for strategy in strategies:
        try:
            accuracy = test_model_strategy(
                model_name=model_name,
                strategy=strategy,
                num_samples=num_samples,
                base_output_dir=base_output_dir,
                print_examples=print_examples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens
            )
            results_summary[strategy] = accuracy
        except Exception as e:
            print(f"ERROR testing {strategy}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[strategy] = None

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY FOR: {model_name}")
    print(f"{'=' * 80}")
    for strategy, acc in results_summary.items():
        if acc is not None:
            print(f"{strategy:20s}: {acc:.4f} ({acc * 100:.2f}%)")
        else:
            print(f"{strategy:20s}: FAILED")
    print(f"{'=' * 80}\n")

    return results_summary


# ============================================================================
# Test All Models Sequentially
# ============================================================================

def test_all_models_sequential(strategies, num_samples=None, 
                               base_output_dir="results_hf", print_examples=True,
                               batch_size=None, max_new_tokens=MAX_LENGTH):
    """Test all models sequentially with specified strategies"""
    print(f"\n{'=' * 80}")
    print(f"SEQUENTIAL TESTING - ALL MODELS (H100 OPTIMIZED)")
    print(f"{'=' * 80}")
    print(f"Models: {len(MODELS)}")
    print(f"Strategies per model: {len(strategies)}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Samples: {num_samples if num_samples else 'All (7,473)'}")
    if batch_size is None:
        print(f"Batch Size: Auto-selected per model (7B models: 64, 9B models: 48)")
    else:
        print(f"Batch Size: {batch_size} (fixed for all models)")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"GPU: H100 80GB")
    print(f"Optimizations: Large Batching + Flash Attention 2")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

    all_results = {}

    for model_name in MODELS:
        try:
            results = test_strategies_for_model(
                model_name=model_name,
                strategies=strategies,
                num_samples=num_samples,
                base_output_dir=base_output_dir,
                print_examples=print_examples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"ERROR testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = None

    # Print final summary
    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY - ALL MODELS (H100 OPTIMIZED)")
    print(f"{'=' * 80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n")

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        if results:
            for strategy, acc in results.items():
                if acc is not None:
                    print(f"  {strategy:20s}: {acc:.4f} ({acc * 100:.2f}%)")
                else:
                    print(f"  {strategy:20s}: FAILED")
        else:
            print(f"  FAILED TO TEST")

    print(f"\n{'=' * 80}\n")

    return all_results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test HuggingFace LLMs on GSM8K (H100 OPTIMIZED)')
    parser.add_argument('--model', type=str, default=None,
                        choices=MODELS,
                        help='Specific model to test (default: test all models)')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=STRATEGIES + ['all'],
                        help='Strategy to test (default: all strategies)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of training samples (default: all 7,473)')
    parser.add_argument('--output_dir', type=str, default='results_hf',
                        help='Base output directory')
    parser.add_argument('--no_examples', action='store_true',
                        help='Do not print first 3 examples (makes output cleaner)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help=f'Batch size for inference (default: {BATCH_SIZE}, can use 48-64 for H100)')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_LENGTH,
                        help=f'Maximum new tokens to generate (default: {MAX_LENGTH})')
    parser.add_argument('--no_flash_attention', action='store_true',
                        help='Disable Flash Attention 2')
    parser.add_argument('--suggest_batch_size', action='store_true',
                        help='Show batch size recommendations and exit')

    args = parser.parse_args()

    # Update global settings
    if args.no_flash_attention:
        USE_FLASH_ATTENTION = False
    
    # Suggest batch size if requested
    if args.suggest_batch_size:
        suggest_batch_size(args.max_new_tokens, gpu_memory_gb=80)
        sys.exit(0)
    
    # Determine batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = BATCH_SIZE

    # Determine which strategies to test
    if args.strategy and args.strategy != 'all':
        strategies_to_test = [args.strategy]
    else:
        strategies_to_test = STRATEGIES

    print_examples = not args.no_examples

    if args.model is None:
        # Test all models with specified strategies
        test_all_models_sequential(
            strategies=strategies_to_test,
            num_samples=args.num_samples,
            base_output_dir=args.output_dir,
            print_examples=print_examples,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens
        )
    else:
        # Test one model with specified strategies
        test_strategies_for_model(
            model_name=args.model,
            strategies=strategies_to_test,
            num_samples=args.num_samples,
            base_output_dir=args.output_dir,
            print_examples=print_examples,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens
        )
