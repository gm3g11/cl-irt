#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test 8 models on GSM8K training set with 5 different prompting strategies
Strategies: zero_shot, zero_shot_cot, few_shot_4, few_shot_cot_4, few_shot_cot_8
"""

import replicate
import json
import re
import time
import os
from datasets import load_dataset
from tqdm import tqdm
import subprocess
import argparse
from datetime import datetime

# ============================================================================
# 8 Models (6 from script 1 + 2 from script 2)
# ============================================================================

MODELS = [
    # From Script 1
    "meta/meta-llama-3-8b-instruct",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-5",
    "meta/meta-llama-3.1-405b-instruct",
    "deepseek-ai/deepseek-r1",
    # From Script 2
    "ibm-granite/granite-3.3-8b-instruct",
    "deepseek-ai/deepseek-v3",
]

# Expected leaderboard results (8-shot CoT on TEST set)
LEADERBOARD_REFERENCE = {
    "meta/meta-llama-3-8b-instruct": 79.6,
    "openai/gpt-4o-mini": 85.0,
    "anthropic/claude-3.5-sonnet": 92.0,
    "openai/gpt-5": 95.0,
    "meta/meta-llama-3.1-405b-instruct": 93.0,
    "deepseek-ai/deepseek-r1": 90.0,
    "ibm-granite/granite-3.3-8b-instruct": 75.0,
    "deepseek-ai/deepseek-v3": 88.0,
}

# Supported strategies
STRATEGIES = [
    "zero_shot",
    "zero_shot_cot",
    "few_shot_4",
    "few_shot_cot_4",
    "few_shot_cot_8",
]

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
# Prompt Construction Functions for Each Strategy
# ============================================================================

FEW_SHOT_EXAMPLES = None

def initialize_few_shot_examples():
    """Initialize few-shot examples (loaded once)"""
    global FEW_SHOT_EXAMPLES
    if FEW_SHOT_EXAMPLES is None:
        FEW_SHOT_EXAMPLES = load_few_shot_examples_from_test(num_examples=8)
    return FEW_SHOT_EXAMPLES


def create_zero_shot_prompt(question):
    """
    Strategy: zero_shot
    Just the question with "Q:" and "A:" format
    """
    prompt = f"Q: {question}\nA:"
    return prompt


def create_zero_shot_cot_prompt(question):
    """
    Strategy: zero_shot_cot
    Question + "Let's think step by step" to encourage reasoning
    """
    prompt = f"Q: {question}\nA: Let's think step by step."
    return prompt


def create_few_shot_4_prompt(question):
    """
    Strategy: few_shot_4
    4 examples with just question and final answer (no reasoning)
    """
    examples = initialize_few_shot_examples()[:4]
    
    prompt = ""
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: The answer is {ex['answer']}.\n\n"
    
    prompt += f"Q: {question}\n"
    prompt += "A:"
    return prompt


def create_few_shot_cot_4_prompt(question):
    """
    Strategy: few_shot_cot_4
    4 examples with full chain-of-thought reasoning
    """
    examples = initialize_few_shot_examples()[:4]
    
    prompt = ""
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"
    
    prompt += f"Q: {question}\n"
    prompt += "A:"
    return prompt


def create_few_shot_cot_8_prompt(question):
    """
    Strategy: few_shot_cot_8
    8 examples with full chain-of-thought reasoning
    """
    examples = initialize_few_shot_examples()[:8]
    
    prompt = ""
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"
    
    prompt += f"Q: {question}\n"
    prompt += "A:"
    return prompt


def create_prompt(question, strategy):
    """
    Create prompt based on strategy
    """
    if strategy == "zero_shot":
        return create_zero_shot_prompt(question)
    elif strategy == "zero_shot_cot":
        return create_zero_shot_cot_prompt(question)
    elif strategy == "few_shot_4":
        return create_few_shot_4_prompt(question)
    elif strategy == "few_shot_cot_4":
        return create_few_shot_cot_4_prompt(question)
    elif strategy == "few_shot_cot_8":
        return create_few_shot_cot_8_prompt(question)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# Model Query
# ============================================================================

def query_model(model_name, prompt, max_tokens=512, temperature=0.0):
    """Query Replicate model"""
    try:
        output = replicate.run(
            model_name,
            input={
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            }
        )
        result = "".join(output)
        return result
    except Exception as e:
        print(f"Error querying model: {e}")
        return None


# ============================================================================
# Main Testing Function
# ============================================================================

def test_single_model(
    model_name,
    strategy="few_shot_cot_8",
    num_samples=None,
    base_output_dir="results"
):
    """
    Test a single model on GSM8K training set with specified strategy
    
    Output structure: results/{model_name}/{strategy_name}/
    - results.json (evaluation results)
    - log.txt (execution log)
    """
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"{'='*80}\n")
    
    # Validate strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {STRATEGIES}")
    
    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    
    if num_samples:
        train_data = train_data.select(range(min(num_samples, len(train_data))))
    
    print(f"Testing on {len(train_data)} training samples\n")
    
    # Initialize few-shot examples if needed
    if strategy in ["few_shot_4", "few_shot_cot_4", "few_shot_cot_8"]:
        initialize_few_shot_examples()
    
    # Results storage
    results = []
    correct = 0
    total = 0
    
    # Test each sample
    for idx, example in enumerate(tqdm(train_data, desc=f"Testing {model_name}")):
        question = example["question"]
        true_answer = normalize_answer(extract_answer(example["answer"]))
        
        # Create prompt based on strategy
        prompt = create_prompt(question, strategy)
        
        # Query model
        response = query_model(model_name, prompt)
        
        if response is None:
            pred_answer = None
        else:
            pred_answer = normalize_answer(extract_answer(response))
        
        # Check correctness
        is_correct = (pred_answer == true_answer) and (pred_answer is not None)
        if is_correct:
            correct += 1
        total += 1
        
        # Store result
        results.append({
            "train_idx": idx,
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": pred_answer,
            "correct": is_correct,
            "full_response": response
        })
        
        # Progress update
        if (idx + 1) % 50 == 0:
            current_acc = correct / total
            print(f"  Progress: {idx + 1}/{len(train_data)} | Accuracy: {current_acc:.4f}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Training Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compare to leaderboard if available (only for few_shot_cot_8)
    if strategy == "few_shot_cot_8" and model_name in LEADERBOARD_REFERENCE:
        test_acc = LEADERBOARD_REFERENCE[model_name]
        expected_train = test_acc + 7  # Expect ~7% higher on training
        diff = (accuracy * 100) - expected_train
        print(f"Test Leaderboard: {test_acc:.2f}%")
        print(f"Expected Train: ~{expected_train:.2f}%")
        print(f"Difference: {diff:+.2f}% {'✓' if abs(diff) < 5 else '⚠️'}")
    
    print(f"{'='*80}\n")
    
    # Save results
    # Create directory structure: results/{model_name}/{strategy_name}/
    model_slug = model_name.replace('/', '_')
    output_dir = os.path.join(base_output_dir, model_slug, strategy)
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "model": model_name,
        "strategy": strategy,
        "split": "train",
        "few_shot_source": "test_set_first_8" if "few_shot" in strategy else "N/A",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}\n")
    
    return accuracy


# ============================================================================
# Parallel Execution
# ============================================================================

def run_all_models_parallel(
    strategies=None,
    num_samples=None,
    base_output_dir="results"
):
    """
    Launch parallel processes for all models and strategies
    Logs are saved to results/{model_name}/{strategy}/log.txt
    
    Args:
        strategies: List of strategies to test (default: all 5 strategies)
        num_samples: Number of samples to test (default: all 7,473)
        base_output_dir: Base directory for results
    """
    
    if strategies is None:
        strategies = STRATEGIES
    
    # Validate strategies
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of {STRATEGIES}")
    
    print(f"\n{'='*80}")
    print(f"PARALLEL EXECUTION - 8 MODELS x {len(strategies)} STRATEGIES")
    print(f"{'='*80}")
    print(f"Models: {len(MODELS)}")
    print(f"Strategies: {strategies}")
    print(f"Samples: {num_samples if num_samples else 'All (7,473)'}")
    print(f"Total jobs: {len(MODELS) * len(strategies)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Launch all processes
    processes = []
    log_files = []
    
    for model in MODELS:
        for strategy in strategies:
            cmd = [
                "python", __file__,
                "--mode", "single",
                "--model", model,
                "--strategy", strategy,
                "--output_dir", base_output_dir,
            ]
            
            if num_samples:
                cmd.extend(["--num_samples", str(num_samples)])
            
            # Create log file in same directory as results
            model_slug = model.replace('/', '_')
            log_dir = os.path.join(base_output_dir, model_slug, strategy)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "log.txt")
            
            log_file = open(log_path, "w")
            log_files.append(log_file)
            
            print(f"Launching: {model} | {strategy}")
            print(f"  Log: {log_path}")
            
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True
            )
            processes.append({
                "model": model,
                "strategy": strategy,
                "process": process,
                "log_file": log_file,
                "log_path": log_path
            })
    
    print(f"\nAll {len(processes)} processes launched!")
    print(f"Logs saved in results/{'{model}'}/{'{strategy}'}/log.txt")
    print(f"Waiting for completion...\n")
    
    # Wait for all to complete
    summary = []
    for p_info in processes:
        model = p_info["model"]
        strategy = p_info["strategy"]
        process = p_info["process"]
        
        print(f"Waiting for {model} | {strategy}...")
        return_code = process.wait()
        
        success = return_code == 0
        
        if success:
            print(f"✓ {model} | {strategy} completed")
        else:
            print(f"✗ {model} | {strategy} failed (check {p_info['log_path']})")
        
        summary.append({
            "model": model,
            "strategy": strategy,
            "success": success,
            "return_code": return_code,
            "log_path": p_info["log_path"]
        })
    
    # Close log files
    for log_file in log_files:
        log_file.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PARALLEL EXECUTION COMPLETED")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = sum(1 for s in summary if s["success"])
    print(f"Successful: {successful}/{len(summary)}")
    
    # Group by model
    print(f"\nResults by model:")
    for model in MODELS:
        model_results = [s for s in summary if s["model"] == model]
        model_success = sum(1 for s in model_results if s["success"])
        status = "✓" if model_success == len(model_results) else "⚠️"
        print(f"  {status} {model}: {model_success}/{len(model_results)}")
        for s in model_results:
            substatus = "✓" if s["success"] else "✗"
            print(f"     {substatus} {s['strategy']}")
    
    print(f"{'='*80}\n")
    
    return summary


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test 8 LLMs on GSM8K with 5 prompting strategies'
    )
    parser.add_argument('--mode', type=str, default='parallel',
                        choices=['parallel', 'single'],
                        help='Run mode: parallel (all models/strategies) or single')
    parser.add_argument('--model', type=str, default=None,
                        help='Single model to test (for single mode)')
    parser.add_argument('--strategy', type=str, default='few_shot_cot_8',
                        choices=STRATEGIES,
                        help='Prompting strategy to use')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                        choices=STRATEGIES,
                        help='Multiple strategies for parallel mode (default: all 5)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of training samples (None = all 7,473)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Base output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'parallel':
        # Run all models in parallel with specified strategies
        run_all_models_parallel(
            strategies=args.strategies,
            num_samples=args.num_samples,
            base_output_dir=args.output_dir
        )
    else:
        # Run single model (for manual control or called by parallel)
        if args.model is None:
            print("Error: --model required for single mode")
            exit(1)
        
        test_single_model(
            model_name=args.model,
            strategy=args.strategy,
            num_samples=args.num_samples,
            base_output_dir=args.output_dir
        )
