#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Accelerated version with async/concurrent API calls
"""

import replicate
import json
import re
import time
import os
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# ============================================================================
# Configuration
# ============================================================================

MODEL = "meta/meta-llama-3.1-405b-instruct"
STRATEGY = "zero_shot"
BASE_OUTPUT_DIR = "results"
MAX_WORKERS = 15  # Number of concurrent requests (you have ~3000/window based on API check)

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
# Prompt Construction
# ============================================================================

def create_zero_shot_prompt(question):
    """Strategy: zero_shot"""
    prompt = f"Q: {question}\nA:"
    return prompt


# ============================================================================
# Model Query (Thread-safe)
# ============================================================================

def query_model(model_name, prompt, max_tokens=512, temperature=0.0, max_retries=3):
    """Query Replicate model (thread-safe with retry logic)"""
    for attempt in range(max_retries):
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
            error_msg = str(e)
            # Check if it's a rate limit error
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\n⚠️  Rate limit hit, waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print(f"\n❌ Max retries reached: {e}")
                    return None
            else:
                print(f"\n❌ Error querying model: {e}")
                return None
    return None


# ============================================================================
# Process Single Sample
# ============================================================================

def process_sample(idx, example):
    """Process a single sample"""
    question = example["question"]
    answer_text = example["answer"]
    
    # Extract true answer
    true_answer = extract_answer(answer_text)
    true_answer = normalize_answer(true_answer)
    
    # Create prompt
    prompt = create_zero_shot_prompt(question)
    
    # Query model
    response = query_model(MODEL, prompt)
    
    # Extract predicted answer
    pred_answer = extract_answer(response) if response else None
    pred_answer = normalize_answer(pred_answer)
    
    # Check correctness
    is_correct = (pred_answer == true_answer) if pred_answer and true_answer else False
    
    return {
        "train_idx": idx,
        "question": question,
        "true_answer": true_answer,
        "predicted_answer": pred_answer,
        "correct": is_correct,
        "full_response": response,
        "prompt": prompt
    }


# ============================================================================
# Main Testing Function with Parallel Execution
# ============================================================================

def test_model_parallel():
    """Test the model with parallel execution"""
    
    print(f"\n{'='*80}")
    print(f"ACCELERATED EXECUTION")
    print(f"{'='*80}")
    print(f"Model: {MODEL}")
    print(f"Strategy: {STRATEGY}")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"Output: {BASE_OUTPUT_DIR}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Load GSM8K training set
    print("Loading GSM8K training set...")
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    print(f"Loaded {len(train_data)} training examples\n")
    
    # Process samples in parallel
    results = [None] * len(train_data)
    correct = 0
    total = 0
    lock = threading.Lock()
    
    def process_and_update(idx, example):
        nonlocal correct, total
        result = process_sample(idx, example)
        
        with lock:
            results[idx] = result
            if result["correct"]:
                correct += 1
            total += 1
            
            # Progress update every 50 samples
            if total % 50 == 0:
                current_acc = correct / total
                elapsed = time.time() - start_time
                rate = total / elapsed
                remaining = (len(train_data) - total) / rate if rate > 0 else 0
                print(f"Progress: {total}/{len(train_data)} | "
                      f"Accuracy: {current_acc:.4f} ({current_acc*100:.2f}%) | "
                      f"Rate: {rate:.2f} samples/sec | "
                      f"ETA: {remaining/60:.1f} min")
    
    print(f"Starting parallel evaluation with {MAX_WORKERS} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = []
        for idx, example in enumerate(train_data):
            future = executor.submit(process_and_update, idx, example)
            futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result()
    
    elapsed_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Model: {MODEL}")
    print(f"Strategy: {STRATEGY}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Training Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total time: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"Average rate: {total/elapsed_time:.2f} samples/sec")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Save results
    model_slug = MODEL.replace('/', '_')
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_slug, STRATEGY)
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "model": MODEL,
        "strategy": STRATEGY,
        "split": "train",
        "few_shot_source": "N/A",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "execution_time_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}\n")
    
    return accuracy


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    test_model_parallel()
