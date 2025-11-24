#!/usr/bin/env python3
"""
Generate py-irt format .jsonlines file from GSM8K evaluation results
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def shorten_model_name(model_name):
    """Shorten model names to more readable versions"""
    name_mapping = {
        # results/ folder models
        'anthropic_claude-3.5-sonnet': 'claude_3_5',
        'meta_meta-llama-3.1-405b-instruct': 'llama_3_1_405b',
        'meta_meta-llama-3-8b-instruct': 'llama_3_8b',
        'openai_gpt-5': 'gpt_5',
        'openai_gpt-4o-mini': 'gpt_4o_mini',
        'deepseek-ai_deepseek-v3': 'deepseek_v3',
        'ibm-granite_granite-3.3-8b-instruct': 'granite_3_3_8b',
        # results_hf/ folder models
        '01-ai_Yi-1.5-9B-Chat': 'yi_1_5_9b',
        'google_gemma-2-9b-it': 'gemma_2_9b',
        'mistralai_Mistral-7B-Instruct-v0.2': 'mistral_7b',
        'Qwen_Qwen2.5-7B-Instruct': 'qwen_2_5_7b'
    }
    # Replace all remaining hyphens with underscores
    return name_mapping.get(model_name, model_name.replace('-', '_'))

def shorten_strategy_name(strategy):
    """Shorten strategy names"""
    strategy_mapping = {
        'few_shot_4': '4_shot',
        'few_shot_cot_4': '4_shot_cot',
        'few_shot_cot_8': '8_shot_cot',
        'zero_shot': '0_shot',
        'zero_shot_cot': '0_shot_cot'
    }
    return strategy_mapping.get(strategy, strategy.replace('-', '_'))

def process_results():
    """Process all results.json files and generate IRT-format data"""
    
    all_data = []
    accuracy_table = []
    
    # Process both result folders
    base_dirs = ['results', 'results_hf']
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} directory not found, skipping...")
            continue
            
        # Iterate through model directories
        for model_dir in sorted(os.listdir(base_dir)):
            model_path = os.path.join(base_dir, model_dir)
            
            if not os.path.isdir(model_path):
                continue
            
            # Iterate through strategy directories
            for strategy_dir in sorted(os.listdir(model_path)):
                strategy_path = os.path.join(model_path, strategy_dir)
                
                if not os.path.isdir(strategy_path):
                    continue
                
                results_file = os.path.join(strategy_path, 'results.json')
                
                if not os.path.exists(results_file):
                    print(f"Warning: {results_file} not found, skipping...")
                    continue
                
                # Read results.json
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
                    continue
                
                # Create subject_id
                short_model = shorten_model_name(model_dir)
                short_strategy = shorten_strategy_name(strategy_dir)
                subject_id = f"{short_model}_{short_strategy}"
                
                # Extract responses
                responses = {}
                for result in data['results']:
                    q_id = f"q{result['train_idx']}"
                    responses[q_id] = 1 if result['correct'] else 0
                
                # Add to data list
                all_data.append({
                    'subject_id': subject_id,
                    'responses': responses
                })
                
                # Add to accuracy table
                accuracy_table.append({
                    'model': short_model,
                    'strategy': short_strategy,
                    'subject_id': subject_id,
                    'accuracy': data['accuracy'],
                    'correct': data['correct'],
                    'total': data['total']
                })
                
                print(f"✓ Processed: {subject_id} ({data['correct']}/{data['total']} = {data['accuracy']:.4f})")
    
    return all_data, accuracy_table

def print_accuracy_table(accuracy_table):
    """Print accuracy table in a readable format"""
    print("\n" + "="*110)
    print("ACCURACY TABLE")
    print("="*110)
    print(f"{'Model':<20} {'Strategy':<15} {'Subject ID':<40} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-"*110)
    
    # Sort by accuracy (descending)
    for entry in sorted(accuracy_table, key=lambda x: x['accuracy'], reverse=True):
        print(f"{entry['model']:<20} {entry['strategy']:<15} {entry['subject_id']:<40} "
              f"{entry['accuracy']:<12.4f} {entry['correct']:>5}/{entry['total']:<5}")
    
    print("="*110)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total subjects: {len(accuracy_table)}")
    if accuracy_table:
        accuracies = [e['accuracy'] for e in accuracy_table]
        print(f"  Mean accuracy: {sum(accuracies)/len(accuracies):.4f}")
        print(f"  Max accuracy:  {max(accuracies):.4f}")
        print(f"  Min accuracy:  {min(accuracies):.4f}")
    print()

def save_jsonlines(data, output_file='irt_data.jsonlines'):
    """Save data to .jsonlines file"""
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"✓ Saved {len(data)} entries to {output_file}")

def main():
    print("="*110)
    print("Processing GSM8K Results for IRT Analysis")
    print("="*110)
    print()
    
    # Process all results
    all_data, accuracy_table = process_results()
    
    if not all_data:
        print("\nError: No data found! Please check your directory structure.")
        return
    
    # Print accuracy table
    print_accuracy_table(accuracy_table)
    
    # Save to .jsonlines
    save_jsonlines(all_data)
    
    # Print additional info
    if all_data:
        n_questions = len(all_data[0]['responses'])
        print(f"\nDataset Information:")
        print(f"  Total subjects: {len(all_data)}")
        print(f"  Total questions: {n_questions}")
        print(f"  Question IDs: q0 to q{n_questions-1}")
        
        # Show example entry
        print(f"\nExample entry:")
        print(f"  Subject: {all_data[0]['subject_id']}")
        print(f"  First 5 responses: {dict(list(all_data[0]['responses'].items())[:5])}")

if __name__ == "__main__":
    main()
