#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract and display IRT results after manual py-irt training
Use this if you ran py-irt commands manually
"""

import json
import pandas as pd
import sys
import os

def load_and_display_results(output_dir, model_type="2pl"):
    """Load and display IRT results from output directory"""
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} not found!")
        print(f"Make sure you've run: py-irt train {model_type} irt_data.jsonlines {output_dir}")
        return
    
    # Check for required files
    item_file = os.path.join(output_dir, "item.json")
    subject_file = os.path.join(output_dir, "subject.json")
    
    if not os.path.exists(item_file) or not os.path.exists(subject_file):
        print(f"Error: Missing item.json or subject.json in {output_dir}")
        return
    
    print("="*80)
    print(f"LOADING {model_type.upper()} MODEL RESULTS")
    print(f"Directory: {output_dir}")
    print("="*80)
    
    # Load item parameters
    with open(item_file, 'r') as f:
        items = json.load(f)
    
    # Load subject parameters
    with open(subject_file, 'r') as f:
        subjects = json.load(f)
    
    print(f"\n✓ Loaded {len(items)} questions and {len(subjects)} models")
    
    # ========================================
    # Parse and display MODEL ABILITIES
    # ========================================
    print("\n" + "="*80)
    print("MODEL ABILITIES (θ) - RANKED")
    print("="*80)
    
    abilities = []
    for subject_id, params in subjects.items():
        abilities.append({
            'subject_id': subject_id,
            'ability': params[0]
        })
    
    df_abilities = pd.DataFrame(abilities).sort_values('ability', ascending=False)
    
    print(f"\n{'Rank':<6} {'Model/Strategy':<45} {'Ability (θ)':<15}")
    print("-"*80)
    for idx, (_, row) in enumerate(df_abilities.iterrows(), 1):
        print(f"{idx:<6} {row['subject_id']:<45} {row['ability']:<15.4f}")
    
    print(f"\n{'='*80}")
    print("ABILITY STATISTICS")
    print(f"{'='*80}")
    print(f"Mean:   {df_abilities['ability'].mean():.4f}")
    print(f"Std:    {df_abilities['ability'].std():.4f}")
    print(f"Min:    {df_abilities['ability'].min():.4f}")
    print(f"Max:    {df_abilities['ability'].max():.4f}")
    print(f"Range:  {df_abilities['ability'].max() - df_abilities['ability'].min():.4f}")
    
    # ========================================
    # Parse and display QUESTION PARAMETERS
    # ========================================
    print("\n" + "="*80)
    print("QUESTION PARAMETERS")
    print("="*80)
    
    questions = []
    for item_id, params in items.items():
        q_data = {'question_id': item_id}
        
        if model_type == "1pl":
            q_data['difficulty'] = params[0]
        elif model_type == "2pl":
            q_data['difficulty'] = params[0]
            q_data['discrimination'] = params[1]
        elif model_type == "3pl":
            q_data['difficulty'] = params[0]
            q_data['discrimination'] = params[1]
            q_data['guessing'] = params[2]
        
        questions.append(q_data)
    
    df_questions = pd.DataFrame(questions)
    df_questions['q_idx'] = df_questions['question_id'].str.replace('q', '').astype(int)
    
    print(f"\nDIFFICULTY STATISTICS (b):")
    print("-"*80)
    print(f"Mean:   {df_questions['difficulty'].mean():.4f}")
    print(f"Std:    {df_questions['difficulty'].std():.4f}")
    print(f"Min:    {df_questions['difficulty'].min():.4f} (easiest)")
    print(f"Max:    {df_questions['difficulty'].max():.4f} (hardest)")
    
    if 'discrimination' in df_questions.columns:
        print(f"\nDISCRIMINATION STATISTICS (a):")
        print("-"*80)
        print(f"Mean:   {df_questions['discrimination'].mean():.4f}")
        print(f"Std:    {df_questions['discrimination'].std():.4f}")
        print(f"Min:    {df_questions['discrimination'].min():.4f}")
        print(f"Max:    {df_questions['discrimination'].max():.4f}")
    
    # Show easiest/hardest questions
    print(f"\nEASIEST 10 QUESTIONS:")
    print("-"*80)
    easiest = df_questions.nsmallest(10, 'difficulty')
    for _, row in easiest.iterrows():
        info = f"q{row['q_idx']:<6} difficulty={row['difficulty']:>7.4f}"
        if 'discrimination' in row:
            info += f" discrimination={row['discrimination']:>7.4f}"
        print(f"  {info}")
    
    print(f"\nHARDEST 10 QUESTIONS:")
    print("-"*80)
    hardest = df_questions.nlargest(10, 'difficulty')
    for _, row in hardest.iterrows():
        info = f"q{row['q_idx']:<6} difficulty={row['difficulty']:>7.4f}"
        if 'discrimination' in row:
            info += f" discrimination={row['discrimination']:>7.4f}"
        print(f"  {info}")
    
    # ========================================
    # Save to CSV
    # ========================================
    ability_file = f'abilities_{model_type}.csv'
    question_file = f'questions_{model_type}.csv'
    
    df_abilities.to_csv(ability_file, index=False)
    df_questions.to_csv(question_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ RESULTS SAVED")
    print(f"{'='*80}")
    print(f"Model abilities:      {ability_file}")
    print(f"Question parameters:  {question_file}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_irt_results.py <output_dir> [model_type]")
        print("")
        print("Examples:")
        print("  python extract_irt_results.py irt_results_1pl 1pl")
        print("  python extract_irt_results.py irt_results_2pl 2pl")
        print("  python extract_irt_results.py irt_results_3pl 3pl")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "2pl"
    
    load_and_display_results(output_dir, model_type)
