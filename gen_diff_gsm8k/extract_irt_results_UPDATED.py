#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract and display IRT results from best_parameters.json
Updated for newer py-irt format
"""

import json
import pandas as pd
import sys
import os

def load_and_display_results(output_dir, model_type="2pl"):
    """Load and display IRT results from best_parameters.json"""
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} not found!")
        return
    
    # Check for best_parameters.json
    params_file = os.path.join(output_dir, "best_parameters.json")
    
    if not os.path.exists(params_file):
        print(f"Error: Missing best_parameters.json in {output_dir}")
        return
    
    print("="*80)
    print(f"LOADING {model_type.upper()} MODEL RESULTS")
    print(f"Directory: {output_dir}")
    print("="*80)
    
    # Load parameters
    with open(params_file, 'r') as f:
        data = json.load(f)
    
    # Extract components
    abilities = data['ability']  # List of ability values
    difficulties = data['diff']  # List of difficulty values
    subject_ids = data['subject_ids']  # Dict: index -> subject_id
    item_ids = data['item_ids']  # Dict: index -> item_id
    
    # Get discrimination if 2PL
    discriminations = data.get('disc', None)  # List of discrimination values (2PL only)
    
    n_subjects = len(abilities)
    n_items = len(difficulties)
    
    print(f"\n✓ Loaded {n_items} questions and {n_subjects} models")
    
    # ========================================
    # Parse and display MODEL ABILITIES
    # ========================================
    print("\n" + "="*80)
    print("MODEL ABILITIES (θ) - RANKED")
    print("="*80)
    
    # Create dataframe for abilities
    ability_data = []
    for idx, ability in enumerate(abilities):
        subject_id = subject_ids[str(idx)]
        ability_data.append({
            'subject_id': subject_id,
            'ability': ability
        })
    
    df_abilities = pd.DataFrame(ability_data).sort_values('ability', ascending=False)
    
    print(f"\n{'Rank':<6} {'Model/Strategy':<45} {'Ability (θ)':<15}")
    print("-"*80)
    for idx, (_, row) in enumerate(df_abilities.iterrows(), 1):
        print(f"{idx:<6} {row['subject_id']:<45} {row['ability']:<15.4f}")
    
    print(f"\n{'='*80}")
    print("ABILITY STATISTICS")
    print(f"{'='*80}")
    print(f"Mean:   {df_abilities['ability'].mean():.4f}")
    print(f"Std:    {df_abilities['ability'].std():.4f}")
    print(f"Min:    {df_abilities['ability'].min():.4f} ({df_abilities.iloc[-1]['subject_id']})")
    print(f"Max:    {df_abilities['ability'].max():.4f} ({df_abilities.iloc[0]['subject_id']})")
    print(f"Range:  {df_abilities['ability'].max() - df_abilities['ability'].min():.4f}")
    
    # ========================================
    # Parse and display QUESTION PARAMETERS
    # ========================================
    print("\n" + "="*80)
    print("QUESTION PARAMETERS")
    print("="*80)
    
    # Create dataframe for questions
    question_data = []
    for idx, difficulty in enumerate(difficulties):
        q_id = item_ids[str(idx)]
        q_data = {
            'question_id': q_id,
            'difficulty': difficulty
        }
        
        # Add discrimination if 2PL
        if discriminations is not None and model_type == "2pl":
            q_data['discrimination'] = discriminations[idx]
        
        question_data.append(q_data)
    
    df_questions = pd.DataFrame(question_data)
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
    
    if 'discrimination' in df_questions.columns:
        print(f"\nMOST DISCRIMINATING 10 QUESTIONS (best at separating abilities):")
        print("-"*80)
        most_disc = df_questions.nlargest(10, 'discrimination')
        for _, row in most_disc.iterrows():
            print(f"  q{row['q_idx']:<6} discrimination={row['discrimination']:>7.4f} difficulty={row['difficulty']:>7.4f}")
        
        print(f"\nLEAST DISCRIMINATING 10 QUESTIONS (poor quality):")
        print("-"*80)
        least_disc = df_questions.nsmallest(10, 'discrimination')
        for _, row in least_disc.iterrows():
            print(f"  q{row['q_idx']:<6} discrimination={row['discrimination']:>7.4f} difficulty={row['difficulty']:>7.4f}")
    
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
        print("  python extract_irt_results.py test-1pl/ 1pl")
        print("  python extract_irt_results.py test-2pl/ 2pl")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "2pl"
    
    load_and_display_results(output_dir, model_type)
