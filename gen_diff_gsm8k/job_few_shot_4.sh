#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu
#$ -m abe
#$ -N gsm8k_few_shot_4

echo "=========================================="
echo "Job: GSM8K - Few Shot 4 Strategy"
echo "8 models, few_shot_4 prompting"
echo "Start time: $(date)"
echo "=========================================="

source cl.env

# Run all 8 models with few_shot_4 strategy
python test_gsm8k_all_strategies.py \
    --mode parallel \
    --strategies few_shot_4 \
    --output_dir results

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
