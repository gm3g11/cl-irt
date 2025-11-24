#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu
#$ -m abe
#$ -N gsm8k_zero_shot

echo "=========================================="
echo "Job: GSM8K - Zero Shot Strategy"
echo "8 models, zero_shot prompting"
echo "Start time: $(date)"
echo "=========================================="

source cl.env

# Run all 8 models with zero_shot strategy
python test_gsm8k_all_strategies.py \
    --mode parallel \
    --strategies zero_shot \
    --output_dir results

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
