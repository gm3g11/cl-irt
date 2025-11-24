#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N ag_word_rarity_pudf_theta

source cl.env
python qwen_ag_ablation.py \
    --difficulty_measurer word_rarity \
    --training_scheduler pudf_theta \
    --num_workers 4 \
    --use_bf16 \
    --pudf_difficulty_file ../gen_difficulty/merged_jsonlines_output/test-1pl/best_parameters.json \
    --output_dir_root ./qwen_ablation_results \
    --seed 63 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 16
