#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N medqa_pudf_irt_linear

source cl.env

python qwen_medqa_ablation.py \
    --difficulty_measurer pudf_irt\
    --training_scheduler linear \
    --pudf_difficulty_file "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/MeD_QA/merged_jsonlines_output/test-1pl/best_parameters.json" \
    --output_dir_root ./medqa_results \
    --seed 63 \
    --model_id Qwen/Qwen2.5-7B \
    --dataset_id "GBaker/MedQA-USMLE-4-options" \
    --prompt_style base \
    --use_bf16 \
    --num_workers 4 \
    --num_epochs 10 \
    --patience_early_stopping 3 \
    --train_batch_size 2 \
    --grad_accum_steps 16
