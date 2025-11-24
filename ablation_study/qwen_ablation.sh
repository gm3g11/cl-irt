#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N glue_pudf_irt_root

source cl.env
python qwen_glue_ablation.py \
    --difficulty_measurer pudf_irt \
    --training_scheduler root \

