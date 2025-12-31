#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N hf_8_shot_cot



source cl.env
python test_hf_models_gsm8k_optimized_v2.py --strategy few_shot_cot_8 --batch_size 160 --no_examples
