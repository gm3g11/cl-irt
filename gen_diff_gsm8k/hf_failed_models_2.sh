#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N hf_gemma



source cl.env
python test_hf_models_gsm8k_optimized_v2.py --model google/gemma-2-9b-it --strategy few_shot_cot_8 --batch_size 32 --no_examples

# Terminal 2
python test_hf_models_gsm8k_optimized_v2.py --model google/gemma-2-9b-it --strategy few_shot_cot_4 --batch_size 32 --no_examples
