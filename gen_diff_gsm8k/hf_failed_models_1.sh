#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N hf_Mistral_Yi



source cl.env
python test_hf_models_gsm8k_optimized_v2.py --model mistralai/Mistral-7B-Instruct-v0.2 --strategy few_shot_cot_8 --batch_size 48 --no_examples

# Terminal 4
python test_hf_models_gsm8k_optimized_v2.py --model 01-ai/Yi-1.5-9B-Chat --strategy few_shot_cot_8 --batch_size 32 --no_examples
