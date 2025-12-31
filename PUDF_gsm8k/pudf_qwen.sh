#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N PUDF_qwen_gsm8k

source cl.env
python qwen2_5_7b_gsm8k_pudf.py \
  --train-batch-size 8 \
  --grad-accum-steps 8 \
  --eval-batch-size 16 \
  --theta-batch-size 96 \
  --pudf-epochs 10 \
  --lr 1e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lower-offset -5.0 \
  --upper-offset 0.5 \
  --min-samples 1000 \
  --theta-max-new-tokens 768 \
  --early-stopping-patience 5 \
  --seed 42
