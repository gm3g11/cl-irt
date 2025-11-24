#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N PUDF_Llama_gsm8k

source cl.env
python llama3_1_8b_gsm8k_pudf.py   --model base \
  --total-epochs 10 \
  --pudf-epochs 8 \
  --lr 1e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --train-batch-size 16 \
  --grad-accum-steps 4 \
  --eval-batch-size 32 \
  --theta-eval-batches 15 \
  --theta-batch-size 128 \
  --theta-max-new-tokens 1024 \
  --early-stopping-patience 5 \
  --seed 42
