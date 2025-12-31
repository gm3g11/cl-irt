#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N glue_Llama3_8B_mix_precision_1



source cl.env
python baseline_GLUE_Llama3.1-8B_mix_precision.py
