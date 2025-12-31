#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N heu_gpt2_medqa



source cl_MedQA.env
python heu_gpt2_medqa.py
