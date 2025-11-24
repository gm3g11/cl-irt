#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N glue_Qwen_3



source cl.env
python baseline_GLUE_Qwen2.5-7B.py
