#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N PUDF_ag_Qwen_3

source cl.env
python pudf_qwen.py
