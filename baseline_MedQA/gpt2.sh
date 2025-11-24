#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N gpt2_MedQA_3



source cl_MedQA.env
python gpt2.py
