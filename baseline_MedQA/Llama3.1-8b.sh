#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N llama_MedQA_data_2



source cl_MedQA.env
python llama_final.py
