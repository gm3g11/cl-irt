#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N ag_news_Qwen2.5_7b_2



source cl.env
python Qwen2.5_7b.py
