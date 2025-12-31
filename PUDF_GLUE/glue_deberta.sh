#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 8
#$ -M YOUR_EMAIL@example.com   # Email address for job notification
#$ -m abe
#$ -N PUDF_glue_DebertaV3_1

source cl.env
python PUDF_glue_DebertaV3.py
