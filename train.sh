#!/bin/bash
#$ -cwd
#$ -o job-logs/
#$ -j y
#$ -pe smp 1

nvidia-smi
python src/train.py