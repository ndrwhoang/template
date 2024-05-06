#!/bin/bash
#$ -N TrainingJob
#$ -cwd
#$ -o job-logs/
#$ -j y

WANDB_API_KEY=$(cat api_keys/wandb.txt)
export WANDB_API_KEY

conda activate nlp
python src/train.py --configs/config.ini
