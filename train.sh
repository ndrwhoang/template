#!/bin/bash
#$ -N TrainingJob
#$ -cwd
#$ -o job-logs/
#$ -j y


conda activate nlp
python src/train.py --configs/config.ini

cp /shared/training_out/TrainingJob ./training_out