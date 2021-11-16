#!/bin/bash
#SBATCH --job-name=10ppl
#SBATCH --output=10output.txt
#SBATCH --error=10error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=3 python count_10.py 2>&1 | tee ./10_output.txt