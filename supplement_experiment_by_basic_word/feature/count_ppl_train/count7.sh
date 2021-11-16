#!/bin/bash
#SBATCH --job-name=7ppl
#SBATCH --output=7output.txt
#SBATCH --error=7error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=0 python count_7.py 2>&1 | tee ./7_output.txt