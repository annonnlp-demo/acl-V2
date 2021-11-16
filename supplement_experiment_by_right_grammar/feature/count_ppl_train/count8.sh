#!/bin/bash
#SBATCH --job-name=8ppl
#SBATCH --output=8output.txt
#SBATCH --error=8error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=3 python count_8.py 2>&1 | tee ./8output.txt