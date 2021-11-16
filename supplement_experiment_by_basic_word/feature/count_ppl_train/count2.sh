#!/bin/bash
#SBATCH --job-name=2ppl
#SBATCH --output=2output.txt
#SBATCH --error=2error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=0 python count_2.py 2>&1 | tee ./2_output.txt