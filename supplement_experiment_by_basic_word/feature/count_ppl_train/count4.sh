#!/bin/bash
#SBATCH --job-name=4ppl
#SBATCH --output=4output.txt
#SBATCH --error=4error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=0 python count_4.py 2>&1 | tee ./4_output.txt