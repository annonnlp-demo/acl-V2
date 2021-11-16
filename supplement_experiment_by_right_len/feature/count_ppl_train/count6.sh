#!/bin/bash
#SBATCH --job-name=6ppl
#SBATCH --output=6output.txt
#SBATCH --error=6error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=3 python count_6.py 2>&1 | tee ./6_output.txt