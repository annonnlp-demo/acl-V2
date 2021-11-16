#!/bin/bash
#SBATCH --job-name=lstm-9-supplement
#SBATCH --output=9-output.txt
#SBATCH --error=9-error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=3 python train9.py 2>&1 | tee ./9_output.txt
