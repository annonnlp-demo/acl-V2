#!/bin/bash
#SBATCH --job-name=lstm-5-supplement
#SBATCH --output=5-output.txt
#SBATCH --error=5-error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=2 python train5.py 2>&1 | tee ./5_output.txt
