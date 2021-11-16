#!/bin/bash
#SBATCH --job-name=lstm-1-supplement
#SBATCH --output=1-output.txt
#SBATCH --error=1-error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
CUDA_VISIBLE_DEVICES=0 python train3.py 2>&1 | tee ./3_output.txt