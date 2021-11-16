#!/bin/bash
#SBATCH --job-name=bert-7-supplement
#SBATCH --output=7-output.txt
#SBATCH --error=7-error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
python train7.py
