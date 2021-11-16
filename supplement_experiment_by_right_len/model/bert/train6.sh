#!/bin/bash
#SBATCH --job-name=bert-6-supplement
#SBATCH --output=6-output.txt
#SBATCH --error=6-error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
python train6.py
