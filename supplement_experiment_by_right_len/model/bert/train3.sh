#!/bin/bash
#SBATCH --job-name=bert-3-supplement
#SBATCH --output=3-output.txt
#SBATCH --error=3-error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
python train3.py
