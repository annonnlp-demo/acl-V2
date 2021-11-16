#!/bin/bash
#SBATCH --job-name=test-ttr
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --mail-type=ALL



#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=CLUSTER
#SBATCH --time=10-24:00:00

source activate calibration
python count.py