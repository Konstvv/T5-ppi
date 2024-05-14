#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=dataset
#SBATCH --cpus-per-task=75
#SBATCH --mem=256G
#SBATCH --time=10-00:00:00

set -x

echo "Starting at: $(date)"

srun python3 test.py 

echo "Finishing at: $(date)"

