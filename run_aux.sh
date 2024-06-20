#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=dataset
#SBATCH --cpus-per-task=64
#SBATCH --mem=758G
#SBATCH --time=10-00:00:00

set -x

echo "Starting at: $(date)"

srun python3 create_dataset.py --int_seq string12.0_combined_score_900.tsv string12.0_combined_score_900.fasta --combined_score 900 --experimental_score 100

echo "Finishing at: $(date)"

