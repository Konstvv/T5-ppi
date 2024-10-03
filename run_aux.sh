#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=dataset
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=10-00:00:00

set -x

echo "Starting at: $(date)"

srun python3 create_dataset.py --int_seq string12.0_combined_score_900.tsv string12.0_combined_score_900.fasta --combined_score 998 --memory_limit 5GB --n_workers 1 --threads_per_worker $SLURM_CPUS_PER_TASK
# srun python3 tokenizer.py

echo "Finishing at: $(date)"

