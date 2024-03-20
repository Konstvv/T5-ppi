#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=t5ankh
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=a100_3g.40gb:4
#SBATCH --nodes=4
#SBATCH --mem 128G
#SBATCH --time=10-10:00:00

srun python model.py --devices 4

# srun python test.py