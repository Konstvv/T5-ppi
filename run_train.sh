#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=t5_ppi_train
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes=9
#SBATCH --gpus=a100_3g.40gb:9
#SBATCH --mem 256G
#SBATCH --time=20-10:00:00

set -x

echo "Starting at: $(date)"

srun python model.py --batch_size 32  --num_nodes $SLURM_NNODES --num_workers $SLURM_CPUS_ON_NODE --accelerator 'gpu' --strategy 'ddp_find_unused_parameters_false'

echo "Finishing at: $(date)"

