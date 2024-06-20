#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=convbert
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=4
#SBATCH --gpus=a100_3g.40gb:4
#SBATCH --time=20-10:00:00
#SBATCH --mem 256G

set -x

echo "Starting at: $(date)"

srun python model.py --lr 1e-4 --batch_size 16 --accumulate_grad_batches 16 --devices 1 --num_nodes $SLURM_NNODES --num_workers $SLURM_CPUS_ON_NODE --accelerator 'gpu' --strategy 'ddp_find_unused_parameters_false'


echo "Finishing at: $(date)"

