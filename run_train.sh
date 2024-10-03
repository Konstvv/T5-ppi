#!/bin/bash
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out
#SBATCH --job-name=transppi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=6
#SBATCH --gpus=a100_3g.40gb:6
#SBATCH --time=20-10:00:00
#SBATCH --mem=500G

set -x

echo "Starting at: $(date)"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# srun python model.py --lr 2e-5 --batch_size 16 --accumulate_grad_batches 16 --devices 1 --num_nodes $SLURM_NNODES --num_workers $SLURM_CPUS_ON_NODE --accelerator 'gpu' --precision 16 --strategy 'ddp_find_unused_parameters_false'
srun python model_esm.py --lr 1e-5 --batch_size 16 --accumulate_grad_batches 16 --devices 1 --num_nodes $SLURM_NNODES --num_workers $SLURM_CPUS_ON_NODE --accelerator 'gpu' --precision 16 --strategy 'ddp_find_unused_parameters_false'

# srun python test.py --devices 1 --num_nodes $SLURM_NNODES --num_workers $SLURM_CPUS_ON_NODE --accelerator 'gpu' --strategy 'ddp_find_unused_parameters_false'

# srun python esm2_emb_caluculate.py string12.0_combined_score_900_esm.fasta

echo "Finishing at: $(date)"
