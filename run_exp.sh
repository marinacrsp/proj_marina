#!/bin/bash
#SBATCH  --output=logs/%j.out
#SBATCH  --cpus-per-task=5
#SBATCH  --gres=gpu:1
#SBATCH  --constraint='titan_xp'
#SBATCH  --mem=50G


source /scratch_net/ken/mcrespo/conda/etc/profile.d/conda.sh # TODO: SET.
conda activate pytcu11

# # Debugging: Check if SLURM_JOB_NODELIST is defined and populated
# echo "SLURM_JOB_NODELIST is: $SLURM_JOB_NODELIST"

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr

# echo "MASTER_ADDR=$MASTER_ADDR"

# # Set up other distributed parameters
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=$SLURM_GPUS_ON_NODE
# export RANK=$SLURM_PROCID

# echo "MASTER_PORT=$MASTER_PORT"
# echo "WORLD_SIZE=$WORLD_SIZE"
# echo "RANK=$RANK"

# torchrun single_vol_multigpu/main.py
# python -u multi_vol/main.py
python -u single_vol/main.py