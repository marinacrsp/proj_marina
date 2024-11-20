#!/bin/bash
#SBATCH  --output=logs/%j.out       
#SBATCH  --gres=gpu:3
#SBATCH  --cpus-per-task=4
#SBATCH  --constraint='titan_xp'
#SBATCH  --mem=50G


source /scratch_net/ken/mcrespo/conda/etc/profile.d/conda.sh # TODO: SET.
conda activate pytcu11

# # Debugging: Check if SLURM_JOB_NODELIST is defined and populated

master_addr=$(echo $SLURM_JOB_NODELIST | sed 's/,.*//' | sed 's/\[.*\]//')
export MASTER_ADDR=$master_addr


# Print debugging info for verification
echo "MASTER_ADDR is $MASTER_ADDR"
echo "Nodes allocated: $SLURM_JOB_NODELIST"
echo "Total GPUs: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

# # # Set up other distributed parameters
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_GPUS_ON_NODE

echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"

# python -u multi_vol/main.py
# python -u multi_vol_hash/main.py
# python -u multi_gpu/main.py
python -u multi_gpu_hash/main.py
# python -u single_vol/main.py
# python -u single_vol_hash/main.py
# python -u single_vol_original/main.py