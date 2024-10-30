#!/bin/bash
#SBATCH  --output=logs/%j.out
#SBATCH  --cpus-per-task=4
#SBATCH  --gres=gpu:1
#SBATCH  --constraint='titan_xp'
#SBATCH  --mem=50G


source path_to_conda/conda.sh # TODO: SET.
conda activate mri-reconstruction


# NOTE: Uncomment when running multi-gpu script.
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# echo "MASTER_PORT="$MASTER_PORT

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR


# python -u multi_gpu/main.py
# python -u multi_vol/main.py
python -u single_vol/main.py