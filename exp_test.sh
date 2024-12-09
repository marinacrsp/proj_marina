#!/bin/bash
#SBATCH  --output=logs/%j.out       
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=4
# SBATCH  --constraint='titan_xp'
#SBATCH  --mem=50G


source /scratch_net/ken/mcrespo/conda/etc/profile.d/conda.sh # TODO: SET.
conda activate pytcu11

# module load cuda/11.7

# Debug GPU availability
echo "Checking GPU status..."
nvidia-smi

# Run your script
python cluster_test.py
