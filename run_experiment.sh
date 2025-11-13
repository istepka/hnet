#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/train_%j.out
#SBATCH --error=slurm/train_%j.err

# Print the hostname of the node you're on
echo "Running on node: $(hostname)"

source ~/.bashrc
conda activate hnet2

echo "Starting training..."

python train.py name=c4_425Mtok_2L \
    training.total_steps=50000 \
    training.save_every=5000 \

echo "Training completed."
