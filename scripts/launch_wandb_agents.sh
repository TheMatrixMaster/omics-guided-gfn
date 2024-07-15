#!/bin/bash

# Purpose: Script to allocate a node and run a wandb sweep agent on it
# Usage: sbatch launch_wandb_agent.sh <SWEEP_ID>

#SBATCH --job-name=wandb_sweep_agent
#SBATCH --array=1-70
#SBATCH --time=11:59:59
#SBATCH --output=slurm_output_files/%x_%N_%A_%a.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --partition long

# Activate conda environment
module --force purge
conda activate gfn

echo "Using environment={$CONDA_DEFAULT_ENV}"

# launch wandb agent
wandb agent --count 1 --entity your.wandb.entity --project omics-guided-gfn $1
