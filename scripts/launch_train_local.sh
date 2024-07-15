#!/bin/bash
#SBATCH --job-name=run_trainer
#SBATCH --output=slurm_output_files/%x_%N_%A_%a.out
#SBATCH --partition=long
#SBATCH --time=11:59:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --partition long

# Activate conda environment
module --force purge
conda activate gfn

python ../gflownet/src/gflownet/tasks/morph_frag.py
