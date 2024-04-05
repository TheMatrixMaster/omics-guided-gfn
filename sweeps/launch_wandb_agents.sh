#!/bin/bash

# Purpose: Script to allocate a node and run a wandb sweep agent on it
# Usage: sbatch launch_wandb_agent.sh <SWEEP_ID>

#SBATCH --job-name=wandb_sweep_agent
#SBATCH --array=1-50
#SBATCH --time=23:59:00
#SBATCH --output=slurm_output_files/%x_%N_%A_%a.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --partition long

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/mila/s/stephen.lu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/mila/s/stephen.lu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/mila/s/stephen.lu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/mila/s/stephen.lu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate conda environment
module --force purge
conda activate gfn

echo "Using environment={$CONDA_DEFAULT_ENV}"

# launch wandb agent
wandb agent --count 1 --entity thematrixmaster --project omics-guided-gfn $1
