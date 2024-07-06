#!/bin/bash
#SBATCH --job-name=run_trainer
#SBATCH --output=slurm_output_files/%x_%N_%A_%a.out
#SBATCH --partition=long
#SBATCH --time=11:59:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
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

python ../gflownet/src/gflownet/tasks/morph_frag.py
