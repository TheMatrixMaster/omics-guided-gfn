#!/bin/bash
#SBATCH --job-name={}
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_error.txt
#SBATCH --partition=long
#SBATCH --time=23:59:59
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

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

cd {}
python run.py "$SLURM_ARRAY_TASK_ID"