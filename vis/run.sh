#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=11:59:59

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

# Set env variables for hydra and cuda
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Start the job
# python /home/mila/s/stephen.lu/gfn_gene/vis/infer.py
# python /home/mila/s/stephen.lu/gfn_gene/vis/main.py
# python /home/mila/s/stephen.lu/gfn_gene/vis/aggr.py

# python aggr.py --config_name cluster_morph.json --plot_individual --norm --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/final_plots --max_k 2000 --num_samples 20000 --ignore_targets 8949,9476 --focus cluster
# python aggr.py --config_name cluster_joint.json --plot_individual --norm --run_name cluster-joint --save_dir /home/mila/s/stephen.lu/scratch/final_plots --max_k 2000 --num_samples 20000 --target_mode joint --ignore_targets 8949,9476
# python aggr.py --config_name assay_morph.json --plot_individual --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/final_plots --max_k 2000 --num_samples 20000 --assay_cutoff 0.0 --ignore_targets 8636,12662,15575 --focus assay

# python aggr.py --config_name assay_morph.json --plot_individual --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/final_plots --max_k 2000 --num_samples 20000 --assay_cutoff 0.0 --ignore_targets 8636,12662,15575 --focus assay
# python aggr.py --config_name cluster_morph_hi.json --plot_individual --norm --run_name cluster-morph-hi --save_dir /home/mila/s/stephen.lu/scratch/final_plots --max_k 2000 --num_samples 20000 --ignore_targets 8949,9476 --focus cluster

# python aggr.py --config_name assay_morph.json --plot_individual --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --ignore_targets 8636,12662,15575 --focus assay --sim_thresh 0.4
# python aggr.py --config_name assay_joint.json --plot_individual --run_name assay-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --ignore_targets 8636,12662,15575 --focus assay --sim_thresh 0.4 --target_mode joint

# python aggr.py --config_name cluster_morph.json --plot_individual --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --ignore_targets 8949,9476 --focus cluster --sim_thresh 0.4
# python aggr.py --config_name cluster_morph_hi.json --plot_individual --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --ignore_targets 8949,9476 --focus cluster --sim_thresh 0.4
# python aggr.py --config_name cluster_joint.json --plot_individual --run_name cluster-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --ignore_targets 8949,9476 --focus cluster --sim_thresh 0.4 --target_mode joint

# python aggr2.py --config_name assay_morph.json --plot_individual --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8
# python aggr2.py --config_name assay_joint.json --plot_individual --run_name assay-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8
# python aggr2.py --config_name cluster_morph.json --plot_individual --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8
python aggr2.py --config_name cluster_joint.json --plot_individual --run_name cluster-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8