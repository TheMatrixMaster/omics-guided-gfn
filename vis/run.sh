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

# python main_plots.py --config_name all_gfn_cluster.json --run_name cluster_joint_vs_morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --focus cluster --use_gneprop --ignore_targets 8949,9476,4331
# python main_plots.py --config_name all_gfn_assay.json --run_name assay_joint_vs_morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --assay_cutoff 0.0 --focus assay --ignore_targets 12662,15575,8636
# python main_plots.py --config_name all_gfn.json --run_name sim-box --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 10000 --num_samples 10000 --focus tsim --sim_to_target_thresh 0.4

# python main_plots.py --config_name assay_morph.json --run_name assay-box-every-k --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --assay_cutoff 0.0 --focus assay --ignore_targets 12662,15575,8636
# python main_plots.py --config_name cluster_morph.json --run_name cluster-box --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --focus cluster --use_gneprop --ignore_targets 8949,9476,4331
# python main_plots.py --config_name cluster_morph.json --run_name cluster-box --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --focus cluster --ignore_targets 8949,9476,4331

# python /home/mila/s/stephen.lu/gfn_gene/vis/infer.py
# python /home/mila/s/stephen.lu/gfn_gene/vis/main.py
# python /home/mila/s/stephen.lu/gfn_gene/vis/aggr.py

# python main_plots.py --config_name all_morph.json --run_name all-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --sim_thresh 0.3
# python main_plots.py --config_name all_joint.json --run_name all-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --sim_thresh 0.3

# python umap_plots.py --config_name all_morph.json --run_name all-morph --save_dir /home/mila/s/stephen.lu/scratch/final_umap_plots --max_k 1000 --num_samples 10000 --sim_thresh 0.3
# python umap_plots.py --config_name all_morph.json --run_name all-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --sim_thresh 0.3

# python main_plots.py --config_name cluster_morph.json --plot_individual --run_name morph_cluster --save_dir /home/mila/s/stephen.lu/scratch/morph_cluster --max_k 1000 --num_samples 10000 --focus cluster --sim_thresh 0.3 --use_gneprop --ignore_targets 8949,9476,4331
# python main_plots.py --config_name cluster_morph_hi.json --plot_individual --norm --run_name dummy --save_dir /home/mila/s/stephen.lu/scratch/dummy --max_k 1000 --num_samples 1000 --focus cluster --sim_thresh 0.3 --ignore_targets 4331,9476
# python main_plots.py --config_name assay_joint.json --run_name assay-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --assay_cutoff 0.0 --focus assay --sim_thresh 0.3 --ignore_targets 12662,15575,8636
# python main_plots.py --config_name assay_morph.json --run_name assay-box --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --focus assay --sim_thresh 0.3
# python main_plots.py --config_name cluster_joint.json --plot_individual --norm --run_name cluster-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --keep_every 8 --focus cluster --sim_thresh 0.3 --ignore_targets 8949,9476
# python main_plots.py --config_name cluster_morph.json --norm --run_name cluster-morph-last-10k --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --focus cluster --sim_thresh 0.3 --ignore_targets 8949,9476

# python main_plots.py --config_name assay_morph.json --plot_individual --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --ignore_targets 8636,12662,15575 --focus assay --sim_thresh 0.3
# python aggr.py --config_name assay_joint.json --plot_individual --run_name assay-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --assay_cutoff 0.0 --ignore_targets 8636,12662,15575 --focus assay --sim_thresh 0.4 --target_mode joint

# python aggr.py --config_name cluster_morph.json --plot_individual --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --ignore_targets 8949,9476 --focus cluster --sim_thresh 0.4
# python aggr.py --config_name cluster_morph_hi.json --plot_individual --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --ignore_targets 8949,9476 --focus cluster --sim_thresh 0.4
# python aggr.py --config_name cluster_joint.json --plot_individual --run_name cluster-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --max_k 1000 --num_samples 10000 --ignore_targets 8949,9476 --focus cluster --sim_thresh 0.4 --target_mode joint

# python num_modes.py --config_name assay_morph.json --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8 --save_modes
# python aggr2.py --config_name assay_joint.json --plot_individual --run_name assay-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8
# python aggr2.py --config_name cluster_morph.json --plot_individual --run_name cluster-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8
# python aggr2.py --config_name cluster_joint.json --plot_individual --run_name cluster-joint --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8

python num_modes.py --config_name all_joint.json --run_name all-joint --save_dir /home/mila/s/stephen.lu/scratch/save_me --sim_thresh 0.3 --keep-every 8
python num_modes.py --config_name all_joint_test.json --run_name all-joint --save_dir /home/mila/s/stephen.lu/scratch/save_me --sim_thresh 0.3 --keep-every 8

# python num_modes.py --config_name all_morph.json --plot_individual --run_name all-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 8
# python num_modes.py --config_name assay_morph.json --run_name assay-morph --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --sim_thresh 0.3 --keep-every 1 --save_modes --ignore_targets 2288,4646,8505

# python oracle_plots.py --config_name assay_morph.json --run_name assay-morph-p90 --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --assay_cutoff 0.0 --reward_percentile 90 --sim_thresh 0.4
# python oracle_plots.py --config_name assay_morph.json --run_name assay-morph-p85 --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --assay_cutoff 0.0 --reward_percentile 85 --sim_thresh 0.3
# python oracle_plots.py --config_name assay_morph.json --run_name assay-morph-p80 --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --assay_cutoff 0.0 --reward_percentile 80 --sim_thresh 0.3

# python oracle_plots.py --config_name cluster_morph.json --run_name assay-cluster-p90 --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --assay_cutoff 0.0 --reward_percentile 90 --sim_thresh 0.3 --focus cluster
# python oracle_plots.py --config_name cluster_morph_hi.json --run_name assay-cluster-p90-hi --save_dir /home/mila/s/stephen.lu/scratch/pdf_plots --assay_cutoff 0.0 --reward_percentile 90 --sim_thresh 0.3 --focus cluster