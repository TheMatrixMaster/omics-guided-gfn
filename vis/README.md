# Plotting Details

This directory contains the code needed to generate all the figures in the paper. Apart from the figures involving the multimodal contrastive model, all the other figures involving GFlowNets were produced post-training, using the samples generated during training trajectories. The `gflownet` submodule saves all training trajectories into a `.db` file containing a SMILES string and a float reward for each sample. Our plotting code then references these `.db` files to generate the figures.

To reproduce our results, we've provided `.db` files for the runs we used in the paper. These files are stored under their training run directory in our [google drive](https://drive.google.com/drive/folders/1CFY9YHJsGGDYggwI7TCJhlssZDe_CuZK?usp=sharing) and must be downloaded to your machine to reproduce the figures. Here is the directory structure of a run folder:

```
run_folder
├── train.log                   # Training output log
├── model_state.pt              # Model checkpoint
├── config.yaml                 # Training configuration
├── train                       
│   ├── generated_mols_0.db     # Training samples
```

Given that our plots often group multiple individual runs (ex: to compare GFlowNet vs. Baselines performance) across multiple `targets`, we provide `.json` config files under the `runs/json` directory that specify paths to the run folders mentioned above. Each config file is an array of objects, where each object specifies the paths to the run directory for a particular set of runs with the same `target`. A `target` is a pickle file that contains the structure and morphology data of the target profile that we want to optimize for during training. The set of `targets` we used in the paper must also be downloaded from our [drive](https://drive.google.com/drive/folders/1CFY9YHJsGGDYggwI7TCJhlssZDe_CuZK?usp=sharing) for plotting.

## Setup
After downloading both the training runs [folder](https://drive.google.com/drive/folders/1CFY9YHJsGGDYggwI7TCJhlssZDe_CuZK?usp=sharing) and the training targets [folder](https://drive.google.com/drive/folders/1q-YqF2F7cvnK4Mr_thmI_CKRDYvjHLRs?usp=sharing) from the drive, please update and source the `env.sh` script in this directory with the paths to these two folders on your machine.

```bash
export RUNS_DIR_PATH="path.to/runs"
export TARGETS_DIR_PATH="path.to/targets"
```

## Number of Modes over Trajectories
To plot the number of modes over trajectories, you can run the following command. This will plot the number of modes over trajectories for each method aggregated across all targets in the joint training setting. The `--sim_thresh` flag specifies the similarity threshold used to determine what constitutes a unique mode. The `--keep-every 8` flag indicates that we should only consider every 8th trajectory in the `.db` file to save on computation time. 
```bash
python plot_num_modes_over_trajs.py --config_name all_joint.json --run_name all-joint --save_dir ~/outdir --sim_thresh 0.3 --keep-every 8
```

## UMAP Visualization
To plot UMAP visualizations of the molecular fingerprints of the training samples, you can run the following command. For each target in the `all_morph.json` config file, this will plot the UMAP of the top `max_k` modes or highest reward samples among the last `num_samples` trajectories in the training data of each method. Once again, the `--sim_thresh` flag specifies the similarity threshold used to determine what constitutes a unique mode. This script produces 4 plots for each target:

- UMAP of the top `max_k` modes for each method colored by reward
- UMAP of the top `max_k` modes for each method colored by tanimoto similarity to the target
- UMAP of the top `max_k` highest reward samples for each method colored by reward
- UMAP of the top `max_k` highest reward samples for each method colored by tanimoto similarity to the target

```bash
python plot_umap.py --config_name all_morph.json --run_name all-morph --save_dir ~/outdir --max_k 1000 --num_samples 10000 --sim_thresh 0.3
```

## Histograms (Reward, Tanimoto Similarity)
To plot the aggregated reward histogram and tanimoto similarity to target histogram, and tanimoto similarity between top-k highest reward samples histogram, you can run the `plot_main.py` script as shown below. This will produce four plots aggregated across all targets in the `all_morph.json` config file.

- Histogram of rewards for the last `num_samples` trajectories for each method
- Histogram of Tanimoto similarity to target of the last `num_samples` trajectories for each method
- Histogram of Tanimoto similarity among the top `max_k` highest reward samples for each method
- Histogram of Tanimoto similarity to target of the top `max_k` highest reward samples for each method

```bash
python plot_main.py --config_name all_morph.json --run_name morph-hist --save_dir ~/outdir --max_k 1000 --num_samples 10000
```

## Oracle Boxplots
The oracle boxplots show the number of samples per method (aggregated across targets) that satisfy some property constraint as determined by some oracle. We have three oracles:
-`cluster` predicts the morphological cluster of a sample from its fingerprint
-`assay` predicts the outcome of a biological assay from the sample's fingerprint
-`tanimoto` is a pseudo-oracle that determines the tanimoto similarity of a sample to the target profile

The command below produces the boxplot using the `oracle` cluster for all morphology targets with an active cluster ground truth label. The `--cluster_pred_thresh` flag specifies the threshold for the cluster prediction to be considered active. The `--focus` flag specifies the oracle to use. We also provide below the commands to produce the boxplots for the other oracles.

```bash
python plot_oracle_box.py --config_name cluster_morph.json --run_name cluster-box --save_dir ~/outdir --max_k 1000 --keep_every 8 --focus cluster --cluster_pred_thresh 0.3

python plot_oracle_box.py --config_name assay_morph.json --run_name assay-box --save_dir ~/outdir --max_k 1000 --keep_every 8 --focus assay --assay_pred_thresh 0.7

python plot_oracle_box.py --config_name all_morph.json --run_name tsim-box --save_dir ~/outdir --max_k 1000 --keep_every 8 --focus tsim --sim_to_target_thresh 0.2
```

## Max Tanimoto Similarity to Target
To produce the table of max tanimoto similarity to target for each method, we simply looked at the highest tanimoto similarity to target for each method across all targets among the last `num_samples` trajectories per run. Here is the command to produce these results:
```bash
python compute_max_tanimoto_sim.py --config_name all_morph.json --num_samples 10000
```

#### GMC Latent Space Visualization
The latent space visualization for the multimodal contrastive model (GMC) are produced in the `../notebooks/evaluate_mmc.ipynb` notebook. The notebook has imports from the `multimodal` submodule, which you can install by following the instructions in the main `README.md` file.
