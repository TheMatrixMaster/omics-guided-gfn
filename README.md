# Omics-Guided GflowNets

## Description
This repository contains the code to run the experiments and visualize the results in the [Cell Morphology-Guided Small Molecule Generation with GFlowNets]() paper. Our codebase builds on top of a fork of the public [recursion gflownet](https://github.com/recursionpharma/gflownet) repo which provides the environment setup to run the gflownet framework on graph domains. We also use a second submodule for training multimodal contrastive learning models (GMC, CLIP) which we use to derive a reward signal for the GFlowNets. See the paper for more details. Please contact the [authors](mailto:stephen.lu@mila.quebec) for further information.

## Setup
To setup the project, first create a conda environment with `python=3.10`
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

Then, pull the submodules and install them accordingly.

```bash
git submodule update --init --recursive --remote

# For the gflownet submodule, run
cd gflownet
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html

# For the gmc submodule, run
cd multimodal_contrastive
pip install -r requirements.txt
pip install -e .
```

## Usage
The current project supports training GFlowNets or any of the baselines in the paper (SAC, SQL, Random) from scratch and reproducing the plots in the paper. The instructions for each are detailed below.

### Training from scratch
Before training, you must download a checkpoint for our reward GMC model and target profiles from our [google drive](). The GMC checkpoint is used to derive the reward signal for the GFlowNets. The target profiles contain an associated morphological and structural profile that we want to optimize for in either the `morph` or `joint` training settings. Refer to the paper for more details.

To train a model from scratch, you may choose to either run the training script directly or run a sweep with wandb. To run the training script directly, modify the hyperparameters at the bottom of the `gflownet/src/gflownet/tasks/morph_frag.py` script and run it directly from the command line.

To run a sweep, please edit the sweep parameters in `sweeps/morph_sim_sweep.py`, then follow the instructions in the `sweeps/README.md` to launch the sweep. We use slurm to launch jobs and have provided shell script under `sweeps/` to do so, but it should be rather simple to modify these to run on your setup.

For both methods, you will need to specify the path to the GMC checkpoint and target profiles that you downloaded earlier.

### Plotting
Plots requires that you download the training trajectories from our [drive]() or that you train your own models from scratch. For detailed instructions on reproducing the plots in our paper, please refer to `vis/README.md`.

## Citation
If you use this codebase in your research, please cite the following paper:
```
@article{lu2022cell,
  title={Cell Morphology-Guided Small Molecule Generation with GFlowNets},
  author={Lu, Stephen}
}
```