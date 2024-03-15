# Omics-Guided GflowNets

## Description
This repository contains the code to run the experiments and visualize the results highlighted in the Omics-Guided GflowNets project.

## Overview
Our codebase builds on top of a fork of the public [recursion gflownet](https://github.com/recursionpharma/gflownet) repo which provides the environment setup to run the gflownet framework on graph domains.

## Setup
To setup the project, first create a conda environment with `python=3.10`
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

Then, install the gflownet package from local source:
```bash
cd gflownet
git submodule update --init
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

or follow the guideline from the public [recursion gflownet](https://github.com/TheMatrixMaster/gflownet) repo

## Usage
The current project supports the following:

### Training from scratch
Our setup is made to support jobs running with the Slurm workload manager. To train a mixture policy from scratch, you must first edit the `utils/template.sh` file to customize the Slurm executable script to your GPU environment. Next, to generate the executables for a given job, simply edit the hyperparameters in the main function of `gen.py` and run the file from the project root directory:

#### Example
First, select the [task] on which you'd like to train the network.
Next, set the hyperparameters in the `gen.py` file by following the [Config](https://github.com/TheMatrixMaster/gflownet/blob/trunk/src/gflownet/config.py) specification.

For every hyperparameter combination you specified, the script will generate a corresponding run folder at `jobs/<current_date>/<run_id-current_time>`. This folder will contain the following files:

- `run.sh`: the Slurm executable script for the job
- `run.py`: the main executable script for the job
- `howto.txt`: a text file containing the command to submit the job to slurm
- `config.json`: a json file containing the full Config object for the job
- `run_object.json`: a json file containing the class instance of the run object, which can be used to re-instantiate the run object for downstream analysis and plotting

To submit the job to slurm, simply run the command specified in `howto.txt` from the run config directory. For example, if the command is `sbatch --array=0-4 run.sh config`, then run the following:

```bash
cd jobs/<current_date>/<run_id-current_time>
sbatch --array=0-4 run.sh config
```