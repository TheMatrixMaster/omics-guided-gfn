Everything is contained in one file; `init_wandb_sweep.py` both defines the search space of the sweep and is the entrypoint of wandb agents.

To launch the search:
1. `python init_wandb_sweep.py` to intialize the sweep
2. `sbatch launch_wandb_agents.sh <SWEEP_ID>` to schedule a jobarray in slurm which will launch wandb agents.
The number of jobs in the sbatch file should reflect the size of the hyperparameter space that is being sweeped.

Alternatively, you can launch a single run locally by calling the `gflownet/src/gflownet/tasks/morph_frag.py` file. An example is provided in the `launch_train_local.sh` script in this directory.