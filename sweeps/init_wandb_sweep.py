import os
import sys
import time

import wandb

from gflownet.config import Config, init_empty, TBVariant
from gflownet.tasks.morph_frag import MorphSimilarityTrainer

TIME = time.strftime("%m-%d-%H-%M")
ENTITY = "thematrixmaster"
PROJECT = "omics-guided-gfn"
SWEEP_NAME = f"{TIME}-morphSimilarity-eps&sampling_tau"
STORAGE_DIR = f"/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps/{SWEEP_NAME}"


# Define the search space of the sweep
sweep_config = {
    "name": SWEEP_NAME,
    "program": "init_wandb_sweep.py",
    "controller": {
        "type": "cloud",
    },
    "method": "grid",
    "parameters": {
        # "config.algo.tb.Z_learning_rate": {"values": [1e-4, 1e-3, 1e-2]},
        # "config.algo.tb.Z_lr_decay": {"values": [2_000, 50_000]},
        "config.algo.sampling_tau": {"values": [0.95, 0.99]},
        "config.algo.train_random_action_prob": {"values": [0.0, 0.01, 0.1]},
        # "config.replay.capacity": {"values": [5000, 10000]},
        # "config.replay.num_from_replay": {"values": [32, 64]},
    },
}


def wandb_config_merger():
    config = init_empty(Config())
    wandb_config = wandb.config

    # Set desired config values
    config.device = "cuda"
    config.log_dir = f"{STORAGE_DIR}/{wandb.run.name}-id-{wandb.run.id}"
    config.print_every = 1
    config.validate_every = 1000
    config.num_final_gen_steps = 1000
    config.num_training_steps = 15_000
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = False
    config.opt.lr_decay = 20_000
    config.algo.tb.Z_learning_rate = 1e-3
    config.algo.tb.Z_lr_decay = 50_000
    config.algo.sampling_tau = 0.99
    config.algo.method = "TB"
    config.algo.train_random_action_prob = 0.01
    config.algo.tb.variant = TBVariant.TB
    config.num_workers = 8
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0.0, 64]
    config.replay.use = False
    config.replay.capacity = 10_000
    config.replay.warmup = 1_000

    # Merge the wandb sweep config with the nested config from gflownet
    # config.algo.tb.Z_learning_rate = wandb_config["config.algo.tb.Z_learning_rate"]
    # config.algo.tb.Z_lr_decay = wandb_config["config.algo.tb.Z_lr_decay"]
    config.algo.sampling_tau = wandb_config["config.algo.sampling_tau"]
    config.algo.train_random_action_prob = wandb_config[
        "config.algo.train_random_action_prob"
    ]
    # config.replay.capacity = wandb_config["config.replay.capacity"]
    # config.replay.num_from_replay = wandb_config["config.replay.num_from_replay"]

    return config


if __name__ == "__main__":
    # if there no arguments, initialize the sweep, otherwise this is a wandb agent
    if len(sys.argv) == 1:
        if os.path.exists(STORAGE_DIR):
            raise ValueError(f"Sweep storage directory {STORAGE_DIR} already exists.")

        wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

    else:
        wandb.init(entity=ENTITY, project=PROJECT)
        config = wandb_config_merger()
        trial = MorphSimilarityTrainer(config)
        trial.run()
