import os
import sys
import time

import wandb

from gflownet.config import Config, init_empty, TBVariant
from gflownet.tasks.morph_frag import MorphSimilarityTrainer

TIME = time.strftime("%m-%d-%H-%M")
ENTITY = "thematrixmaster"
PROJECT = "omics-guided-gfn"
SWEEP_NAME = f"{TIME}-morphSimilarity-more_nodes&morph_target"
STORAGE_DIR = f"/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps/{SWEEP_NAME}"


# Define the search space of the sweep
sweep_config = {
    "name": SWEEP_NAME,
    "program": "morph_sim_sweep.py",
    "controller": {
        "type": "cloud",
    },
    "method": "grid",
    "parameters": {
        # "config.opt.learning_rate": {"values": [1e-4, 1e-3]},
        # "config.algo.tb.Z_learning_rate": {"values": [1e-4, 1e-3]},
        # "config.algo.tb.Z_lr_decay": {"values": [2_000, 50_000]},
        # "config.algo.sampling_tau": {"values": [0.0, 0.95, 0.99]},
        # "config.algo.train_random_action_prob": {"values": [0.0, 0.05, 0.1, 0.2]},
        # "config.cond.temperature.dist_params": {
        #     "values": [
        #         [0, 128.0],
        #         [0, 64.0],
        #         [0, 32.0],
        #     ]
        # },
        # "config.replay.capacity": {"values": [5000, 10000]},
        # "config.replay.num_from_replay": {"values": [32, 64]},
        # "config.task.morph_sim.reduced_frag": {"values": [True, False]},
        # "config.algo.max_nodes": {"values": [9, 10, 11, 12, 13, 14]},
        "config.task.morph_sim.target_path": {
            "values": [
                "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_0.pkl",
                "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_92.pkl",
                "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_112.pkl",
                "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_1047.pkl",
                "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_1413.pkl",
                "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_1429.pkl",
            ]
        },
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
    config.num_training_steps = 5000
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = False
    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 20_000
    config.algo.tb.Z_learning_rate = 1e-4
    config.algo.tb.Z_lr_decay = 50_000
    config.algo.max_nodes = 9
    config.algo.sampling_tau = 0.95
    config.algo.method = "TB"
    config.algo.train_random_action_prob = 0.01
    config.algo.tb.variant = TBVariant.TB
    config.num_workers = 0
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [64.0]
    config.cond.temperature.num_thermometer_dim = 1
    # look into different types of temperature conditioning and their parameters (look at constant)
    config.replay.use = False
    config.replay.capacity = 50
    config.replay.warmup = 10

    # task specific hyperparameters
    config.task.morph_sim.target_path = (
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_0.pkl"
    )
    config.task.morph_sim.proxy_path = (
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/morph_struct.ckpt"
    )
    config.task.morph_sim.config_dir = (
        "/home/mila/s/stephen.lu/gfn_gene/multimodal_contrastive/configs"
    )
    config.task.morph_sim.config_name = "puma_sm_gmc.yaml"
    config.task.morph_sim.reduced_frag = False
    config.task.morph_sim.target_mode = "joint"

    ##  Merge the wandb sweep config with the nested config from gflownet

    # config.opt.learning_rate = wandb_config["config.opt.learning_rate"]
    # config.algo.tb.Z_learning_rate = wandb_config["config.algo.tb.Z_learning_rate"]
    # config.algo.tb.Z_lr_decay = wandb_config["config.algo.tb.Z_lr_decay"]
    # config.algo.sampling_tau = wandb_config["config.algo.sampling_tau"]
    # config.algo.max_nodes = wandb_config["config.algo.max_nodes"]
    # config.algo.train_random_action_prob = wandb_config[
    #     "config.algo.train_random_action_prob"
    # ]
    # config.cond.temperature.dist_params = wandb_config[
    #     "config.cond.temperature.dist_params"
    # ]
    # config.replay.capacity = wandb_config["config.replay.capacity"]
    # config.replay.num_from_replay = wandb_config["config.replay.num_from_replay"]

    config.task.morph_sim.target_path = wandb_config[
        "config.task.morph_sim.target_path"
    ]
    # config.task.morph_sim.reduced_frag = wandb_config[
    #     "config.task.morph_sim.reduced_frag"
    # ]

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
