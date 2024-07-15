import os
import sys
import time

import wandb

from gflownet.config import Config, init_empty, TBVariant
from gflownet.tasks.morph_frag import MorphSimilarityTrainer

TIME = time.strftime("%m-%d-%H-%M")
ENTITY = "your.wandb.entity"
PROJECT = "omics-guided-gfn"
SWEEP_NAME = f"{TIME}-morph-sim"
STORAGE_DIR = f"~/wandb_sweeps/{SWEEP_NAME}"


# Define the search space of the sweep
sweep_config = {
    "name": SWEEP_NAME,
    "program": "init_wandb_sweep.py",
    "controller": {
        "type": "cloud",
    },
    "method": "grid",
    "parameters": {
        # "config.opt.learning_rate": {"values": [1e-4, 1e-3]},
        # "config.algo.method": {"values": ["SQL", "SAC"]},
        # "config.algo.tb.Z_learning_rate": {"values": [1e-4, 1e-3]},
        # "config.algo.tb.Z_lr_decay": {"values": [2_000, 50_000]},
        # "config.algo.sampling_tau": {"values": [0.0, 0.95, 0.99]},
        # "config.algo.train_random_action_prob": {"values": [0.01]},
        # "config.cond.temperature.dist_params": {
        #   "values": [[128.0], [256.0], [512.0]]
        # },
        # "config.replay.capacity": {"values": [5000, 10000]},
        # "config.replay.num_from_replay": {"values": [32, 64]},
        # "config.task.morph_sim.reduced_frag": {"values": [True, False]},
        # "config.algo.max_nodes": {"values": [4, 6, 8]},
        "config.task.morph_sim.target_path": {
            "values": [
                # Old Assay based targets
                "6~/path.to.targets/sample_39.pkl",
                "4~/path.to.targets/sample_903.pkl",
                "6~/path.to.targets/sample_1847.pkl",
                "5~/path.to.targets/sample_2288.pkl",
                "7~/path.to.targets/sample_6888.pkl",
                "6~/path.to.targets/sample_8838.pkl",
                "4~/path.to.targets/sample_10075.pkl",
                "7~/path.to.targets/sample_13905.pkl",

                New Assay based targets
                "5~/path.to.targets/sample_2288.pkl",
                "7~/path.to.targets/sample_4646.pkl",
                "5~/path.to.targets/sample_8505.pkl",
                "4~/path.to.targets/sample_8636.pkl",
                "4~/path.to.targets/sample_10075.pkl",
                "7~/path.to.targets/sample_10816.pkl",
                "7~/path.to.targets/sample_12662.pkl",
                "5~/path.to.targets/sample_15575.pkl",

                # Cluster based targets
                "7~/path.to.targets/sample_4331.pkl",
                "7~/path.to.targets/sample_8206.pkl",
                "4~/path.to.targets/sample_338.pkl",
                "6~/path.to.targets/sample_8949.pkl",
                "5~/path.to.targets/sample_9277.pkl",
                "7~/path.to.targets/sample_9300.pkl",
                "6~/path.to.targets/sample_9445.pkl",
                "7~/path.to.targets/sample_9476.pkl",
                "5~/path.to.targets/sample_12071.pkl",
            ]
        },
        # "config.task.morph_sim.target_mode": {"values": ["morph", "joint"]},
    },
}


def wandb_config_merger():
    config = init_empty(Config())
    wandb_config = wandb.config

    # Set desired config values
    config.device = "cuda"
    config.log_dir = f"{STORAGE_DIR}/{wandb.run.name}-id-{wandb.run.id}"
    config.print_every = 1
    config.validate_every = 0
    config.num_final_gen_steps = 0
    config.num_training_steps = 10_000
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = False
    config.opt.learning_rate = 1e-4
    config.opt.lr_decay = 20_000
    config.algo.tb.Z_learning_rate = 1e-4
    config.algo.tb.Z_lr_decay = 50_000
    config.algo.max_nodes = 7
    config.algo.sampling_tau = 0.95
    config.algo.method = "TB"
    config.algo.a2c.entropy = 0.2
    config.algo.sql.alpha = 0.01
    config.algo.train_random_action_prob = 0.01
    config.algo.tb.variant = TBVariant.TB
    config.num_workers = 0
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [128.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.replay.use = False
    config.replay.capacity = 50
    config.replay.warmup = 10

    # task specific hyperparameters
    config.task.morph_sim.target_path = (
        "/path.to.targets/sample_10852.pkl"
    )
    config.task.morph_sim.proxy_path = "path.to/gmc_proxy.ckpt"
    config.task.morph_sim.config_dir = "../multimodal_contrastive/configs"
    config.task.morph_sim.config_name = "puma_sm_gmc.yaml"
    config.task.morph_sim.reduced_frag = False
    config.task.morph_sim.target_mode = "morph"

    ##  Merge the wandb sweep config with the nested config from gflownet

    # config.opt.learning_rate = wandb_config["config.opt.learning_rate"]
    # config.algo.tb.Z_learning_rate = wandb_config["config.algo.tb.Z_learning_rate"]
    # config.algo.tb.Z_lr_decay = wandb_config["config.algo.tb.Z_lr_decay"]
    # config.algo.sampling_tau = wandb_config["config.algo.sampling_tau"]
    # config.algo.train_random_action_prob = wandb_config[
    #     "config.algo.train_random_action_prob"
    # ]
    # config.replay.capacity = wandb_config["config.replay.capacity"]
    # config.replay.num_from_replay = wandb_config["config.replay.num_from_replay"]
    # config.task.morph_sim.reduced_frag = wandb_config[
    #     "config.task.morph_sim.reduced_frag"
    # ]

    # method = wandb_config["config.algo.method"]
    # config.algo.method = method

    # if method == "SAC":
    #     config.algo.sampling_tau = 0.995
    #     config.algo.train_random_action_prob = 0.05
    # else:
    #     config.algo.sampling_tau = 0.95
    #     config.algo.train_random_action_prob = 0.01

    # config.task.morph_sim.target_mode = wandb_config[
    #     "config.task.morph_sim.target_mode"
    # ]

    config.cond.temperature.dist_params = wandb_config[
        "config.cond.temperature.dist_params"
    ]
    
    ## Split max_nodes from target path
    max_nodes, target_path = wandb_config["config.task.morph_sim.target_path"].split("~")
    config.algo.max_nodes = int(max_nodes)
    config.task.morph_sim.target_path = target_path

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
