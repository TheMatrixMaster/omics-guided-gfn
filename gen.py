"""
This is a generative script that will generate the executables and log directories 
for a set of hyperparameters provided to a GFlowNet trainer object. To use this script,
simply modify the hyperparameters in the main function below and run the script.
"""

import os
from gflownet.algo.config import TBVariant
from gflownet.config import *
from utils.runs import RunObject

TASK = "morph"
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_ROOT = "/home/mila/s/stephen.lu/scratch/gfn_gene"

BASE_HPS = init_empty(Config())
BASE_HPS.desc = "GFlowNet training"
BASE_HPS.log_dir = ""
BASE_HPS.print_every = 1
BASE_HPS.device = "cuda"
BASE_HPS.pickle_mp_messages = True
BASE_HPS.overwrite_existing_exp = True
BASE_HPS.num_training_steps = 5000
BASE_HPS.validate_every = 1000
BASE_HPS.num_final_gen_steps = 1000
BASE_HPS.num_validation_gen_steps = 0
# BASE_HPS.num_workers = 8
BASE_HPS.num_workers = 0
BASE_HPS.opt.lr_decay = 20_000
BASE_HPS.algo.sampling_tau = 0.99
BASE_HPS.algo.train_random_action_prob = 0.01
BASE_HPS.algo.method = "TB"
BASE_HPS.algo.tb.variant = TBVariant.TB
BASE_HPS.algo.tb.Z_learning_rate = 1e-3
BASE_HPS.algo.tb.Z_lr_decay = 50_000

BASE_HPS.cond.temperature.sample_dist = "constant"
BASE_HPS.cond.temperature.dist_params = [32.0]
BASE_HPS.cond.temperature.num_thermometer_dim = 1

# task specific hyperparameters
BASE_HPS.task.morph_sim.target_path = (
    "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample.pkl"
)
BASE_HPS.task.morph_sim.proxy_path = (
    "/home/mila/s/stephen.lu/gfn_gene/res/mmc/morph_struct.ckpt"
)
BASE_HPS.task.morph_sim.config_dir = (
    "/home/mila/s/stephen.lu/gfn_gene/multimodal_contrastive/configs"
)
BASE_HPS.task.morph_sim.config_name = "puma_sm_gmc.yaml"

# look into different types of temperature conditioning and their parameters (look at constant)
BASE_HPS.replay.use = False
BASE_HPS.replay.capacity = 50
BASE_HPS.replay.warmup = 10
BASE_HPS.replay.num_from_replay = None

if __name__ == "__main__":
    assert TASK in ["seh", "qm9", "toy", "morph"], f"Invalid task: {TASK}"

    # Define Config tree of hyperparameters to test
    # Each field needs to be a List of values to test
    SUPER_HPS = init_empty(Config())
    # SUPER_HPS.replay.use = [True]
    # SUPER_HPS.replay.capacity = [1000]
    # SUPER_HPS.replay.warmup = [100]
    # SUPER_HPS.num_workers = [8]
    # SUPER_HPS.num_training_steps = [10_000]
    # SUPER_HPS.opt.lr_decay = [10_000]
    # SUPER_HPS.algo.sampling_tau = [0.0]

    SUPER_HPS.task.morph_sim.target_path = [
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_0.pkl",
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_1.pkl",
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_2.pkl",
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_3.pkl",
        "/home/mila/s/stephen.lu/gfn_gene/res/mmc/sample_4.pkl",
    ]

    cfgs = dfs_config_tree(SUPER_HPS)
    cfgs = [merge_cfgs(BASE_HPS, cfg) for cfg in cfgs]

    for cfg in cfgs:
        run_obj = RunObject(
            task=TASK,
            cfg=cfg,
            num_seeds=1,
            LOG_ROOT=LOG_ROOT,
        )
        run_obj.print_obj()
        run_obj.generate_scripts(CUR_DIR=CUR_DIR)
