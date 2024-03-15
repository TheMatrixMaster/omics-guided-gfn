"""
This is a generative script that will generate the executables and log directories 
for a set of hyperparameters provided to a GFlowNet trainer object. To use this script,
simply modify the hyperparameters in the main function below and run the script.
"""

import os
from gflownet.algo.config import TBVariant
from gflownet.config import Config, init_empty, dfs_config_tree, merge_cfgs
from utils.runs import RunObject

TASK = "qm9"
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_ROOT = "/home/mila/s/stephen.lu/scratch/gfn_gene"

BASE_HPS = init_empty(Config())
BASE_HPS.desc = "GFlowNet training"
BASE_HPS.log_dir = ""
BASE_HPS.device = "cuda"
BASE_HPS.overwrite_existing_exp = True
BASE_HPS.num_training_steps = 10_000
BASE_HPS.validate_every = 0
BASE_HPS.num_workers = 0
BASE_HPS.opt.lr_decay = 10_000
BASE_HPS.cond.temperature.sample_dist = "uniform"
BASE_HPS.cond.temperature.dist_params = [0.5, 32]
BASE_HPS.cond.temperature.num_thermometer_dim = 32
BASE_HPS.algo.sampling_tau = 0.0
BASE_HPS.algo.train_random_action_prob = 0.0
BASE_HPS.algo.tb.variant = TBVariant.TB
BASE_HPS.task.qm9.h5_path = "/home/mila/s/stephen.lu/gfn_gene/res/qm9/qm9.h5"
BASE_HPS.task.qm9.model_path = (
    "/home/mila/s/stephen.lu/gfn_gene/res/qm9/mxmnet_gap_model.pt"
)

if __name__ == "__main__":
    assert TASK in ["seh", "qm9"], f"Invalid task: {TASK}"

    # Define Config tree of hyperparameters to test
    # Each field needs to be a List of values to test
    SUPER_HPS = init_empty(Config())
    SUPER_HPS.num_workers = [0]
    SUPER_HPS.num_training_steps = [10_000]
    SUPER_HPS.opt.lr_decay = [10_000]
    SUPER_HPS.algo.sampling_tau = [0.0]

    cfgs = dfs_config_tree(SUPER_HPS)
    cfgs = [merge_cfgs(BASE_HPS, cfg) for cfg in cfgs]

    for cfg in cfgs:
        run_obj = RunObject(task=TASK, cfg=cfg, LOG_ROOT=LOG_ROOT)
        run_obj.print_obj()
        run_obj.generate_scripts(CUR_DIR=CUR_DIR)
