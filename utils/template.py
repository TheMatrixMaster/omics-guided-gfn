import sys, os
from typing import List

import wandb
from omegaconf import OmegaConf
from gflownet.config import Config, init_from_dict
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.tasks.qm9 import QM9GapTrainer
from gflownet.tasks.toy_frag import ToySimilarityTrainer

hps = init_from_dict(Config(), {CONFIG})

configs: List[Config] = []

for seed in range({SEEDS}):
    hps.log_dir = "{LOG_DIR}/seed-" + str(seed + 1) + "/"
    configs.append(hps)

if __name__ == "__main__":
    hps = configs[int(sys.argv[1])]
    seed = int(sys.argv[1]) + 1
    os.makedirs(hps.log_dir, exist_ok=True)

    run = wandb.init(
        project="omics-guided-gfn",
        group="gfn-{TASK}-task",
        entity="thematrixmaster",
        dir=hps.log_dir,
        allow_val_change=True,
        name=f"{RUN_ID}-seed-" + str(seed),
    )

    if "{TASK}" == "qm9":
        trial = QM9GapTrainer(hps)
    elif "{TASK}" == "seh":
        trial = SEHFragTrainer(hps)
    elif "{TASK}" == "toy":
        trial = ToySimilarityTrainer(hps)

    run.config.update(OmegaConf.to_container(trial.cfg))

    trial.print_every = 1
    trial.verbose = True
    trial.run()

    run.finish()
