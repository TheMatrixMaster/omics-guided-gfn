import sys, os
from typing import List

from gflownet.config import Config, init_from_dict
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.tasks.qm9 import QM9GapTrainer

hps = init_from_dict(Config(), {CONFIG})

configs: List[Config] = []

for seed in range({SEEDS}):
    hps.log_dir = "{LOG_DIR}/seed-" + str(seed + 1) + "/"
    configs.append(hps)

if __name__ == "__main__":
    hps = configs[int(sys.argv[1])]
    os.makedirs(hps.log_dir, exist_ok=True)

    if "{TASK}" == "qm9":
        trial = QM9GapTrainer(hps)
    elif "{TASK}" == "seh":
        trial = SEHFragTrainer(hps)

    trial.print_every = 1
    trial.verbose = True
    trial.run()
