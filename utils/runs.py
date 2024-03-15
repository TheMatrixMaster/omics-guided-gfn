"""
Class methods for generating runs
"""

import os
import json
from uuid import uuid4
from datetime import datetime
from .helper import EnhancedJSONEncoder
from gflownet.config import Config, strip_missing


def make_py_script(
    task: str, hps: Config, log_dir: str, run_name: str, seeds=5, CUR_DIR: str = None
):
    with open(f"{CUR_DIR}/utils/template.py", "r") as f:
        script = f.read()
        f.close()
    hps = strip_missing(hps)
    script = script.format(
        CONFIG=hps, LOG_DIR=log_dir, SEEDS=seeds, TASK=task, RUN_ID=run_name
    )
    return script


def make_sh_script(run_name: str, log_dir: str, CUR_DIR: str = None):
    with open(f"{CUR_DIR}/utils/template.sh", "r") as f:
        script = f.read()
        f.close()
    script = script.format(run_name, log_dir)
    return script


class RunObject:
    cfg: Config
    log_dir: str
    run_name: str
    num_seeds: int
    task: str

    def __init__(
        self,
        task: str = "seh",
        cfg: Config = None,
        run_name: str = None,
        num_seeds: int = 5,
        from_config: bool = False,
        config_path: str = None,
        LOG_ROOT: str = None,
    ):
        if from_config:
            with open(f"{config_path}/run_object.json", "r") as f:
                config = json.load(f, cls=EnhancedJSONEncoder)
                f.close()
            self.__dict__ = config
            self.cfg = Config(**self.cfg)
            print(f"Loaded run object from {config_path}.\n")
        else:
            assert task in [
                "seh",
                "qm9",
                "toy",
            ], "Task must be one of 'seh', 'qm9', or 'toy'."
            self.cfg = cfg
            self.task = task
            self.num_seeds = num_seeds
            self.run_name = run_name if run_name else str(uuid4())
            self.log_dir = self.make_log_dir(LOG_ROOT=LOG_ROOT)

    def make_log_dir(self, LOG_ROOT) -> str:
        cur_date = datetime.today().strftime("%Y-%m-%d")
        cur_time = datetime.today().strftime("%H-%M-%S")
        log_dir = f"{LOG_ROOT}/{cur_date}/{self.run_name}-{cur_time}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def generate_scripts(self, save_run: bool = True, CUR_DIR: str = None):
        print(f"Generating scripts for run {self.run_name}...\n")

        py_script = make_py_script(
            task=self.task,
            hps=self.cfg,
            log_dir=self.log_dir,
            run_name=self.run_name,
            seeds=self.num_seeds,
            CUR_DIR=CUR_DIR,
        )
        sh_script = make_sh_script(self.run_name, self.log_dir, CUR_DIR=CUR_DIR)
        sh_cmd = f"sbatch --array=0-{self.num_seeds-1} run.sh"

        with open(f"{self.log_dir}/run.py", "w") as f:
            f.write(py_script)

        with open(f"{self.log_dir}/run.sh", "w") as f:
            f.write(sh_script)

        with open(f"{self.log_dir}/config.json", "w") as f:
            json.dump(self.cfg, f, indent=4, cls=EnhancedJSONEncoder)

        with open(f"{self.log_dir}/howto.txt", "w") as f:
            f.write(sh_cmd)

        if save_run:
            with open(f"{self.log_dir}/run_object.json", "w") as f:
                json.dump(self.__dict__, f, indent=4, cls=EnhancedJSONEncoder)

        print("Scripts generated successfully in the following directory:\n")
        print(self.log_dir)
        print("\n\n")

    def print_obj(self):
        print(f"Setting up run {self.run_name} with the following hyperparameters:\n")
        print(self.cfg)
        print()
