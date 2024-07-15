import os
import argparse
from utils import *
from plotting import *
import json

TARGET_DIR = os.getenv("TARGETS_DIR_PATH")


def load_run(run):
    target_idx, run_paths = run["target_idx"], run["run_paths"]
    target_sample_path = f"{TARGET_DIR}/sample_{target_idx}.pkl"
    _, target_fp, _, _, _, _ = load_target_from_path(target_sample_path)
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        full_fps, _, _ = load_datum_from_run(RUNDIR, run_id, remove_duplicates=False, 
                                             fps_from_file=False, save_fps=False, last_k=NUM_SAMPLES)
        full_tsim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, full_fps))
        runs_datum[run_name] = np.max(full_tsim_to_target)
        print(f"{run_name} has max tanimoto similarity to target {np.max(full_tsim_to_target)}")
        del full_fps
        del full_tsim_to_target
    return runs_datum
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_name", type=str, default="cluster_morph.json", help="JSON config to use")
    parser.add_argument("--ignore_targets", type=str, default="", help="Comma-separated list of targets to ignore")
    parser.add_argument("--run_dir", type=str, default=os.getenv("RUNS_DIR_PATH"), help="Run directory for runs")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to load from each run")

    args = parser.parse_args()
    CONFIG_NAME = args.config_name
    IGNORE_TARGETS = args.ignore_targets.split(",")
    RUNDIR = args.run_dir
    NUM_SAMPLES = args.num_samples
    
    # Load runs from JSON config
    with open(f"../runs/{CONFIG_NAME}") as f:
        RUNS = json.load(f)
        
    NUM_RUNS = 0
    joint_datum = {}
    for run in RUNS:
        target_idx = run['target_idx']
        if str(target_idx).strip() in IGNORE_TARGETS: continue
        print(f"Processing run for target {target_idx}")
        runs_datum = load_run(run)
        print()
        if runs_datum is None: continue
        NUM_RUNS += 1

        for run_name, run_datum in runs_datum.items():
            if run_name not in joint_datum: joint_datum[run_name] = []
            joint_datum[run_name].append(run_datum)

    print(f"Processed {NUM_RUNS} runs with proper proxy alignment")
    print(joint_datum.keys())

    # Produce mean and std for the top tanimoto similarity to target for each method
    for run_name, run_datum in joint_datum.items():
        run_datum = np.array(run_datum)
        print(f"Run {run_name}: {np.mean(run_datum)} +- {np.std(run_datum)}")