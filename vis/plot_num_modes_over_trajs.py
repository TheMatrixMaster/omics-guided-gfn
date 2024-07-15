import os
import argparse
from utils import *
from plotting import *
import json


def compute_rew_thresh(run, n=5000, percentile=90):
    runs_datum_sm, _ = load_run(run, num_samples=n)
    merged_rewards = []
    for run_name, run_datum in runs_datum_sm.items():
        merged_rewards.append(run_datum["rewards"])
    merged_rewards = np.hstack(merged_rewards)
    return round(np.percentile(merged_rewards, percentile), 2)

def load_run(run, num_samples=None, every_k=None, check_len=False):
    # Load baseline data for runs
    should_skip_run = False
    run_paths = run["run_paths"]
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        full_fps, full_rewards, full_smis = load_datum_from_run(RUNDIR, run_id, remove_duplicates=False,
                                                                fps_from_file=False, save_fps=False,
                                                                last_k=num_samples, every_k=every_k)
        # Create run_datum object with duplicates
        run_datum = { "smis": full_smis, "rewards": full_rewards, "fps": full_fps }
        assert len(full_smis) == len(full_rewards) == len(full_fps)
        
        if check_len and len(full_smis) != (640000/every_k):
            # should_skip_run = True
            continue
        else:
            runs_datum[run_name] = run_datum
    return runs_datum, should_skip_run

def compute_num_modes(runs_datum, rew_thresh):
    res = {}
    for run_name, run_datum in runs_datum.items():
        num_modes, avg_rew = num_modes_lazy(run_datum, rew_thresh, SIM_THRESH, bs=64//KEEP_EVERY)
        print(f"Run {run_name} has {num_modes[-1]} modes with average reward {avg_rew[-1]}")
        res[run_name] = { "num_modes": num_modes, "avg_rew": avg_rew }
    return res

def get_modes_smis(run_datum, rew_thresh):
    res = {}
    for run_name, run_datum in runs_datum.items():
        num_modes, avg_rew, modes_smis = num_modes_lazy(run_datum, rew_thresh, SIM_THRESH, 
                                            bs=64//KEEP_EVERY, return_smis=True)
        print(f"Run {run_name} has {num_modes[-1]} modes with average reward {avg_rew[-1]}")
        res[run_name] = { "modes_smis": modes_smis }
    return res

def merge(runs_datum, joint_datum, keys_to_keep=[]):
    """Merge runs datum runs into joint_datum"""
    for run_name, run_datum in runs_datum.items():
        tmp_datum = joint_datum[run_name] if run_name in joint_datum else {}
        for k, v in run_datum.items():
            if k not in keys_to_keep: continue
            if k not in tmp_datum: tmp_datum[k] = []
            tmp_datum[k].append(v)
        joint_datum[run_name] = tmp_datum
    return joint_datum

def compute_spread(joint_datum):
    """Computes the mean and std ranges for each method"""
    for run_name, run_datum in joint_datum.items():
        all_num_modes = run_datum["num_modes"]
        all_avg_rew = run_datum["avg_rew"]
        run_datum["num_modes_median"] = np.median(all_num_modes, axis=0)
        run_datum["num_modes_lo"] = np.percentile(all_num_modes, 25, axis=0)
        run_datum["num_modes_hi"] = np.percentile(all_num_modes, 75, axis=0)
        run_datum["avg_rew_mean"] = np.mean(all_avg_rew, axis=0)
        run_datum["avg_rew_std"] = np.std(all_avg_rew, axis=0)
        print(f"Run {run_name} has median num modes {run_datum['num_modes_median'][-1]} with iqr {run_datum['num_modes_lo'][-1]}-{run_datum['num_modes_hi'][-1]}")
        print(f"Run {run_name} has mean avg rew {run_datum['avg_rew_mean'][-1]} with std {run_datum['avg_rew_std'][-1]}")

def go(joint_datum, num_runs, target_idx=None, rew_thresh=None, save_dir=None, is_joint=False):
    plot_format = "pdf"
    os.makedirs(save_dir, exist_ok=True)
    plot_modes_over_trajs(
        joint_datum,
        num_runs,
        target_idx,
        is_joint=is_joint,
        rew_thresh=rew_thresh,
        sim_thresh=SIM_THRESH,
        ignore=["PUMA_test"],
        save_path=f"{save_dir}/num_modes_{rew_thresh}.{plot_format}"
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--plot_individual", action="store_true", help="Plot individual runs")
    parser.add_argument("--target_mode", type=str, default="morph", help="Target mode to use")
    parser.add_argument("--config_name", type=str, default="cluster_morph.json", help="JSON config to use")
    parser.add_argument("--run_name", type=str, default="cluster-morph", help="Run name to use")
    parser.add_argument("--ignore_targets", type=str, default="", help="Comma-separated list of targets to ignore")
    parser.add_argument("--save_dir", type=str, default="~/plots", help="Save directory for plots")
    parser.add_argument("--run_dir", type=str, default=os.getenv("RUNS_DIR_PATH"), help="Run directory for runs")
    parser.add_argument("--sim_thresh", type=float, default=0.7, help="Similarity threshold for mode finding")
    parser.add_argument("--keep-every", type=int, default=8, help="Keep every k samples from the run")
    parser.add_argument("--save_memory", action="store_true", help="Save memory by not storing all runs")
    parser.add_argument("--save_modes", action="store_true", help="Save modes SMILES")

    args = parser.parse_args()
    RUN_NAME = args.run_name
    CONFIG_NAME = args.config_name
    TARGET_MODE = args.target_mode
    IGNORE_TARGETS = args.ignore_targets.split(",")
    SAVEDIR = args.save_dir
    RUNDIR = args.run_dir
    SIM_THRESH = args.sim_thresh
    KEEP_EVERY = args.keep_every
    
    # Load runs from JSON config
    with open(f"../runs/{CONFIG_NAME}") as f:
        RUNS = json.load(f)
        
    NUM_RUNS = 0
    joint_datum = {}
    for run in RUNS:
        target_idx = run['target_idx']
        if str(target_idx).strip() in IGNORE_TARGETS: continue

        rew_thresh = compute_rew_thresh(run, percentile=90)
        print(f"Computed reward threshold for {target_idx} as {rew_thresh}")

        runs_datum, should_skip = load_run(run, every_k=KEEP_EVERY, check_len=True)
        if runs_datum is None or should_skip:
            print(f"Skipping {target_idx} due to missing data")
            continue
        else: NUM_RUNS += 1

        save_dir = f"{SAVEDIR}/{target_idx}-{RUN_NAME}"
        os.makedirs(save_dir, exist_ok=True)

        if args.save_modes:
            result = get_modes_smis(runs_datum, rew_thresh)
            np.save(f"{save_dir}/modes_smis.npy", result)
            continue
        else:
            result = compute_num_modes(runs_datum, rew_thresh)

        if args.plot_individual:
            go(result, 1, target_idx=target_idx, rew_thresh=rew_thresh, save_dir=save_dir)
            
        if args.save_memory:
            np.save(f"{save_dir}/runs_datum.npy", runs_datum)
        else:
            print(f"Finished plotting for {target_idx}. Now merging into joint datum...")
            joint_datum = merge(result, joint_datum, keys_to_keep=["num_modes", "avg_rew"])
            
        del runs_datum

    print(f"Processed {NUM_RUNS} runs")
    print(joint_datum.keys())

    compute_spread(joint_datum)

    # Produce plots for the joint runs
    save_dir = f"{SAVEDIR}/aggr-{RUN_NAME}"
    go(joint_datum, NUM_RUNS, save_dir=save_dir, is_joint=True)