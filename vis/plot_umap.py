"""
This script produces all the relevant aggregate plots for gfn analysis by combining datum
across multiple targets

Usage:
python aggr.py --config_name morph_assay_t=64.json --run_name assay-t=64 --target_mode morph --num_samples 10000 --max_k 5000 --assay_cutoff 0.5 --ignore_targets 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
"""

import argparse
from utils import *
from plotting import *
import json


def load_puma():
    # Load models and ground truth data
    assay_dataset = load_assay_matrix_from_csv()
    cluster_labels = load_cluster_labels_from_csv()
    assay_model = load_assay_pred_model(use_gneprop=USE_GNEPROP)
    cluster_model = load_cluster_pred_model(use_gneprop=USE_GNEPROP)
    mmc_model = None
    return assay_dataset, cluster_labels, assay_model, cluster_model, mmc_model


def load_run(run):
    target_idx, run_paths = run["target_idx"], run["run_paths"]
    target_sample_path = f"/home/mila/s/stephen.lu/gfn_gene/res/mmc/targets/sample_{target_idx}.pkl"

    # Load target fingerprint, smiles, latents, active assay cols (if any)
    target_smi, target_fp, _,_,_, target_reward =\
        load_target_from_path(target_sample_path, mmc_model, target_mode=TARGET_MODE)

    # print(f"Target struct~{TARGET_MODE} alignment: ", target_reward)
    print(f"Processing samples for {target_idx}")
    print(f"Target smi: ", target_smi)

    # Load baseline data for runs
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        run_path = os.path.join(RUNDIR, run_id)
        if NUM_SAMPLES == None:
            full_fps, full_rewards, full_smis = load_datum_from_run(
                RUNDIR, run_id, remove_duplicates=False, fps_from_file=False,
                save_fps=False, every_k=KEEP_EVERY)
        else:
            full_fps, full_rewards, full_smis = load_datum_from_run(
                RUNDIR, run_id, remove_duplicates=False, fps_from_file=False,
                save_fps=False, last_k=NUM_SAMPLES)
        full_mols = list(map(Chem.MolFromSmiles, tqdm(full_smis)))
        full_tsim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, full_fps))

        # Create run_datum object with duplicates
        run_datum = {
            "smis": full_smis, "rewards": full_rewards, "mols": full_mols,
            "fps": full_fps, "tsim_to_target": full_tsim_to_target,
        }
        assert len(full_smis) == len(full_rewards) == len(full_mols) ==\
            len(full_fps) == len(full_tsim_to_target)

        # Remove duplicates
        run_datum = remove_duplicates_from_run(run_datum)
        rewards, tsim, smis, fps = run_datum["rewards"], run_datum["tsim_to_target"],\
            run_datum["smis"], run_datum["fps"]
        assert len(rewards) == len(smis) == len(fps) == len(tsim)
        
        # Compute top-k modes by reward and by sim
        top_k_reward_idx = np.argsort(rewards)[::-1][:MAX_K]
        top_k_reward_fps = [fps[j] for j in top_k_reward_idx]
        top_k_reward_tsim_to_target = tsim[top_k_reward_idx]
        top_k_modes_idx, top_k_modes_fps = find_modes_from_arrays(rewards, smis, fps, k=MAX_K, sim_threshold=SIM_THRESH, return_fps=True)
        top_k_tsim_idx = np.argsort(tsim)[::-1][:MAX_K]
        top_k_modes_tsim_to_target = tsim[top_k_modes_idx]
        top_k_tsim_to_target = tsim[top_k_tsim_idx]

        print(f"Found {len(top_k_modes_idx)} modes for method {run_name}")

        # Save final run datum object
        run_datum["top_k_reward_idx"] = top_k_reward_idx
        run_datum["top_k_reward_fps"] = top_k_reward_fps
        run_datum["top_k_reward_tsim_to_target"] = top_k_reward_tsim_to_target
        run_datum["top_k_modes_idx"] = top_k_modes_idx
        run_datum["top_k_modes_fps"] = top_k_modes_fps
        run_datum["top_k_modes_tsim_to_target"] = top_k_modes_tsim_to_target
        run_datum["top_k_tsim_idx"] = top_k_tsim_idx
        run_datum["full_rewards"] = full_rewards
        run_datum["full_tsim_to_target"] = full_tsim_to_target
        runs_datum[run_name] = run_datum
        
        print(f"{run_name} topk tsim to target: ", np.mean(top_k_reward_tsim_to_target), np.quantile(top_k_reward_tsim_to_target, 0.75), np.max(top_k_reward_tsim_to_target))
        print(f"{run_name} full tsim to target: ", np.mean(full_tsim_to_target), np.quantile(full_tsim_to_target, 0.75), np.max(full_tsim_to_target))
        print(f"{run_name} topk tsim: ", np.min(top_k_tsim_to_target), np.mean(top_k_tsim_to_target), np.max(top_k_tsim_to_target))
        print()

    return runs_datum, target_fp, target_reward

def go(joint_datum, num_runs, target_fp=None, target_rew=None,
       assay_cols=None, cluster_id=None, save_dir=None, is_joint=False):
    plot_format = "pdf"
    os.makedirs(save_dir, exist_ok=True)
    if target_fp != None:
        plot_umap_from_runs_datum(joint_datum, target_fp, target_rew, n_neigh=10,
                                  k=MAX_K, ignore=["PUMA_test"],
                                  save_path=f"{save_dir}/umap.{plot_format}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--norm", action="store_true", help="Normalize assay and cluster predictions")
    parser.add_argument("--plot_individual", action="store_true", help="Plot individual runs")
    parser.add_argument("--use_gneprop", action="store_true", help="Use GNEPROP for assay preds")
    parser.add_argument("--target_mode", type=str, default="morph", help="Target mode to use")
    parser.add_argument("--config_name", type=str, default="cluster_morph.json", help="JSON config to use")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--max_k", type=int, default=5000, help="Max number of modes to consider")
    parser.add_argument("--run_name", type=str, default="cluster-morph", help="Run name to use")
    parser.add_argument("--assay_cutoff", type=float, default=0.5, help="Assay cutoff for active cols")
    parser.add_argument("--ignore_targets", type=str, default="", help="Comma-separated list of targets to ignore")
    parser.add_argument("--save_dir", type=str, default="/home/mila/s/stephen.lu/scratch/plots", help="Save directory for plots")
    parser.add_argument("--keep_every", type=int, default=8, help="Keep every k samples from the run")
    parser.add_argument("--run_dir", type=str, default="/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps", help="Run directory for runs")
    parser.add_argument("--focus", type=str, default="assay", help="Focus on assay or cluster preds")
    parser.add_argument("--sim_thresh", type=float, default=0.7, help="Similarity threshold for mode finding")

    args = parser.parse_args()
    RUN_NAME = args.run_name
    CONFIG_NAME = args.config_name
    USE_GNEPROP = args.use_gneprop
    TARGET_MODE = args.target_mode
    NUM_SAMPLES = args.num_samples
    MAX_K = args.max_k
    ASSAY_CUTOFF = args.assay_cutoff
    IGNORE_TARGETS = args.ignore_targets.split(",")
    SAVEDIR = args.save_dir
    RUNDIR = args.run_dir
    FOCUS = args.focus
    KEEP_EVERY = args.keep_every
    SIM_THRESH = args.sim_thresh
    
    # Load runs from JSON config
    with open(f"json/{CONFIG_NAME}") as f:
        RUNS = json.load(f)

    # Load models and ground truth data
    assay_dataset, cluster_labels, assay_model, cluster_model, mmc_model = load_puma()
        
    NUM_RUNS = 0
    joint_datum = {}
    for run in RUNS:
        target_idx = run['target_idx']
        if str(target_idx).strip() in IGNORE_TARGETS: continue
        runs_datum, target_fp, target_rew = load_run(run)
        if runs_datum is None: continue
        NUM_RUNS += 1
        print(f"Finished loading data for target {target_idx}. Now plotting individual plots...")
        save_dir = f"{SAVEDIR}/{target_idx}-{RUN_NAME}"
        if USE_GNEPROP: save_dir += "-gneprop"
        if args.norm: save_dir += "-norm"
        go(runs_datum, 1, target_fp=target_fp, target_rew=target_rew, save_dir=save_dir)
