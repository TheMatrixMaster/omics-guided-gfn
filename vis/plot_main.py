import os
import argparse
from utils import *
from plotting import *
import json

TARGET_DIR = os.getenv("TARGETS_DIR_PATH")


def load_puma():
    # Load models and ground truth data
    assay_dataset = load_assay_matrix_from_csv()
    cluster_labels = load_cluster_labels_from_csv()
    assay_model = load_assay_pred_model()
    cluster_model = load_cluster_pred_model()
    mmc_model = None
    return assay_dataset, cluster_labels, assay_model, cluster_model, mmc_model


def load_run(run):
    target_idx, run_paths, rew_thresh = run["target_idx"], run["run_paths"], run["reward_thresh"]
    bs = run["bs"] if "bs" in run.keys() else 64
    target_sample_path = f"{TARGET_DIR}/sample_{target_idx}.pkl"

    # Load target fingerprint, smiles, latents, active assay cols (if any)
    should_plot_assay_preds, should_plot_cluster_preds = True, True
    target_smi, target_fp, target_struct_latent, target_morph_latent, target_joint_latent, target_reward =\
        load_target_from_path(target_sample_path, mmc_model, target_mode=TARGET_MODE)
    target_active_assay_cols = get_active_assay_cols(assay_dataset, target_smi)
    if target_active_assay_cols == None: should_plot_assay_preds = False
    else: target_active_assay_cols = target_active_assay_cols.tolist()
    target_cluster_id = cluster_labels.loc[target_smi]["Activity"]
    
    target_assay_preds = predict_assay_logits_from_smi(None, [target_smi], assay_model, target_active_assay_cols, save_preds=False, skip=(not should_plot_assay_preds))
    target_active_cluster_pred = predict_cluster_logits_from_smi(None, [target_smi], cluster_model, target_cluster_id, save_preds=False, skip=False)
    
    # Only keep target active assay cols where the target assay pred is > 0.5
    if should_plot_assay_preds:
        if type(target_active_assay_cols) != list:
            target_active_assay_cols = [target_active_assay_cols]

        if len(target_active_assay_cols) == 1:
            target_active_assay_cols = target_active_assay_cols if target_assay_preds > ASSAY_CUTOFF else []
        elif len(target_active_assay_cols) > 1:
            target_active_assay_cols = [c for (c, p) in zip(target_active_assay_cols, target_assay_preds) if p > ASSAY_CUTOFF]
        if len(target_active_assay_cols) == 0:
            print(f"Skipping target {target_idx} as it has no active assay cols with pred > 0.5")
            should_plot_assay_preds = False
    
    # Only keep target cluster cols where the target cluster pred
    if target_cluster_id == None or target_active_cluster_pred <= CLUSTER_PRED_THRESH:
        print(f"Skipping target {target_idx} as it has no active cluster with pred > {CLUSTER_PRED_THRESH}")
        should_plot_cluster_preds = False

    if FOCUS == "assay":
        if not should_plot_assay_preds: return None, None, None, None, None
        should_plot_cluster_preds = False
    elif FOCUS == "cluster":
        if not should_plot_cluster_preds: return None, None, None, None, None
        should_plot_assay_preds = False
    elif FOCUS == "tsim":
        should_plot_assay_preds = False
        should_plot_cluster_preds = False
        
    # print(f"Target struct~{TARGET_MODE} alignment: ", target_reward)
    print(f"Processing samples for {target_idx}")
    print(f"Target smi: ", target_smi)
    print(f"Target assay {target_active_assay_cols} predicted logits: ", target_assay_preds)
    print(f"Target cluster {target_cluster_id} active logit: ", target_active_cluster_pred)
    
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
        top_k_modes_idx, top_k_modes_fps = find_modes_from_arrays(rewards, smis, fps, k=MAX_K, sim_threshold=SIM_THRESH, return_fps=True)

        # Count number of high sim to target samples
        num_high_sim_to_target_topk_rew = np.sum(tsim[top_k_reward_idx] >= SIM_TO_TARGET_THRESH)
        num_high_sim_to_target_topk_modes = np.sum(tsim[top_k_modes_idx] >= SIM_TO_TARGET_THRESH)

        # Infer assay and cluster logit predictions
        if not should_plot_assay_preds: 
            top_k_reward_assay_preds = top_k_modes_assay_preds = top_k_tsim_assay_preds = [] 
        else:
            top_k_reward_assay_preds = predict_assay_logits_from_smi(
                None, smis[top_k_reward_idx], assay_model, target_active_assay_cols,
                force_recompute=True, save_preds=False)
            top_k_modes_assay_preds = predict_assay_logits_from_smi(
                None, smis[top_k_modes_idx], assay_model, target_active_assay_cols,
                force_recompute=True, save_preds=False)
            
            # Count number of high assay preds
            num_high_assay_preds_by_rew = np.sum(top_k_reward_assay_preds >= ASSAY_PRED_THRESH, axis=-1)
            num_high_assay_preds_by_modes = np.sum(top_k_modes_assay_preds >= ASSAY_PRED_THRESH, axis=-1)
            
        if not should_plot_cluster_preds:
            top_k_reward_cluster_preds = top_k_modes_cluster_preds = top_k_tsim_cluster_preds = [] 
        else:
            top_k_reward_cluster_preds = predict_cluster_logits_from_smi(
                None, smis[top_k_reward_idx], cluster_model, target_cluster_id,
                force_recompute=True, save_preds=False).flatten()
            top_k_modes_cluster_preds = predict_cluster_logits_from_smi(
                None, smis[top_k_modes_idx], cluster_model, target_cluster_id,
                force_recompute=True, save_preds=False).flatten()
            
            num_high_cluster_preds_by_rew = np.sum(top_k_reward_cluster_preds >= CLUSTER_PRED_THRESH)
            num_high_cluster_preds_by_modes = np.sum(top_k_modes_cluster_preds >= CLUSTER_PRED_THRESH)
            
        # Save final run datum object
        run_datum["top_k_reward_idx"] = top_k_reward_idx
        
        if should_plot_assay_preds:
            run_datum["num_high_assay_preds_by_rew"] = num_high_assay_preds_by_rew
            run_datum["num_high_assay_preds_by_modes"] = num_high_assay_preds_by_modes
            
        if should_plot_cluster_preds:
            run_datum["num_high_cluster_preds_by_rew"] = [num_high_cluster_preds_by_rew]
            run_datum["num_high_cluster_preds_by_modes"] = [num_high_cluster_preds_by_modes]
            
        run_datum["top_k_modes_idx"] = top_k_modes_idx
        run_datum["full_rewards"] = full_rewards
        run_datum["full_tsim_to_target"] = full_tsim_to_target
        run_datum["num_high_sim_to_target_topk_rew"] = [num_high_sim_to_target_topk_rew]
        run_datum["num_high_sim_to_target_topk_modes"] = [num_high_sim_to_target_topk_modes]
        
        runs_datum[run_name] = run_datum
        print(f"{run_name} num high sim to target topk rew: ", num_high_sim_to_target_topk_rew)
        print(f"{run_name} num high sim to target topk modes: ", num_high_sim_to_target_topk_modes)
        
        if should_plot_assay_preds:
            print(f"{run_name} num high assay preds by rew: ", num_high_assay_preds_by_rew)
            print(f"{run_name} num high assay preds by modes: ", num_high_assay_preds_by_modes)
            
        if should_plot_cluster_preds:
            print(f"{run_name} num high cluster preds by rew: ", num_high_cluster_preds_by_rew)
            print(f"{run_name} num high cluster preds by modes: ", num_high_cluster_preds_by_modes)
            
        print()

    return runs_datum, target_fp, target_reward, target_active_assay_cols, target_cluster_id


def merge(runs_datum, joint_datum, keys_to_flatten=[]):
    """Merge runs datum runs into joint_datum"""
    for run_name, run_datum in runs_datum.items():
        tmp_datum = joint_datum[run_name] if run_name in joint_datum else {}
        for k, v in run_datum.items():
            # if k in keys_to_flatten:
            #     v = v.reshape(1, -1)
            if type(v) == list and len(v) > 0:
                tmp_datum[k] = tmp_datum[k] + v if k in tmp_datum else v
            else:
                # tmp_datum[k] = np.hstack([tmp_datum[k], v]) if k in tmp_datum else v
                tmp_datum[k] = np.concatenate([tmp_datum[k], v]) if k in tmp_datum else v
        joint_datum[run_name] = tmp_datum
    return joint_datum


def go(joint_datum, num_runs, target_fp=None, target_rew=None,
       assay_cols=None, cluster_id=None, save_dir=None, is_joint=False):
    plot_format = "pdf"
    os.makedirs(save_dir, exist_ok=True)
    
    plot_tsim_and_reward_full_hist(joint_datum, bins=50, is_joint=True,
                                   rew_key="full_rewards", sim_key="full_tsim_to_target",
                                   save_path=f"{save_dir}/full_tsim_reward_hist.{plot_format}")
    plot_tsim_between_modes_and_to_target(joint_datum, is_joint=is_joint, 
                                          k1=MAX_K*num_runs, k2=100, bins=50, 
                                          save_path=f"{save_dir}/tsim_modes_hist.{plot_format}")
    
    if is_joint:
        plot_pooled_boxplot_sim_and_rew(joint_datum, nbins1=15, nbins2=15, nsamples1=1000, nsamples2=1000,
                                        save_path=f"{save_dir}/pooled_boxplot_sim_rew.{plot_format}")
        plot_unpooled_boxplot_sim_and_rew(joint_datum, bins1=15, bins2=15, n1=1000, n2=1000, ignore=[],
                                        save_path=f"{save_dir}/unpooled_boxplot_sim_rew.{plot_format}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--norm", action="store_true", help="Normalize assay and cluster predictions")
    parser.add_argument("--plot_individual", action="store_true", help="Plot individual runs")
    parser.add_argument("--target_mode", type=str, default="morph", help="Target mode to use")
    parser.add_argument("--config_name", type=str, default="cluster_morph.json", help="JSON config to use")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--max_k", type=int, default=5000, help="Max number of modes to consider")
    parser.add_argument("--run_name", type=str, default="cluster-morph", help="Run name to use")
    parser.add_argument("--assay_cutoff", type=float, default=0.5, help="Assay cutoff for active cols")
    parser.add_argument("--ignore_targets", type=str, default="", help="Comma-separated list of targets to ignore")
    parser.add_argument("--save_dir", type=str, default="~/plots", help="Save directory for plots")
    parser.add_argument("--keep_every", type=int, default=8, help="Keep every k samples from the run")
    parser.add_argument("--run_dir", type=str, default=os.getenv("RUNS_DIR_PATH"), help="Run directory for runs")
    parser.add_argument("--focus", type=str, default="assay", help="Focus on assay or cluster preds")
    parser.add_argument("--sim_thresh", type=float, default=0.3, help="Similarity threshold for mode finding")
    parser.add_argument("--cluster_pred_thresh", type=float, default=0.3, help="Used to count number of high cluster preds")
    parser.add_argument("--assay_pred_thresh", type=float, default=0.7, help="Used to count number of high assay preds")
    parser.add_argument("--sim_to_target_thresh", type=float, default=0.2, help="Used to count number of highly similar to target samples")

    args = parser.parse_args()
    RUN_NAME = args.run_name
    CONFIG_NAME = args.config_name
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
    CLUSTER_PRED_THRESH = args.cluster_pred_thresh
    ASSAY_PRED_THRESH = args.assay_pred_thresh
    SIM_TO_TARGET_THRESH = args.sim_to_target_thresh
    
    # Load runs from JSON config
    with open(f"../runs/{CONFIG_NAME}") as f:
        RUNS = json.load(f)

    # Load models and ground truth data
    assay_dataset, cluster_labels, assay_model, cluster_model, mmc_model = load_puma()
        
    NUM_RUNS = 0
    joint_datum = {}
    for run in RUNS:
        target_idx = run['target_idx']
        if str(target_idx).strip() in IGNORE_TARGETS: continue
        runs_datum, target_fp, target_rew, assay_cols, cluster_id = load_run(run)

        if runs_datum is None: continue
        NUM_RUNS += 1

        print(f"Finished plotting for {target_idx}. Now merging into joint datum...")
        joint_datum = merge(runs_datum, joint_datum)

    print(f"Processed {NUM_RUNS} runs with proper proxy alignment")
    print(joint_datum.keys())

    # Produce plots for the joint runs
    save_dir = f"{SAVEDIR}/aggr-{RUN_NAME}"
    if args.norm: save_dir += "-norm"
    go(joint_datum, NUM_RUNS, save_dir=save_dir, is_joint=True)