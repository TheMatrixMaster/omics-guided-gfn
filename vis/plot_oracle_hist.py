import os
import argparse
from utils import *
from plotting import *
import json

TARGET_DIR = os.getenv("TARGETS_DIR_PATH")


def load_sm_run(run, num_samples=None, every_k=None):
    # Load baseline data for runs
    run_paths = run["run_paths"]
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        full_fps, full_rewards, full_smis = load_datum_from_run(RUNDIR, run_id, remove_duplicates=False,
                                                                fps_from_file=False, save_fps=False,
                                                                last_k=num_samples, every_k=every_k)
        # Create run_datum object with duplicates
        run_datum = { "smis": full_smis, "rewards": full_rewards, "fps": full_fps }
        assert len(full_smis) == len(full_rewards) == len(full_fps)
        runs_datum[run_name] = run_datum
    return runs_datum


def compute_rew_thresh(run, n=5000, percentile=90):
    runs_datum_sm = load_sm_run(run, num_samples=n)
    merged_rewards = []
    for run_name, run_datum in runs_datum_sm.items():
        merged_rewards.append(run_datum["rewards"])
    merged_rewards = np.hstack(merged_rewards)
    return round(np.percentile(merged_rewards, percentile), 2)


def load_puma():
    # Load models and ground truth data
    assay_dataset = load_assay_matrix_from_csv()
    cluster_labels = load_cluster_labels_from_csv()
    assay_model = load_assay_pred_model(use_gneprop=USE_GNEPROP)
    cluster_model = load_cluster_pred_model(use_gneprop=USE_GNEPROP)
    mmc_model = None
    return assay_dataset, cluster_labels, assay_model, cluster_model, mmc_model


def plot_cluster_preds_hist(runs_datum, cluster_id=None, k=10000, bins=50, is_joint=False,
                            ignore=[], plot_all=True, save_path="cluster_preds_hist.pdf"):
    clabel = cluster_id if cluster_id != None else "Aggregated"
    fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
    dfc = pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        mod_cp = run_datum['modes_cluster_preds']
        dfc = pd.concat([dfc, pd.DataFrame({
            "method": [run_name] * len(mod_cp),
            "value": mod_cp,
        })])
    if len(dfc) > 0:
        sns.histplot(data=dfc, x="value", hue="method", bins=bins, 
                     ax=ax2, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
    if is_joint:
        ax2.set_title(f"Aggregated Cluster Activity Predictions of Top {k}\nModes per Target")
    else:
        ax2.set_title(f"Cluster ({clabel}) Activity Predictions of Top {k} Modes")
    ax2.set_xlabel("Probability")
    ax2.set_yscale('log')
    sns.move_legend(ax2, "lower left")
    plt.tight_layout()
    fig2.savefig(save_path.replace(".pdf", "_mod.pdf"), dpi=300)


def plot_assay_preds_hist(runs_datum, assay_cols=[], k=10000, bins=50, is_joint=False,
                          ignore=[], plot_all=True, save_path="assay_cluster_preds_hist.pdf"):
    assay_cols = [None] if assay_cols == None else assay_cols
    n = None if plot_all else k
    for i, col in enumerate(assay_cols):
        fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
        dfa = pd.DataFrame()
        col_label = col if col != None else "Aggregated"
        for run_name, run_datum in runs_datum.items():
            if run_name in ignore: continue
            mod_ap = run_datum['modes_assay_preds'][i]
            dfa = pd.concat([dfa, pd.DataFrame({
                "method": [run_name] * len(mod_ap),
                "value": mod_ap,
            })])
        if len(dfa) > 0:
            sns.histplot(data=dfa, x="value", hue="method", bins=bins,
                         ax=ax2, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
        if is_joint:
            ax2.set_title(f"Aggregated Assay Activity Predictions of Top {k}\nModes per Target")
        else:
            ax2.set_title(f"Assay ({col_label}) Activity Predictions of Top {k} Modes")
        ax2.set_xlabel("Probability")
        ax2.set_yscale('log')
        sns.move_legend(ax2, "lower left")
        plt.tight_layout()
        fig2.savefig(save_path.replace(".pdf", f"_mod_{col}.pdf"), dpi=300)


def load_run(run, rew_thresh=None, every_k=8):
    target_idx, run_paths, _ = run["target_idx"], run["run_paths"], run["reward_thresh"]
    target_sample_path = f"{TARGET_DIR}/sample_{target_idx}.pkl"

    # Load target fingerprint, smiles, latents, active assay cols (if any)
    should_plot_assay_preds, should_plot_cluster_preds = True, True
    target_smi, target_fp, target_struct_latent, target_morph_latent, target_joint_latent, target_reward =\
        load_target_from_path(target_sample_path, mmc_model, target_mode=TARGET_MODE)
    target_active_assay_cols = get_active_assay_cols(assay_dataset, target_smi)
    if target_active_assay_cols == None: should_plot_assay_preds = False
    else: target_active_assay_cols = target_active_assay_cols.tolist()
    target_cluster_id = cluster_labels.loc[target_smi]["Activity"]
    target_assay_preds = predict_assay_logits_from_smi(None, [target_smi], assay_model, target_active_assay_cols, save_preds=False, use_gneprop=USE_GNEPROP, skip=(not should_plot_assay_preds))
    target_active_cluster_pred = predict_cluster_logits_from_smi(None, [target_smi], cluster_model, target_cluster_id, save_preds=False, use_gneprop=USE_GNEPROP, skip=False)
    
    # Only keep target active assay cols where the target assay pred is > 0.5
    if should_plot_assay_preds:
        if type(target_active_assay_cols) != list:
            target_active_assay_cols = [target_active_assay_cols]
        if USE_GNEPROP:
            if 2 not in target_active_assay_cols:
                print(f"Skipping target {target_idx} as it has no active assay cols covered by GNEPROP")
                should_plot_assay_preds = False
            else: target_active_assay_cols = [2]
        if len(target_active_assay_cols) == 1:
            target_active_assay_cols = target_active_assay_cols if target_assay_preds > ASSAY_CUTOFF else []
        elif len(target_active_assay_cols) > 1:
            target_active_assay_cols = [c for (c, p) in zip(target_active_assay_cols, target_assay_preds) if p > ASSAY_CUTOFF]
        if len(target_active_assay_cols) == 0:
            print(f"Skipping target {target_idx} as it has no active assay cols with pred > 0.5")
            should_plot_assay_preds = False
    
    if FOCUS == "assay": should_plot_cluster_preds = False
    elif FOCUS == "cluster": should_plot_assay_preds = False
        
    print(f"Processing samples for {target_idx}")
    print(f"Target smi: ", target_smi)
    print(f"Target assay {target_active_assay_cols} predicted logits: ", target_assay_preds)
    print(f"Target cluster {target_cluster_id} active logit: ", target_active_cluster_pred)
    
    # Load baseline data for runs
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        full_fps, full_rewards, full_smis = load_datum_from_run(RUNDIR, run_id, remove_duplicates=False,
                                                                fps_from_file=False, save_fps=False,
                                                                every_k=every_k)
        run_datum = { "smis": full_smis, "rewards": full_rewards, "fps": full_fps }
        num_modes, avg_rew, modes_smis_local = num_modes_lazy(run_datum, rew_thresh, SIM_THRESH, 
                                                        bs=64//every_k, return_smis=True)
        print(f"Run {run_name} has {num_modes[-1]} modes with average reward {avg_rew[-1]}")
        
        # Infer assay and cluster logit predictions
        if not should_plot_assay_preds or len(modes_smis_local) == 0: 
            modes_assay_preds = []
        else:
            modes_assay_preds = predict_assay_logits_from_smi(
                None, modes_smis_local, assay_model, target_active_assay_cols,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP)
        
        if not should_plot_cluster_preds:
            modes_cluster_preds = [] 
        else:
            modes_cluster_preds = predict_cluster_logits_from_smi(
                None, modes_smis_local, cluster_model, target_cluster_id,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP).flatten()
        
        # Save final run datum object
        run_datum = {}
        if should_plot_assay_preds and len(modes_assay_preds) > 0:
            run_datum["modes_assay_preds"] = modes_assay_preds
        if should_plot_cluster_preds and len(modes_cluster_preds) > 0:
            run_datum["modes_cluster_preds"] = modes_cluster_preds

        runs_datum[run_name] = run_datum
        
        if should_plot_assay_preds and len(modes_assay_preds) > 0:
            print(f"{run_name} modes assay preds: ", np.mean(modes_assay_preds, axis=-1), np.quantile(modes_assay_preds, 0.75, axis=-1), np.max(modes_assay_preds, axis=-1))
        if should_plot_cluster_preds and len(modes_cluster_preds) > 0:
            print(f"{run_name} modes cluster preds: ", np.mean(modes_cluster_preds), np.quantile(modes_cluster_preds, 0.75), np.max(modes_cluster_preds))
        print()

    return runs_datum, target_fp, target_reward, target_active_assay_cols, target_cluster_id


def minmax_norm_datum(run_datum, cols=[]):
    """Min-max normalize the datum"""
    col_vals = {}
    for _, run_datum in runs_datum.items():
        for col in cols:
            if col not in col_vals: col_vals[col] = []
            if col in run_datum: col_vals[col].extend(run_datum[col])
    for col in cols:
        col_vals[col] = np.array(col_vals[col])
        col_vals[col] = (col_vals[col] - np.min(col_vals[col])) / (np.max(col_vals[col]) - np.min(col_vals[col]))
        for _, run_datum in runs_datum.items():
            if col not in run_datum: continue
            run_datum[col] = col_vals[col][:len(run_datum[col])]
            col_vals[col] = col_vals[col][len(run_datum[col]):]
    return runs_datum


def merge(runs_datum, joint_datum, keys_to_flatten=[]):
    """Merge runs datum runs into joint_datum"""
    for run_name, run_datum in runs_datum.items():
        tmp_datum = joint_datum[run_name] if run_name in joint_datum else {}
        for k, v in run_datum.items():
            if k in keys_to_flatten:
                v = v.reshape(1, -1)
            if type(v) == list and len(v) > 0:
                tmp_datum[k] = tmp_datum[k] + v if k in tmp_datum else v
            else:
                tmp_datum[k] = np.hstack([tmp_datum[k], v]) if k in tmp_datum else v
        joint_datum[run_name] = tmp_datum
    return joint_datum


def go(joint_datum, num_runs, target_fp=None, target_rew=None,
       assay_cols=None, cluster_id=None, save_dir=None, is_joint=False):
    plot_format = "pdf"
    os.makedirs(save_dir, exist_ok=True)
    if FOCUS == "cluster":
        plot_cluster_preds_hist(joint_datum, cluster_id=cluster_id, is_joint=is_joint, 
                                k=MAX_K*num_runs, bins=50, ignore=["PUMA_test"],
                                save_path=f"{save_dir}/cluster_preds_hist.{plot_format}")
    elif FOCUS == "assay":
        plot_assay_preds_hist(joint_datum, assay_cols=assay_cols, is_joint=is_joint, 
                              k=MAX_K*num_runs, bins=50, ignore=["PUMA_test"],
                              save_path=f"{save_dir}/assay_cluster_preds_hist.{plot_format}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--norm", action="store_true", help="Normalize assay and cluster predictions")
    parser.add_argument("--plot_individual", action="store_true", help="Plot individual runs")
    parser.add_argument("--use_gneprop", action="store_true", help="Use GNEPROP for assay preds")
    parser.add_argument("--target_mode", type=str, default="morph", help="Target mode to use")
    parser.add_argument("--config_name", type=str, default="cluster_morph.json", help="JSON config to use")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--max_k", type=int, default=5000, help="Max number of modes to consider")
    parser.add_argument("--run_name", type=str, default="cluster-morph", help="Run name to use")
    parser.add_argument("--assay_cutoff", type=float, default=0.5, help="Assay cutoff for active cols")
    parser.add_argument("--ignore_targets", type=str, default="", help="Comma-separated list of targets to ignore")
    parser.add_argument("--save_dir", type=str, default="~/plots", help="Save directory for plots")
    parser.add_argument("--run_dir", type=str, default=os.getenv("RUNS_DIR_PATH"), help="Run directory for runs")
    parser.add_argument("--focus", type=str, default="assay", help="Focus on assay or cluster preds")
    parser.add_argument("--sim_thresh", type=float, default=0.7, help="Similarity threshold for mode finding")
    parser.add_argument("--reward_percentile", type=int, default=90, help="Reward percentile for thresholding")

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
    SIM_THRESH = args.sim_thresh
    
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
        save_dir = f"{SAVEDIR}/{target_idx}-{RUN_NAME}"
        os.makedirs(save_dir, exist_ok=True)
        
        # rew_thresh = None
        rew_thresh = compute_rew_thresh(run, percentile=args.reward_percentile)
        print(f"Computed reward threshold for {target_idx} as {rew_thresh}")

        runs_datum, target_fp, target_rew, assay_cols, cluster_id = load_run(run, rew_thresh)
        if args.norm and FOCUS == "cluster":
            runs_datum = minmax_norm_datum(runs_datum, cols=["modes_cluster_preds"])
            
        if runs_datum is None: continue
        NUM_RUNS += 1

        if args.plot_individual:
            print(f"Finished loading data for target {target_idx}. Now plotting individual plots...")
            if USE_GNEPROP: save_dir += "-gneprop"
            if args.norm: save_dir += "-norm"
            go(runs_datum, 1, target_fp=target_fp, target_rew=target_rew,
               assay_cols=assay_cols, cluster_id=cluster_id, save_dir=save_dir)

        print(f"Finished plotting for {target_idx}. Now merging into joint datum...")
        joint_datum = merge(runs_datum, joint_datum, keys_to_flatten=["modes_assay_preds"])

    print(f"Processed {NUM_RUNS} runs with proper proxy alignment")
    print(joint_datum.keys())

    # Produce plots for the joint runs
    save_dir = f"{SAVEDIR}/aggr-{RUN_NAME}"
    if USE_GNEPROP: save_dir += "-gneprop"
    if args.norm: save_dir += "-norm"
    go(joint_datum, NUM_RUNS, save_dir=save_dir, is_joint=True)