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
    # mmc_model = load_mmc_model(cfg)

    # Run inference on PUMA test split
    # Get full PUMA dataset latents, fingerprints, and smiles
    # datamodule, cfg = setup_puma()
    # representations = get_representations()
    # dataset_smis = np.array([x["inputs"]["struct"].mols for x in datamodule.dataset])
    # dataset_fps = load_puma_dataset_fps(dataset_smis, save_fps=True)
    # _, _, test_idx = datamodule.get_split_idx()
    # puma_test_assay_preds = predict_assay_logits_from_smi(None, dataset_smis[test_idx], assay_model, None, save_preds=False)
    # puma_test_cluster_preds = predict_cluster_logits_from_smi(None, dataset_smis[test_idx], cluster_model, None, save_preds=False, use_gneprop=False)
    return assay_dataset, cluster_labels, assay_model, cluster_model, mmc_model


def load_run(run):
    target_idx, run_paths, rew_thresh = run["target_idx"], run["run_paths"], run["reward_thresh"]
    bs = run["bs"] if "bs" in run.keys() else 64
    target_sample_path = f"/home/mila/s/stephen.lu/gfn_gene/res/mmc/targets/sample_{target_idx}.pkl"

    # Load target fingerprint, smiles, latents, active assay cols (if any)
    should_plot_assay_preds, should_plot_cluster_preds = True, True
    target_smi, target_fp, target_struct_latent, target_morph_latent, target_joint_latent, target_reward =\
        load_target_from_path(target_sample_path, mmc_model, target_mode=TARGET_MODE)
    target_active_assay_cols = get_active_assay_cols(assay_dataset, target_smi)
    if target_active_assay_cols == None: should_plot_assay_preds = False
    else: target_active_assay_cols = target_active_assay_cols.tolist()
    target_cluster_id = cluster_labels.loc[target_smi]["Activity"]
    # target_latent = target_morph_latent if TARGET_MODE == "morph" else target_joint_latent
    # print(target_active_assay_cols, target_cluster_id)

    # # Infer target & dataset reward, target assay logits, and target cluster logits
    # target_reward = cosine_similarity(target_struct_latent, target_latent)[0][0]
    # dataset_rewards = ((cosine_similarity(representations['struct'], target_latent) + 1) / 2).reshape(-1,)
    # dataset_sim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, dataset_fps))
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

    # Only keep target cluster cols where the target cluster pred is > 0.01
    # if target_cluster_id == None or target_active_cluster_pred <= 0.01:
    #     print(f"Skipping target {target_idx} as it has no active cluster with pred > 0.01")
    #     should_plot_cluster_preds = False

    if FOCUS == "assay":
        should_plot_cluster_preds = False
    elif FOCUS == "cluster":
        should_plot_assay_preds = False
        
    # print(f"Target struct~{TARGET_MODE} alignment: ", target_reward)
    print(f"Processing samples for {target_idx}")
    print(f"Target smi: ", target_smi)
    print(f"Target assay {target_active_assay_cols} predicted logits: ", target_assay_preds)
    print(f"Target cluster {target_cluster_id} active logit: ", target_active_cluster_pred)
    
    # Load baseline data for runs
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        run_path = os.path.join(RUNDIR, run_id)
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

        # Get number of modes and avg reward over trajs
        # bsl = bs if type(bs) == int else bs[run_name]
        # num_modes, avg_rew = num_modes_lazy(run_datum, rew_thresh, 0.7, bsl)

        # Remove duplicates
        run_datum = remove_duplicates_from_run(run_datum)
        rewards, tsim, smis, fps = run_datum["rewards"], run_datum["tsim_to_target"],\
            run_datum["smis"], run_datum["fps"]
        assert len(rewards) == len(smis) == len(fps) == len(tsim)

        # Compute top-k modes by reward and by sim
        top_k_reward_idx = np.argsort(rewards)[::-1][:MAX_K]
        top_k_reward_fps = [fps[j] for j in top_k_reward_idx]
        top_k_reward_tsim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, top_k_reward_fps))
        top_k_modes_idx, top_k_modes_fps = find_modes_from_arrays(rewards, smis, fps, k=MAX_K, sim_threshold=SIM_THRESH, return_fps=True)
        top_k_tsim_idx = np.argsort(tsim)[::-1][:MAX_K]
        top_k_tsim_to_target = tsim[top_k_tsim_idx]

        # Compute Tanimoto Sim between top-k highest reward molecules
        top_sk_reward_fps = top_k_reward_fps[:100]
        tani_sim_between_modes = []
        for i in tqdm(range(len(top_sk_reward_fps))):
            tani_sim_between_modes.extend(AllChem.DataStructs.BulkTanimotoSimilarity(top_sk_reward_fps[i], top_sk_reward_fps[i+1:]))

        # Infer assay and cluster logit predictions
        if not should_plot_assay_preds: 
            top_k_reward_assay_preds = top_k_modes_assay_preds = top_k_tsim_assay_preds = [] 
        else:
            top_k_reward_assay_preds = predict_assay_logits_from_smi(
                None, smis[top_k_reward_idx], assay_model, target_active_assay_cols,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP)
            top_k_modes_assay_preds = predict_assay_logits_from_smi(
                None, smis[top_k_modes_idx], assay_model, target_active_assay_cols,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP)
            top_k_tsim_assay_preds = predict_assay_logits_from_smi(
                None, smis[top_k_tsim_idx], assay_model, target_active_assay_cols,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP)
        if not should_plot_cluster_preds:
            top_k_reward_cluster_preds = top_k_modes_cluster_preds = top_k_tsim_cluster_preds = [] 
        else:
            top_k_reward_cluster_preds = predict_cluster_logits_from_smi(
                None, smis[top_k_reward_idx], cluster_model, target_cluster_id,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP).flatten()
            top_k_modes_cluster_preds = predict_cluster_logits_from_smi(
                None, smis[top_k_modes_idx], cluster_model, target_cluster_id,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP).flatten()
            top_k_tsim_cluster_preds = predict_cluster_logits_from_smi(
                None, smis[top_k_tsim_idx], cluster_model, target_cluster_id,
                force_recompute=True, save_preds=False, use_gneprop=USE_GNEPROP).flatten()
        
        # Save final run datum object
        run_datum["top_k_reward_idx"] = top_k_reward_idx
        run_datum["top_k_reward_fps"] = top_k_reward_fps
        run_datum["top_k_reward_tsim_to_target"] = top_k_reward_tsim_to_target
        run_datum["top_k_cross_tsim"] = tani_sim_between_modes
        if should_plot_assay_preds:
            run_datum["top_k_reward_assay_preds"] = top_k_reward_assay_preds
            run_datum["top_k_modes_assay_preds"] = top_k_modes_assay_preds
            run_datum["top_k_tsim_assay_preds"] = top_k_tsim_assay_preds
        if should_plot_cluster_preds:
            run_datum["top_k_tsim_cluster_preds"] = top_k_tsim_cluster_preds
            run_datum["top_k_modes_cluster_preds"] = top_k_modes_cluster_preds
            run_datum["top_k_reward_cluster_preds"] = top_k_reward_cluster_preds
        run_datum["top_k_modes_idx"] = top_k_modes_idx
        run_datum["top_k_modes_fps"] = top_k_modes_fps
        run_datum["top_k_tsim_idx"] = top_k_tsim_idx
        run_datum["full_rewards"] = full_rewards
        run_datum["full_tsim_to_target"] = full_tsim_to_target
        # run_datum["num_modes_over_trajs"] = num_modes
        # run_datum["avg_reward_over_trajs"] = avg_rew
        runs_datum[run_name] = run_datum
        
        print(f"{run_name} tsim between modes: ", np.mean(tani_sim_between_modes), np.quantile(tani_sim_between_modes, 0.75), np.max(tani_sim_between_modes))
        print(f"{run_name} topk tsim to target: ", np.mean(top_k_reward_tsim_to_target), np.quantile(top_k_reward_tsim_to_target, 0.75), np.max(top_k_reward_tsim_to_target))
        print(f"{run_name} full tsim to target: ", np.mean(full_tsim_to_target), np.quantile(full_tsim_to_target, 0.75), np.max(full_tsim_to_target))
        print(f"{run_name} topk tsim: ", np.min(top_k_tsim_to_target), np.mean(top_k_tsim_to_target), np.max(top_k_tsim_to_target))
        if should_plot_assay_preds:
            print(f"{run_name} topk rew assay preds: ", np.mean(top_k_reward_assay_preds, axis=-1), np.quantile(top_k_reward_assay_preds, 0.75, axis=-1), np.max(top_k_reward_assay_preds, axis=-1))
            print(f"{run_name} topk modes assay preds: ", np.mean(top_k_modes_assay_preds, axis=-1), np.quantile(top_k_modes_assay_preds, 0.75, axis=-1), np.max(top_k_modes_assay_preds, axis=-1))
            print(f"{run_name} topk tsim assay preds: ", np.mean(top_k_tsim_assay_preds, axis=-1), np.quantile(top_k_tsim_assay_preds, 0.75, axis=-1), np.max(top_k_tsim_assay_preds, axis=-1))
        if should_plot_cluster_preds:
            print(f"{run_name} topk rew cluster preds: ", np.mean(top_k_reward_cluster_preds), np.quantile(top_k_reward_cluster_preds, 0.75), np.max(top_k_reward_cluster_preds))
            print(f"{run_name} topk modes cluster preds: ", np.mean(top_k_modes_cluster_preds), np.quantile(top_k_modes_cluster_preds, 0.75), np.max(top_k_modes_cluster_preds))
            print(f"{run_name} topk tsim cluster preds: ", np.mean(top_k_tsim_cluster_preds), np.quantile(top_k_tsim_cluster_preds, 0.75), np.max(top_k_tsim_cluster_preds))
        print()

    # runs_datum["PUMA_test"] = {
    #     "smis": dataset_smis[test_idx],
    #     "rewards": dataset_rewards[test_idx],
    #     "mols": list(map(Chem.MolFromSmiles, dataset_smis[test_idx])),
    #     "fps": [dataset_fps[j] for j in test_idx],
    #     "tsim_to_target": dataset_sim_to_target[test_idx],
    #     "top_k_reward_idx": np.arange(len(test_idx)),
    #     "top_k_reward_fps": [dataset_fps[j] for j in test_idx],
    #     "top_k_reward_tsim_to_target": dataset_sim_to_target[test_idx],
    #     "top_k_reward_assay_preds": puma_test_assay_preds,
    #     "top_k_reward_cluster_preds": puma_test_cluster_preds,
    #     "top_k_modes_idx": np.arange(len(test_idx)),
    #     "top_k_modes_fps": [dataset_fps[j] for j in test_idx],
    # }
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
    # PLOTS THAT REQUIRE DUPLICATES OVER THE ENTIRE TRAINING SAMPLES
    # plot_modes_over_trajs(joint_datum, rew_thresh=rew_thresh, sim_thresh=0.7, bs=bs, ignore=["PUMA_test"], 
    #                     save_path=f"{save_dir}/num_modes_{rew_thresh}.{plot_format}")
    plot_tsim_and_reward_full_hist(joint_datum, bins=50,
                                   rew_key="full_rewards", sim_key="full_tsim_to_target",
                                   save_path=f"{save_dir}/full_tsim_reward_hist.{plot_format}")
    
    # PLOTS THAT REQUIRE DUPLICATES REMOVED
    if FOCUS == "cluster":
        plot_cluster_preds_hist(joint_datum, cluster_id=cluster_id, k=MAX_K*num_runs, bins=50, ignore=["PUMA_test"],
                                save_path=f"{save_dir}/cluster_preds_hist.{plot_format}")
    elif FOCUS == "assay":
        plot_assay_preds_hist(joint_datum, assay_cols=assay_cols, k=MAX_K*num_runs, bins=50, ignore=["PUMA_test"],
                                save_path=f"{save_dir}/assay_cluster_preds_hist.{plot_format}")
        
    plot_tsim_between_modes_and_to_target(joint_datum, k1=MAX_K*num_runs, k2=100, bins=50, 
                                    save_path=f"{save_dir}/tsim_modes_hist.{plot_format}")
    if is_joint:
        plot_pooled_boxplot_sim_and_rew(joint_datum, nbins1=15, nbins2=15, nsamples1=1000, nsamples2=1000,
                                        save_path=f"{save_dir}/pooled_boxplot_sim_rew.{plot_format}")
        plot_unpooled_boxplot_sim_and_rew(joint_datum, bins1=15, bins2=15, n1=1000, n2=1000, ignore=[],
                                        save_path=f"{save_dir}/unpooled_boxplot_sim_rew.{plot_format}")
    # if target_fp != None:
    #     plot_umap_from_runs_datum(joint_datum, target_fp, target_rew, n_neigh=30, k=MAX_K,
    #                                 ignore=["PUMA_test"], save_path=f"{save_dir}/umap.{plot_format}")
    
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
    parser.add_argument("--save_dir", type=str, default="/home/mila/s/stephen.lu/scratch/plots", help="Save directory for plots")
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
        runs_datum, target_fp, target_rew, assay_cols, cluster_id = load_run(run)
        if args.norm and FOCUS == "cluster":
            runs_datum = minmax_norm_datum(runs_datum, cols=[
                "top_k_reward_cluster_preds",
                "top_k_modes_cluster_preds",
                "top_k_tsim_cluster_preds"
            ])
        if runs_datum is None: continue
        NUM_RUNS += 1

        if args.plot_individual:
            print(f"Finished loading data for target {target_idx}. Now plotting individual plots...")
            save_dir = f"{SAVEDIR}/{target_idx}-{RUN_NAME}"
            if USE_GNEPROP: save_dir += "-gneprop"
            if args.norm: save_dir += "-norm"
            go(runs_datum, 1, target_fp=target_fp, target_rew=target_rew,
               assay_cols=assay_cols, cluster_id=cluster_id, save_dir=save_dir)

        print(f"Finished plotting for {target_idx}. Now merging into joint datum...")
        joint_datum = merge(runs_datum, joint_datum, keys_to_flatten=[
            "top_k_reward_assay_preds",
            "top_k_modes_assay_preds",
            "top_k_tsim_assay_preds",
        ])

    print(f"Processed {NUM_RUNS} runs with proper proxy alignment")
    print(joint_datum.keys())

    # Produce plots for the joint runs
    save_dir = f"{SAVEDIR}/aggr-{RUN_NAME}"
    if USE_GNEPROP: save_dir += "-gneprop"
    if args.norm: save_dir += "-norm"
    go(joint_datum, NUM_RUNS, save_dir=save_dir, is_joint=True)