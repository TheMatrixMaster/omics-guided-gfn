"""
This script produces all the relevant plots for the gflownet analysis
"""

from utils import *
from plotting import *
import json

MAX_K = 5000
TARGET_MODE = "morph"
RUNDIR = "/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps"
# SAVEDIR = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/plots"
SAVEDIR = "/home/mila/s/stephen.lu/scratch/plots"

# Get full PUMA dataset latents, fingerprints, and smiles
datamodule, cfg = setup_puma()
representations = get_representations()
dataset_smis = np.array([x["inputs"]["struct"].mols for x in datamodule.dataset])
dataset_fps = load_puma_dataset_fps(dataset_smis, save_fps=True)

# Load models and ground truth data
assay_dataset = load_assay_matrix_from_csv()
assay_model = load_assay_pred_model().to(device)
cluster_labels = load_cluster_labels_from_csv()
cluster_model = load_cluster_pred_model().to(device)
mmc_model = load_mmc_model(cfg).to(device)

# Run inference on PUMA test split
_, _, test_idx = datamodule.get_split_idx()
puma_test_assay_preds = predict_assay_logits_from_smi(None, dataset_smis[test_idx], assay_model, None, save_preds=False)
puma_test_cluster_preds = predict_cluster_logits_from_smi(None, dataset_smis[test_idx], cluster_model, None, save_preds=False, use_gneprop=False)

def go(run):
    target_idx, run_paths, rew_thresh = run["target_idx"], run["run_paths"], run["reward_thresh"]
    bs = run["bs"] if "bs" in run.keys() else 64
    print(f"Producing plots for target {target_idx} under mode {TARGET_MODE}...")
    target_sample_path = f"/home/mila/s/stephen.lu/gfn_gene/res/mmc/targets/sample_{target_idx}.pkl"
    save_dir = f"{SAVEDIR}/{target_idx}"
    os.makedirs(save_dir, exist_ok=True)

    # Load target fingerprint, smiles, latents, active assay cols (if any)
    target_smi, target_fp, target_struct_latent, target_morph_latent, target_joint_latent, target_reward =\
        load_target_from_path(target_sample_path, mmc_model, target_mode=TARGET_MODE)
    target_active_assay_cols = get_active_assay_cols(assay_dataset, target_smi)
    target_cluster_id = cluster_labels.loc[target_smi]["Activity"]
    target_latent = target_morph_latent if TARGET_MODE == "morph" else target_joint_latent
    print(target_active_assay_cols, target_cluster_id)

    # Infer target & dataset reward, target assay logits, and target cluster logits
    target_reward = cosine_similarity(target_struct_latent, target_latent)[0][0]
    dataset_rewards = ((cosine_similarity(representations['struct'], target_latent) + 1) / 2).reshape(-1,)
    dataset_sim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, dataset_fps))
    target_assay_preds = predict_assay_logits_from_smi(None, [target_smi], assay_model, target_active_assay_cols, save_preds=False)
    target_active_cluster_pred = predict_cluster_logits_from_smi(None, [target_smi], cluster_model, target_cluster_id, save_preds=False, use_gneprop=False)

    print(f"Target struct~{TARGET_MODE} alignment: ", target_reward)
    print(f"Target assay predicted logits: ", target_assay_preds)
    print(f"Target cluster {target_cluster_id} active logit: ", target_active_cluster_pred)

    # Load baseline data for runs
    runs_datum = {}
    for run_name, run_id in run_paths.items():
        run_path = os.path.join(RUNDIR, run_id)
        fps, rewards, smis = load_datum_from_run(RUNDIR, run_id, remove_duplicates=False,
                                                 save_fps=False, fps_from_file=False, last_k=10000)
        mols = list(map(Chem.MolFromSmiles, tqdm(smis)))
        tsim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, fps))
        run_datum = {
            "path": run_path, "smis": smis, "rewards": rewards, "mols": mols,
            "fps": fps, "tsim_to_target": tsim_to_target,
        }
        assert len(smis) == len(rewards) == len(mols) == len(fps) == len(tsim_to_target)
        runs_datum[run_name] = run_datum

    assert len(test_idx) <= MAX_K
    runs_datum["PUMA_test"] = {
        "smis": dataset_smis[test_idx],
        "rewards": dataset_rewards[test_idx],
        "mols": list(map(Chem.MolFromSmiles, dataset_smis[test_idx])),
        "fps": [dataset_fps[j] for j in test_idx],
        "tsim_to_target": dataset_sim_to_target[test_idx],
    }

    # PLOTS THAT REQUIRE DUPLICATES OVER THE ENTIRE TRAINING SAMPLES
    plot_modes_over_trajs(runs_datum, rew_thresh=rew_thresh, sim_thresh=0.7, bs=bs, ignore=["PUMA_test"], 
                        save_path=f"{save_dir}/num_modes_{rew_thresh}.png")
    plot_tsim_and_reward_full_hist(runs_datum, bins=50, save_path=f"{save_dir}/tsim_and_reward_hist.png")

    # Remove duplicates and compute top-k rewards, modes, and assay/cluster predictions for runs
    for run_name in runs_datum.keys():
        if run_name == "PUMA_test": continue
        run_datum = remove_duplicates_from_run(runs_datum[run_name])
        rewards, smis, fps = run_datum["rewards"], run_datum["smis"], run_datum["fps"]
        top_k_reward_idx = np.argsort(rewards)[::-1][:MAX_K]
        top_k_modes_idx, top_k_modes_fps = find_modes_from_arrays(rewards, smis, fps, sim_threshold=0.7, return_fps=True)
        top_k_reward_fps = [fps[j] for j in top_k_reward_idx]
        top_k_reward_tsim_to_target = np.array(AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, top_k_reward_fps))

        if not target_active_assay_cols or len(target_active_assay_cols) == 0: top_k_reward_assay_preds = []
        else: top_k_reward_assay_preds = predict_assay_logits_from_smi(run_path, smis[top_k_reward_idx], assay_model, target_active_assay_cols, force_recompute=True, save_preds=False)
        if not target_cluster_id: top_k_reward_cluster_preds = [] 
        else: top_k_reward_cluster_preds = predict_cluster_logits_from_smi(run_path, smis[top_k_reward_idx], cluster_model, target_cluster_id, force_recompute=True, save_preds=False)
        
        run_datum["top_k_reward_idx"] = top_k_reward_idx
        run_datum["top_k_reward_fps"] = top_k_reward_fps
        run_datum["top_k_reward_tsim_to_target"] = top_k_reward_tsim_to_target
        run_datum["top_k_reward_assay_preds"] = top_k_reward_assay_preds
        run_datum["top_k_reward_cluster_preds"] = top_k_reward_cluster_preds
        run_datum["top_k_modes_idx"] = top_k_modes_idx
        run_datum["top_k_modes_fps"] = top_k_modes_fps
        runs_datum[run_name] = run_datum

    # Update PUMA test split with top-k rewards, modes, and assay/cluster predictions
    runs_datum["PUMA_test"]["top_k_reward_idx"] = np.arange(len(test_idx))
    runs_datum["PUMA_test"]["top_k_reward_fps"] = [dataset_fps[j] for j in test_idx]
    runs_datum["PUMA_test"]["top_k_reward_tsim_to_target"] = dataset_sim_to_target[test_idx]
    runs_datum["PUMA_test"]["top_k_reward_assay_preds"] = puma_test_assay_preds[:,target_active_assay_cols]
    runs_datum["PUMA_test"]["top_k_reward_cluster_preds"] = puma_test_cluster_preds[:,target_cluster_id]
    runs_datum["PUMA_test"]["top_k_modes_idx"] = np.arange(len(test_idx))
    runs_datum["PUMA_test"]["top_k_modes_fps"] = [dataset_fps[j] for j in test_idx]

    # PLOTS THAT REQUIRE DUPLICATES REMOVED
    plot_tsim_between_modes_and_to_target(runs_datum, k1=MAX_K, k2=2000, bins=50, 
                                    save_path=f"{save_dir}/tanimoto_sim_hist.png")
    plot_pooled_boxplot_sim_and_rew(runs_datum, nbins1=15, nbins2=15, nsamples1=1000, nsamples2=1000,
                                    save_path=f"{save_dir}/pooled_boxplot_sim_rew.png")
    plot_unpooled_boxplot_sim_and_rew(runs_datum, bins1=15, bins2=15, n1=1000, n2=1000, ignore=[],
                                    save_path=f"{save_dir}/unpooled_boxplot_sim_rew.png")
    plot_assay_cluster_preds_hist(runs_datum, k1=MAX_K, k2=MAX_K, bins=50,
        assay_model=assay_model, cluster_model=cluster_model, ignore=["PUMA_test"],
        assay_cols=target_active_assay_cols, cluster_id=target_cluster_id,
        save_path=f"{save_dir}/assay_cluster_preds_hist.png")
    plot_unpooled_boxplot_oracle(runs_datum, bins1=10, bins2=10, n1=2000, n2=1000,
        assay_model=assay_model, assay_cols=target_active_assay_cols,
        cluster_model=cluster_model, cluster_id=target_cluster_id, use_gneprop=False,
        ignore=["PUMA_test"], save_path=f"{save_dir}/unpooled_boxplot_oracle.png")
    plot_umap_from_runs_datum(runs_datum, target_fp, target_reward, sim_thresh=0.7, n_neigh=15,
                            k=MAX_K, ignore=["PUMA_test"], save_path=f"{save_dir}/umap_mols.png")
    
if __name__ == "__main__":
    with open(f'{TARGET_MODE}_runs.json') as f:
        RUNS = json.load(f)
    
    for run in RUNS:
        go(run)