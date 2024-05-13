"""
Script to perform actual plotting
"""
from utils import *
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def plot_modes_over_trajs(runs_datum, rew_thresh=0.9, sim_thresh=0.7, bs=64, ignore=[], save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    maxbs = bs if type(bs) == int else max(bs.values())
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        if type(bs) == int: bsl = bs
        else: bsl = bs[run_name]
        num_modes, avg_rew = num_modes_lazy(run_datum, rew_thresh, sim_thresh, bsl)
        sns.lineplot(x=np.arange(len(num_modes)), y=num_modes, ax=ax[0], label=run_name)
        sns.lineplot(x=np.arange(len(avg_rew)), y=avg_rew, ax=ax[1], label=run_name)
    ax[0].set_title(f"Number of modes found w/ reward >= {rew_thresh} & Tanimoto sim <= {sim_thresh}")
    ax[0].set_xlabel(f"Num Trajectories (x{maxbs})")
    ax[0].set_ylabel("Num Modes")
    ax[1].set_title(f"Average reward")
    ax[1].set_xlabel(f"Num Trajectories (x{maxbs})")
    ax[1].set_ylabel("Average Reward")
    plt.legend()
    plt.savefig(save_path if save_path else f"num_modes_{rew_thresh}_{sim_thresh}.png")

def plot_tsim_between_modes_and_to_target(runs_datum, k1=10000, k2=1000, bins=50, ignore=[], save_path="tanimoto_sim_hist.png"):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    df1, df2 = pd.DataFrame(), pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        top_k1_tsim_to_target = run_datum['top_k_reward_tsim_to_target'][:k1]
        top_k2_fps = run_datum['top_k_reward_fps'][:k2]
        tani_sim_between_modes = []
        for i in tqdm(range(len(top_k2_fps))):
            tani_sim_between_modes.extend(AllChem.DataStructs.BulkTanimotoSimilarity(top_k2_fps[i], top_k2_fps[i+1:]))
        df1 = pd.concat([df1, pd.DataFrame({ "method": [run_name] * len(top_k1_tsim_to_target), "value": top_k1_tsim_to_target })], ignore_index=True)
        df2 = pd.concat([df2, pd.DataFrame({ "method": [run_name] * len(tani_sim_between_modes), "value": tani_sim_between_modes })], ignore_index=True)
    sns.histplot(data=df1, x="value", hue="method", bins=bins, ax=ax[0], stat="density", common_norm=False, alpha=0.5)
    sns.histplot(data=df2, x="value", hue="method", bins=bins, ax=ax[1], stat="density", common_norm=False, alpha=0.5)
    ax[0].set_title(f"Tanimoto Similarity to Target of Top-{k1} Samples by Reward")
    ax[0].set_xlabel("Tanimoto Similarity")
    ax[1].set_title(f"Tanimoto Similarity between Top-{k2} Samples by Reward")
    ax[1].set_xlabel("Tanimoto Similarity")
    ax[0].set_yscale('log', base=10)
    plt.savefig(save_path)

def plot_tsim_and_reward_full_hist(runs_datum, bins=50, rew_key="rewards", sim_key="tsim_to_target", 
                                   ignore=[], save_path="tsim_and_reward_hist.png"):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    df = pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        df = pd.concat([df, pd.DataFrame({
            "method": [run_name] * len(run_datum[rew_key]),
            "rew": run_datum[rew_key],
            "sim": run_datum[sim_key]
        })], ignore_index=True)
    sns.histplot(data=df, x="sim", hue="method", bins=bins, ax=ax[0], stat='density', common_norm=False)
    sns.histplot(data=df, x="rew", hue="method", bins=bins, ax=ax[1], stat='density', common_norm=False)
    ax[0].set_title("Tanimoto Similarity to Target of all Samples")
    ax[1].set_title("Gflownet Rewards of all Samples")
    ax[0].set_xlabel("Tanimoto Similarity")
    ax[1].set_xlabel("Reward")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.savefig(save_path)

def plot_pooled_boxplot_sim_and_rew(runs_datum, nbins1=15, nbins2=15, nsamples1=1000, nsamples2=1000, ignore=[], save_path="pooled_boxplot_sim_rew.png"):
    pooled_datum = pool_datum(runs_datum, avoid_keys=ignore)
    binned_datum_by_reward, rew_bins, rew_empty_bins, sper_rew_bin = bin_datum_by_col("rewards", 
        pooled_datum, nbins1, return_bins=True, samples_per_method=nsamples1,
        toss_bins_with_less_methods=True, subset=["rewards", "tsim_to_target"])
    binned_datum_by_tan_sim, sim_bins, sim_empty_bins, sper_sim_bin = bin_datum_by_col("tsim_to_target",
        pooled_datum, nbins2, return_bins=True, samples_per_method=nsamples2,
        toss_bins_with_less_methods=True, subset=["rewards", "tsim_to_target"])
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    global_df_by_rew = pd.DataFrame()
    global_df_by_sim = pd.DataFrame()
    for run_name, run_datum in binned_datum_by_reward.items():
        dfs = [pd.DataFrame({**{'Bin': bin}, **subdict}) for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        global_df_by_rew = pd.concat([global_df_by_rew, df], ignore_index=True)
    rew_bins_ne = [i for i in range(1, len(rew_bins)+1) if i not in rew_empty_bins]
    global_df_by_rew = global_df_by_rew[global_df_by_rew['Bin'].isin(rew_bins_ne)]
    rew_bin_labels = [f"{rew_bins[i]:.2f}-{rew_bins[i+1]:.2f}\n({sper_rew_bin[i]})"
                        for i in range(len(rew_bins)-1)] + [f"{rew_bins[-1]:.2f}-1.0\n({sper_rew_bin[-1]})"]
    sns.boxplot(x="Bin", y="tsim_to_target", data=global_df_by_rew, ax=ax[0],\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1)
    for run_name, run_datum in binned_datum_by_tan_sim.items():
        if run_name in ignore: continue
        dfs = [pd.DataFrame({**{'Bin': bin}, **subdict}) for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        global_df_by_sim = pd.concat([global_df_by_sim, df], ignore_index=True)
    sim_bins_ne = [i for i in range(1, len(sim_bins)+1) if i not in sim_empty_bins]
    global_df_by_sim = global_df_by_sim[global_df_by_sim['Bin'].isin(sim_bins_ne)]
    sim_bin_labels = [f"{sim_bins[i]:.2f}-{sim_bins[i+1]:.2f}\n({sper_sim_bin[i]})"
                        for i in range(len(sim_bins)-1)] + [f"{sim_bins[-1]:.2f}-1.0\n({sper_sim_bin[-1]})"]
    sns.boxplot(x="Bin", y="rewards", data=global_df_by_sim, ax=ax[1],\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1)
    ax[0].set_title("Tanimoto Similarity to Target by Reward Bins")
    ax[0].set_xlabel("Reward Bin")
    ax[0].set_ylabel("Tanimoto Similarity")
    ax[0].xaxis.set_tick_params(rotation=45)
    ax[1].set_title("Reward by Tanimoto Similarity to Target Bins")
    ax[1].set_xlabel("Tanimoto Similarity Bin")
    ax[1].set_ylabel("Reward")
    ax[1].xaxis.set_tick_params(rotation=45)
    plt.savefig(save_path)

def get_preds_for_bin(subdict, assay_model, assay_cols, cluster_model, cluster_id, use_gneprop=False):
    assert "smis" in subdict.keys()
    subdict["assay_preds"] = predict_assay_logits_from_smi(None, subdict["smis"], assay_model,
                                assay_cols, save_preds=False, verbose=False)
    subdict["cluster_preds"] = predict_cluster_logits_from_smi(None, subdict["smis"], cluster_model,
                                cluster_id, save_preds=False, use_gneprop=use_gneprop, verbose=False)
    return subdict

def plot_unpooled_boxplot_sim_and_rew(runs_datum, bins1=10, bins2=10, n1=2000, n2=1000, ignore=[],
                                 save_path="unpooled_boxplot_sim_rew.png"):
    binned_datum_by_reward, rew_bins, rew_empty_bins, sper_rew_bin = bin_datum_by_col(
        "rewards", runs_datum, bins1, return_bins=True, samples_per_method=n1, ignore_runs=ignore, subset=["tsim_to_target"])
    binned_datum_by_tan_sim, sim_bins, sim_empty_bins, sper_sim_bin = bin_datum_by_col(
        "tsim_to_target", runs_datum, bins2, return_bins=True, samples_per_method=n2, ignore_runs=ignore, subset=["rewards"])    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
    global_df_by_rew = pd.DataFrame()
    global_df_by_sim = pd.DataFrame()
    for run_name, run_datum in binned_datum_by_reward.items():
        if run_name in ignore: continue
        dfs = [pd.DataFrame({**{'Bin': bin}, **subdict}) for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        df["method"] = run_name
        global_df_by_rew = pd.concat([global_df_by_rew, df], ignore_index=True)
    for run_name, run_datum in binned_datum_by_tan_sim.items():
        if run_name in ignore: continue
        dfs = [pd.DataFrame({**{'Bin': bin}, **subdict}) for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        df["method"] = run_name
        global_df_by_sim = pd.concat([global_df_by_sim, df], ignore_index=True)
    rew_bin_labels = [f"{rew_bins[i]:.2f}-{rew_bins[i+1]:.2f}\n({sper_rew_bin[i]})"
                        for i in range(len(rew_bins)-1)] + [f"{rew_bins[-1]:.2f}-1.0\n({sper_rew_bin[-1]})"]
    sim_bin_labels = [f"{sim_bins[i]:.2f}-{sim_bins[i+1]:.2f}\n({sper_sim_bin[i]})"
                        for i in range(len(sim_bins)-1)] + [f"{sim_bins[-1]:.2f}-1.0\n({sper_sim_bin[-1]})"]
    # Only keep the bottom 5 and top 5 bins
    rew_bins_ne = [i for i in range(1, len(rew_bins)+1) if i not in rew_empty_bins]
    sim_bins_ne = [i for i in range(1, len(sim_bins)+1) if i not in sim_empty_bins]
    rew_bins_to_keep = rew_bins_ne[:5] + rew_bins_ne[-5:]
    sim_bins_to_keep = sim_bins_ne[:5] + sim_bins_ne[-5:]
    global_df_by_rew = global_df_by_rew[global_df_by_rew['Bin'].isin(rew_bins_to_keep)]
    global_df_by_sim = global_df_by_sim[global_df_by_sim['Bin'].isin(sim_bins_to_keep)]
    sns.boxplot(x="Bin", y="tsim_to_target", hue="method", data=global_df_by_rew, ax=ax[0,0],\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1)
    sns.boxplot(x="Bin", y="rewards", hue="method", data=global_df_by_sim, ax=ax[0,1],\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1)
    ax[0,0].set_title("Tanimoto Sim to Target by Reward Bins")
    ax[0,0].set_xlabel("Reward Bin")
    ax[0,0].set_ylabel("Tanimoto Sim")
    ax[0,0].xaxis.set_tick_params(rotation=45)
    ax[0,1].set_title("Reward by Tanimoto Sim to Target Bins")
    ax[0,1].set_xlabel("Tanimoto Similarity Bin")
    ax[0,1].set_ylabel("Reward")
    ax[0,1].xaxis.set_tick_params(rotation=45)
    plt.savefig(save_path)

def plot_unpooled_boxplot_oracle(runs_datum, bins1=10, bins2=10, n1=2000, n2=1000, ignore=[],
                                 save_path="unpooled_boxplot_oracle.png", **kwargs):
    binned_datum_by_reward, rew_bins, rew_empty_bins, sper_rew_bin = bin_datum_by_col(
        "rewards", runs_datum, bins1, return_bins=True, samples_per_method=n1, ignore_runs=ignore, subset=["smis"])
    binned_datum_by_tan_sim, sim_bins, sim_empty_bins, sper_sim_bin = bin_datum_by_col(
        "tsim_to_target", runs_datum, bins2, return_bins=True, samples_per_method=n2, ignore_runs=ignore, subset=["smis"])    
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    global_df_by_rew = pd.DataFrame()
    global_df_by_sim = pd.DataFrame()
    for run_name, run_datum in binned_datum_by_reward.items():
        if run_name in ignore: continue
        dfs = [pd.DataFrame({**{'Bin': bin}, **get_preds_for_bin(subdict, **kwargs)})
               for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        df["method"] = run_name
        global_df_by_rew = pd.concat([global_df_by_rew, df], ignore_index=True)
    for run_name, run_datum in binned_datum_by_tan_sim.items():
        if run_name in ignore: continue
        dfs = [pd.DataFrame({**{'Bin': bin}, **get_preds_for_bin(subdict, **kwargs)})
               for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        df["method"] = run_name
        global_df_by_sim = pd.concat([global_df_by_sim, df], ignore_index=True)
    rew_bin_labels = [f"{rew_bins[i]:.2f}-{rew_bins[i+1]:.2f}\n({sper_rew_bin[i]})"
                        for i in range(len(rew_bins)-1)] + [f"{rew_bins[-1]:.2f}-1.0\n({sper_rew_bin[-1]})"]
    sim_bin_labels = [f"{sim_bins[i]:.2f}-{sim_bins[i+1]:.2f}\n({sper_sim_bin[i]})"
                        for i in range(len(sim_bins)-1)] + [f"{sim_bins[-1]:.2f}-1.0\n({sper_sim_bin[-1]})"]
    # Only keep the bottom 5 and top 5 bins
    rew_bins_ne = [i for i in range(1, len(rew_bins)+1) if i not in rew_empty_bins]
    sim_bins_ne = [i for i in range(1, len(sim_bins)+1) if i not in sim_empty_bins]
    rew_bins_to_keep = rew_bins_ne[:5] + rew_bins_ne[-5:]
    sim_bins_to_keep = sim_bins_ne[:5] + sim_bins_ne[-5:]
    global_df_by_rew = global_df_by_rew[global_df_by_rew['Bin'].isin(rew_bins_to_keep)]
    global_df_by_sim = global_df_by_sim[global_df_by_sim['Bin'].isin(sim_bins_to_keep)]
    sns.boxplot(x="Bin", y="assay_preds", hue="method", data=global_df_by_rew, ax=ax[0,0],\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1)
    sns.boxplot(x="Bin", y="assay_preds", hue="method", data=global_df_by_sim, ax=ax[0,1],\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1)
    sns.boxplot(x="Bin", y="cluster_preds", hue="method", data=global_df_by_rew, ax=ax[1,0],\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1)
    sns.boxplot(x="Bin", y="cluster_preds", hue="method", data=global_df_by_sim, ax=ax[1,1],\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1)
    ax[0,0].set_title("Assay Logits by Reward Bins")
    ax[0,0].set_xlabel("Reward Bin")
    ax[0,0].set_ylabel("Assay Logit Preds")
    ax[0,0].xaxis.set_tick_params(rotation=45)
    ax[0,1].set_title("Assay Logits by Tanimoto Sim to Target Bins")
    ax[0,1].set_xlabel("Tanimoto Similarity Bin")
    ax[0,1].set_ylabel("Assay Logit Preds")
    ax[0,1].xaxis.set_tick_params(rotation=45)
    ax[1,0].set_title("Cluster Logits by Reward Bins")
    ax[1,0].set_xlabel("Reward Bin")
    ax[1,0].set_ylabel("Cluster Logit Preds")
    ax[1,0].xaxis.set_tick_params(rotation=45)
    ax[1,1].set_title("Cluster Logits by Tanimoto Sim to Target Bins")
    ax[1,1].set_xlabel("Tanimoto Similarity Bin")
    ax[1,1].set_ylabel("Cluster Logit Preds")
    ax[1,1].xaxis.set_tick_params(rotation=45)
    plt.savefig(save_path)

def plot_assay_cluster_preds_hist(runs_datum, k=10000, bins=50,
                                  ignore=[], save_path="assay_cluster_preds_hist.png"):
    fig, ax = plt.subplots(2, 3, figsize=(30, 20))
    dfa, dfc = pd.DataFrame(), pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        rew_ap = run_datum['top_k_reward_assay_preds'][:k]
        rew_cp = run_datum['top_k_reward_cluster_preds'][:k]
        mod_ap = run_datum['top_k_modes_assay_preds'][:k]
        mod_cp = run_datum['top_k_modes_cluster_preds'][:k]
        sim_ap = run_datum['top_k_tsim_assay_preds'][:k]
        sim_cp = run_datum['top_k_tsim_cluster_preds'][:k]
        ap_merged = np.concatenate([rew_ap, mod_ap, sim_ap])
        cp_merged = np.concatenate([rew_cp, mod_cp, sim_cp])
        ap_type = ["rew"] * len(rew_ap) + ["mod"] * len(mod_ap) + ["sim"] * len(sim_ap)
        cp_type = ["rew"] * len(rew_cp) + ["mod"] * len(mod_cp) + ["sim"] * len(sim_cp)
        dfa = pd.concat([dfa, pd.DataFrame({
            "method": [run_name] * len(ap_merged),
            "value": ap_merged,
            "type": ap_type,
        })])
        dfc = pd.concat([dfc, pd.DataFrame({
            "method": [run_name] * len(cp_merged),
            "value": cp_merged,
            "type": cp_type,
        })])
    if len(dfa) > 0:
        sns.histplot(data=dfa[dfa['type'] == 'rew'], x="value", hue="method", bins=bins, ax=ax[0,0], stat='density', alpha=0.5, common_norm=False)
        sns.histplot(data=dfa[dfa['type'] == 'mod'], x="value", hue="method", bins=bins, ax=ax[0,1], stat='density', alpha=0.5, common_norm=False)
        sns.histplot(data=dfa[dfa['type'] == 'sim'], x="value", hue="method", bins=bins, ax=ax[0,2], stat='density', alpha=0.5, common_norm=False)
    if len(dfc) > 0:
        sns.histplot(data=dfc[dfc['type'] == 'rew'], x="value", hue="method", bins=bins, ax=ax[1,0], stat='density', alpha=0.5, common_norm=False)
        sns.histplot(data=dfc[dfc['type'] == 'mod'], x="value", hue="method", bins=bins, ax=ax[1,1], stat='density', alpha=0.5, common_norm=False)
        sns.histplot(data=dfc[dfc['type'] == 'sim'], x="value", hue="method", bins=bins, ax=ax[1,2], stat='density', alpha=0.5, common_norm=False)
    ax[0,0].set_title(f"Predicted Assay Logits of Top-{k} Samples by Reward")
    ax[0,1].set_title(f"Predicted Assay Logits of Top-{k} Modes")
    ax[0,2].set_title(f"Predicted Assay Logits of Top-{k} Samples by TanSim to Target")
    ax[1,0].set_title(f"Predicted Cluster Logits of Top-{k} Samples by Reward")
    ax[1,1].set_title(f"Predicted Cluster Logits of Top-{k} Modes")
    ax[1,2].set_title(f"Predicted Cluster Logits of Top-{k} Samples by TanSim to Target")
    ax[0,0].set_xlabel("Assay Logit")
    ax[1,0].set_xlabel("Cluster Logit")
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[0,2].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[1,2].set_yscale('log')
    plt.savefig(save_path)

def plot_umap_from_runs_datum(runs_datum, target_fp, target_rew, sim_thresh=0.7, n_neigh=30, k=5000, 
    ignore=[], save_path="umap-mols.png"):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    df1, df2 = pd.DataFrame(), pd.DataFrame()
    model = umap.UMAP(n_components=2, n_neighbors=n_neigh, random_state=42, verbose=False, 
                      min_dist=0.01, metric="euclidean")
    max_reward = max([max(run_datum['rewards']) for run_datum in runs_datum.values()])
    min_reward = min([min(run_datum['rewards']) for run_datum in runs_datum.values()])
    norm_color_by_reward = lambda x: ((x - min_reward) / (max_reward - min_reward))**4
    target_color = norm_color_by_reward(target_rew)
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        rewards = run_datum['rewards']
        top_k_modes_fps = run_datum['top_k_modes_fps'][:k]
        top_k_modes_rew = rewards[run_datum['top_k_modes_idx'][:k]]
        top_k_reward_fps = run_datum['top_k_reward_fps'][:k]
        top_k_reward_rew = rewards[run_datum['top_k_reward_idx'][:k]]
        df1 = pd.concat([df1, pd.DataFrame({
            "method": [run_name] * len(top_k_modes_fps),
            "fps": top_k_modes_fps,
            "rewards": top_k_modes_rew,
            "alpha": list(map(norm_color_by_reward, top_k_modes_rew)),
        })], ignore_index=True)
        df2 = pd.concat([df2, pd.DataFrame({
            "method": [run_name] * len(top_k_reward_fps),
            "fps": top_k_reward_fps,
            "rewards": top_k_reward_rew,
            "alpha": list(map(norm_color_by_reward, top_k_reward_rew)),
        })], ignore_index=True)
    umap_top_modes = model.fit_transform(list(df1["fps"]))
    target_umap = model.transform([target_fp])
    df1["umap_0"], df1["umap_1"] = umap_top_modes[:,0], umap_top_modes[:,1]
    sns.scatterplot(x=target_umap[:, 0], y=target_umap[:, 1], ax=ax[0], label="Target", color="black", s=100, alpha=target_color)
    for run_name, _ in runs_datum.items():
        df1_method = df1[df1['method'] == run_name]
        sns.scatterplot(x=df1_method["umap_0"], y=df1_method["umap_1"], ax=ax[0],
                        alpha=df1_method['alpha'], label=run_name)
    umap_top_rewards = model.fit_transform(list(df2["fps"]))
    target_umap = model.transform([target_fp])
    df2["umap_0"], df2["umap_1"] = umap_top_rewards[:,0], umap_top_rewards[:,1]
    sns.scatterplot(x=target_umap[:, 0], y=target_umap[:, 1], ax=ax[1], label="Target", color="black", s=100, alpha=target_color)
    for run_name, _ in runs_datum.items():
        df2_method = df2[df2['method'] == run_name]
        sns.scatterplot(x=df2_method["umap_0"], y=df2_method["umap_1"], ax=ax[1],
                        alpha=df2_method['alpha'], label=run_name)
    ax[0].set_title(f"UMAP of Top-{k} modes with Tanimoto sim <= {sim_thresh}")
    ax[1].set_title(f"UMAP of Top-{k} samples by reward")
    ax[0].set_xlabel("UMAP 1")
    ax[0].set_ylabel("UMAP 2")
    ax[1].set_xlabel("UMAP 1")
    ax[1].set_ylabel("UMAP 2")
    norm = Normalize(vmin=min_reward, vmax=max_reward)
    reward_to_rgba = lambda x: (0, 0, 0, norm_color_by_reward(x))
    colors = [reward_to_rgba(norm(x)) for x in np.linspace(min_reward, max_reward, 100)]
    cmap = LinearSegmentedColormap.from_list('reward_cmap', colors)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax[1], location="right")
    cbar.set_label('Reward')
    plt.tight_layout()
    plt.savefig(save_path)