"""
Methods for plotting results
"""
from utils import *
from umap import umap_ as umap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

run_name_to_color = {
    "RND": "b",
    "GFN": "darkorange",
    "SAC": "g",
    "SQL": "r",
    # "PUMA_test": "purple"
}
hue_palette = sns.color_palette(list(run_name_to_color.values()))
hue_order = run_name_to_color.keys()


def smooth(x, n=100):
  idx = np.int32(np.linspace(0, n-1e-3, len(x)))
  return np.linspace(0, len(x), n), np.bincount(idx, weights=x)/np.bincount(idx)


def smooth_ci(lo, hi, n=100):
  assert len(lo) == len(hi)
  idx = np.int32(np.linspace(0, n-1e-3, len(lo)))
  return (
    np.linspace(0, len(lo), n),
    np.bincount(idx, weights=lo)/np.bincount(idx),
    np.bincount(idx, weights=hi)/np.bincount(idx)
  )

def plot_umap_from_runs_datum(runs_datum, target_fp=None, target_rew=None, n_neigh=30, k=5000,
                                ignore=[], save_path="umap-mols.pdf"):
    fig1, ax1 = plt.subplots(figsize=(6.75, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
    fig3, ax3 = plt.subplots(figsize=(6.75, 4.5))
    fig4, ax4 = plt.subplots(figsize=(6.75, 4.5))

    df1, df2 = pd.DataFrame(), pd.DataFrame()
    model1 = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, verbose=False, min_dist=0.1, metric="jaccard")
    model2 = umap.UMAP(n_components=2, n_neighbors=30, random_state=42, verbose=False, min_dist=0.1, metric="jaccard")
    
    # max_reward = max([max(run_datum['rewards']) for run_datum in runs_datum.values()])
    # min_reward = max_reward - 0.1
    # max_tsim = max([max(run_datum['tsim_to_target']) for run_datum in runs_datum.values()])
    # all_tsim = np.concatenate([run_datum['tsim_to_target'] for run_datum in runs_datum.values()])
    # min_tsim = np.percentile(all_tsim, 25)
    min_rew_ax1, max_rew_ax1 = 1, 0
    min_rew_ax2, max_rew_ax2 = 1, 0
    min_tsim_ax1, max_tsim_ax1 = 1, 0
    min_tsim_ax2, max_tsim_ax2 = 1, 0

    # print(f"Min reward: {min_reward}, Max reward: {max_reward}")
    # print(f"Min tsim: {min_tsim}, Max tsim: {max_tsim}")

    def norm_color(x, min_, max_):
        assert min_ <= max_
        if x < min_: return 0
        c_tmp = round(((x - min_) / (max_ - min_))**2, 3)
        return max(0, c_tmp)
    
    # target_color = norm_color_by_reward(target_rew if target_rew else max_reward)
    target_color = 1

    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        rewards = run_datum['rewards']
        top_k_modes_fps = run_datum['top_k_modes_fps'][:k]
        top_k_modes_rew = rewards[run_datum['top_k_modes_idx'][:k]]
        top_k_modes_tsim = run_datum['top_k_modes_tsim_to_target'][:k]
        top_k_reward_fps = run_datum['top_k_reward_fps'][:k]
        top_k_reward_rew = rewards[run_datum['top_k_reward_idx'][:k]]
        top_k_reward_tsim = run_datum['top_k_reward_tsim_to_target'][:k]
        df1 = pd.concat([df1, pd.DataFrame({
            "method": [run_name] * len(top_k_modes_fps),
            "fps": top_k_modes_fps,
            "rewards": top_k_modes_rew,
            "tsim": top_k_modes_tsim,
        })], ignore_index=True)
        df2 = pd.concat([df2, pd.DataFrame({
            "method": [run_name] * len(top_k_reward_fps),
            "fps": top_k_reward_fps,
            "rewards": top_k_reward_rew,
            "tsim": top_k_reward_tsim,
        })], ignore_index=True)
        max_rew_ax1 = max(max_rew_ax1, max(top_k_modes_rew))
        max_rew_ax2 = max(max_rew_ax2, max(top_k_reward_rew))
        min_tsim_ax1, max_tsim_ax1 = min(min_tsim_ax1, min(top_k_modes_tsim)), max(max_tsim_ax1, max(top_k_modes_tsim))
        min_tsim_ax2, max_tsim_ax2 = min(min_tsim_ax2, min(top_k_reward_tsim)), max(max_tsim_ax2, max(top_k_reward_tsim))

    min_rew_ax1 = max_rew_ax1 - 0.1
    min_rew_ax2 = max_rew_ax2 - 0.1

    df1["alpha"] = df1["rewards"].apply(lambda x: norm_color(x, min_rew_ax1, max_rew_ax1))
    df1["alpha_t"] = df1["tsim"].apply(lambda x: norm_color(x, min_tsim_ax1, max_tsim_ax1))
    df2["alpha"] = df2["rewards"].apply(lambda x: norm_color(x, min_rew_ax2, max_rew_ax2))
    df2["alpha_t"] = df2["tsim"].apply(lambda x: norm_color(x, min_tsim_ax2, max_tsim_ax2))

    model1.fit(list((df1[df1['alpha'] != 0])['fps']))
    umap_top_modes = model1.transform(list(df1["fps"]))
    df1["umap_0"], df1["umap_1"] = umap_top_modes[:,0], umap_top_modes[:,1]
    for run_name, _ in runs_datum.items():
        df1_method = df1[df1['method'] == run_name]
        sns.scatterplot(x=df1_method["umap_0"], y=df1_method["umap_1"], ax=ax1, s=20,
                        alpha=df1_method['alpha'], label=run_name, color=run_name_to_color[run_name])
        sns.scatterplot(x=df1_method["umap_0"], y=df1_method["umap_1"], ax=ax3, s=20,
                        alpha=df1_method['alpha_t'], label=run_name, color=run_name_to_color[run_name])
    if target_fp:
        target_umap = model1.transform([target_fp])
        sns.scatterplot(x=target_umap[:, 0], y=target_umap[:, 1], ax=ax1, label="Target", 
                        color="black", s=50, alpha=target_color, marker='x')
        sns.scatterplot(x=target_umap[:, 0], y=target_umap[:, 1], ax=ax3, label="Target", 
                        color="black", s=50, alpha=target_color, marker='x')

    model2.fit(list((df2[df2['alpha'] != 0])['fps']))
    umap_top_rewards = model2.transform(list(df2["fps"]))
    df2["umap_0"], df2["umap_1"] = umap_top_rewards[:,0], umap_top_rewards[:,1]
    for run_name, _ in runs_datum.items():
        df2_method = df2[df2['method'] == run_name]
        sns.scatterplot(x=df2_method["umap_0"], y=df2_method["umap_1"], ax=ax2, s=20,
                        alpha=df2_method['alpha'], label=run_name, color=run_name_to_color[run_name])
        sns.scatterplot(x=df2_method["umap_0"], y=df2_method["umap_1"], ax=ax4, s=20,
                        alpha=df2_method['alpha_t'], label=run_name, color=run_name_to_color[run_name])
    if target_fp:
        target_umap = model2.transform([target_fp])
        sns.scatterplot(x=target_umap[:, 0], y=target_umap[:, 1], ax=ax2, label="Target", 
                        color="black", s=50, alpha=target_color, marker='x')
        sns.scatterplot(x=target_umap[:, 0], y=target_umap[:, 1], ax=ax4, label="Target", 
                        color="black", s=50, alpha=target_color, marker='x')
    ax1.set_title(f"UMAP of Top {k} Modes with Tanimoto Similarity ≤ 0.3")
    ax2.set_title(f"UMAP of Top {k} Highest Reward Samples")
    ax3.set_title(f"UMAP of Top {k} Modes with Tanimoto Similarity ≤ 0.3 Colored by\nTanimoto Similarity to Target")
    ax4.set_title(f"UMAP of Top {k} Highest Reward Samples Colored by\nTanimoto Similarity to Target")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax3.set_xlabel("UMAP 1")
    ax3.set_ylabel("UMAP 2")
    ax4.set_xlabel("UMAP 1")
    ax4.set_ylabel("UMAP 2")
    norm_ax1 = Normalize(vmin=min_rew_ax1, vmax=max_rew_ax1)
    norm_ax2 = Normalize(vmin=min_rew_ax2, vmax=max_rew_ax2)
    norm_ax3 = Normalize(vmin=min_tsim_ax1, vmax=max_tsim_ax1)
    norm_ax4 = Normalize(vmin=min_tsim_ax2, vmax=max_tsim_ax2)
    value_to_rgba = lambda x, min_, max_: (0, 0, 0, norm_color(x, min_, max_))
    colors_ax1 = [value_to_rgba(x, min_rew_ax1, max_rew_ax1) for x in np.linspace(min_rew_ax1, max_rew_ax1, 20)]
    colors_ax2 = [value_to_rgba(x, min_rew_ax2, max_rew_ax2) for x in np.linspace(min_rew_ax2, max_rew_ax2, 20)]
    colors_ax3 = [value_to_rgba(x, min_tsim_ax1, max_tsim_ax1) for x in np.linspace(min_tsim_ax1, max_tsim_ax1, 20)]
    colors_ax4 = [value_to_rgba(x, min_tsim_ax2, max_tsim_ax2) for x in np.linspace(min_tsim_ax2, max_tsim_ax2, 20)]
    cmap_ax1 = LinearSegmentedColormap.from_list('rew_cmap', colors_ax1)
    cmap_ax2 = LinearSegmentedColormap.from_list('rew_cmap', colors_ax2)
    cmap_ax3 = LinearSegmentedColormap.from_list('tsim_cmap', colors_ax3)
    cmap_ax4 = LinearSegmentedColormap.from_list('tsim_cmap', colors_ax4)
    sm_ax1 = plt.cm.ScalarMappable(norm=norm_ax1, cmap=cmap_ax1)
    sm_ax2 = plt.cm.ScalarMappable(norm=norm_ax2, cmap=cmap_ax2)
    sm_ax3 = plt.cm.ScalarMappable(norm=norm_ax3, cmap=cmap_ax3)
    sm_ax4 = plt.cm.ScalarMappable(norm=norm_ax4, cmap=cmap_ax4)
    cbar1 = plt.colorbar(sm_ax1, ax=ax1, location="right"); cbar1.set_label('Reward')
    cbar2 = plt.colorbar(sm_ax2, ax=ax2, location="right"); cbar2.set_label('Reward')
    cbar3 = plt.colorbar(sm_ax3, ax=ax3, location="right"); cbar3.set_label('Tanimoto Similarity')
    cbar4 = plt.colorbar(sm_ax4, ax=ax4, location="right"); cbar4.set_label('Tanimoto Similarity')
    plt.tight_layout()
    for lh in ax1.get_legend().legendHandles:
        if type(lh.get_alpha()) in [float, int]: continue
        else: lh.set_alpha([1] * len(lh.get_alpha()))
    for lh in ax2.get_legend().legendHandles:
        if type(lh.get_alpha()) in [float, int]: continue
        else: lh.set_alpha([1] * len(lh.get_alpha()))
    for lh in ax3.get_legend().legendHandles:
        if type(lh.get_alpha()) in [float, int]: continue
        else: lh.set_alpha([1] * len(lh.get_alpha()))
    for lh in ax4.get_legend().legendHandles:
        if type(lh.get_alpha()) in [float, int]: continue
        else: lh.set_alpha([1] * len(lh.get_alpha()))
    fig1.savefig(save_path.replace(".pdf", "_modes.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_rew.pdf"), dpi=300)
    fig3.savefig(save_path.replace(".pdf", "_modes_tsim.pdf"), dpi=300)
    fig4.savefig(save_path.replace(".pdf", "_rew_tsim.pdf"), dpi=300)


def plot_modes_over_trajs(runs_datum, nruns, target_idx=None, is_joint=False, rew_thresh=None,
                          sim_thresh=0.7, n=2000, ignore=[], save_path="num_modes_over_trajs.pdf"):
    fig1, ax1 = plt.subplots(figsize=(6.75, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        if is_joint:
            num_modes = run_datum["num_modes_median"]
            num_modes_lo, num_modes_hi = run_datum["num_modes_lo"], run_datum["num_modes_hi"]
            x_modes_ci, num_modes_lo, num_modes_hi = smooth_ci(num_modes_lo, num_modes_hi, n=n)
            x_modes, num_modes = smooth(num_modes, n=n)

            avg_rew, avg_rew_std = run_datum["avg_rew_mean"], run_datum["avg_rew_std"]
            x_rew_ci, avg_rew_lo, avg_rew_hi = smooth_ci(avg_rew - avg_rew_std, avg_rew + avg_rew_std, n=n)
            x_rew, avg_rew = smooth(avg_rew, n=n)
            
            sns.lineplot(x=x_modes, y=num_modes, ax=ax1, label=run_name, color=run_name_to_color[run_name])
            sns.lineplot(x=x_rew, y=avg_rew, ax=ax2, label=run_name, color=run_name_to_color[run_name])
            ax1.fill_between(x_modes_ci, num_modes_lo, num_modes_hi, alpha=0.2, color=run_name_to_color[run_name])
            ax2.fill_between(x_rew_ci, avg_rew_lo, avg_rew_hi, alpha=0.2, color=run_name_to_color[run_name])
        else:
            num_modes, avg_rew = run_datum["num_modes"], run_datum["avg_rew"]
            x_modes, num_modes = smooth(num_modes, n=n)
            x_rew, avg_rew = smooth(avg_rew, n=n)
            sns.lineplot(x=x_modes, y=num_modes, ax=ax1, label=run_name, color=run_name_to_color[run_name])
            sns.lineplot(x=x_rew, y=avg_rew, ax=ax2, label=run_name, color=run_name_to_color[run_name])

    if is_joint:
        ax1.set_title(f"Number of Modes for {nruns} Aggregated Targets with\nTanimoto Similarity ≤ {sim_thresh}")
        ax2.set_title(f"Average Reward for {nruns} Aggregated Targets")
    elif target_idx:
        ax1.set_title(f"Number of Modes for Target {target_idx} with Reward ≥ {rew_thresh}\nand Tanimoto Similarity ≤ {sim_thresh}")
        ax2.set_title(f"Average Reward for Target {target_idx}")
    else:
        ax1.set_title(f"Number of modes with Tanimoto Similarity ≤ {sim_thresh}")
        ax2.set_title(f"Average Reward")
    ax1.set_xlabel("Num Trajectories (x64)")
    ax1.set_ylabel("Num Modes")
    ax2.set_xlabel("Num Trajectories (x64)")
    ax2.set_ylabel("Average Reward")
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_modes.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_rew.pdf"), dpi=300)


def plot_tsim_between_modes_and_to_target(runs_datum, k1=10000, k2=1000, bins=50, is_joint=False,
                                          ignore=[], save_path="tanimoto_sim_hist.pdf"):
    fig1, ax1 = plt.subplots(figsize=(6.75, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
    df1, df2 = pd.DataFrame(), pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        top_k1_tsim_to_target = run_datum['top_k_reward_tsim_to_target']
        tani_sim_between_modes = run_datum['top_k_cross_tsim']
        df1 = pd.concat([df1, pd.DataFrame({ "method": [run_name] * len(top_k1_tsim_to_target), "value": top_k1_tsim_to_target })], ignore_index=True)
        df2 = pd.concat([df2, pd.DataFrame({ "method": [run_name] * len(tani_sim_between_modes), "value": tani_sim_between_modes })], ignore_index=True)
    sns.histplot(data=df1, x="value", hue="method", bins=bins, ax=ax1, stat="density", 
                 common_norm=False, alpha=0.5, hue_order=hue_order)
    sns.histplot(data=df2, x="value", hue="method", bins=bins, ax=ax2, stat="density", 
                 common_norm=False, alpha=0.5, hue_order=hue_order)
    if is_joint:
        ax1.set_title(f"Tanimoto Similarity to Target of Top 1000\nHighest Reward Samples per Target")
        ax2.set_title(f"Tanimoto Similarity between Top 100 Highest\nReward Samples per Target")
    else:
        ax1.set_title(f"Tanimoto Similarity to Target of Top {k1} Highest Reward Samples")
        ax2.set_title(f"Tanimoto Similarity between Top {k2} Highest Reward Samples")
    ax1.set_xlabel("Tanimoto Similarity")
    ax2.set_xlabel("Tanimoto Similarity")
    ax1.set_yscale('log', base=10)
    ax2.set_yscale('log', base=10)
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_tsim_to_target.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_tsim_between_modes.pdf"), dpi=300)


def plot_tsim_and_reward_full_hist(runs_datum, bins=50, rew_key="rewards", sim_key="tsim_to_target", 
                                   is_joint=False, ignore=[], save_path="tsim_and_reward_hist.pdf"):
    fig1, ax1 = plt.subplots(figsize=(6.75, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
    df = pd.DataFrame()
    k = 0
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        k = max(k, len(run_datum[rew_key]))
        df = pd.concat([df, pd.DataFrame({
            "method": [run_name] * len(run_datum[rew_key]),
            "rew": run_datum[rew_key],
            "sim": run_datum[sim_key]
        })], ignore_index=True)
    sns.histplot(data=df, x="sim", hue="method", bins=bins, ax=ax1, stat='density', 
                 common_norm=False, hue_order=hue_order)
    sns.histplot(data=df, x="rew", hue="method", bins=bins, ax=ax2, stat='density', 
                 common_norm=False, hue_order=hue_order)
    if is_joint:
        ax1.set_title(f"Tanimoto Similarity to Target of Last 10000 Samples per Target")
        ax2.set_title(f"Reward of Last 10000 Samples per Target")
    else:
        ax1.set_title(f"Tanimoto Similarity to Target of Last {k} Samples")
        ax2.set_title(f"Reward of Last {k} Samples")
    ax1.set_xlabel("Tanimoto Similarity")
    ax2.set_xlabel("Reward")
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_tsim.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_rew.pdf"), dpi=300)


def plot_pooled_boxplot_sim_and_rew(runs_datum, nbins1=15, nbins2=15, nsamples1=1000, nsamples2=1000, ignore=[], save_path="pooled_boxplot_sim_rew.pdf"):
    pooled_datum = pool_datum(runs_datum, avoid_runs=ignore, keep_keys=['rewards', 'tsim_to_target'])
    binned_datum_by_reward, rew_bins, rew_empty_bins, sper_rew_bin = bin_datum_by_col("rewards", 
        pooled_datum, nbins1, return_bins=True, samples_per_method=nsamples1,
        toss_bins_with_less_methods=True, subset=["rewards", "tsim_to_target"])
    binned_datum_by_tan_sim, sim_bins, sim_empty_bins, sper_sim_bin = bin_datum_by_col("tsim_to_target",
        pooled_datum, nbins2, return_bins=True, samples_per_method=nsamples2,
        toss_bins_with_less_methods=True, subset=["rewards", "tsim_to_target"])
    fig1, ax1 = plt.subplots(figsize=(6.75, 6.75))
    fig2, ax2 = plt.subplots(figsize=(6.75, 6.75))
    global_df_by_rew = pd.DataFrame()
    global_df_by_sim = pd.DataFrame()
    for run_name, run_datum in binned_datum_by_reward.items():
        dfs = [pd.DataFrame({**{'Bin': bin}, **subdict}) for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        global_df_by_rew = pd.concat([global_df_by_rew, df], ignore_index=True)
    rew_bins_ne = [i for i in range(1, len(rew_bins)+1) if i not in rew_empty_bins]
    global_df_by_rew = global_df_by_rew[global_df_by_rew['Bin'].isin(rew_bins_ne)]
    rew_bin_labels = [f"{rew_bins[i]:.2f}-{rew_bins[i+1]:.2f}"
                        for i in range(len(rew_bins)-1)] + [f"{rew_bins[-1]:.2f}-1.0"]
    sns.boxplot(x="Bin", y="tsim_to_target", data=global_df_by_rew, ax=ax1,\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    for run_name, run_datum in binned_datum_by_tan_sim.items():
        if run_name in ignore: continue
        dfs = [pd.DataFrame({**{'Bin': bin}, **subdict}) for bin, subdict in run_datum.items()]
        if len(dfs) == 0: continue
        df = pd.concat(dfs, ignore_index=True)
        global_df_by_sim = pd.concat([global_df_by_sim, df], ignore_index=True)
    sim_bins_ne = [i for i in range(1, len(sim_bins)+1) if i not in sim_empty_bins]
    global_df_by_sim = global_df_by_sim[global_df_by_sim['Bin'].isin(sim_bins_ne)]
    sim_bin_labels = [f"{sim_bins[i]:.2f}-{sim_bins[i+1]:.2f}"
                        for i in range(len(sim_bins)-1)] + [f"{sim_bins[-1]:.2f}-1.0"]
    sns.boxplot(x="Bin", y="rewards", data=global_df_by_sim, ax=ax2,\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    ax1.set_title("Tanimoto Similarity to Target by Reward Bins")
    ax1.set_xlabel("Reward Bin")
    ax1.set_ylabel("Tanimoto Similarity")
    ax1.xaxis.set_tick_params(rotation=45)
    ax2.set_title("Reward by Tanimoto Similarity to Target Bins")
    ax2.set_xlabel("Tanimoto Similarity Bin")
    ax2.set_ylabel("Reward")
    ax2.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_rew.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_sim.pdf"), dpi=300)


def get_preds_for_bin(subdict, assay_model, assay_cols, cluster_model, cluster_id, use_gneprop=False):
    assert "smis" in subdict.keys()
    subdict["assay_preds"] = predict_assay_logits_from_smi(None, subdict["smis"], assay_model,
                                assay_cols, save_preds=False, verbose=False)
    subdict["cluster_preds"] = predict_cluster_logits_from_smi(None, subdict["smis"], cluster_model,
                                cluster_id, save_preds=False, use_gneprop=use_gneprop, verbose=False)
    return subdict


def plot_unpooled_boxplot_sim_and_rew(runs_datum, bins1=10, bins2=10, n1=2000, n2=1000, ignore=[],
                                 save_path="unpooled_boxplot_sim_rew.pdf"):
    binned_datum_by_reward, rew_bins, rew_empty_bins, sper_rew_bin = bin_datum_by_col(
        "rewards", runs_datum, bins1, return_bins=True, samples_per_method=n1, ignore_runs=ignore, subset=["tsim_to_target"])
    binned_datum_by_tan_sim, sim_bins, sim_empty_bins, sper_sim_bin = bin_datum_by_col(
        "tsim_to_target", runs_datum, bins2, return_bins=True, samples_per_method=n2, ignore_runs=ignore, subset=["rewards"])    
    fig1, ax1 = plt.subplots(figsize=(6.75, 6.75))
    fig2, ax2 = plt.subplots(figsize=(6.75, 6.75))
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
    rew_bin_labels = [f"{rew_bins[i]:.2f}-{rew_bins[i+1]:.2f}"
                        for i in range(len(rew_bins)-1)] + [f"{rew_bins[-1]:.2f}-1.0"]
    sim_bin_labels = [f"{sim_bins[i]:.2f}-{sim_bins[i+1]:.2f}"
                        for i in range(len(sim_bins)-1)] + [f"{sim_bins[-1]:.2f}-1.0"]
    # Only keep the bottom 5 and top 5 bins
    rew_bins_ne = [i for i in range(1, len(rew_bins)+1) if i not in rew_empty_bins]
    sim_bins_ne = [i for i in range(1, len(sim_bins)+1) if i not in sim_empty_bins]
    rew_bins_to_keep = rew_bins_ne[:5] + rew_bins_ne[-5:]
    sim_bins_to_keep = sim_bins_ne[:5] + sim_bins_ne[-5:]
    global_df_by_rew = global_df_by_rew[global_df_by_rew['Bin'].isin(rew_bins_to_keep)]
    global_df_by_sim = global_df_by_sim[global_df_by_sim['Bin'].isin(sim_bins_to_keep)]
    sns.boxplot(x="Bin", y="tsim_to_target", hue="method", data=global_df_by_rew, ax=ax1,\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    sns.boxplot(x="Bin", y="rewards", hue="method", data=global_df_by_sim, ax=ax2,\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    ax1.set_title("Tanimoto Similarity to Target by Reward Bins")
    ax1.set_xlabel("Reward Bin")
    ax1.set_ylabel("Tanimoto Similarity")
    ax1.xaxis.set_tick_params(rotation=45)
    ax2.set_title("Reward by Tanimoto Similarity to Target Bins")
    ax2.set_xlabel("Tanimoto Similarity Bin")
    ax2.set_ylabel("Reward")
    ax2.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_rew.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_sim.pdf"), dpi=300)


def plot_unpooled_boxplot_oracle(runs_datum, bins1=10, bins2=10, n1=2000, n2=1000, ignore=[],
                                 save_path="unpooled_boxplot_oracle.pdf", **kwargs):
    binned_datum_by_reward, rew_bins, rew_empty_bins, sper_rew_bin = bin_datum_by_col(
        "rewards", runs_datum, bins1, return_bins=True, samples_per_method=n1, ignore_runs=ignore, subset=["smis"])
    binned_datum_by_tan_sim, sim_bins, sim_empty_bins, sper_sim_bin = bin_datum_by_col(
        "tsim_to_target", runs_datum, bins2, return_bins=True, samples_per_method=n2, ignore_runs=ignore, subset=["smis"])    
    fig1, ax1 = plt.subplots(figsize=(6.75, 6.75))
    fig2, ax2 = plt.subplots(figsize=(6.75, 6.75))
    fig3, ax3 = plt.subplots(figsize=(6.75, 6.75))
    fig4, ax4 = plt.subplots(figsize=(6.75, 6.75))
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
    rew_bin_labels = [f"{rew_bins[i]:.2f}-{rew_bins[i+1]:.2f}"
                        for i in range(len(rew_bins)-1)] + [f"{rew_bins[-1]:.2f}-1.0"]
    sim_bin_labels = [f"{sim_bins[i]:.2f}-{sim_bins[i+1]:.2f}"
                        for i in range(len(sim_bins)-1)] + [f"{sim_bins[-1]:.2f}-1.0"]
    # Only keep the bottom 5 and top 5 bins
    rew_bins_ne = [i for i in range(1, len(rew_bins)+1) if i not in rew_empty_bins]
    sim_bins_ne = [i for i in range(1, len(sim_bins)+1) if i not in sim_empty_bins]
    rew_bins_to_keep = rew_bins_ne[:5] + rew_bins_ne[-5:]
    sim_bins_to_keep = sim_bins_ne[:5] + sim_bins_ne[-5:]
    global_df_by_rew = global_df_by_rew[global_df_by_rew['Bin'].isin(rew_bins_to_keep)]
    global_df_by_sim = global_df_by_sim[global_df_by_sim['Bin'].isin(sim_bins_to_keep)]
    sns.boxplot(x="Bin", y="assay_preds", hue="method", data=global_df_by_rew, ax=ax1,\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    sns.boxplot(x="Bin", y="assay_preds", hue="method", data=global_df_by_sim, ax=ax2,\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    sns.boxplot(x="Bin", y="cluster_preds", hue="method", data=global_df_by_rew, ax=ax3,\
                formatter=lambda x: rew_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    sns.boxplot(x="Bin", y="cluster_preds", hue="method", data=global_df_by_sim, ax=ax4,\
                formatter=lambda x: sim_bin_labels[int(x)-1], gap=.1, hue_order=hue_order)
    ax1.set_title("Assay Activity Probabilities by Reward Bins")
    ax1.set_xlabel("Reward Bin")
    ax1.set_ylabel("Probability")
    ax1.xaxis.set_tick_params(rotation=45)
    ax2.set_title("Assay Activity Probabilities by Tanimoto Similarity to Target Bins")
    ax2.set_xlabel("Tanimoto Similarity Bin")
    ax2.set_ylabel("Probability")
    ax2.xaxis.set_tick_params(rotation=45)
    ax3.set_title("Cluster Activity Probabilities by Reward Bins")
    ax3.set_xlabel("Reward Bin")
    ax3.set_ylabel("Probability")
    ax3.xaxis.set_tick_params(rotation=45)
    ax3.set_title("Cluster Activity Probabilities by Tanimoto Similarity to Target Bins")
    ax3.set_xlabel("Tanimoto Similarity Bin")
    ax3.set_ylabel("Probability")
    ax3.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_assay_rew.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_assay_sim.pdf"), dpi=300)
    fig3.savefig(save_path.replace(".pdf", "_cluster_rew.pdf"), dpi=300)
    fig4.savefig(save_path.replace(".pdf", "_cluster_sim.pdf"), dpi=300)


def plot_preds_boxplot(runs_datum, key, ignore=[], hue_order=None):
    fig, ax = plt.subplots(figsize=(6.75, 4.5))
    df = pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        assert key in run_datum.keys()
        df = pd.concat([df, pd.DataFrame({
            "method": [run_name] * len(run_datum[key]),
            "value": run_datum[key]
        })])
    hue_order = hue_order if hue_order != None else list(runs_datum.keys())
    sns.boxplot(x="method", y="value", hue="method", data=df, ax=ax, hue_order=hue_order)
    return fig, ax


def plot_cluster_preds_hist(runs_datum, cluster_id=None, k=10000, bins=50, is_joint=False,
                            ignore=[], plot_all=True, save_path="cluster_preds_hist.pdf"):
    n = None if plot_all else k
    clabel = cluster_id if cluster_id != None else "Aggregated"
    fig1, ax1 = plt.subplots(figsize=(6.75, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
    fig3, ax3 = plt.subplots(figsize=(6.75, 4.5))
    dfc = pd.DataFrame()
    for run_name, run_datum in runs_datum.items():
        if run_name in ignore: continue
        rew_cp = run_datum['top_k_reward_cluster_preds'][:n]
        mod_cp = run_datum['top_k_modes_cluster_preds'][:n]
        sim_cp = run_datum['top_k_tsim_cluster_preds'][:n]
        cp_merged = np.concatenate([rew_cp, mod_cp, sim_cp])
        cp_type = ["rew"] * len(rew_cp) + ["mod"] * len(mod_cp) + ["sim"] * len(sim_cp)
        dfc = pd.concat([dfc, pd.DataFrame({
            "method": [run_name] * len(cp_merged),
            "value": cp_merged,
            "type": cp_type,
        })])
    if len(dfc) > 0:
        sns.histplot(data=dfc[dfc['type'] == 'rew'], x="value", hue="method", bins=bins, 
                     ax=ax1, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
        sns.histplot(data=dfc[dfc['type'] == 'mod'], x="value", hue="method", bins=bins, 
                     ax=ax2, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
        sns.histplot(data=dfc[dfc['type'] == 'sim'], x="value", hue="method", bins=bins, 
                     ax=ax3, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
    if is_joint:
        ax1.set_title(f"Aggregated Cluster Activity Predictions of Top 1000\nHighest Reward Samples per Target")
        ax2.set_title(f"Aggregated Cluster Activity Predictions of Top 1000\nModes per Target")
        ax3.set_title(f"Aggregated Cluster Activity Predictions of Top 1000\nSamples by Tanimoto Similarity to Target")
    else:
        ax1.set_title(f"Cluster ({clabel}) Activity Predictions of Top {k}\nHighest Reward Samples")
        ax2.set_title(f"Cluster ({clabel}) Activity Predictions of Top {k} Modes")
        ax3.set_title(f"Cluster ({clabel}) Activity Predictions of Top {k}\nSamples by Tanimoto Similarity to Target")
    ax1.set_xlabel("Probability")
    ax2.set_xlabel("Probability")
    ax3.set_xlabel("Probability")
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    sns.move_legend(ax1, "lower left")
    sns.move_legend(ax2, "lower left")
    sns.move_legend(ax3, "lower left")
    plt.tight_layout()
    fig1.savefig(save_path.replace(".pdf", "_rew.pdf"), dpi=300)
    fig2.savefig(save_path.replace(".pdf", "_mod.pdf"), dpi=300)
    fig3.savefig(save_path.replace(".pdf", "_sim.pdf"), dpi=300)


def plot_assay_preds_hist(runs_datum, assay_cols=[], k=10000, bins=50, is_joint=False,
                          ignore=[], plot_all=True, save_path="assay_cluster_preds_hist.pdf"):
    assay_cols = [None] if assay_cols == None else assay_cols
    n = None if plot_all else k
    for i, col in enumerate(assay_cols):
        fig1, ax1 = plt.subplots(figsize=(6.75, 4.5))
        fig2, ax2 = plt.subplots(figsize=(6.75, 4.5))
        fig3, ax3 = plt.subplots(figsize=(6.75, 4.5))
        dfa = pd.DataFrame()
        col_label = col if col != None else "Aggregated"
        for run_name, run_datum in runs_datum.items():
            if run_name in ignore: continue
            rew_ap = run_datum['top_k_reward_assay_preds'][i][:n]
            mod_ap = run_datum['top_k_modes_assay_preds'][i][:n]
            sim_ap = run_datum['top_k_tsim_assay_preds'][i][:n]
            ap_merged = np.concatenate([rew_ap, mod_ap, sim_ap])
            ap_type = ["rew"] * len(rew_ap) + ["mod"] * len(mod_ap) + ["sim"] * len(sim_ap)
            dfa = pd.concat([dfa, pd.DataFrame({
                "method": [run_name] * len(ap_merged),
                "value": ap_merged,
                "type": ap_type,
            })])
        if len(dfa) > 0:
            sns.histplot(data=dfa[dfa['type'] == 'rew'], x="value", hue="method", bins=bins, 
                         ax=ax1, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
            sns.histplot(data=dfa[dfa['type'] == 'mod'], x="value", hue="method", bins=bins,
                         ax=ax2, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
            sns.histplot(data=dfa[dfa['type'] == 'sim'], x="value", hue="method", bins=bins,
                         ax=ax3, stat='density', alpha=0.4, common_norm=False, hue_order=hue_order)
        if is_joint:
            ax1.set_title(f"Aggregated Assay Activity Predictions of Top 1000\nHighest Reward Samples per Target")
            ax2.set_title(f"Aggregated Assay Activity Predictions of Top 1000\nModes per Target")
            ax3.set_title(f"Aggregated Assay Activity Predictions of Top 1000\nSamples by Tanimoto Similarity to Target")
        else:
            ax1.set_title(f"Assay ({col_label}) Activity Predictions of Top {k}\nHighest Reward Samples")
            ax2.set_title(f"Assay ({col_label}) Activity Predictions of Top {k} Modes")
            ax3.set_title(f"Assay ({col_label}) Activity Predictions of Top {k}\nSamples by Tanimoto Similarity to Target")
        ax1.set_xlabel("Probability")
        ax2.set_xlabel("Probability")
        ax3.set_xlabel("Probability")
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        sns.move_legend(ax1, "lower left")
        sns.move_legend(ax2, "lower left")
        sns.move_legend(ax3, "lower left")
        plt.tight_layout()
        fig1.savefig(save_path.replace(".pdf", f"_rew_{col}.pdf"), dpi=300)
        fig2.savefig(save_path.replace(".pdf", f"_mod_{col}.pdf"), dpi=300)
        fig3.savefig(save_path.replace(".pdf", f"_sim_{col}.pdf"), dpi=300)
