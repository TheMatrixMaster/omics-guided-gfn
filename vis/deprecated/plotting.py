import numpy as np
import pandas as pd
import hydra
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)

import umap

from tqdm import tqdm
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintsFromSmiles

from multimodal_contrastive.utils import utils
from multimodal_contrastive.networks.models import MultiTask_FP_PL
from multimodal_contrastive.data.dataset import TestDataset
from multimodal_contrastive.analysis.utils import make_eval_data_loader

import torch_geometric.data as gd
from gflownet.models.mmc import mol2graph

# register custom resolvers if not already registered
OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list), replace=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

sqlite_cols = (
    ["smi", "r"] + [f"{a}_{i}" for a in ["fr"] for i in range(1)] + ["ci_beta"]
)

import sqlite3
from collections import defaultdict
from tqdm import tqdm


def get_bulk_fingerprints(smis):
    filtered_fps = FingerprintsFromSmiles(
        dataSource=[{'id': idx, 'smi': s} for idx, s in enumerate(smis)],
        idCol='id',
        smiCol='smi',
        fingerprinter=Chem.RDKFingerprint,
        reportFreq=10000,
    )
    filtered_fps = [x[1] for x in filtered_fps]
    return filtered_fps


def plot_sim_to_target_hist(fps, target_fp, rews, k=100, by="similarity", filename="tanimoto-sim-to-target.png"):
    # Plots the tanimoto similarity of the top-k most similar sampled molecules 
    # for each method or top-k highest by reward
    for model_name, fp in fps.items():
        if len(fp) == 0 or model_name == "Target":
            continue
        tan_sim_to_target = AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, fp)
        if by == "similarity":
            top_k_idx = np.argsort(tan_sim_to_target)[::-1][:k]
        else:
            top_k_idx = np.argsort(rews[model_name])[::-1][:k]
        tan_sim_to_target = [tan_sim_to_target[i] for i in top_k_idx]
        plt.hist(tan_sim_to_target, bins=50, label=model_name, alpha=0.4, density=True)

    plt.legend()
    plt.title(f"Tanimoto Similarity to Target (Top-k highest by {by})")
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Density")
    plt.savefig(filename)
    plt.clf()

def plot_pairwise_sim_hist(fps, rews, k=50, filename="tanimoto-sim-between-samples.png"):
    for model_name, fp in fps.items():
        if len(fp) < k or model_name in ["Target", "PUMA"]:
            continue

        # choose the top-k fps with highest rew
        top_k_idx = np.argsort(rews[model_name])[::-1][:k]
        top_k_fps = [fp[i] for i in top_k_idx]
        
        tanimoto_sim = []
        for i, j in combinations(top_k_fps, 2):
            tanimoto_sim.append(AllChem.DataStructs.TanimotoSimilarity(i, j))

        plt.hist(tanimoto_sim, bins=50, label=model_name, alpha=0.4, density=True)

    plt.legend()
    plt.title("Pairwise Tanimoto Similarity")
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Density")
    plt.savefig(filename)
    plt.clf()

def plot_reward_hist(rewards, filename="reward-hist.png"):
    for model_name, rew in rewards.items():
        if len(rew) == 0 or model_name == "Target":
            continue
        plt.hist(rew, bins=50, label=model_name, alpha=0.4, density=True)

    plt.legend()
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.savefig(filename)
    plt.clf()

def plot_umap(fps, target_fp, rews, k=200, sim_threshold=0.7, filename="umap-fps.png"):
    # Plot umap of molecular fingerprints of top-k highest reward molecules per method
    n_neighbors = [20, 30, 40, 50, 60]
    all_fps = []
    for model_name, fp in fps.items():
        if model_name == "Target": continue
        elif model_name == "PUMA":
            # randomly sample 10*k molecules from the dataset
            # top_k_idx = np.random.choice(len(fp), min(10*k, len(fp)), replace=False)
            top_k_idx = np.argsort(rews[model_name])[::-1][:10*k]
        else:
            # get top-k most dissimilar molecules with highest reward (modes)
            top_k_idx = get_top_k_dissimilar_modes(
                fp, target_fp, rews[model_name],
                sim_threshold=sim_threshold, k=k
            )
            print(f"Found {len(top_k_idx)} modes for {model_name}")
        fps[model_name] = [fp[i] for i in top_k_idx]
        rews[model_name] = [rews[model_name][i] for i in top_k_idx]
        all_fps.extend(fps[model_name])

    for n_neigh in n_neighbors:
        print(f"Running umap with {n_neigh} neighbors")
        reducer = umap.UMAP(n_neighbors=n_neigh, random_state=42, verbose=True, min_dist=0.1, metric="jaccard")
        reducer = reducer.fit(all_fps)

        for model_name, fp in fps.items():
            if len(fp) == 0: continue
            fp_reduced = reducer.transform(fp)
            s = 2
            a = [r**8 for r in rews[model_name]]
            if model_name == "Target":
                s = 25
                a = 1
                
            plt.scatter(fp_reduced[:, 0], fp_reduced[:, 1], label=model_name, alpha=a, s=s)

        plt.legend()
        plt.title("UMAP of molecular fps")
        plt.xlabel("umap1")
        plt.ylabel("umap2")
        plt.savefig(f"results/{n_neigh}n_{filename}")
        plt.clf()

def plot_assay_logit_hist(smis, rews, model, active_cols, k=500, filename="assay-logit-hist.png"):
    # Plots a histogram of the logit distribution of top-k highest reward molecules from each method
    active_cols = torch.tensor(active_cols) if isinstance(active_cols, list) else active_cols
    fig, ax = plt.subplots(1, len(active_cols), figsize=(5*len(active_cols), 5), squeeze=False)

    for model_name, smi in smis.items():
        if len(smi) < k or model_name in ["Target"]:
            continue

        # choose the top-k smi with highest rew
        top_k_idx = np.argsort(rews[model_name])[::-1][:k]
        top_k_smi = [smi[i] for i in top_k_idx]
        
        # create a test dataset using the smiles and run inference with assay model
        test_df = pd.DataFrame(top_k_smi, columns=["smiles"])
        dataset = TestDataset(test_df, mol_col="smiles", label_col=None)
        y_hat = model(next(iter(make_eval_data_loader(dataset, batch_size=k))))

        # keep the logit values for the active columns
        logit_values = torch.index_select(y_hat[0].detach().cpu(), 1, active_cols).numpy()

        # for each active column, we produce a separate hist plot of the logit values
        for i, col in enumerate(active_cols):
            ax[0,i].hist(logit_values[:, i], bins=50, label=model_name, alpha=0.4, density=True)
    
    for i, col in enumerate(active_cols):
        ax[0,i].legend()
        ax[0,i].set_title(f"Logit Distribution for assay {col}")
        ax[0,i].set_xlabel("Logit Value")
        ax[0,i].set_ylabel("Density")
        
    plt.savefig(filename)
    plt.clf()


def setup_puma():
    # Load config for MMC model
    config_name = "puma_sm_gmc"
    configs_path = "../multimodal_contrastive/configs"

    with hydra.initialize(version_base=None, config_path=configs_path):
        cfg = hydra.compose(config_name=config_name)

    print(cfg.datamodule.split_type)
    if cfg.get("seed"): seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("test")
    return datamodule, cfg


def load_assay_pred_model():
    ckpt = '/home/mila/s/stephen.lu/gfn_gene/res/mmc/models/puma_assay_epoch=139.ckpt'
    model = MultiTask_FP_PL.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    return model

def get_active_assay_cols(dataset, smi):
    # returns the column indices of the active assays == 1 for a given target smile
    if smi not in dataset.ids: return None
    target_idx = np.where(np.array(dataset.ids)==smi)[0][0]
    return torch.where(dataset.y[target_idx] == 1)[0]

def load_assay_matrix_from_csv():
    data_dir = '/home/mila/s/stephen.lu/scratch/mmc/datasets/'
    dataset = TestDataset(data_dir + 'assay_matrix_discrete_37_assays_canonical.csv')
    return dataset

def load_mmc_model(cfg):
    # Load model from checkpoint
    ckpt_path = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/models/epoch=72-step=7738.ckpt"
    model = utils.instantiate_model(cfg)
    model = model.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.eval()
    return model

def inference_puma(datamodule, cfg):
    # Get latent representations for full dataset
    model = load_mmc_model(cfg)
    representations = model.compute_representation_dataloader(
        make_eval_data_loader(datamodule.dataset),
        device=device,
        return_mol=False
    )
    return representations


def get_representations():
    data_dir = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/data"
    try:
        res = np.load(f"{data_dir}/puma_embeddings.npz", allow_pickle=True)
    except FileNotFoundError:
        datamodule, cfg = setup_puma()
        res = inference_puma(datamodule, cfg)
        np.savez(f"{data_dir}/puma_embeddings.npz", **res)
    return res


def get_fp_from_base64(base64_fp):
    fp_from_base64 = ExplicitBitVect(2048)
    fp_from_base64.FromBase64(base64_fp)
    return fp_from_base64


def get_fp_from_bit_array(bit_array):
    fp = ExplicitBitVect(2048)
    fp.SetBitsFromList((np.where(bit_array)[0].tolist()))
    return fp


def get_fp_from_base64_or_bit_array(fp):
    if isinstance(fp, str):
        return get_fp_from_base64(fp)
    return get_fp_from_bit_array(fp)


def get_fingerprints(fpgen):
    data_dir = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/data"
    try:
        res = np.load(f"{data_dir}/puma_fingerprints.npy", allow_pickle=True)
        res = list(map(get_fp_from_base64, res))
    except FileNotFoundError:
        res = []
        datamodule, _ = setup_puma()
        for idx in tqdm(range(len(datamodule.dataset))):
            smi = datamodule.dataset[idx]["inputs"]["struct"].mols
            mol = Chem.MolFromSmiles(smi)
            fp = fpgen.GetFingerprint(mol)
            res.append(fp)
    return res


def sqlite_load(root, columns, num_workers=8, upto=None, begin=0):
    try:
        bar = tqdm(smoothing=0)
        values = defaultdict(lambda: [[] for i in range(num_workers)])
        for i in range(num_workers):
            con = sqlite3.connect(
                f"file:{root}generated_mols_{i}.db?mode=ro", uri=True, timeout=6
            )
            cur = con.cursor()
            cur.execute("pragma mmap_size = 134217728")
            cur.execute("pragma cache_size = -1024000;")
            r = cur.execute(
                f'select {",".join(columns)} from results where rowid >= {begin}'
            )
            n = 0
            for j, row in enumerate(r):
                bar.update()
                for value, col_name in zip(row, columns):
                    values[col_name][i].append(value)
                n += 1
                if upto is not None and n * num_workers > upto:
                    break
            con.close()
        return values
    finally:
        bar.close()


def is_new_mode(modes_fp, new_fp, sim_threshold=0.7):
    """Returns True if obj is a new mode, False otherwise"""
    if len(modes_fp) == 0:
        return True

    if new_fp is None:
        return False
    
    sim_scores = AllChem.DataStructs.BulkTanimotoSimilarity(new_fp, modes_fp)
    return all(s < sim_threshold for s in sim_scores)

def get_top_k_dissimilar_modes(fps, target_fp, rews, sim_threshold=0.7, k=100):
    """
    Returns the index of the top k molecules with the highest reward with mutual 
    tanimoto similarity <= sim_threshold
    """
    # first, sort the molecules by reward
    sorted_idx = np.argsort(rews)[::-1]
    sorted_fps = [fps[i] for i in sorted_idx]
    # while we haven't reached k modes, or if we've exhausted molecules,
    # check if the new molecule is a new mode
    modes_fp = []
    modes_idx = []
    for i, fp in enumerate(sorted_fps):
        if len(modes_fp) >= k: break
        if is_new_mode(modes_fp, fp, sim_threshold):
            modes_fp.append(fp)
            modes_idx.append(sorted_idx[i])
    return modes_idx


def get_data_from_run(base_dir, run_id, target_fp):
    # Obtain sampled data from the run
    run_dir = f"{base_dir}/{run_id}"
    values = sqlite_load(f"{run_dir}/train/", sqlite_cols, 1)
    smis, rewards = values['smi'][0], values['fr_0'][0]
    high = 0

    try:
        filtered_fps = np.load(f"{run_dir}/fps.npy", allow_pickle=True)
        filtered_fps = list(map(get_fp_from_base64_or_bit_array, tqdm(filtered_fps)))
        assert len(filtered_fps) == len(smis)
    except FileNotFoundError:
        raise FileNotFoundError(f"Filtered fps not found for {model_name}")

    for idx, (smi, r) in tqdm(enumerate(zip(smis, rewards))):
        fp = filtered_fps[idx]
        tanimoto_sim = AllChem.DataStructs.TanimotoSimilarity(target_fp, fp)
        if tanimoto_sim > high:
            high = tanimoto_sim
            print(high, r)
        filtered_fps.append(fp)
    
    return filtered_fps, rewards, smis


def get_data_for_baseline_run(base_dir, run_id, target_latent, cfg):
    filtered_fps, _, smis = get_data_from_run(base_dir, run_id, target_fp)
    # now we need to recompute latents for these random samples and obtain their rewards
    # first, I don't wanna run inference on 640000 samples, so let's randomly take a subset
    rnd_idx = np.random.choice(len(smis), 10000, replace=False)
    smis = [smis[i] for i in rnd_idx]
    filtered_fps = [filtered_fps[i] for i in rnd_idx]
    # now, we need to build a dataloader from these smis that we can pass to the mmc model
    mmc_model = load_mmc_model(cfg)
    graphs = [mol2graph(Chem.MolFromSmiles(smi)) for smi in smis]
    batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
    batch.to(mmc_model.device if hasattr(mmc_model, 'device') else device)
    preds = mmc_model({"inputs": {"struct": batch}}, mod_name="struct")
    preds = preds.data.cpu().detach().numpy()
    rewards = (cosine_similarity(target_latent, preds) + 1) / 2
    return filtered_fps, rewards, smis


if __name__ == "__main__":
    base_dir = "/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps"
    sim_threshold = 0.2
    
    models = {
        "6888": "04-24-01-46-morph-sim-final-targets/amber-sweep-7-id-dxn50jrg",
        "2288": "04-24-01-46-morph-sim-final-targets/dandy-sweep-5-id-ongkpkty",
        "338": "04-24-01-46-morph-sim-final-targets/logical-sweep-2-id-kl0ygljq",
        "4331": "04-24-01-46-morph-sim-final-targets/pleasant-sweep-6-id-w9d4thq9",
        "8949": "04-25-08-32-morph-sim-run-failed-8949/true-sweep-1-id-xodl84ba",
        "903": "04-24-01-46-morph-sim-final-targets/snowy-sweep-3-id-in1e3736",
        "1847": "04-24-01-46-morph-sim-final-targets/splendid-sweep-4-id-o4l26cxc",
        "8838": "04-24-01-46-morph-sim-final-targets/stilted-sweep-9-id-e0h282g0",
        "9277": "04-24-01-46-morph-sim-final-targets/sunny-sweep-11-id-66qel5x0",
        "8206": "04-24-01-46-morph-sim-final-targets/sweet-sweep-8-id-fr5fx186",
        "39": "04-24-01-46-morph-sim-final-targets/young-sweep-1-id-6ifys7pm",
        "9476": "04-24-01-49-morph-sim-final-targets/astral-sweep-14-id-pih6w90m",
        "13905": "04-24-01-49-morph-sim-final-targets/azure-sweep-17-id-fgtpgfuz",
        "12071": "04-24-01-49-morph-sim-final-targets/deep-sweep-16-id-lij3uk3z",
        "9445": "04-24-01-49-morph-sim-final-targets/ethereal-sweep-13-id-fcbo3yhr",
        "10075": "04-24-01-49-morph-sim-final-targets/fiery-sweep-15-id-3fj60c6m",
        "9300": "04-24-01-49-morph-sim-final-targets/twilight-sweep-12-id-laveyrao",
    }

    fpgen = AllChem.GetRDKitFPGenerator(
        maxPath=7,
        branchedPaths=False,
    )

    # Obtain representations and fingerprints for puma dataset
    datamodule, _ = setup_puma()
    representations = get_representations()
    dataset_fps = get_fingerprints(fpgen)
    dataset_smis = [x["inputs"]["struct"].mols for x in datamodule.dataset]
    assay_dataset = load_assay_matrix_from_csv()
    assay_model = load_assay_pred_model()

    for model_name, run_id in models.items():
        print(f"Running for model: ", model_name)

        # Obtain representations, fingerprints, and active assay columns for target and dataset
        target_idx = int(model_name)
        target_fp = dataset_fps[target_idx]
        target_smi = datamodule.dataset[target_idx]["inputs"]["struct"].mols.decode("utf-8")
        target_struct_latent = representations['struct'][target_idx]
        target_morph_latent = representations['morph'][target_idx]
        target_active_assay_cols = get_active_assay_cols(assay_dataset, target_smi)
        target_reward = cosine_similarity(target_struct_latent.reshape(1, -1), target_morph_latent.reshape(1, -1))[0][0]
        dataset_rewards = ((cosine_similarity(representations['struct'], target_morph_latent.reshape(1, -1)) + 1) / 2).reshape(-1,)

        print('Target has morph~struct cosine sim: ', target_reward)

        # Collect the data we need for plotting
        fps = { "Target": [target_fp], "PUMA": dataset_fps }
        rews = { "Target": [target_reward], "PUMA": dataset_rewards }
        smiles = { "Target": [target_smi], "PUMA":  dataset_smis }

        # Get data from run
        filtered_fps, rewards, smis = get_data_from_run(base_dir, run_id, target_fp)

        fps[model_name] = filtered_fps
        rews[model_name] = rewards
        smiles[model_name] = smis

        print("Finished processing for model: ", model_name)
    
        if target_active_assay_cols is not None:
            plot_assay_logit_hist(smiles, rews, assay_model, target_active_assay_cols, k=500, filename=f"results/assay-logit-hist_{target_idx}.png")
        
        plot_reward_hist(rews, f"results/reward-hist-target_{target_idx}.png")
        plot_pairwise_sim_hist(fps, rews, k=50, filename=f"results/tanimoto-sim-betw-samples_{target_idx}.png")
        plot_sim_to_target_hist(fps, target_fp, rews, k=200, by="similarity", filename=f"results/tan-sim-to-target_{target_idx}_by_sim.png")
        plot_sim_to_target_hist(fps, target_fp, rews, k=200, by="reward", filename=f"results/tan-sim-to-target_{target_idx}_by_rew.png")
        plot_umap(fps, target_fp, rews, k=200, sim_threshold=sim_threshold, filename=f"umap_top_200_<{sim_threshold}_target_{target_idx}.png")

        print("Finished plotting for model: ", model_name)
