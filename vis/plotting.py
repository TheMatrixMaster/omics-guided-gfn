import numpy as np
import hydra
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)

import umap
from itertools import combinations
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintsFromSmiles
from tqdm import tqdm
from utils import make_eval_data_loader
from multimodal_contrastive.utils import utils
from sklearn.metrics.pairwise import cosine_similarity

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


def plot_sim_to_target_hist(fps, target_fp, filename="tanimoto-sim-to-target.png"):
    for model_name, fp in fps.items():
        if len(fp) == 0 or model_name == "Target":
            continue
        tan_sim_to_target = AllChem.DataStructs.BulkTanimotoSimilarity(target_fp, fp)
        plt.hist(tan_sim_to_target, bins=50, label=model_name, alpha=0.4, density=True)

    plt.legend()
    plt.title("Tanimoto Similarity to Target")
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Density")
    plt.savefig(filename)
    plt.clf()

def plot_pairwise_sim_hist(top_k_fps, k=50, filename="tanimoto-sim-between-samples.png"):
    for model_name, fp in top_k_fps.items():
        if len(fp) < k or model_name in ["Target", "PUMA"]:
            continue
        
        tanimoto_sim = []
        for i, j in combinations(fp[:k], 2):
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

def plot_umap(fps, filename="umap-fps.png"):
    n_neighbors = [20, 30, 40, 50, 60]
    all_fps = []
    for fp in fps.values():
        all_fps.extend(fp)

    for n_neigh in n_neighbors:
        print(f"Running umap with {n_neigh} neighbors")
        # reducer = PCA(n_components=2)
        reducer = umap.UMAP(n_neighbors=n_neigh, random_state=42, verbose=True, min_dist=0.1, metric="jaccard")
        reducer = reducer.fit(all_fps)

        for model_name, fp in fps.items():
            if len(fp) == 0: continue
            fp_reduced = reducer.transform(fp)
            s, a = 2, 0.2
            if model_name == "Target":
                s = 25
                a = 1
                
            plt.scatter(fp_reduced[:, 0], fp_reduced[:, 1], label=model_name, alpha=a, s=s)

        plt.legend()
        plt.title("UMAP of molecular fps")
        plt.xlabel("umap1")
        plt.ylabel("umap2")
        plt.savefig(f"{n_neigh}n_{filename}")
        plt.clf()


def setup_puma():
    # Load config for CLIP model
    config_name = "puma_sm_gmc"
    configs_path = "../../configs"

    with hydra.initialize(version_base=None, config_path=configs_path):
        cfg = hydra.compose(config_name=config_name)

    print(cfg.datamodule.split_type)

    # Set seed for random number generators in pytorch, numpy and python.random
    # and especially for generating the same data splits for the test set
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Load test data split
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("test")
    return datamodule, cfg


def inference_puma(datamodule, cfg):
    # Load model from checkpoint
    ckpt_path = "/home/mila/s/stephen.lu/gfn_gene/res/mmc/models/morph_struct_90_step_val_loss.ckpt"
    model = utils.instantiate_model(cfg)
    model = model.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.eval()

    # Get latent representations for full dataset
    representations = model.compute_representation_dataloader(
        make_eval_data_loader(datamodule.dataset),
        device=device,
        return_mol=False
    )
    return representations


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


if __name__ == "__main__":
    base_dir = "/home/mila/s/stephen.lu/scratch/gfn_gene/wandb_sweeps"
    reward_threshold = 0.88
    sim_threshold = 0.25
    target_idx = 10852

    # models = {
    #     "RND": "04-12-03-58-morph-sim-target-67-algo/swift-sweep-4-id-08lkh6fe",
    #     "A2C": "04-12-03-58-morph-sim-target-67-algo/jolly-sweep-3-id-4dj1ezru",
    #     "GFN": "04-12-03-58-morph-sim-target-67-algo/confused-sweep-1-id-qzlflioh",
    #     "SQL": "04-12-03-58-morph-sim-target-67-algo/absurd-sweep-2-id-6gjx9a58",
    # }

    models = {
        "8_frags": "04-18-14-51-morph-sim-target-10852-algo/prime-sweep-10-id-xpqchkgk",
        "6_frags": "04-18-14-50-morph-sim-target-10852-algo/magic-sweep-8-id-gzkms9dw",
        "4_frags": "04-18-14-42-morph-sim-target-10852-algo/classic-sweep-3-id-j5jzvjc2"
    }

    fpgen = AllChem.GetRDKitFPGenerator(
        maxPath=7,
        branchedPaths=False,
    )

    # Load PUMA dataset and infer representations
    datamodule, cfg = setup_puma()

    # Load target molecule and its latent representations
    target_smi = datamodule.dataset[target_idx]["inputs"]["struct"].mols
    target_mol = Chem.MolFromSmiles(bytes(target_smi))
    target_fp = fpgen.GetFingerprint(target_mol)

    ## Compute reward for target molecule and puma dataset
    # representations = inference_puma(datamodule, cfg)
    # target_struct_latent = representations['struct'][target_idx]
    # target_morph_latent = representations['morph'][target_idx]
    # target_reward = cosine_similarity(target_struct_latent.reshape(1, -1), target_morph_latent.reshape(1, -1))[0][0]
    # dataset_rewards = (cosine_similarity(representations['struct'], target_morph_latent.reshape(1, -1)) + 1) / 2
    # print('Target has morph~struct cosine sim: ', target_reward)

    # Obtain molecular fingerprints for PUMA dataset
    dataset_fps = []
    rnd_idx = np.random.choice(len(datamodule.dataset), 2000, replace=False)
    for idx in tqdm(range(len(datamodule.dataset))):
    # for idx in tqdm(rnd_idx):
        smi = datamodule.dataset[idx]["inputs"]["struct"].mols
        mol = Chem.MolFromSmiles(smi)
        fp = fpgen.GetFingerprint(mol)
        dataset_fps.append(fp)

    # Sort molecular fingerprints by decreasing reward
    # dataset_fps = [x for _, x in sorted(zip(dataset_rewards, dataset_fps), reverse=True)]

    fps = {
        "Target": [target_fp],
        "PUMA": dataset_fps,
        # "PUMA": [dataset_fps[i] for i in rnd_idx],
        # "PUMA_top_k": dataset_fps[:50]
    }
    rews = {
        # "Target": [target_reward],
        # "PUMA": dataset_rewards,
    }

    for model_name, run_id in models.items():
        print(f"Running for model: ", model_name)

        # Obtain sampled data from the run
        run_dir = f"{base_dir}/{run_id}"
        values = sqlite_load(f"{run_dir}/train/", sqlite_cols, 1)
        smis, rewards = values['smi'][0], values['fr_0'][0]
        filtered_samples = []
        filtered_fps = []
        high = 0

        # sort_idx = np.argsort(rewards)
        # sorted_smis = np.array(smis)[sort_idx[::-1]]
        # for smi in sorted_smis[:50]:
        #     mol = Chem.MolFromSmiles(smi)
        #     filtered_fps.append(fpgen.GetFingerprint(mol))

        for smi, r in tqdm(zip(smis, rewards)):
            # if r >= reward_threshold:
            #     mol = Chem.MolFromSmiles(smi)
            #     fp = fpgen.GetFingerprint(mol)
            #     filtered_fps.append(fp)
                # if is_new_mode(filtered_fps, fp, sim_threshold=sim_threshold):
                #     filtered_fps.append(fp)
            
            fp = fpgen.GetFingerprint(Chem.MolFromSmiles(smi))
            tanimoto_sim = AllChem.DataStructs.TanimotoSimilarity(target_fp, fp)
            if tanimoto_sim > high:
                high = tanimoto_sim
                print(high, r)
            filtered_fps.append(fp)

        fps[model_name] = filtered_fps
        rews[model_name] = rewards
        print(f"{model_name} found {len(filtered_fps)} modes")
    
    # plot_umap(fps, f"umap_>{reward_threshold}_<{sim_threshold}_target_{target_idx}.png")
    plot_sim_to_target_hist(fps, target_fp, f"tanimoto-sim-to-target_{target_idx}.png")
    # plot_pairwise_sim_hist(fps, k=50, filename=f"tanimoto-sim-betw-samples_{target_idx}.png")
    # plot_reward_hist(rews, f"reward-hist-target_{target_idx}.png")
