import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra
import torch
import pickle
import sqlite3
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintsFromSmiles

from multimodal_contrastive.utils import utils
from multimodal_contrastive.networks.models import MultiTask_FP_PL
from multimodal_contrastive.data.dataset import TestDataset
from multimodal_contrastive.analysis.utils import make_eval_data_loader

import torch_geometric.data as gd
from gflownet.models.mmc import mol2graph, to_device
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_lightning import LightningDataModule, seed_everything

from gneprop.gneprop_pyg import predict_single

# register custom resolvers if not already registered
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list), replace=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

sqlite_cols = (
    ["smi", "r"] + [f"{a}_{i}" for a in ["fr"] for i in range(1)] + ["ci_beta"]
)

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
    data_dir = "path.to/data_folder"
    try:
        res = np.load(f"{data_dir}/puma_embeddings.npz", allow_pickle=True)
    except FileNotFoundError:
        datamodule, cfg = setup_puma()
        res = inference_puma(datamodule, cfg)
        np.savez(f"{data_dir}/puma_embeddings.npz", **res)
    return res

def load_mmc_model(cfg):
    # Load model from checkpoint
    ckpt_path = "path.to/gmc_proxy.ckpt"
    model = utils.instantiate_model(cfg)
    model = model.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.eval()
    return model.to(device)

def load_assay_pred_model(use_gneprop=False):
    ckpt = 'path.to/assay_oracle.ckpt'
    model = MultiTask_FP_PL.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    return model.to(device)

def load_cluster_pred_model(use_gneprop=False):
    ckpt = 'path.to/cluster_oracle.ckpt'
    model = MultiTask_FP_PL.load_from_checkpoint(ckpt, map_location=device)
    model.eval() 
    return model.to(device)

def get_active_assay_cols(dataset, smi):
    # returns the column indices of the active assays == 1 for a given target smile
    if smi not in dataset.ids: return None
    target_idx = np.where(np.array(dataset.ids)==smi)[0][0]
    active_cols = torch.where(dataset.y[target_idx] == 1)[0]
    if len(active_cols) == 0: return None
    return active_cols

def load_assay_matrix_from_csv():
    data_dir = 'path.to/data_folder'
    dataset = TestDataset(data_dir + 'assay_matrix_discrete_37_assays_canonical.csv')
    return dataset

def load_cluster_labels_from_csv():
    data_dir = 'path.to/data_folder'
    return pd.read_csv(data_dir + 'cluster_matrix.csv', index_col=1)

def sqlite_load(root, columns, num_workers=8, upto=None, begin=0):
    try:
        bar = tqdm(smoothing=0)
        values = defaultdict(lambda: [[] for i in range(num_workers)])
        for i in range(num_workers):
            filename = f"{root}generated_mols_{i}.db"
            if not os.path.exists(filename):
                filename = f"{root}generated_objs_{i}.db"
            con = sqlite3.connect(f"file:{filename}?mode=ro", uri=True, timeout=6)
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

def minmax_norm_datum(runs_datum, cols=[]):
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

def load_datum_from_run(run_dir, run_id, remove_duplicates=True, save_fps=True,
                        load_fps=True, fps_from_file=True, last_k=None, every_k=None):
    run_path = os.path.join(run_dir, run_id)
    if not os.path.exists(run_path):
        print(f"Run {run_id} does not exist")
        return None
    values = sqlite_load(f"{run_path}/train/", sqlite_cols, 1)
    smis, rewards = np.array(values['smi'][0]), np.array(values['fr_0'][0])
    if not load_fps:
        return smis, rewards, None
    original_len = len(smis)
    fps_file = os.path.join(run_path, "fps.npy")

    if remove_duplicates:
        smis, idx = np.unique(smis, return_index=True)
        rewards = rewards[idx]
        print(f"Removed {original_len - len(smis)} duplicates")

    if last_k:
        smis, rewards = smis[-last_k:], rewards[-last_k:]
    
    if every_k:
        # Only keep every k-th element
        smis, rewards = smis[::every_k], rewards[::every_k]

    if os.path.exists(fps_file) and fps_from_file:
        fps = np.load(fps_file)
        fps = list(map(get_fp_from_base64, fps))
        if len(fps) == original_len and remove_duplicates:
            fps = [fps[i] for i in idx]
        if last_k:
            fps = fps[-last_k:]
        assert len(fps) == len(smis), f"fps len {len(fps)} != smis len {len(smis)}"
        print(f"Loaded fps from {fps_file}!")
    else:
        print("Generating fps...")
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        mols = list(map(Chem.MolFromSmiles, tqdm(smis)))
        fps = list(fpgen.GetFingerprints(mols, numThreads=8))
        if save_fps:
            print("Saving fps to file...")
            to_save_fps = np.array([x.ToBase64() for x in tqdm(fps)])
            np.save(fps_file, to_save_fps)
            print(f"Saved fps to {fps_file}")
    
    return fps, rewards, smis

def load_puma_dataset_fps(smis, save_fps=True, force_recompute=False):
    data_dir = "path.to/data_folder"
    try:
        if force_recompute: raise FileNotFoundError
        fps = np.load(f"{data_dir}/puma_fingerprints.npy", allow_pickle=True)
        fps = list(map(get_fp_from_base64, fps))
    except FileNotFoundError:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        mols = list(map(Chem.MolFromSmiles, tqdm(smis)))
        fps = fpgen.GetFingerprints(mols, numThreads=8)
        if save_fps:
            to_save_fps = np.array([x.ToBase64() for x in tqdm(fps)])
            np.save(f"{data_dir}/puma_fingerprints.npy", to_save_fps)
    return fps

def predict_assay_logits_from_smi(run_path, smis, assay_model, target_active_assay_cols=None, 
                                  save_preds=True, force_recompute=False, use_gneprop=False, verbose=True, skip=False):
    if skip: return None
    try:
        if run_path is None or force_recompute: raise FileNotFoundError
        y_hat = np.load(f"{run_path}/assay_preds.npy")
        print(f"Loaded assay preds from {run_path}/assay_preds.npy")
        print(y_hat.shape)
        return y_hat
    except FileNotFoundError:
        full_df = pd.DataFrame(smis, columns=["SMILES"])
        dataset = TestDataset(full_df, mol_col="SMILES", label_col=None, device=device)
        dataloader = make_eval_data_loader(dataset, batch_size=2048)
        if use_gneprop:
            y_hat = predict_single(assay_model, full_df, aggr="none", batch_size=2048).squeeze()
            print(y_hat.shape)
        else:
            y_hat = []
            for batch in tqdm(dataloader, disable=(not verbose)):
                y_hat.append(assay_model(batch)[0].detach().cpu().numpy())
            y_hat = np.vstack(y_hat)
        if save_preds:
            np.save(f"{run_path}/assay_preds.npy", logit_values)
            print(f"Saved assay preds to {run_path}/assay_preds.npy")
    if target_active_assay_cols == None or use_gneprop:
        return y_hat.reshape(1, -1)
    target_active_assay_cols = target_active_assay_cols.tolist()\
        if torch.is_tensor(target_active_assay_cols) else target_active_assay_cols
    if y_hat.ndim == 1:
        logit_values = y_hat[target_active_assay_cols].reshape(1, -1)
    else:
        logit_values = y_hat[:, target_active_assay_cols].T
    print(logit_values.shape)
    return logit_values

def predict_cluster_logits_from_smi(run_path, smis, cluster_model, target_cluster_col=None, 
                                    save_preds=True, force_recompute=False, use_gneprop=False, verbose=True, skip=False):
    if skip: return None
    try:
        if run_path is None or force_recompute: raise FileNotFoundError
        y_hat = np.load(f"{run_path}/cluster_preds.npy")
        print(f"Loaded cluster preds from {run_path}/cluster_preds.npy")
    except FileNotFoundError:
        full_df = pd.DataFrame(smis, columns=["SMILES"])
        dataset = TestDataset(full_df, mol_col="SMILES", label_col=None, device=device)
        dataloader = make_eval_data_loader(dataset, batch_size=2048)
        if use_gneprop:
            y_hat = predict_single(cluster_model, full_df, aggr="none", batch_size=2048)
            print(y_hat.shape)
        else:
            y_hat = []
            for batch in tqdm(dataloader, disable=(not verbose)):
                y_hat.append(cluster_model(batch)[0].detach().cpu().numpy())
            y_hat = np.vstack(y_hat)
        if save_preds:
            np.save(f"{run_path}/cluster_preds.npy", y_hat)
            print(f"Saved cluster preds to {run_path}/cluster_preds.npy")
    if target_cluster_col == None:
        return y_hat
    if y_hat.ndim == 1:
        logit_values = y_hat[target_cluster_col]
    else:
        logit_values = y_hat[:, target_cluster_col].squeeze()
    print(logit_values.shape)
    return logit_values

def predict_latents_from_smi(cfg, smis, mmc_model=None):
    mmc_model = load_mmc_model(cfg) if mmc_model is None else mmc_model
    graphs = [mol2graph(Chem.MolFromSmiles(smi)) for smi in tqdm(smis)]
    preds = []
    batch_size = 2048
    for i in range(0, len(graphs), batch_size):
        batch = gd.Batch.from_data_list([i for i in graphs[i:i+batch_size] if i is not None])
        batch.to(mmc_model.device if hasattr(mmc_model, 'device') else device)
        pred = mmc_model({"inputs": {"struct": batch}}, mod_name="struct")
        pred = pred.data.cpu().detach().numpy()
        preds.append(pred)
    return np.vstack(preds)

def select_from(array, indices):
    if type(array) is list:
        return [array[i] for i in indices]
    return array[indices]

def remove_duplicates_from_run(run_datum):
    original_len = len(run_datum['smis'])
    smis, idx = np.unique(run_datum['smis'], return_index=True)
    print(f"Removed {original_len - len(smis)} duplicates")
    run_datum = {
        k: select_from(v, idx) for k, v in run_datum.items()
        if type(v) not in [int, float, str]
    }
    return run_datum

def find_modes_from_run_datum(run_datum, sim_threshold=0.7, k=100, min_reward=None, max_reward=None):
    "Returns the indices of the top-k modes in the run_datum"
    return find_modes_from_arrays(
        run_datum['rewards'],
        run_datum['smis'],
        run_datum['fps'] if 'fps' in run_datum else None,
        sim_threshold=sim_threshold,
        k=k,
        min_reward=min_reward,
        max_reward=max_reward
    )

def find_modes_from_arrays(rewards, smis, fps=None, sim_threshold=0.7, k=100, min_reward=None, max_reward=None, return_fps=False):
    assert len(rewards) == len(smis)
    assert fps is None or len(fps) == len(smis)
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    sorted_idx = np.argsort(rewards)[::-1]
    if max_reward:
        sorted_idx = sorted_idx[rewards[sorted_idx] <= max_reward]
    modes_idx = [sorted_idx[0]]
    modes_fps = [fps[sorted_idx[0]]] if fps is not None else [fpgen.GetFingerprint(mols[sorted_idx[0]])]
    for idx in tqdm(sorted_idx):
        if min_reward and rewards[idx] < min_reward: continue
        if len(modes_idx) >= k: break
        fp = fps[idx] if fps is not None else fpgen.GetFingerprint(mols[idx])
        if is_new_mode(modes_fps, fp, sim_threshold=sim_threshold):
            modes_fps.append(fp)
            modes_idx.append(idx)
    if return_fps:
        return modes_idx, modes_fps
    return modes_idx

def num_modes_over_trajs(run_datum, rew_thresh=0.9, sim_thresh=0.7, batch_size=64):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    rewards, smis = run_datum['rewards'], run_datum['smis']
    mols = list(map(Chem.MolFromSmiles, smis))
    fps = run_datum['fps'] if 'fps' in run_datum else None
    num_traj = len(rewards) // batch_size
    num_modes_in_each_traj, modes_fps = np.zeros(num_traj), []
    num_unique_scafs, unique_scafs = np.zeros(num_traj), []
    for i in tqdm(range(num_traj)):
        start, end = i * batch_size, (i + 1) * batch_size
        for j, (r, smi, mol) in enumerate(zip(rewards[start:end], smis[start:end], mols[start:end])):
            fp = fps[start:end][j] if fps is not None else fpgen.GetFingerprint(mol)
            # scaf = Chem.MolToSmiles(MakeScaffoldGeneric(mol))
            if r >= rew_thresh and is_new_mode(modes_fps, fp, sim_threshold=sim_thresh):
                modes_fps.append(fp)
                num_modes_in_each_traj[i] += 1
            # if scaf not in unique_scafs:
                # unique_scafs.append(scaf)
                # num_unique_scafs[i] += 1
    num_modes_overall = np.cumsum(num_modes_in_each_traj)
    num_scafs_overall = np.cumsum(num_unique_scafs)
    return num_modes_overall, num_scafs_overall

def num_modes_lazy(run_datum, rew_thresh=0.9, sim_thresh=0.7, bs=64, return_smis=False):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    rewards, smis = run_datum['rewards'], run_datum['smis']
    fps = run_datum['fps'] if 'fps' in run_datum else None
    num_traj = len(rewards) // bs
    num_modes_in_each_traj, modes_fps, modes_smi = np.zeros(num_traj), [], []
    avg_rew_per_traj = np.zeros(num_traj)
    for i in tqdm(range(num_traj)):
        start, end = i * bs, (i + 1) * bs
        avg_rew_per_traj[i] = np.mean(rewards[start:end])
        for j, (r, smi) in enumerate(zip(rewards[start:end], smis[start:end])):
            if r < rew_thresh: continue
            if fps is not None: fp = fps[start:end][j]
            else: fpgen.GetFingerprint(Chem.MolFromSmiles(smi))
            if is_new_mode(modes_fps, fp, sim_threshold=sim_thresh):
                modes_fps.append(fp)
                modes_smi.append(smi)
                num_modes_in_each_traj[i] += 1
    num_modes_overall = np.cumsum(num_modes_in_each_traj)
    if return_smis:
        return num_modes_overall, avg_rew_per_traj, modes_smi
    return num_modes_overall, avg_rew_per_traj

def load_target_from_path(target_path, mmc_model=None, target_mode="morph"):
    with open(target_path, "rb") as f:
        target = pickle.load(f)
    target_smi = target['inputs']['struct'].mols.decode('utf-8')
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    target_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(target_smi))

    if mmc_model is None:
        return target_smi, target_fp, None, None, None, None

    target["inputs"]["morph"] = target["inputs"]["morph"][None, ...]
    target["inputs"]["joint"]["morph"] = target["inputs"]["joint"]["morph"][None, ...]
    target["inputs"] = to_device(target["inputs"], device=device)
    with torch.no_grad():
        morph_latent = mmc_model(target, mod_name="morph").cpu().numpy().reshape(1, -1)
        struct_latent = mmc_model(target, mod_name="struct").cpu().numpy().reshape(1, -1)
        joint_latent = mmc_model(target, mod_name="joint").cpu().numpy().reshape(1, -1)

    target_latent = morph_latent if target_mode == "morph" else joint_latent
    target_reward = cosine_similarity(struct_latent, target_latent)[0][0]
    return target_smi, target_fp, struct_latent, morph_latent, joint_latent, target_reward
    
def pool_datum(runs_datum, avoid_runs=None, keep_keys=[]):
    pooled_datum = {'pooled': {}}
    for run_name, run_datum in runs_datum.items():
        if run_name in avoid_runs: continue
        for key, value in run_datum.items():
            if key not in keep_keys: continue
            if key not in pooled_datum['pooled']:
                pooled_datum['pooled'][key] = value
            elif type(value) == list:
                pooled_datum['pooled'][key].extend(value)
            elif type(value).__module__ == np.__name__:
                print(key)
                pooled_datum['pooled'][key] = np.concatenate([pooled_datum['pooled'][key], value])
    return pooled_datum

def bin_datum_by_col(bin_col: str, datum, k_bins: int, bin_min=None, bin_max=None, return_bins=False, 
                     samples_per_method=None, toss_bins_with_less_methods=False, ignore_runs=None, subset=[]):
    if ignore_runs is not None:
        datum = {k: v for k, v in datum.items() if k not in ignore_runs}
    if bin_col not in subset:
        subset.append(bin_col)
    # First, we assert that the bin_col is present among all the runs
    for run_name, run_datum in datum.items():
        assert bin_col in run_datum.keys(), f"{bin_col} not found in {run_name} keys"
    # Next, we find the min and max among all runs in the datum for the bin_col
    min_val = min([min(run_datum[bin_col]) for run_datum in datum.values()])
    max_val = max([max(run_datum[bin_col]) for run_datum in datum.values()])
    # If bin_min and bin_max are provided, we use them instead, only if they are tighter than the min and max
    if bin_min is not None: min_val = max(min_val, bin_min)
    if bin_max is not None: max_val = min(max_val, bin_max)
    # We then create k bins between min and max
    bins = np.linspace(min_val, max_val, k_bins)
    # Finally, for each run, we save the bin index for each data point
    print(f"Binning {bin_col} into {k_bins} bins between {min_val} and {max_val}: ")
    print(f"Bins: {bins}")
    datum_bins = {}
    for run_name, run_datum in datum.items():
        bin_indices = np.digitize(run_datum[bin_col], bins)
        datum_bins[run_name] = bin_indices
    # Finally, we compute the bins if return_bins is True, or just return the bin indices
    if not return_bins:
        return datum_bins, bins
    # if samples_per_method is 'auto', we sample the same number of samples per method per bin
    if samples_per_method == 'auto':
        mode = min if toss_bins_with_less_methods else max
        mode_samples_per_bin = [
            mode([ len(np.where(datum_bins[run_name] == i)[0]) for run_name in datum.keys() ])
            for i in range(1, k_bins)
        ]
        samples_per_method = min([x for x in mode_samples_per_bin if x != 0])
    print(f"Sampling {samples_per_method} samples per method per bin")
    binned_datum = {}
    empty_bins = [False] * k_bins
    samples_per_bin = [0] * k_bins
    for run_name, run_datum in datum.items():
        binned_datum[run_name] = {}
        for i in range(1, k_bins+1):
            if empty_bins[i-1]: continue
            idx = np.where(datum_bins[run_name] == i)[0]
            if samples_per_method:
                idx = idx[np.random.choice(len(idx), size=min(samples_per_method, len(idx)), replace=False)]
                if len(idx) < samples_per_method:
                    if toss_bins_with_less_methods: empty_bins[i-1] = True
                    continue
            samples_per_bin[i-1] += len(idx)
            binned_datum[run_name][i] = {
                k: [v[j] for j in idx]
                for k, v in run_datum.items()
                if type(v) not in [float, int, str] and k in subset
            }
    empty_bins = [x for x in range(1,k_bins+1) if empty_bins[x-1] == True]
    return binned_datum, bins, empty_bins, samples_per_bin