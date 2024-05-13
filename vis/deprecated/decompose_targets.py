import numpy as np
import hydra
import torch
import pandas as pd
from omegaconf import OmegaConf

from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)

from rdkit import Chem
from tqdm import tqdm
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext, _recursive_decompose

# register custom resolvers if not already registered
OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list), replace=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

if __name__ == "__main__":
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

    # Setup a frag building environment, so that we can use the mol_to_graph() function to attempt to 
    # factorize the molecules in the puma dataset into their constituent fragments
    env = FragMolBuildingEnvContext()

    def mol_to_graph(environ, mol):
        """Convert an RDMol to a Graph"""
        assert type(mol) is Chem.Mol
        all_matches = {}
        for fragidx, frag in environ.sorted_frags:
            all_matches[fragidx] = mol.GetSubstructMatches(frag, uniquify=False)
        return _recursive_decompose(environ, mol, all_matches, {}, [], [], 9)

    def go(dataset):
        decomposed_samples = pd.DataFrame(columns=["smiles", "num_nodes"])
        for idx, sample in enumerate(tqdm(dataset)):
            smiles = sample["inputs"]["struct"].mols
            mol = Chem.MolFromSmiles(smiles)
            try:
                g = mol_to_graph(env, mol)
                if g != None:
                    print(idx, f"Factorized with {g.number_of_nodes()} nodes.")
                    decomposed_samples.loc[idx] = [smiles, g.number_of_nodes()]
            except ValueError as e:
                continue
        return decomposed_samples

    decomposed_samples = go(datamodule.dataset)

    # save the samples that were successfully factorized into pandas dataframe
    decomposed_samples.to_csv("decomposed_samples_puma.csv", index=True)
    
