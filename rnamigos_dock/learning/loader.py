import os
import sys

import dgl
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
from rnaglib.utils import graph_io
from rnaglib.utils import NODE_FEATURE_MAP
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import torch
from torch.utils.data import Dataset

RDLogger.DisableLog('rdApp.*')  # disable warnings

script_dir = os.path.dirname(__file__)


def mol_encode_one(smiles, fp_type):
    success = False
    assert fp_type in {'MACCS', 'morgan'}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if fp_type == 'MACCS':
            # for some reason RDKit maccs is 167 bits
            fp = list(map(int, MACCSkeys.GenMACCSKeys(mol).ToBitString()))[1:]
        else:
            fp = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()))
        success = True
    except:
        if fp_type == 'MACCS':
            fp = [0] * 166
        else:
            fp = [0] * 1024
    fp = np.asarray(fp)
    return fp, success


def mol_encode_list(smiles_list, fp_type, encoding_func=mol_encode_one):
    fps = []
    ok_inds = []
    for i, sm in enumerate(smiles_list):
        fp, success = encoding_func(sm, fp_type=fp_type)
        fps.append(fp)
        if success:
            ok_inds.append(i)
    return np.array(fps), ok_inds


class MolEncoder:
    """
    Stateful encoder for using cashed computations
    """

    def __init__(self, fp_type='MACCS'):
        self.fp_type = fp_type
        cashed_path = os.path.join(script_dir, f'../../data/ligands/{"maccs" if fp_type == "MACCS" else "morgan"}.p')
        self.cashed_fps = pickle.load(open(cashed_path, 'rb'))

    def encode_mol(self, smiles, fp_type=None):
        if smiles in self.cashed_fps:
            return self.cashed_fps[smiles], True
        return mol_encode_one(smiles, self.fp_type)

    def encode_list(self, smiles_list):
        return mol_encode_list(smiles_list, fp_type=self.fp_type, encoding_func=self.encode_mol)


def get_systems(target='dock', split=None, fp_split=None, fp_split_train=True, get_migos1_only=False):
    """
    :param target: The systems to load 
    :param split: None or one of 'TRAIN', 'VALIDATION', 'TEST'
    :param get_migos1_only: Only use the systems present in RNAmigos1
    :param fp_split: For fp, and following RNAmigos1, there is a special splitting procedure that uses 10 fixed splits.
    :param fp_split_train: For a given fp split, the test systems have a one label. Set this param to False to get test
    systems. 
    :return:
    """
    assert split in {None, 'TRAIN', 'VALIDATION', 'TEST'}
    assert fp_split is None or fp_split.startswith("split_test_")
    if target == 'dock':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/docking_data.csv')
    elif target == 'native_fp':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/fp_data.csv')
    elif target == 'is_native':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/binary_data.csv')
    else:
        raise ValueError("train.target should be in {dock, native_fp, is_native}, received : " + target)
    systems = pd.read_csv(interactions_csv, index_col=0)
    if split is not None:
        systems = systems.loc[systems['SPLIT'] == split]
    if fp_split is not None:
        if get_migos1_only:
            systems = systems.loc[systems['IN_MIGOS_1'] == 1]
        systems = systems.loc[systems['SPLIT'] == (not fp_split_train)]
    return systems


def load_rna_graph(rna_path, edge_map):
    pocket_graph = graph_io.load_json(rna_path)
    one_hot = {edge: torch.tensor(edge_map[label.upper()]) for edge, label in
               (nx.get_edge_attributes(pocket_graph, 'LW')).items()}
    nx.set_edge_attributes(pocket_graph, name='edge_type', values=one_hot)
    one_hot_nucs = {node: NODE_FEATURE_MAP['nt_code'].encode(label) for node, label in
                    (nx.get_node_attributes(pocket_graph, 'nt_code')).items()}
    nx.set_node_attributes(pocket_graph, name='nt_features', values=one_hot_nucs)
    pocket_graph_dgl = dgl.from_networkx(nx_graph=pocket_graph,
                                         edge_attrs=['edge_type'],
                                         node_attrs=['nt_features'])
    return pocket_graph_dgl


class DockingDataset(Dataset):

    def __init__(self,
                 pockets_path,
                 systems,
                 edge_types=None,
                 target='dock',
                 fp_type='MACCS',
                 shuffle=False,
                 seed=0,
                 debug=False,
                 cache_graphs=True
                 ):
        """
            Setup for data loader.

            Arguments:
                pockets_path (str): path to annotated graphs (see `annotator.py`).
                get_sim_mat (bool): whether to compute a node similarity matrix (deault=True).
                nucs (bool): whether to include nucleotide ID in node (default=False).
        """
        print(f">>> fetching data from {pockets_path}")
        self.systems = systems
        if debug:
            self.all_interactions = self.systems[:100]
        self.pockets_path = pockets_path
        self.target = target

        self.edge_map = {e: i for i, e in enumerate(sorted(edge_types))}
        self.num_edge_types = len(self.edge_map)

        self.fp_type = fp_type
        self.ligand_encoder = MolEncoder(fp_type=fp_type)

        self.cache_graphs = cache_graphs
        if cache_graphs:
            all_pockets = set(self.systems['PDB_ID_POCKET'].unique())
            self.all_pockets = {pocket_id: load_rna_graph(rna_path=os.path.join(self.pockets_path, f"{pocket_id}.json"),
                                                          edge_map=self.edge_map) for pocket_id in all_pockets}

        if seed:
            print(f">>> shuffling with random seed {seed}")
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.systems.values)

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        """
            Returns one training item at index `idx`.
        """
        # t0 = time.perf_counter()
        row = self.systems.iloc[idx].values
        pocket_id, ligand_smiles = row[0], row[1]
        if self.cache_graphs:
            pocket_graph = self.all_pockets[pocket_id]
        else:
            pocket_graph = load_rna_graph(rna_path=os.path.join(self.pockets_path, f"{pocket_id}.json"),
                                          edge_map=self.edge_map)
        ligand_fp, success = self.ligand_encoder.encode_mol(smiles=ligand_smiles)
        target = ligand_fp if self.target == 'native_fp' else row[2]
        # print("1 : ", time.perf_counter() - t0)
        return pocket_graph, ligand_fp, target, [idx]


class VirtualScreenDataset(DockingDataset):
    def __init__(self,
                 pockets_path,
                 ligands_path,
                 systems,
                 decoy_mode='pdb',
                 fp_type='MACCS',
                 edge_types=None,
                 ):
        super().__init__(pockets_path, systems=systems, edge_types=edge_types, fp_type=fp_type, shuffle=False)
        self.ligands_path = ligands_path
        self.decoy_mode = decoy_mode
        self.all_pockets_id = list(self.systems['PDB_ID_POCKET'].unique())
        pass

    def __len__(self):
        return len(self.all_pockets_id)

    def parse_smiles(self, smiles_path):
        return list(open(smiles_path).readlines())

    def __getitem__(self, idx):
        try:
            pocket_id = self.all_pockets_id[idx]
            if self.cache_graphs:
                pocket_graph = self.all_pockets[pocket_id]
            else:
                pocket_graph = load_rna_graph(rna_path=os.path.join(self.pockets_path, f"{pocket_id}.json"),
                                              edge_map=self.edge_map)
            actives_smiles = self.parse_smiles(Path(self.ligands_path, pocket_id, self.decoy_mode, 'actives.txt'))
            decoys_smiles = self.parse_smiles(Path(self.ligands_path, pocket_id, self.decoy_mode, 'decoys.txt'))

            is_active = np.zeros((len(actives_smiles) + len(decoys_smiles)))
            is_active[:len(actives_smiles)] = 1.

            all_fps, ok_inds = self.ligand_encoder.encode_list(actives_smiles + decoys_smiles)

            return pocket_graph, torch.tensor(all_fps[ok_inds]), torch.tensor(is_active[ok_inds])
        except FileNotFoundError:
            return None, None, None


if __name__ == '__main__':
    pockets_path = '../../data/json_pockets'
    test_systems = get_systems(target='dock', split='TEST')
    dataset = DockingDataset(pockets_path=pockets_path,
                             systems=test_systems,
                             target='dock')
    a = dataset[0]
    pass
