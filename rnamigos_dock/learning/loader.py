import pickle
import sys
import os
import time
from collections import Counter
import itertools
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem

import networkx as nx
from tqdm import tqdm
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from dgl.dataloading import GraphDataLoader
import pandas as pd

from rnaglib.utils import graph_io
from rnaglib.utils import NODE_FEATURE_MAP

RDLogger.DisableLog('rdApp.*')  # disable warnings

script_dir = os.path.dirname(__file__)


def mol_encode_one(smiles, fp_type):
    success = False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if fp_type == 'MACCS':
            # for some reason RDKit maccs is 167 bits
            fp = list(map(int, MACCSkeys.GenMACCSKeys(mol).ToBitString()))[1:]
        if fp_type == 'morgan':
            fp = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()))
        success = True
    except:
        if fp_type == 'MACCS':
            fp = [0] * 166
        if fp_type == 'morgan':
            fp = [0] * 1024
    fp = np.asarray(fp)
    return fp, success


def mol_encode_list(smiles_list, fp_type, encoding_func=mol_encode_one):
    fps = []
    ok_inds = []
    for i, sm in tqdm(enumerate(smiles_list), total=len(smiles_list)):
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

    def encode_mol(self, smiles):
        if smiles in self.cashed_fps:
            return self.cashed_fps[smiles], True
        return mol_encode_one(smiles, self.fp_type)

    def encode_list(self, smiles_list):
        return mol_encode_list(smiles_list, fp_type=self.fp_type, encoding_func=self.encode_mol)


class DockingDataset(Dataset):

    def __init__(self,
                 annotated_path,
                 edge_types=None,
                 nuc_types=None,
                 shuffle=False,
                 target='dock',
                 seed=0,
                 debug=False
                 ):
        """
            Setup for data loader.

            Arguments:
                annotated_path (str): path to annotated graphs (see `annotator.py`).
                get_sim_mat (bool): whether to compute a node similarity matrix (deault=True).
                nucs (bool): whether to include nucleotide ID in node (default=False).
        """
        print(f">>> fetching data from {annotated_path}")
        self.path = annotated_path
        self.all_graphs = sorted(os.listdir(annotated_path))
        if debug:
            self.all_graphs = self.all_graphs[:100]
        if seed:
            print(f">>> shuffling with random seed {seed}")
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.all_graphs)
        # build edge map
        self.edge_map = {e: i for i, e in enumerate(sorted(edge_types))}
        self.num_edge_types = len(self.edge_map)
        self.target = target

        self.n = len(self.all_graphs)

    def __len__(self):
        return self.n

    def load_rna_graph(self, idx, rna_only=True):
        data = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))

        graph = data[1]
        graph = nx.to_directed(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label.upper()]) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}
        nx.set_edge_attributes(graph, name='edge_type', values=one_hot)

        node_attrs = None
        one_hot_nucs = {node: NODE_FEATURE_MAP['nt_code'].encode(label) for node, label in
                        (nx.get_node_attributes(graph, 'nt')).items()}

        nx.set_node_attributes(graph, name='nt_features', values=one_hot_nucs)

        # g_dgl = dgl.DGLGraph()
        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['edge_type'], node_attrs=['nt_features'])
        g_dgl.title = self.all_graphs[idx]

        if rna_only:
            return g_dgl
        else:
            # _, graph, _, ring, fp_nat, fp, inter_score, inter_score_trans, score_native_ligand, label_native_lig, label_1std, label_2std, label_thr_min30, label_thr_min17, label_thr_min12, label_thr_min8, label_thr_0, sample_type, is_native
            fp_nat = data[4]
            fp_docked = data[5]
            is_native = data[-1]
            inter_score_trans = data[6]
            return g_dgl, fp_nat, fp_docked, is_native, inter_score_trans

    def __getitem__(self, idx):
        """
            Returns one training item at index `idx`.
        """
        g_dgl, fp_nat, fp_docked, is_native, inter_score_trans = self.load_rna_graph(idx, rna_only=False)

        if self.target == 'native_fp':
            target = fp_nat
        if self.target == 'is_native':
            target = is_native
        if self.target == 'dock':
            target = inter_score_trans
        else:
            target = torch.tensor(0, dtype=torch.float)

        return g_dgl, fp_docked, torch.tensor(target, dtype=torch.float), [idx]


def get_systems(target='dock', split=None, fp_split=None, fp_split_train=True):
    """
    
    :param target: The systems to load 
    :param split: None or one of 'train', 'val', 'test'
    :param fp_split: For fp, and following RNAmigos1, there is a special splitting procedure that uses 10 fixed splits.
    :param fp_split_train: For a given fp split, the test systems have a one label. Set this param to False to get test
    systems. 
    :return:
    """
    assert split in {None, 'train', 'val', 'test'}
    assert split is None or fp_split.startswith("split_test_")
    if target == 'dock':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/docking_data.csv')
    elif target == 'fp':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/fp_data.csv')
    elif target == 'binary':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/binary_data.csv')
    else:
        raise ValueError
    systems = pd.read_csv(interactions_csv, index_col=0)
    if split is not None:
        systems = systems.loc[systems['SPLIT'] == split]
    if fp_split is not None:
        systems = systems.loc[systems['SPLIT'] == (not fp_split_train)]
    return systems


class DockingDatasetVincent(Dataset):

    def __init__(self,
                 pockets_path,
                 edge_types=None,
                 target='dock',
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
        self.systems = get_systems(target=target)
        if debug:
            self.all_graphs = self.systems[:100]
        self.pockets_path = pockets_path
        self.target = target

        self.edge_map = {e: i for i, e in enumerate(sorted(edge_types))}
        self.num_edge_types = len(self.edge_map)

        self.ligand_encoder = MolEncoder(fp_type='MACCS')

        self.cache_graphs = cache_graphs
        if cache_graphs:
            all_pockets = set(self.systems['PDB_ID_POCKET'].unique())
            self.all_pockets = {pocket_id: self.load_rna_graph(pocket_id) for pocket_id in all_pockets}

        if seed:
            print(f">>> shuffling with random seed {seed}")
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.systems.values)

    def __len__(self):
        return len(self.systems)

    def load_rna_graph(self, rna_name):
        rna_path = os.path.join(self.pockets_path, f"{rna_name}.json")
        pocket_graph = graph_io.load_json(rna_path)
        one_hot = {edge: torch.tensor(self.edge_map[label.upper()]) for edge, label in
                   (nx.get_edge_attributes(pocket_graph, 'LW')).items()}
        nx.set_edge_attributes(pocket_graph, name='edge_type', values=one_hot)
        one_hot_nucs = {node: NODE_FEATURE_MAP['nt_code'].encode(label) for node, label in
                        (nx.get_node_attributes(pocket_graph, 'nt_code')).items()}
        nx.set_node_attributes(pocket_graph, name='nt_features', values=one_hot_nucs)
        pocket_graph_dgl = dgl.from_networkx(nx_graph=pocket_graph,
                                             edge_attrs=['edge_type'],
                                             node_attrs=['nt_features'])
        return pocket_graph_dgl

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
            pocket_graph = self.load_rna_graph(pocket_id)
        ligand_fp, success = self.ligand_encoder.encode_mol(smiles=ligand_smiles)
        target = ligand_fp if self.target == 'fp' else row[2]
        # print("1 : ", time.perf_counter() - t0)
        return pocket_graph, ligand_fp, target, [idx]


class VirtualScreenDataset(DockingDataset):
    def __init__(self,
                 pockets_path,
                 ligands_path,
                 decoy_mode='pdb',
                 fp_type='MACCS',
                 edge_types=None,
                 nuc_types=None,
                 ):
        super().__init__(pockets_path, edge_types=edge_types, nuc_types=nuc_types)
        self.all_graphs = sorted(os.listdir(pockets_path))
        self.pockets_path = pockets_path
        self.ligands_path = ligands_path
        self.decoy_mode = decoy_mode
        self.fp_type = fp_type
        self.edge_map = {e: i for i, e in enumerate(sorted(edge_types))}
        pass

    def parse_smiles(self, smiles_path):
        return list(open(smiles_path).readlines())

    def __len__(self):
        return len(self.all_graphs)

    def get_pocket_id(self, filename):
        # 1ARJ_#0.1_N_ARG_1_1PE_BIND.nx.p_annot.p
        pieces = filename.split("_")
        return "_".join([pieces[0], pieces[2], pieces[3], pieces[4]])

    def __getitem__(self, idx):
        g_dgl = self.load_rna_graph(idx, rna_only=True)
        pocket_id = self.get_pocket_id(self.all_graphs[idx])
        actives_smiles = self.parse_smiles(Path(self.ligands_path, pocket_id, self.decoy_mode, 'actives.txt'))
        decoys_smiles = self.parse_smiles(Path(self.ligands_path, pocket_id, self.decoy_mode, 'decoys.txt'))

        is_active = np.zeros((len(actives_smiles) + len(decoys_smiles)))
        is_active[:len(actives_smiles)] = 1.

        all_fps, ok_inds = mol_encode_list(actives_smiles + decoys_smiles, fp_type=self.fp_type)

        return g_dgl, torch.tensor(all_fps[ok_inds]), torch.tensor(is_active[ok_inds])


class Loader():
    def __init__(self,
                 dataset,
                 batch_size=128,
                 num_workers=20,
                 shuffle=False,
                 seed=0,
                 debug=False
                 ):
        """
        Wrapper class to call with all arguments and that returns appropriate data_loaders
        :param pocket_path:
        :param ligand_path:
        :param batch_size:
        :param num_workers:
        :param augment_flips: perform numpy flips
        :param ram: store whole thing in RAM
        :param siamese: for the batch siamese technique
        :param full_siamese for the true siamese one
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle

    def get_data_strat(self, k_fold=0):
        n = len(self.dataset)
        indices = list(range(n))
        labels = []
        for idx, item in enumerate(self.dataset):
            labels.append(item[2])

        # collate_block = collate_wrapper()

        if k_fold > 1:
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=k_fold)
            # from sklearn.model_selection import KFold
            # kf = KFold(n_splits=k_fold)
            for train_indices, test_indices in kf.split(np.array(indices), np.array(labels)):
                train_set = Subset(self.dataset, train_indices)
                test_set = Subset(self.dataset, test_indices)

                train_loader = GraphDataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                               num_workers=self.num_workers, collate_fn=None)
                test_loader = GraphDataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                              num_workers=self.num_workers, collate_fn=None)

                yield train_loader, test_loader

        else:
            from sklearn.model_selection import train_test_split

            train_indices, test_indices, train_indices_labels, test_indices_labels = train_test_split(np.array(indices),
                                                                                                      np.array(labels),
                                                                                                      test_size=0.2,
                                                                                                      train_size=0.8,
                                                                                                      random_state=None,
                                                                                                      shuffle=True,
                                                                                                      stratify=np.array(
                                                                                                          labels))

            train_set = Subset(self.dataset, train_indices)
            test_set = Subset(self.dataset, test_indices)

            train_loader = GraphDataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                           num_workers=self.num_workers, collate_fn=None)
            test_loader = GraphDataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                          num_workers=self.num_workers, collate_fn=None)

            # return train_loader, valid_loader, test_loader
            yield train_loader, test_loader

    def get_data(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        train_loader = GraphDataLoader(dataset=train_set, shuffle=self.shuffle, batch_size=self.batch_size,
                                       num_workers=self.num_workers, collate_fn=None)
        test_loader = GraphDataLoader(dataset=test_set, shuffle=self.shuffle, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=None)

        return train_loader, test_loader


def describe_dataset(annotated_path='../data/annotated/pockets_docking_annotated'):
    rdock_scores = pd.DataFrame(columns=['POCKET_ID', 'LABEL_NAT', 'LABEL_1STD', 'LABEL_2STD', 'TOTAL'])
    path = annotated_path
    all_graphs = sorted(os.listdir(annotated_path))
    for g in tqdm(all_graphs):
        # p = pickle.load(open(os.path.join(path, g), 'rb'))
        _, graph, _, ring, fp_nat, _, fp, total_score, label_nat, label_1std, label_2std = pickle.load(
            open(os.path.join(path, g), 'rb'))
        rdock_scores = rdock_scores.append({'POCKET_ID': g,
                                            'LABEL_NAT': str(label_nat),
                                            'LABEL_1STD': str(label_1std),
                                            'LABEL_2STD': str(label_2std),
                                            'TOTAL': str(total_score)}, ignore_index=True)

    pickle.dump(rdock_scores, open('dataset_labels_and_score.p', 'wb'))
    rdock_scores.to_csv('dataset_labels_and_score.csv')


class InferenceLoader(Loader):
    def __init__(self,
                 annotated_path,
                 batch_size=5,
                 num_workers=20):
        super().__init__(
            annotated_path=annotated_path,
            batch_size=batch_size,
            num_workers=num_workers)
        self.dataset.all_graphs = sorted(os.listdir(annotated_path))

    def get_data(self):
        collate_block = collate_wrapper()
        train_loader = GraphDataLoader(dataset=self.dataset,
                                       shuffle=False,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       collate_fn=None)
        return train_loader


if __name__ == '__main__':
    pockets_path = '../../data/json_pockets'
    dataset = DockingDatasetVincent(pockets_path=pockets_path,
                                    target='dock')
    a = dataset[0]
    loader = Loader(dataset=dataset, shuffle=False, seed=99, batch_size=1, num_workers=1)
    data = loader.get_data(k_fold=1)
    pass
