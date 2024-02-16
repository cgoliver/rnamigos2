import os

import pandas
import dgl
from dgl.dataloading import GraphCollator
import itertools
import numpy as np
from numpy import random
from pathlib import Path
import pandas as pd
import pickle
from rdkit import RDLogger
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB
import torch
from torch.utils.data import Dataset, Sampler

from rnamigos_dock.learning.ligand_encoding import MolFPEncoder, MolGraphEncoder
from rnamigos_dock.tools.graph_utils import load_rna_graph

RDLogger.DisableLog('rdApp.*')  # disable warnings

ROBIN_SYSTEMS = """2GDI	TPP
6QN3	GLN
5BTP	AMZ
2QWY	SAM
3FU2	PRF
"""


def rnamigos_1_split(systems, rnamigos1_test_split=0, return_test=False,
                     use_rnamigos1_train=False, use_rnamigos1_ligands=False):
    """

    :param systems: a dataframe to filter
    :param use_rnamigos1_train: if True, only pockets in the original data are used for training
    :param rnamigos1_test_split: An integer between 0 and 10
    :param return_test: If True return the test systems, else return the train set
    :return:
    """
    script_dir = os.path.dirname(__file__)
    interactions_csv_migos1 = os.path.join(script_dir, '../../data/csvs/rnamigos1_dataset.csv')
    systems_migos_1 = pd.read_csv(interactions_csv_migos1)
    train_split = set()
    test_split = set()
    rnamigos1_ligands = set()
    for i, row in systems_migos_1.iterrows():
        pocket_id = f"{row['pdbid'].upper()}_{row['chain']}_{row['ligand_id']}_{row['ligand_resnum']}"
        rnamigos1_ligands.add(row['native_smiles'])
        if row[-10 + rnamigos1_test_split]:
            test_split.add(pocket_id)
        else:
            train_split.add(pocket_id)
    if use_rnamigos1_ligands:
        systems = systems.loc[systems['LIGAND_SMILES'].isin(rnamigos1_ligands)]
    if return_test:
        systems = systems.loc[systems['PDB_ID_POCKET'].isin(test_split)]
    else:
        # SOME LIGANDS DON'T EXIST ANYMORE (filtered out)
        # in_set = set(systems['PDB_ID_POCKET'].unique())
        # neither = {elt for elt in train_split if elt not in in_set}
        # >>> '5U3G_B_GAI_125', '6S0X_A_ERY_3001', '5J5B_DB_EDO_210', '5T83_A_GAI_116',
        # len(neither) = 283
        if use_rnamigos1_train:
            systems = systems.loc[systems['PDB_ID_POCKET'].isin(train_split)]
        else:
            systems = systems.loc[~systems['PDB_ID_POCKET'].isin(test_split)]
    return systems


def get_systems(target='dock', rnamigos1_split=-1, return_test=False, use_rnamigos1_train=False,
                use_rnamigos1_ligands=False, filter_robin=False, group_pockets=False):
    """
    :param target: The systems to load 
    :param split: None or one of 'TRAIN', 'VALIDATION', 'TEST'
    :param use_rnamigos1_train: Only use the systems present in RNAmigos1
    :param rnamigos1_split: For fp, and following RNAmigos1, there is a special splitting procedure that uses 10 fixed
     splits.
    :param get_rnamigos1_train: For a given fp split, the test systems have a one label. Set this param to False to get test
    systems.
    :param group_pockets: When using RMScores, we end up with redundant clusters. Should we train on all elements of
     the clusters ?
    :return:
    """
    # Can't split twice
    assert rnamigos1_split in {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    script_dir = os.path.dirname(__file__)
    if target == 'dock':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/docking_data.csv')
    elif target == 'native_fp':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/fp_data.csv')
    elif target == 'is_native':
        interactions_csv = os.path.join(script_dir, '../../data/csvs/binary_data.csv')
    else:
        raise ValueError("train.target should be in {dock, native_fp, is_native}, received : " + target)
    systems = pd.read_csv(interactions_csv, index_col=0)
    if rnamigos1_split == -2:
        splits_file = os.path.join(script_dir, '../../data/train_test_75.p')
        train_names, test_names, train_names_grouped, test_names_grouped = pickle.load(open(splits_file, 'rb'))
        if group_pockets:
            train_names = list(train_names_grouped.keys())
            test_names = list(test_names_grouped.keys())
        if return_test:
            systems = systems[systems['PDB_ID_POCKET'].isin(test_names)]
        else:
            systems = systems[systems['PDB_ID_POCKET'].isin(train_names)]
    elif rnamigos1_split == -1:
        split = 'TEST' if return_test else 'TRAIN'
        # systems_train = systems.loc[systems['SPLIT'] == 'TRAIN']
        # systems_val = systems.loc[systems['SPLIT'] == 'VALIDATION']
        # systems_test = systems.loc[systems['SPLIT'] == 'TEST']
        # unique_train = set(systems_train['PDB_ID_POCKET'].unique())
        # unique_val = set(systems_val['PDB_ID_POCKET'].unique())
        # unique_test = set(systems_test['PDB_ID_POCKET'].unique())
        # all_sys = unique_train.union(unique_val).union(unique_test)
        systems = systems.loc[systems['SPLIT'] == split]
        # Remove robin systems from the train
        if filter_robin and split == 'TRAIN':
            unique_train = set(systems['PDB_ID_POCKET'].unique())
            shortlist_to_avoid = set()
            for robin_sys in ROBIN_SYSTEMS.splitlines():
                robin_pdb_id = robin_sys.split()[0]
                to_avoid = {s for s in unique_train if s.startswith(robin_pdb_id)}
                shortlist_to_avoid = shortlist_to_avoid.union(to_avoid)
            systems = systems[~systems['PDB_ID_POCKET'].isin(shortlist_to_avoid)]
    else:
        systems = rnamigos_1_split(systems,
                                   rnamigos1_test_split=rnamigos1_split,
                                   return_test=return_test,
                                   use_rnamigos1_train=use_rnamigos1_train,
                                   use_rnamigos1_ligands=use_rnamigos1_ligands)
    return systems


def stretch_values(value):
    """
    Takes a value in 0,1 and strech lower ones
    :param value:
    :return:
    """
    return value ** (1 / 3)


class DockingDataset(Dataset):

    def __init__(self,
                 pockets_path,
                 systems,
                 target='dock',
                 use_normalized_score=False,
                 stretch_scores=False,
                 fp_type='MACCS',
                 use_graphligs=False,
                 shuffle=False,
                 seed=0,
                 debug=False,
                 cache_graphs=True,
                 undirected=False,
                 use_rings=False
                 ):
        """
            Setup for data loader.

            Arguments:
                pockets_path (str): path to annotated graphs (see `annotator.py`).
                get_sim_mat (bool): whether to compute a node similarity matrix (deault=True).
                nucs (bool): whether to include nucleotide ID in node (default=False).
        """

        # Get systems
        self.systems = systems
        if debug:
            self.all_interactions = self.systems[:100]
        if seed:
            print(f">>> shuffling with random seed {seed}")
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.systems.values)

        # Setup task and values
        self.target = target
        self.use_normalized_score = use_normalized_score
        self.stretch_scores = stretch_scores

        # Setup Ligands
        self.fp_type = fp_type
        self.use_graphligs = use_graphligs
        self.ligand_encoder = MolFPEncoder(fp_type=fp_type)
        self.ligand_graph_encoder = MolGraphEncoder() if use_graphligs else None

        # Setup pockets
        self.undirected = undirected
        self.num_edge_types = len(EDGE_MAP_RGLIB)
        self.pockets_path = pockets_path
        self.cache_graphs = cache_graphs
        self.use_rings = use_rings
        if cache_graphs:
            all_pockets = set(self.systems['PDB_ID_POCKET'].unique())
            self.all_pockets = {pocket_id: load_rna_graph(rna_path=os.path.join(self.pockets_path, f"{pocket_id}.json"),
                                                          undirected=self.undirected,
                                                          use_rings=self.use_rings) for pocket_id in all_pockets}
        print('done caching')

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        """
            Returns one training item at index `idx`.
        """
        # t0 = time.perf_counter()
        row = self.systems.iloc[idx]
        pocket_id, ligand_smiles = row['PDB_ID_POCKET'], row['LIGAND_SMILES']
        if self.cache_graphs:
            pocket_graph, rings = self.all_pockets[pocket_id]
        else:
            pocket_graph, rings = load_rna_graph(rna_path=os.path.join(self.pockets_path, f"{pocket_id}.json"),
                                                 undirected=self.undirected,
                                                 use_rings=self.use_rings)
        ligand_fp = self.ligand_encoder.smiles_to_fp_one(smiles=ligand_smiles)

        # Maybe return ligand as a graph.
        if self.use_graphligs:
            lig_graph = self.ligand_graph_encoder.smiles_to_graph_one(smiles=ligand_smiles)
        else:
            lig_graph = None
        if self.target == 'native_fp':
            target = ligand_fp
        elif self.target == 'dock':
            if not self.use_normalized_score:
                target = row['INTER'] / 40
            else:
                target = row['normalized_values']
                if self.stretch_scores:
                    target = stretch_values(target)
        else:
            target = row['IS_NATIVE']
        # print("1 : ", time.perf_counter() - t0)
        return {'graph': pocket_graph,
                'ligand_input': lig_graph if self.use_graphligs else ligand_fp,
                'target': target,
                'rings': rings,
                'idx': [idx]}


class NativeSampler(Sampler):
    def __init__(self, systems_dataframe):
        super().__init__(data_source=None)
        positive = (systems_dataframe['IS_NATIVE'] == 1).values
        self.positive_rows = np.where(positive)[0]
        self.negative_rows = np.where(1 - positive)[0]
        self.num_pos = len(self.positive_rows)

    def __iter__(self):
        selected_neg_rows = np.random.choice(self.negative_rows,
                                             self.num_pos,
                                             replace=False)
        systems = np.concatenate((selected_neg_rows, self.positive_rows))
        np.random.shuffle(systems)
        yield from systems

    def __len__(self) -> int:
        return self.num_pos * 2


class RingCollater():
    def __init__(self, node_simfunc=None, max_size_kernel=None):
        self.node_simfunc = node_simfunc
        self.max_size_kernel = max_size_kernel
        self.graph_collator = GraphCollator()

    def k_block(self, node_rings):
        # We need to reimplement because current one expects dicts
        block = np.zeros((len(node_rings), len(node_rings)))
        assert self.node_simfunc.compare(node_rings[0],
                                         node_rings[0]) > 0.99, "Identical rings giving non 1 similarity."
        sims = [self.node_simfunc.compare(n1, n2)
                for i, (n1, n2) in enumerate(itertools.combinations(node_rings, 2))]
        block[np.triu_indices(len(node_rings), 1)] = sims
        block += block.T
        block += np.eye(len(node_rings))
        return block

    def collate(self, items):
        batch = {}
        for key, value in items[0].items():
            values = [d[key] for d in items]
            if key == 'rings':
                if self.node_simfunc is None:
                    batch[key] = None, None
                    continue
                # Taken from rglib
                flat_rings = list()
                for ring in values:
                    flat_rings.extend(ring)
                if self.max_size_kernel is None or len(flat_rings) < self.max_size_kernel:
                    # Just take them all
                    node_ids = [1 for _ in flat_rings]
                else:
                    # Take only 'max_size_kernel' elements
                    node_ids = [1 for _ in range(self.max_size_kernel)] + \
                               [0 for _ in range(len(flat_rings) - self.max_size_kernel)]
                    random.shuffle(node_ids)
                    flat_rings = [node for i, node in enumerate(flat_rings) if node_ids[i] == 1]
                k_block = self.k_block(flat_rings)
                batch[key] = torch.from_numpy(k_block).detach().float(), node_ids
            else:
                batch[key] = self.graph_collator.collate(items=values)
        return batch


class VirtualScreenDataset(DockingDataset):
    def __init__(self,
                 pockets_path,
                 ligands_path,
                 systems,
                 decoy_mode='pdb',
                 rognan=False,
                 **kwargs
                 ):
        super().__init__(pockets_path, systems=systems, shuffle=False, **kwargs)
        self.ligands_path = ligands_path
        self.decoy_mode = decoy_mode
        self.all_pockets_names = list(self.systems['PDB_ID_POCKET'].unique())
        self.rognan = rognan
        pass

    def __len__(self):
        return len(self.all_pockets_names)

    def parse_smiles(self, smiles_path):
        return list(open(smiles_path).readlines())

    def __getitem__(self, idx):
        try:
            if self.rognan:
                pocket_name = self.all_pockets_names[np.random.randint(0, len(self.all_pockets_names))]
            else:
                pocket_name = self.all_pockets_names[idx]
            if self.cache_graphs:
                pocket_graph, _ = self.all_pockets[pocket_name]
            else:
                pocket_graph, _ = load_rna_graph(rna_path=os.path.join(self.pockets_path, f"{pocket_name}.json"),
                                                 use_rings=False)

            actives_smiles = self.parse_smiles(
                Path(self.ligands_path, self.all_pockets_names[idx], self.decoy_mode, 'actives.txt'))
            decoys_smiles = self.parse_smiles(
                Path(self.ligands_path, self.all_pockets_names[idx], self.decoy_mode, 'decoys.txt'))
            all_smiles = actives_smiles + decoys_smiles

            is_active = np.zeros(len(all_smiles))
            is_active[:len(actives_smiles)] = 1.

            if self.use_graphligs:
                all_inputs = self.ligand_graph_encoder.smiles_to_graph_list(all_smiles)
            else:
                all_inputs = self.ligand_encoder.smiles_to_fp_list(all_smiles)
                all_inputs = torch.tensor(all_inputs)
            return pocket_name, pocket_graph, all_inputs, torch.tensor(is_active), all_smiles
        except FileNotFoundError:
            return None, None, None, None, None


class InferenceDataset(Dataset):
    def __init__(self,
                 smiles_list,
                 ):
        self.smiles_list = smiles_list
        self.ligand_graph_encoder = MolGraphEncoder(cache=False)
        self.ligand_encoder = MolFPEncoder()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        ligand = self.smiles_list[idx]
        encoded_ligand_graphs = self.ligand_graph_encoder.smiles_to_graph_one(ligand)
        encoded_ligand_fp = self.ligand_encoder.smiles_to_fp_one(ligand)
        return encoded_ligand_graphs, encoded_ligand_fp

    def collate(self, ligands):
        batch_graph = dgl.batch([x[0] for x in ligands])
        batch_vector = np.array([x[1] for x in ligands])
        batch_vector = torch.tensor(batch_vector)
        return batch_graph, batch_vector


if __name__ == '__main__':
    pockets_path = '../../data/json_pockets_load'
    test_systems = get_systems(target='dock', return_test=True)
    dataset = DockingDataset(pockets_path=pockets_path,
                             systems=test_systems,
                             target='dock',
                             use_graphligs=True)
    a = dataset[0]
    b = 1
    pass
