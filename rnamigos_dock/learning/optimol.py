import os
import sys
import torch
import dgl
import pandas as pd

import pickle
import json
import networkx as nx
from rdkit import Chem

from torch.utils.data import Dataset


def smiles_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles)

    mol_graph = nx.Graph()

    for atom in mol.GetAtoms():
        mol_graph.add_node(atom.GetIdx(),
                           atomic_num=atom.GetAtomicNum(),
                           formal_charge=atom.GetFormalCharge(),
                           chiral_tag=atom.GetChiralTag(),
                           num_explicit_hs=atom.GetNumExplicitHs(),
                           is_aromatic=atom.GetIsAromatic())

    for bond in mol.GetBonds():
        mol_graph.add_edge(bond.GetBeginAtomIdx(),
                           bond.GetEndAtomIdx(),
                           bond_type=bond.GetBondType())
    return mol_graph

def oh_tensor(category, n):
    # One-hot float tensor construction
    t = torch.zeros(n, dtype=torch.float)
    t[category] = 1.0
    return t


class molDataset(Dataset):
    """
    pytorch Dataset for training on small molecules graphs + smiles
    """

    def __init__(self, csv_path,
                 maps_path, ):

        # =========== 2/ Graphs handling ====================
        with open(os.path.join(maps_path, 'edges_and_nodes_map.pickle'), "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)




if __name__ == '__main__':
    dataset = molDataset(
        csv_path="../../data/csvs/binary_data.csv",
        maps_path="../../data/map_files",
    )
    item = dataset[0]
    h = item.ndata['h']
    # for x in item.nodes(data=True):
    #     a=1
    a = 1
