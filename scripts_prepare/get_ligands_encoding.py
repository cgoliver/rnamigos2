import os
import sys

import dgl
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from tqdm import tqdm

from rnamigos.learning.ligand_encoding import MolGraphEncoder

interactions_csv_dock = 'data/csvs/docking_data.csv'
systems = pd.read_csv(interactions_csv_dock)

ligands = set(systems['LIGAND_SMILES'].unique())

# GET FPs
morgan_map = {}
maccs_map = {}
morgan_path = 'data/ligands/morgan.p'
maccs_path = 'data/ligands/maccs.p'
failed_smiles = 0
failed_maccs = 0
failed_morgan = 0
for sm in tqdm(ligands):
    try:
        mol = Chem.MolFromSmiles(sm)
    except:
        failed_smiles += 1
        continue
    try:
        # for some reason RDKit maccs is 167 bits
        maccs = (list(map(int, MACCSkeys.GenMACCSKeys(mol).ToBitString()))[1:])
    except:
        maccs = [0] * 166
        failed_maccs += 1
        print(sm)
    try:
        morgan = list(map(int, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString()))
    except:
        morgan = [0] * 1024
        failed_morgan += 1
    morgan = np.asarray(morgan, dtype=bool)
    maccs = np.asarray(maccs, dtype=bool)
    morgan_map[sm] = morgan
    maccs_map[sm] = maccs
print('Failed parsing smiles', failed_smiles)
print('Failed for maccs', failed_maccs)
print('Failed for morgan', failed_morgan)
pickle.dump(morgan_map, open(morgan_path, 'wb'))
pickle.dump(maccs_map, open(maccs_path, 'wb'))

# GET GRAPHS
lig_graphs_map = {}
lig_graphs = 'data/ligands/lig_graphs.p'
failed = 0
mol_graph_encoder = MolGraphEncoder(cache=False)
for i, sm in enumerate(tqdm(ligands)):
    out_graph = mol_graph_encoder.smiles_to_graph_one(sm)
    # buggy smiles
    # CCCCC(=O)NCCOCCOCCOCCNC(=O)C[N]1=C(Sc2c1cccc2)CC1=CC=[N](c2c1cccc2)C
    # This is not parsed by rdkit, and also caused an error for MACCS and morgan, we returned whole zeroes
    if out_graph.num_nodes() <= 1:
        failed += 1
        print("Failed for smiles : ", sm)
        graph = dgl.graph(([], []))
    else:
        graph = out_graph
    lig_graphs_map[sm] = graph
print("Failed on ", failed)
pickle.dump(lig_graphs_map, open(lig_graphs, 'wb'))
