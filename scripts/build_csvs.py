import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from tqdm import tqdm

interactions_csv_original = '../data/rnamigos2_dataset_consolidated.csv'
interactions_csv_migos1 = '../data/rnamigos_1_data/rnamigos1_dataset.csv'

interactions_csv_dock = '../data/csvs/docking_data.csv'
interactions_csv_fp = '../data/csvs/fp_data.csv'
interactions_csv_binary = '../data/csvs/binary_data.csv'

systems = pd.read_csv(interactions_csv_original)
systems = systems.rename({'TYPE': 'SPLIT'}, axis='columns')

# Get PDB, SMILES for fp predictions
natives = systems.loc[systems['IS_NATIVE'] == 'YES']
systems_fp = natives[['PDB_ID_POCKET', 'LIGAND_SMILES', 'SPLIT']]
systems_fp.to_csv(interactions_csv_fp)

# Get PDB, SMILES, SCORE
systems_dock = systems[['PDB_ID_POCKET', 'LIGAND_SMILES', 'TOTAL', 'SPLIT']]
systems_dock.to_csv(interactions_csv_dock)

# Get PDB, SMILES, 0/1
systems_binary = systems[['PDB_ID_POCKET', 'LIGAND_SMILES', 'IS_NATIVE', 'SPLIT']]
systems_binary['IS_NATIVE'] = systems_binary['IS_NATIVE'].apply(lambda x: 1 if x == 'YES' else 0)
systems_binary.to_csv(interactions_csv_binary)

