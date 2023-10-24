"""
Map smiles to a unique ID that the loader can pass
"""

df = pd.read_csv("../data/rnamigos2_dataset_consolidated.csv")
all_smiles = df['LIGAND_SMILES']


