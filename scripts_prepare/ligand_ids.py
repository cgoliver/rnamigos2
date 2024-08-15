""" Map all ligands in ligand_db to an integer ID """

import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":

    decoy_modes = ['pdb', 'pdb_chembl', 'decoy_finder']

    db_path = '../data/ligand_db/'

    all_smiles = set()
    all_pockets = list(os.listdir(Path(db_path)))

    for pocket in tqdm(all_pockets):
        for mode in decoy_modes:
            try:
                actives = [s.lstrip().rstrip() for s in
                           open(Path(db_path, pocket, mode, 'actives.txt', ), 'r').readlines()]
                decoys = [s.lstrip().rstrip() for s in open(Path(db_path, pocket, mode, 'decoys.txt'), 'r').readlines()]
                all_smiles |= set(actives)
                all_smiles |= set(decoys)
            except FileNotFoundError:
                continue

    all_smiles = sorted(list(all_smiles))
    pd.DataFrame({"smiles": all_smiles, "id": list(range(len(all_smiles)))}).to_csv("../data/lig_id.csv")
