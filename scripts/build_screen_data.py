from pathlib import Path 

from tqdm import tqdm
import pandas as pd

def build_actives_decoys(pdb_data_path='data/rnamigos2_dataset_consolidated.csv', save_path='data/ligand_db'):
    """ Build active and decoy lists for every pocket in master dataset `rnamigos2_dataset_consolidated.csv`

    Each pocket ID gets a folder:
        pocket_id
            [pdb|pdb+chembl|decoy_finder|robin]
                actives.txt
                decoys.txt
    """

    pdb_df = pd.read_csv(pdb_data_path)

    methods = ['pdb', 'decoy_finder']
    pockets = pdb_df['PDB_ID_POCKET'].unique()

    pdb_ligands = set(pdb_df.loc[pdb_df['LIGAND_SOURCE'] == 'PDB']['LIGAND_SMILES'].unique())
    chembl_ligands = set(pdb_df.loc[pdb_df['LIGAND_SOURCE'] == 'CHEMBL']['LIGAND_SMILES'].unique())

    natives = pdb_df.loc[pdb_df['IS_NATIVE'] == 'YES']

    for pocket in tqdm(natives.itertuples(), total=len(natives)):
        pdb_path = Path(save_path, pocket.PDB_ID_POCKET, 'pdb')
        chembl_path = Path(save_path, pocket.PDB_ID_POCKET, 'pdb_chembl')
        pdb_path.mkdir(parents=True, exist_ok=True)
        chembl_path.mkdir(parents=True, exist_ok=True)

        with open(pdb_path / 'actives.txt', 'w') as ac:
            ac.write(pocket.LIGAND_SMILES)
        with open(chembl_path/ 'actives.txt', 'w') as ac:
            ac.write(pocket.LIGAND_SMILES)
        with open(pdb_path / 'decoys.txt', 'w') as de:
            de.write("\n".join(pdb_ligands - set(pocket.LIGAND_SMILES)))
        with open(chembl_path / 'decoys.txt', 'w') as de:
            de.write("\n".join(list((pdb_ligands | chembl_ligands) - set(pocket.LIGAND_SMILES))))
            
pass

if __name__ == "__main__":
    build_actives_decoys()
