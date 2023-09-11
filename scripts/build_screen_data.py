import argparse
from pathlib import Path 
import tempfile

from tqdm import tqdm
import pandas as pd
from loguru import logger
from rdkit import Chem

from decoy_finder import find_decoys


def get_decoyfinder_decoys(smiles, decoy_db="data/decoy_libraries/in-vitro.csv"):
    logger.info(f"Getting decoys for {smiles}")
    with tempfile.TemporaryDirectory() as tdir:
        with open(Path(tdir, "input.txt"), 'w') as inp:
            inp.write(smiles)
        out_path = Path(tdir, 'decoys.sdf')
        for _ in find_decoys(query_files=[(Path(tdir, 'input.txt'))], db_files=[decoy_db], outputfile=str(out_path)):
            pass
        # parse output SDF.
        try:
            decoys = [Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier(str(out_path))]
            logger.info(f"Found {len(decoys)} decoys.")
            return decoys
        except:
            logger.info(f"Failed on {smiles}")
            return []

def build_actives_decoys(pdb_data_path='data/rnamigos2_dataset_consolidated.csv', save_path='data/ligand_db', no_decoyfinder=True):
    """ Build active and decoy lists for every pocket in master dataset `rnamigos2_dataset_consolidated.csv`

    Each pocket ID gets a folder:
        pocket_id
            [pdb|pdb+chembl|decoy_finder|robin]
                actives.txt
                decoys.txt
    """
    if not no_decoyfinder:
        from decoy_finder import find_decoys

    pdb_df = pd.read_csv(pdb_data_path)

    pockets = pdb_df['PDB_ID_POCKET'].unique()

    pdb_ligands = set(pdb_df.loc[pdb_df['LIGAND_SOURCE'] == 'PDB']['LIGAND_SMILES'].unique())
    chembl_ligands = set(pdb_df.loc[pdb_df['LIGAND_SOURCE'] == 'CHEMBL']['LIGAND_SMILES'].unique())

    natives = pdb_df.loc[pdb_df['IS_NATIVE'] == 'YES']

    for pocket in tqdm(natives.itertuples(), total=len(natives)):
        pdb_path = Path(save_path, pocket.PDB_ID_POCKET, 'pdb')
        chembl_path = Path(save_path, pocket.PDB_ID_POCKET, 'pdb_chembl')
        decoyfinder_path = Path(save_path, pocket.PDB_ID_POCKET, 'decoy_finder')

        pdb_path.mkdir(parents=True, exist_ok=True)
        chembl_path.mkdir(parents=True, exist_ok=True)

        if not no_decoyfinder:
            decoyfinder_path.mkdir(parents=True, exist_ok=True)
            decoyfinder_decoys = get_decoyfinder_decoys(pocket.LIGAND_SMILES)

        with open(pdb_path / 'actives.txt', 'w') as ac:
            ac.write(pocket.LIGAND_SMILES)
        with open(chembl_path/ 'actives.txt', 'w') as ac:
            ac.write(pocket.LIGAND_SMILES)
        with open(pdb_path / 'decoys.txt', 'w') as de:
            de.write("\n".join(pdb_ligands - set(pocket.LIGAND_SMILES)))
        with open(chembl_path / 'decoys.txt', 'w') as de:
            de.write("\n".join(list((pdb_ligands | chembl_ligands) - set(pocket.LIGAND_SMILES))))
        with open(chembl_path / 'decoys.txt', 'w') as de:
            de.write("\n".join(list((pdb_ligands | chembl_ligands) - set(pocket.LIGAND_SMILES))))

        if not no_decoyfinder:
            with open(decoyfinder_path / 'decoys.txt', 'w') as de:
                de.write("\n".join(get_decoyfinder_decoys(pocket.LIGAND_SMILES)))

pass

def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-decoyfinder', action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = cline()
    build_actives_decoys(no_decoyfinder=args.no_decoyfinder)
