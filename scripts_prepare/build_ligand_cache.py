import os
import sys
import pickle
from pathlib import Path
from joblib import delayed, Parallel
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.ligand_encoding import MolGraphEncoder

ROBIN_POCKETS = {'TPP': '2GDI_Y_TPP_100',
                 'ZTP': '5BTP_A_AMZ_106',
                 'SAM_ll': '2QWY_B_SAM_300',
                 'PreQ1': '3FU2_A_PRF_101'
                 }


def parse_smiles(smiles_path):
    sm_list = list(open(smiles_path).readlines())
    sm_list = [sm.strip() for sm in sm_list]
    return sm_list


def do_one(smiles):
    dgl_graph = MolGraphEncoder().smiles_to_graph_one(smiles)
    return smiles, dgl_graph


if __name__ == "__main__":
    all_ligs = []
    for pocket in ROBIN_POCKETS.values():
        active = parse_smiles(Path('data', 'ligand_db', pocket, 'robin', 'actives.txt'))
        decoy = parse_smiles(Path('data', 'ligand_db', pocket, 'robin', 'decoys.txt'))
        all_ligs.extend(active)
        all_ligs.extend(decoy)
    unique_ligs = list(set(all_ligs))
    result = Parallel(n_jobs=20)(delayed(do_one)(sm) for sm in tqdm(unique_ligs, total=len(unique_ligs)))

    with open('data/ligands/robin_lig_graphs.p', 'wb') as cache:
        pickle.dump(dict(result), cache)
