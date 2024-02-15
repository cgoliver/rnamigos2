""" Convert consolidated csv with RDOCK to output/ format """
import numpy as np
from pathlib import Path
import pandas as pd


def mean_active_rank(scores, is_active, lower_is_better=True, **kwargs):
    """ Compute the average rank of actives in the scored ligand set

    Arguments
    ----------
    scores (list): list of scalar scores for each ligand in the library
    is_active (list): binary vector with 1 if ligand is active or 0 else, one for each of the scores
    lower_is_better (bool): True if a lower score means higher binding likelihood, False v.v.

    Returns
    ---------
    int
        Mean rank of the active ligand [0, 1], 1 is the best score.
        

    >>> mean_active_rank([-1, -5, 1], [0, 1, 0], lower_is_better=True)
    >>> 1.0

    """
    is_active_sorted = sorted(zip(scores, is_active), reverse=lower_is_better)
    return (np.mean([rank for rank, (score, is_active) in enumerate(is_active_sorted) if is_active]) + 1) / len(scores)


df = pd.read_csv("../data/rnamigos2_dataset_consolidated.csv")
df = df.loc[df['TYPE'] == 'TEST']
decoy_db = Path("../data/ligand_db")

# ,score,metric,data_idx,decoys,pocket_id
# ,raw_score,is_active,pocket_id

ef_rows = []
raw_rows = []
grouped_by_pocket = df.groupby('PDB_ID_POCKET')
for i, (pocket_id, pocket) in enumerate(grouped_by_pocket):
    print("Processing", i, len(grouped_by_pocket))
    for decoy_set in ['pdb', 'pdb_chembl', 'chembl']:
        try:
            actives = open(decoy_db / pocket_id / decoy_set / 'actives.txt', 'r').readlines()
            decoys = open(decoy_db / pocket_id / decoy_set / 'decoys.txt', 'r').readlines()
        except FileNotFoundError:
            print("missing ", pocket_id)
        else:
            scores, is_active, all_smiles = [], [], []
            for sm in actives:
                s = list(pocket.loc[pocket['LIGAND_SMILES'] == sm.strip()]['INTER'])[0]
                # s = list(pocket.loc[pocket['LIGAND_SMILES'] == sm.strip()]['TOTAL'])[0]
                scores.append(s)
                is_active.append(1)
                all_smiles.append(sm)
                raw_rows.append({'raw_score': s,
                                 'is_active': 1,
                                 'smiles': sm.lstrip().rstrip(),
                                 'pocket_id': pocket_id,
                                 'decoys': decoy_set})
            missing_sms = []
            for sm in decoys:
                is_active.append(0)
                all_smiles.append(sm)
                try:
                    s = list(pocket.loc[pocket['LIGAND_SMILES'] == sm.strip()]['INTER'])[0]
                    # s = list(pocket.loc[pocket['LIGAND_SMILES'] == sm.strip()]['TOTAL'])[0]
                    scores.append(s)
                except IndexError:
                    s = np.nan

                raw_rows.append({'raw_score': s,
                                 'is_active': 0,
                                 'smiles': sm.lstrip().rstrip(),
                                 'pocket_id': pocket_id,
                                 'decoys': decoy_set})

            ef = mean_active_rank(scores, is_active)
            ef_rows.append({'score': ef,
                            'metric': 'MAR',
                            'data_idx': 0,
                            'decoys': decoy_set,
                            'pocket_id': pocket_id
                            }
                           )

pd.DataFrame(ef_rows).to_csv("../outputs/rdock.csv")
pd.DataFrame(raw_rows).to_csv("../outputs/rdock_raw.csv")
# pd.DataFrame(ef_rows).to_csv("../outputs/rdock_total.csv")
# pd.DataFrame(raw_rows).to_csv("../outputs/rdock_total_raw.csv")
