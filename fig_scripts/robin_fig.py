from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from rnaglib.drawing import rna_draw
from rnaglib.utils import load_json


def enrichment_factor(scores, is_active, lower_is_better=True, frac=0.01):
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=not lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    return (n_actives_screened / n_screened) / (n_actives / len(scores))


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


# raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinx3dna_raw.csv')
# ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinx3dna.csv')
#
# raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robin_raw.csv')
# ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robin.csv')
#
# raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinprebuilt_raw.csv')
# ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinprebuilt.csv')
#
# raw_df = pd.read_csv('outputs/final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin_x3dna_raw.csv')
# ef_df = pd.read_csv('outputs/final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin_x3dna.csv')
#
# raw_df_2 = pd.read_csv('outputs/final_chembl_native_graphligs_dim64_optimol1_robinprebuilt_raw.csv')
# ef_df_2 = pd.read_csv('outputs/final_chembl_native_graphligs_dim64_optimol1_robinprebuilt.csv')

# raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_raw.csv')
# ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0.csv')


# nt_key = 'nt_code'
# colors = {'C': 'red', 'G': 'yellow', 'A': 'blue', 'U': 'green'}
#

import os

pocket_names = [
    "2GDI_Y_TPP_100",
    "5BTP_A_AMZ_106",
    "2QWY_A_SAM_100",
    "3FU2_C_PRF_101",
]
ligand_names = [
    "TPP",
    "ZTP",
    "SAM_ll",
    "PreQ1",
]


def get_dfs_docking(ligand_name):
    out_dir = 'outputs/robin'

    # Get relevant mapping smiles : normalized score
    docking_df = pd.read_csv(os.path.join(out_dir, "robin_targets_docking.csv"))
    # docking_df = pd.read_csv(os.path.join(out_dir, "robin_targets_docking_consolidated.csv"))
    docking_df = docking_df[docking_df["TARGET"] == ligand_name]
    scores = -docking_df[["TOTAL"]].values.squeeze()

    # DEAL WITH NANS, ACTUALLY WHEN SORTING NANS, THEY GO THE END
    # count = np.count_nonzero(np.isnan(scores))
    # scores = np.sort(scores)
    # cropped_scores = np.concatenate((scores[:10], scores[-10:]))
    # print(cropped_scores)

    scores = np.nan_to_num(scores, nan=np.nanmin(scores))
    # mi = np.nanmin(scores)
    # ma = np.nanmax(scores)
    # print(ma, mi)
    # normalized_scores = (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores))
    # normalized_scores = scores
    normalized_scores = scores / 80
    mapping = {}
    for smiles, score in zip(docking_df[["SMILE"]].values, normalized_scores):
        mapping[smiles[0]] = score
    mapping = defaultdict(int, mapping)

    # Use this mapping to create our actives/inactives distribution dataframe
    active_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "actives.txt")
    # out_path = os.path.join(out_dir, f"{pocket_name}_actives.txt")
    smiles_list = [s.lstrip().rstrip() for s in list(open(active_ligands_path).readlines())]
    actives_df = pd.DataFrame({"docking_score": [mapping[sm] for sm in smiles_list]})
    actives_df['split'] = 'actives'

    # scores = actives_df[["docking_score"]].values.squeeze()
    # ma = np.nanmax(scores)
    # mi = np.nanmin(scores)
    # count = np.count_nonzero(np.isnan(scores))
    # print(f"actives max/min : {ma} {mi}, nancount : {count} "
    #       f"scores over 200 : {np.sum(scores > 200)} length : {len(scores)} ")

    inactives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "decoys.txt")
    # out_path = os.path.join(out_dir, f"{pocket_name}_inactives.txt")
    smiles_list = [s.lstrip().rstrip() for s in list(open(inactives_ligands_path).readlines())]
    inactives_df = pd.DataFrame({"docking_score": [mapping[sm] for sm in smiles_list]})
    inactives_df['split'] = 'inactives'

    # scores = inactives_df[["docking_score"]].values.squeeze()
    # ma = np.nanmax(scores)
    # mi = np.nanmin(scores)
    # count = np.count_nonzero(np.isnan(scores))
    # print(f"inactives max/min : {ma} {mi}, nancount : {count} "
    #       f"scores over 200 : {np.sum(scores > 200)} length : {len(scores)} ")

    merged = pd.concat([actives_df, inactives_df]).reset_index()
    return merged


def get_dfs_migos(pocket_name):
    names = ['smiles', 'dock', 'is_native', 'native_fp', 'merged']
    out_dir = 'outputs/robin'
    actives_df = pd.read_csv(os.path.join(out_dir, f"{pocket_name}_actives.txt"), names=names, sep=' ')
    actives_df['split'] = 'actives'
    inactives_df = pd.read_csv(os.path.join(out_dir, f"{pocket_name}_inactives.txt"), names=names, sep=' ')
    inactives_df['split'] = 'inactives'
    # decoys_df = pd.read_csv(os.path.join(out_dir, f"{pocket_name}_decoys.txt"), names=names, sep=' ')
    # decoys_df['split'] = 'decoys'
    # merged = pd.concat([actives_df, inactives_df, decoys_df])
    # merged = pd.concat([actives_df, decoys_df])
    merged = pd.concat([actives_df, inactives_df])
    return merged


if __name__ == '__main__':

    all_efs = list()
    all_aurocs = list()
    fig, axs = plt.subplots(4)
    for i, (pocket_name, ligand_name) in enumerate(zip(pocket_names, ligand_names)):
        # FOR DOCKING
        # merged = get_dfs_docking(ligand_name=ligand_name)
        # score_to_use = 'docking_score'
        # break

        # FOR MIGOS
        merged = get_dfs_migos(pocket_name=pocket_name)
        score_to_use = 'merged'
        # score_to_use = 'dock'
        # score_to_use = 'is_native'
        # score_to_use = 'native_fp'
        ax = axs[i]
        sns.kdeplot(data=merged, x=score_to_use, hue='split', common_norm=False, clip=(0, 1), ax=ax)
        ax.set_title(pocket_name)
        #     g = load_json(f"data/robin_graphs_x3dna/{name}.json")
        #     g = g.subgraph([n for n,d in g.nodes(data=True) if d['in_pocket'] == True])
        #     print(g.nodes(data=True))
        #     rna_draw(g,
        #              node_colors=[colors[d[nt_key]] for n,d in g.nodes(data=True)],
        #              ax=axs[i][1])
        scores = merged[score_to_use]
        actives = merged['split'].isin(['actives'])

        # GET EFS
        frac = 0.01
        ef = enrichment_factor(scores=scores, is_active=actives,
                               lower_is_better=False, frac=frac)
        all_efs.append(ef)
        print(f'EF@{frac} : ', pocket_name, ef)

        # GET AUROC
        fpr, tpr, thresholds = metrics.roc_curve(actives, scores)
        auroc = metrics.auc(fpr, tpr)
        all_aurocs.append(auroc)
        # print('AuROC : ', pocket_name, auroc)
        # print()

        # mar = mean_active_rank(scores, actives, lower_is_better=False)
        # print(mar)
    #     #ef = f"EF@1\% {list(ef_df.loc[ef_df['pocket_id'] == name]['score'])[0]:.3f}"
    #     axs[i][0].text(0, 0, f"{name} EF: {ef:.3} MAR: {mar:.3}")
    #     axs[i][0].axis("off")
    #     axs[i][1].axis("off")
    #     sns.despine()
    #
    print(np.mean(all_efs))
    print(np.mean(all_aurocs))
    plt.show()
