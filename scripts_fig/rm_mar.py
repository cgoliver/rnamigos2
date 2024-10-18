"""
Plot relationship between dissimilarity from train set and performance
"""
import pickle
import itertools

import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

from scripts_fig.plot_utils import get_rmscores

rm_scores = get_rmscores()

robin_pockets = ['2GDI_Y_TPP_100',
                 '2QWY_A_SAM_100',
                 '3FU2_C_PRF_101',
                 '5BTP_A_AMZ_106']


def get_score(pocket_id, sm, df):
    try:
        return scores.loc[(df['pocket_id'] == pocket_id) & (df['smiles'] == sm)]['mixed'].iloc[0]
    except IndexError:
        return np.nan


def one_corr(pocket_1, pocket_2, scores, ref_ligs, percentile=.05):
    scores_1 = [get_score(pocket_1, sm, scores) for sm in ref_ligs]
    scores_2 = [get_score(pocket_2, sm, scores) for sm in ref_ligs]

    sm_sorted_1 = sorted(zip(scores_1, ref_ligs), key=lambda x: x[0], reverse=True)
    sm_sorted_2 = sorted(zip(scores_2, ref_ligs), key=lambda x: x[0], reverse=True)

    top_K = int(len(ref_ligs) * percentile)

    sm_keep_1 = set([sm for _, sm in sm_sorted_1[:top_K]])
    sm_keep_2 = set([sm for _, sm in sm_sorted_2[:top_K]])

    overlap = len(sm_keep_1 & sm_keep_2) / len(sm_keep_1 | sm_keep_2)

    return pocket_1, pocket_2, overlap


if __name__ == "__main__":
    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))
    scores = pd.read_csv("outputs/mixed_rdock.csv")
    test_pockets = list(scores['pocket_id'])
    # test_pockets = test_pockets[:5]
    pocket_pairs = list(itertools.combinations(test_pockets, 2))

    """
    for rob in robin_pockets:
        max_sim = rm_scores.loc[rob,rm_scores.index.isin(names_train)].max()
        mean_sim = rm_scores.loc[rob,rm_scores.index.isin(names_train)].mean()
        print(f"{rob} max sim: {max_sim} mean sim: {mean_sim}")

    scores['train_sim_max'] = scores['pocket_id'].apply(lambda x: rm_scores.loc[x,rm_scores.index.isin(names_train)].max())
    scores['train_sim_mean'] = scores['pocket_id'].apply(lambda x: rm_scores.loc[x,rm_scores.index.isin(names_train)].mean())
    
    sns.scatterplot(data=scores, x='train_sim_mean', y='score')
    plt.show()

    sns.scatterplot(data=scores, x='train_sim_max', y='score')
    plt.show()

    

    pair_dists = [rm_scores.loc[p1,p2] for p1, p2 in pocket_pairs]
    sns.distplot(pair_dists)
    plt.title("Test set pairwise pocket similarity")
    plt.show()
    """

    scores = pd.read_csv("outputs/mixed_raw.csv")
    corrs = []
    rows = []

    pocket_names = [f"{n.split('_')[0]}-{n.split('_')[2]}" for n in test_pockets]

    ref_ligs = list(set(scores.loc[scores['pocket_id'] == '1BYJ_A_GE3_30']['smiles']))

    """
    for i, (p1, p2) in enumerate(pocket_pairs):
        _,_, r = one_corr(p1, p2, scores, ref_ligs) 
        corrs.append(r)
        rows.append({'pocket_1': p1, 'pocket_2': p2, 'overlap': r})
        print(i, p1, p2, r, len(pocket_pairs))


    pd.DataFrame(rows).to_csv("outputs/pred_mixed_overlap.csv")
    """
    corrs = list(pd.read_csv("outputs/pred_mixed_overlap.csv")['overlap'])

    mat = np.zeros((len(test_pockets), len(test_pockets)))
    mat[np.triu_indices(len(test_pockets), 1)] = corrs
    mat += mat.T
    np.fill_diagonal(mat, 1)
    ax = sns.heatmap(mat, vmin=0, vmax=1, annot=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title("Pairwise Jaccard overlap of top 5% ligands for test set pockets")
    plt.show()

    sns.distplot(corrs)
    plt.show()

    # scatter: jaccard vs rm score

    rms = [rm_scores.loc[p1, p2] for p1, p2 in pocket_pairs]
    plt.scatter(rms, corrs)
    plt.xlabel("Pocket RM score")
    plt.ylabel("Prediction Similarity")
    plt.show()

    # scatter: jaccard vs fp sim of natives

    smiles_list = [scores.loc[(scores['pocket_id'] == p) & scores['is_active']]['smiles'].iloc[0] for p in test_pockets]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]

    rms = [DataStructs.TanimotoSimilarity(fp_1, fp_2) for fp_1, fp_2 in itertools.combinations(fps, 2)]
    plt.scatter(rms, corrs)
    plt.xlabel("Native Ligand Similarity")
    plt.ylabel("Prediction Similarity")
    plt.show()

    # fp novelty vs performance 
    tree = KDTree(fps, leaf_size=2)
    for p in test_pockets:
        print(p)
