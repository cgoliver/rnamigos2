"""
Plot relationship between dissimilarity from train set and performance
"""
import os
import sys

from collections import Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from scipy.spatial.distance import squareform, cdist
import seaborn as sns

from fig_scripts.plot_utils import group_df, get_rmscores, get_smooth_order, rotate_2D_coords


def train_sim_perf_plot(grouped=True):
    """
    Make the scatter plot of performance as a function of similarity to train set
    """
    rmscores = get_rmscores()
    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))
    mixed_res = pd.read_csv(f'outputs/mixed.csv')
    if grouped:
        mixed_res = group_df(mixed_res)
    mixed_res['train_sim_max'] = mixed_res['pocket_id'].apply(
        lambda x: rmscores.loc[x, rmscores.index.isin(names_train)].max())

    sns.scatterplot(data=mixed_res, x='train_sim_max', y='score', alpha=0.7)
    sns.despine()
    plt.xlabel("Max RMscore to train set")
    plt.ylabel("AuROC")
    plt.savefig("figs/train_max_perf.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def get_predictions_pocket(pocket, scores, ref_ligs=None, percentile=0.01):
    pocket_scores = scores.loc[scores['pocket_id'] == pocket]
    pocket_lig_scores = pocket_scores.loc[pocket_scores['smiles'].isin(ref_ligs)] if ref_ligs is not None \
        else pocket_scores
    scores, ligs = pocket_lig_scores[['mixed', 'smiles']].values.T
    sm_sorted = sorted(zip(scores, ligs), key=lambda x: x[0], reverse=True)
    top_K = int(len(ligs) * percentile)
    sm_keep = set([sm for _, sm in sm_sorted[:top_K]])
    return sm_keep


def compute_pred_distances(big_df_raw,
                           out_name="outputs/pred_mixed_overlap_vincent.csv",
                           percentile=0.01,
                           recompute=False):
    """
    Compute the pairwise distance between all pockets
    The similarity between two pockets is based on our predictions: it is computed as the IoU of the top 5%
    """
    if not recompute and os.path.exists(out_name):
        return
    test_pockets = sorted(big_df_raw['pocket_id'].unique())
    pocket_pairs = list(itertools.combinations(test_pockets, 2))
    ref_ligs = list(set(big_df_raw.loc[big_df_raw['pocket_id'] == '1BYJ_A_GE3_30']['smiles']))

    pocket_preds = {}
    for pocket in test_pockets:
        pocket_preds[pocket] = get_predictions_pocket(pocket,
                                                      ref_ligs=ref_ligs,
                                                      scores=big_df_raw,
                                                      percentile=percentile)

    # from collections import Counter
    # all_pred = Counter()
    # for pred in pocket_preds.values():
    #     all_pred += Counter(pred)

    rows = []
    for p1, p2 in (pocket_pairs):
        sm_keep_1 = pocket_preds[p1]
        sm_keep_2 = pocket_preds[p2]
        overlap = len(sm_keep_1 & sm_keep_2) / len(sm_keep_1 | sm_keep_2)
        rows.append({'pocket_1': p1, 'pocket_2': p2, 'overlap': overlap})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_name)


def double_heatmap(corr1, corr2=None,
                   kwargs1={},
                   kwargs2={}):
    default_kwargs_1 = {'cmap': sns.light_palette('royalblue', as_cmap=True),
                        'linewidths': 2,
                        'square': True,
                        'cbar_kws': {"shrink": .95}}
    default_kwargs_2 = {'cmap': sns.light_palette('forestgreen', as_cmap=True),
                        'linewidths': 2,
                        'square': True,
                        'cbar_kws': {"shrink": .95}}
    for default, updated in ((default_kwargs_1, kwargs1), (default_kwargs_2, kwargs2)):
        for key, value in updated.items():
            default[key] = value
    ax = plt.gca()
    if corr2 is None:
        corr2 = corr1
        default_kwargs_2 = default_kwargs_1.copy()
        default_kwargs_2['cbar'] = False
    mask1 = np.tril(np.ones_like(corr1, dtype=bool))
    mask2 = np.triu(np.ones_like(corr2, dtype=bool))
    sns.heatmap(corr1, mask=mask1, ax=ax, **default_kwargs_1)
    sns.heatmap(corr2, mask=mask2, ax=ax, **default_kwargs_2)
    return ax


def sims(grouped=True):
    # PLOT 1
    # train_sim_perf_plot(grouped=grouped)
    # plt.rcParams['figure.figsize'] = (10, 5)

    # Get raw values
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_raw.csv')
    big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'mixed']]
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles', 'is_active'])

    test_pockets = sorted(big_df_raw['pocket_id'].unique())
    pocket_pairs = list(itertools.combinations(test_pockets, 2))

    # Get smooth ordering
    smooth = True
    rmscores = get_rmscores()
    if smooth:
        order = get_smooth_order(pockets=test_pockets, rmscores=rmscores)
    else:
        order = np.arange(len(test_pockets))

    # HEATMAPS

    # COMPUTE PDIST based on overlap of predictions
    out_name = "outputs/pred_mixed_overlap_vincent.csv"
    compute_pred_distances(big_df_raw=big_df_raw, out_name=out_name, percentile=0.1, recompute=False)
    results_df = pd.read_csv(out_name)
    corrs = list(results_df['overlap'])
    square_corrs = squareform(corrs)
    square_corrs = square_corrs[order][:, order]

    # # TEMP PLOT: what is the best percentile to correlate with RMScores ?
    # import scipy
    # rms = [rmscores.loc[p1, p2] for p1, p2 in pocket_pairs]
    # all_corrs = []
    # every_n = 3
    # for thresh in range(0, 120, every_n):
    #     compute_pred_distances(big_df_raw=big_df_raw, out_name=out_name, percentile=(thresh + 3) / 501)
    #     results_df = pd.read_csv(out_name)
    #     corrs = list(results_df['overlap'])
    #     correlation = scipy.stats.pearsonr(corrs, rms)[0]
    #     all_corrs.append(correlation)
    # plt.plot(every_n * np.arange(len(all_corrs)), all_corrs)
    # plt.show()

    # PLOT THOSE DISTS
    # ax = double_heatmap(corr1=square_corrs)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # plt.tight_layout()
    # sns.despine()
    # # plt.savefig("figs/fig_2c.pdf", format="pdf")
    # plt.show()

    # COMPARE jaccard vs RMscore
    rms = [rmscores.loc[p1, p2] for p1, p2 in pocket_pairs]
    square_rms = squareform(rms)
    square_rms = square_rms[order][:, order]

    # plt.scatter(rms, corrs, alpha=0.7)
    # plt.xlabel("Pocket RM score")
    # plt.ylabel("Prediction Similarity")
    # sns.despine()
    # plt.tight_layout()
    # plt.savefig("figs/fig_2d.pdf", format="pdf")
    # plt.show()

    ax = double_heatmap(corr1=square_corrs, corr2=square_rms, kwargs2={'vmax': 0.7, 'vmin': 0.2})
    plt.savefig("figs/rmscores_preds.pdf")
    plt.show()

    # COMPARE: jaccard vs fp sim of natives
    smiles_list = [big_df_raw.loc[(big_df_raw['pocket_id'] == p) & big_df_raw['is_active']]['smiles'].iloc[0] for p in
                   test_pockets]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    tani = [DataStructs.TanimotoSimilarity(fp_1, fp_2) for fp_1, fp_2 in itertools.combinations(fps, 2)]
    square_tani = squareform(tani)
    square_tani = square_tani[order][:, order]
    palette_lig = sns.light_palette('navy', as_cmap=True)
    ax = double_heatmap(corr1=square_corrs, corr2=square_tani, kwargs2={'cmap': palette_lig})
    plt.savefig("figs/tanimotos_preds.pdf")
    plt.show()

    ax = double_heatmap(corr1=square_tani,
                        corr2=square_rms,
                        kwargs1={'cmap': palette_lig},
                        kwargs2={'cmap': sns.light_palette('forestgreen'), 'vmax': 0.7, 'vmin': 0.2}, )
    plt.savefig("figs/rmscores_ligands.pdf")
    plt.show()

    # plt.scatter(rms, corrs, alpha=0.7)
    # plt.xlabel("Native Ligand Similarity")
    # plt.ylabel("Prediction Similarity")
    # plt.savefig("figs/fig_2e.pdf", format="pdf")
    # sns.despine()
    # plt.tight_layout()
    # plt.show()
    pass


def tsne(grouped=True):
    # GET POCKET SPACE EMBEDDINGS
    # Get the test_pockets that were used
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_raw.csv')
    big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'mixed']]
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles', 'is_active'])
    test_pockets = set(big_df_raw['pocket_id'].unique())

    # # Complement with train pockets for MDS
    # (train_names, test_names, train_names_grouped, _) = pickle.load(open("data/train_test_75.p", 'rb'))
    # if grouped:
    #     all_pockets = set(train_names_grouped.keys()).union(test_pockets)
    # else:
    #     all_pockets = train_names.union(test_names)
    #
    # # Subset RMscores to get only train+test pockets
    # rmscores = get_rmscores()
    # rmscores_idx = [pocket in all_pockets for pocket in rmscores.columns]
    # rmscores = rmscores.iloc[rmscores_idx, rmscores_idx]
    # all_pockets=rmscores.columns
    # dists = 1 - rmscores.values
    # distance_symmetrized = (dists + dists.T) / 2
    # # X_embedded_pocket = TSNE(n_components=2, init='random', metric='precomputed', learning_rate='auto',
    # #                          ).fit_transform(distance_symmetrized)
    # X_embedded_pocket = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distance_symmetrized)
    # X_embedded_pocket = preprocessing.MinMaxScaler().fit_transform(X_embedded_pocket)
    # pickle.dump((X_embedded_pocket, all_pockets, test_pockets), open('temp_pockets.p', 'wb'))
    X_embedded_pocket, all_pockets, test_pockets = pickle.load(open('temp_pockets.p', 'rb'))
    # plt.scatter(X_embedded_pocket[:, 0], X_embedded_pocket[:, 1],
    #             c=['green' if pocket in test_pockets else 'grey' for pocket in all_pockets],
    #             s=[20 if pocket in test_pockets else 0.5 for pocket in all_pockets],
    #             alpha=.7)
    # plt.show()

    # # GET LIGANDS SPACE EMBEDDINGS
    # smiles_list = sorted(big_df_raw['smiles'].unique())
    # active_smiles = set(big_df_raw.loc[big_df_raw['is_active'] == 1]['smiles'].unique())
    # mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    # fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    # tani = [DataStructs.TanimotoSimilarity(fp_1, fp_2) for fp_1, fp_2 in itertools.combinations(fps, 2)]
    # dists = 1 - squareform(tani)
    # distance_symmetrized = (dists + dists.T) / 2
    # # X_embedded_lig = TSNE(n_components=2, learning_rate='auto', metric='precomputed', init='random').fit_transform(
    # #     distance_symmetrized)
    # X_embedded_lig = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distance_symmetrized)
    # X_embedded_lig = preprocessing.MinMaxScaler().fit_transform(X_embedded_lig)
    # pickle.dump((X_embedded_lig, smiles_list, active_smiles), open('temp_ligs.p', 'wb'))
    X_embedded_lig, smiles_list, active_smiles = pickle.load(open('temp_ligs.p', 'rb'))
    # plt.scatter(X_embedded_lig[:, 0], X_embedded_lig[:, 1],
    #             c=['blue' if sm in active_smiles else 'grey' for sm in smiles_list],
    #             s=[20 if sm in active_smiles else 0.5 for sm in smiles_list],
    #             alpha=.7)
    # plt.show()

    x_offset = -0.5
    # x_offset = 0.
    y_offset = -0.5
    # y_offset = 0.
    z_offset = 2
    X_embedded_pocket[:, 0] += x_offset
    X_embedded_pocket[:, 1] += y_offset

    # GET LINKS
    smiles_to_ind = {smiles: i for i, smiles in enumerate(smiles_list)}
    pocket_to_ind = {pocket: i for i, pocket in enumerate(all_pockets)}
    pred_links = []
    found_gt_links = []
    missed_gt_links = []
    for i, pocket in enumerate(test_pockets):
        # if i > 3: break
        # pred links
        pocket_id = pocket_to_ind[pocket]
        pred_smiles = get_predictions_pocket(pocket,
                                             scores=big_df_raw,
                                             percentile=0.05)
        for smiles in pred_smiles:
            smile_id = smiles_to_ind[smiles]
            pred_links.append((pocket_id, smile_id))
        # GT links
        pocket_df = big_df_raw.loc[big_df_raw['pocket_id'] == pocket]
        actives_smiles = pocket_df.loc[pocket_df['is_active'].astype(bool)]['smiles']
        for smiles in actives_smiles:
            smile_id = smiles_to_ind[smiles]
            if smiles in pred_smiles:
                found_gt_links.append((pocket_id, smile_id))
            else:
                missed_gt_links.append((pocket_id, smile_id))

    pred_links = np.asarray(pred_links)
    found_gt_links = np.asarray(found_gt_links)
    missed_gt_links = np.asarray(missed_gt_links)

    def find_best_angle(X_pocket, X_ligand, links):
        """
        Rotate one of the spaces to minimize crossings.
        """
        fixed_x_embs = X_pocket.copy()
        all_angles = list()
        all_dists = list()
        for angle in range(0, 360, 10):
            rotated_X_pocket = rotate_2D_coords(fixed_x_embs, angle=angle)
            pocket_coords = rotated_X_pocket[links[:, 0]]
            ligand_coords = X_ligand[links[:, 1]]
            dists = cdist(pocket_coords, ligand_coords)
            all_angles.append(angle)
            all_dists.append(np.mean(dists))
        plt.plot(all_angles, all_dists)
        plt.show()
        return all_angles[np.argmin(all_dists)]

    # all_links = np.concatenate((pred_links, found_gt_links, missed_gt_links))
    # best_angle = find_best_angle(X_embedded_pocket, X_embedded_lig, all_links)
    # print(best_angle)
    best_angle = 300

    # PLOT 3D
    ax = plt.axes(projection='3d')
    X_embedded_pocket = rotate_2D_coords(X_embedded_pocket, angle=best_angle)
    ax.scatter(X_embedded_pocket[:, 0], X_embedded_pocket[:, 1], z_offset * np.ones(len(X_embedded_pocket)),
               c=['forestgreen' if pocket in test_pockets else 'grey' for pocket in all_pockets],
               s=[20 if pocket in test_pockets else 0.5 for pocket in all_pockets],
               alpha=.9)
    ax.scatter(X_embedded_lig[:, 0], X_embedded_lig[:, 1], np.zeros(len(X_embedded_lig)),
               c=['navy' if sm in active_smiles else 'grey' for sm in smiles_list],
               s=[20 if sm in active_smiles else 0.5 for sm in smiles_list],
               alpha=.9)

    # Get coords for each kind of link
    pred_links_pocket_coords = X_embedded_pocket[pred_links[:, 0]]
    pred_links_ligand_coords = X_embedded_lig[pred_links[:, 1]]
    found_gt_links_pocket_coords = X_embedded_pocket[found_gt_links[:, 0]]
    found_gt_links_ligand_coords = X_embedded_lig[found_gt_links[:, 1]]
    missed_gt_links_pocket_coords = X_embedded_pocket[missed_gt_links[:, 0]]
    missed_gt_links_ligand_coords = X_embedded_lig[missed_gt_links[:, 1]]

    # Plot them with different colors
    for pocket_coord, ligand_coord in zip(pred_links_pocket_coords, pred_links_ligand_coords):
        ax.plot([pocket_coord[0], ligand_coord[0]], [pocket_coord[1], ligand_coord[1]], zs=[z_offset, 0],
                color='gray', alpha=0.1)
    for pocket_coord, ligand_coord in zip(found_gt_links_pocket_coords, found_gt_links_ligand_coords):
        ax.plot([pocket_coord[0], ligand_coord[0]], [pocket_coord[1], ligand_coord[1]], zs=[z_offset, 0],
                color='forestgreen', alpha=0.3)
    for pocket_coord, ligand_coord in zip(missed_gt_links_pocket_coords, missed_gt_links_ligand_coords):
        ax.plot([pocket_coord[0], ligand_coord[0]], [pocket_coord[1], ligand_coord[1]], zs=[z_offset, 0],
                color='firebrick', alpha=0.3)
    ax.set_axis_off()
    ax.azim = 41
    ax.elev = 67
    plt.savefig("figs/tsne_mappings.pdf", format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    sims()
    tsne()
