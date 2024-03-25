"""
Plot relationship between dissimilarity from train set and performance
"""
import pickle
import itertools
from collections import Counter

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jaccard, squareform
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
import seaborn as sns
import scienceplots


from fig_scripts.plot_utils import group_df, get_rmscores, get_smooth_order


def compute_old():
    def get_score(pocket_id, sm, df):
        try:
            return df.loc[(df['pocket_id'] == pocket_id) & (df['smiles'] == sm)]['mixed'].iloc[0]
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

    # names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))
    # scores = pd.read_csv("outputs/mixed.csv")
    # test_pockets = list(scores['pocket_id'])
    # pocket_pairs = list(itertools.combinations(test_pockets, 2))
    # # test_pockets = test_pockets[:5]
    # # pocket_names = [f"{n.split('_')[0]}-{n.split('_')[2]}" for n in test_pockets]
    #
    # scores_raw = pd.read_csv("outputs/mixed_raw.csv")
    # ref_ligs = list(set(scores_raw.loc[scores_raw['pocket_id'] == '1BYJ_A_GE3_30']['smiles']))
    # rows = []
    # for i, (p1, p2) in enumerate(pocket_pairs):
    #     _,_, r = one_corr(p1, p2, scores, ref_ligs)
    #     rows.append({'pocket_1': p1, 'pocket_2': p2, 'overlap': r})
    #     print(i, p1, p2, r, len(pocket_pairs))
    # pd.DataFrame(rows).to_csv("outputs/pred_mixed_overlap.csv")

    # OLD ? TO GET NEGATIVES ?
    # smiles_list = pd.read_csv("data/csvs/fp_data.csv")['LIGAND_SMILES']
    # mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    # clean_mols = []
    # clean_smiles = []
    # for mol, sm in zip(mols, smiles_list):
    #     if mol is None:
    #         continue
    #     clean_mols.append(mol)
    #     clean_smiles.append(sm)
    # smiles_to_ind = {sm: i for i, sm in enumerate(clean_smiles)}
    # fps = [MACCSkeys.GenMACCSKeys(m) for m in clean_mols]
    # fps = [np.array(fp.ToList()) for fp in fps]
    # fps = np.vstack(fps)
    # row_sums = fps.sum(axis=1, keepdims=True)
    # fps = fps / row_sums
    # print(fps.shape)

    # OLD ? TO GET PERF VS I DON'T KNOW
    # # fp novelty vs performance
    # tree = KDTree(fps, leaf_size=2, metric='euclidean')
    # colors = ['blue', 'red', 'green']
    # epss = [0.1, 0.15, 0.2]
    # for i, eps in enumerate(epss):
    #     perfs = []
    #     neis = []
    #     for p in test_pockets:
    #         native = scores_raw.loc[(scores_raw['pocket_id'] == p) & (scores_raw['is_active'] == 1)].iloc[0]['smiles']
    #         native_fp = fps[smiles_to_ind[native]].reshape(-1, 1).T
    #         num_nei = tree.query_radius(native_fp, eps, count_only=True)[0]
    #         neis.append(num_nei - 1)
    #         perfs.append(scores.loc[(scores['pocket_id'] == p) & (scores['decoys'] == 'chembl')].iloc[0]['score'])
    #
    #     perfs = np.array(perfs)
    #     neis = np.array(neis)
    #
    #     bins = np.linspace(neis.min(), neis.max(), 10)  # 20 bins from min to max of x
    #     bin_indices = np.digitize(neis, bins)  # Assign each x to a bin
    #
    #     # Calculate mean and standard deviation for y-values in each bin
    #     bin_means = [perfs[bin_indices == i].mean() for i in range(1, len(bins))]
    #     bin_stds = [perfs[bin_indices == i].std() for i in range(1, len(bins))]
    #
    #     # Use the mid-point of each bin for plotting
    #     bin_centers = 0.5 * (bins[:-1] + bins[1:])
    #
    #     y_jitter_strength = 0.02  # Adjust this value to increase/decrease jitter
    #     x_jitter_strength = 0.03  # Adjust this value to increase/decrease jitter
    #
    #     # Adding random jitter to x and y
    #     neis = neis + np.random.normal(0, x_jitter_strength, neis.shape)
    #     perfs = perfs + np.random.normal(0, y_jitter_strength, perfs.shape)
    #
    #     # plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o', ecolor=colors[i], elinewidth=3, capsize=0, label=eps)
    #     plt.scatter(neis, perfs, c=colors[i], alpha=0.5, label=f"{eps}", s=20)
    #
    # plt.gca().set_xlim(plt.gca().get_xlim()[::-1])
    # plt.xlabel("Number of neighbors")
    # plt.ylabel("Performance")
    # plt.legend()
    # plt.show()
    pass


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
    plt.tight_layout()
    plt.savefig("figs/2b.pdf", format="pdf")
    plt.show()


def compute_pred_distances(big_df_raw, out_name="outputs/pred_mixed_overlap_vincent.csv", percentile=0.01):
    """
    Compute the pairwise distance between all pockets
    The similarity between two pockets is based on our predictions: it is computed as the IoU of the top 5%
    """
    test_pockets = sorted(big_df_raw['pocket_id'].unique())
    pocket_pairs = list(itertools.combinations(test_pockets, 2))
    ref_ligs = list(set(big_df_raw.loc[big_df_raw['pocket_id'] == '1BYJ_A_GE3_30']['smiles']))

    # Factorize the 'Category' column
    codes, uniques = pd.factorize(big_df_raw['smiles'])

    # Update the 'Category' column with the encoded integers
    big_df_raw['lig_id'] = codes


    def get_predictions_pocket(pocket, scores, ref_ligs, percentile=percentile):
        pocket_scores = scores.loc[scores['pocket_id'] == pocket]
        pocket_lig_scores = pocket_scores.loc[pocket_scores['smiles'].isin(ref_ligs)]
        scores, ligs = pocket_lig_scores[['mixed', 'smiles']].values.T
        sm_sorted = sorted(zip(scores, ligs), key=lambda x: x[0], reverse=True)
        top_K = int(len(ligs) * percentile)
        sm_keep = set([sm for _, sm in sm_sorted[:top_K]])
        return sm_keep

    pocket_preds = {}
    for pocket in test_pockets:
        pocket_preds[pocket] = get_predictions_pocket(pocket, ref_ligs=ref_ligs, scores=big_df_raw)
        #print(pocket, pocket_preds[pocket])

    all_pred = Counter()
    for pred in pocket_preds.values():
        all_pred += Counter(pred)

    orders = {sm:i for i, sm in enumerate(all_pred.keys())}
    df_ridge = big_df_raw.loc[big_df_raw['smiles'].isin(all_pred)]
    df_ridge['rank'] = df_ridge.groupby('pocket_id')['mixed'].rank(pct=True, ascending=True)
    df_ridge['order'] = df_ridge['smiles'].apply(lambda x:orders[x])

    df_ridge = df_ridge.sort_values(by='order')
    print(df_ridge)

    sns.set(font_scale = .5)

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df_ridge, row="smiles", hue="smiles", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "rank",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "rank", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.despine(bottom=True, left=True)
    plt.show()


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
    default_kwargs_1 = {'cmap': sns.light_palette('blue', as_cmap=True),
                        'linewidths': 2,
                        'square': True,
                        'cbar_kws': {"shrink": .95}}
    default_kwargs_2 = {'cmap': sns.light_palette('green', as_cmap=True),
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

    # Get raw values
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_raw.csv')
    big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'mixed']]
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles', 'is_active'])

    test_pockets = sorted(big_df_raw['pocket_id'].unique())
    for p in test_pockets:
        print(p)
    pocket_pairs = list(itertools.combinations(test_pockets, 2))

    # Get smooth ordering
    smooth = True

    rmscores = get_rmscores()
    if smooth:
        rmscores_values = rmscores.values
        rmscores_labels = rmscores.columns

        # Select testset
        test_set = set(test_pockets)
        test_index = np.array([name in test_set for name in rmscores_labels])
        test_rmscores_labels = rmscores_labels[test_index]
        test_rmscores_values = rmscores_values[test_index][:, test_index]

        # Order following test_pocket sort
        sorter = np.argsort(test_rmscores_labels)
        test_rmscores_labels = test_rmscores_labels[sorter]
        test_rmscores_values = test_rmscores_values[sorter][:, sorter]
        order = get_smooth_order(rmscores=test_rmscores_values)
    else:
        order = np.arange(len(test_pockets))

    # HEATMAPS

    # COMPUTE PDIST based on overlap of predictions
    out_name = "outputs/pred_mixed_overlap_vincent.csv"
    compute_pred_distances(big_df_raw=big_df_raw, out_name=out_name, percentile=0.01)
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
    plt.show()

    # COMPARE: jaccard vs fp sim of natives
    smiles_list = [big_df_raw.loc[(big_df_raw['pocket_id'] == p) & big_df_raw['is_active']]['smiles'].iloc[0] for p in
                   test_pockets]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    tani = [DataStructs.TanimotoSimilarity(fp_1, fp_2) for fp_1, fp_2 in itertools.combinations(fps, 2)]
    square_tani = squareform(tani)
    square_tani = square_tani[order][:, order]
    palette_lig = sns.light_palette('orange', as_cmap=True)
    ax = double_heatmap(corr1=square_corrs, corr2=square_tani, kwargs2={'cmap': palette_lig})
    plt.show()

    # ax = double_heatmap(corr1=square_rms,
    #                     corr2=square_tani,
    #                     kwargs1={'cmap': sns.light_palette('green'), 'vmax': 0.7, 'vmin': 0.2},
    #                     kwargs2={'cmap': palette_lig})
    # plt.show()

    # plt.scatter(rms, corrs, alpha=0.7)
    # plt.xlabel("Native Ligand Similarity")
    # plt.ylabel("Prediction Similarity")
    # plt.savefig("figs/fig_2e.pdf", format="pdf")
    # sns.despine()
    # plt.tight_layout()
    # plt.show()
    pass


def tsne():
    df_fp = pd.read_csv("data/csvs/fp_data.csv")
    df = pd.read_csv("outputs/mixed_raw.csv")

    decoys = 'pdb'

    smiles_list = sorted(list(set(df['smiles'])))
    # smiles_list = sorted(list(set(df_fp['LIGAND_SMILES'])))
    # natives = set(list(df.loc[(df['is_active'] == 1) & (df['decoys'] == 'chembl')]['smiles']))

    print(f"Making fps {len(smiles_list)}")
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    clean_mols = []
    clean_smiles = []
    for mol, sm in zip(mols, smiles_list):
        if mol is None:
            continue
        clean_mols.append(mol)
        clean_smiles.append(sm)

    smiles_to_ind = {sm: i for i, sm in enumerate(clean_smiles)}
    fps = np.array([MACCSkeys.GenMACCSKeys(m) for m in clean_mols])

    print("lig tsne")
    X_embedded_lig = TSNE(n_components=2, learning_rate='auto',
                          init='pca').fit_transform(fps)

    print("lig tsne done")

    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))

    # keep = list(grouped_train.keys()) + list(grouped_test.keys())
    keep = list(grouped_test.keys())

    # pocket_list = df.loc[df['PDB_ID_POCKET'].isin(keep)]['PDB_ID_POCKET']
    pocket_list = df.loc[df['pocket_id'].isin(keep)]['pocket_id']
    pocket_list = sorted(list(set(pocket_list)))

    c = Counter([p.split("_")[2] for p in pocket_list])
    print(c)

    pocket_pairs = list(itertools.combinations(pocket_list, 2))
    dists = [1 - rmscores.loc[p1, p2] for p1, p2 in pocket_pairs]
    print(len(pocket_list), len(pocket_pairs), len(dists))
    DM = np.zeros((len(pocket_list), len(pocket_list)))
    DM[np.triu_indices(len(pocket_list), 1)] = dists
    DM += DM.T
    print("pocket tsne")
    X_embedded_pocket = TSNE(n_components=2, init='random', metric='precomputed', learning_rate='auto',
                             ).fit_transform(DM)

    X_embedded_pocket = preprocessing.MinMaxScaler().fit_transform(X_embedded_pocket)
    X_embedded_lig = preprocessing.MinMaxScaler().fit_transform(X_embedded_lig)

    # X_embedded_pocket -= np.mean(X_embedded_pocket)
    # X_embedded_lig -= np.mean(X_embedded_lig)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    # Plot the solid plane
    # ax1.plot_surface(x_plane, y_plane, z_plane, alpha=0.2, color='grey')
    # ax1.plot_surface(x_plane, y_plane, z_plane_lig, alpha=0.2, color='grey')

    # ax2.plot_surface(x_plane, y_plane, z_plane, alpha=0.2, color='grey')
    # ax2.plot_surface(x_plane, y_plane, z_plane_lig, alpha=0.2, color='grey')

    clustering_ligs = AgglomerativeClustering(distance_threshold=0.35, metric='cosine', linkage='average',
                                              n_clusters=None).fit(fps)
    clustering_pockets = AgglomerativeClustering(metric='precomputed', linkage='single', n_clusters=None,
                                                 distance_threshold=0.4).fit(DM)

    offset = -10

    active_smiles = set(df.loc[df['is_active'] == 1]['smiles'].unique())
    print(active_smiles)

    ax.scatter(X_embedded_pocket[:, 0], X_embedded_pocket[:, 1], marker='^', alpha=1, c='black', s=25)
    # ax2.scatter(X_embedded_pocket[:,0], X_embedded_pocket[:,1],  alpha=.8, c='red', s=10)
    # ax.scatter(X_embedded_pocket[:,0], X_embedded_pocket[:,1], [0] * X_embedded_pocket.shape[0], alpha=.8, c=clustering_pockets.labels_, cmap='Set2', s=5)
    # ax1.scatter(X_embedded_lig[:,0], X_embedded_lig[:,1], [-2] * X_embedded_lig.shape[0], c=clustering_ligs.labels_, alpha=.7, s=1, cmap='Set2')
    # ax.scatter(X_embedded_lig[:,0], X_embedded_lig[:,1] + offset, c=clustering_ligs.labels_, alpha=.7, s=3, cmap='Set2')
    ax.scatter(X_embedded_lig[:, 0], X_embedded_lig[:, 1] + offset,
               c=['blue' if sm in active_smiles else 'grey' for sm in clean_smiles],
               alpha=.7, s=[20 if sm in active_smiles else 0.5 for sm in clean_smiles])
    # ax2.scatter(X_embedded_lig[:,0], X_embedded_lig[:,1], [-2] * X_embedded_lig.shape[0], c=clustering_ligs.labels_, alpha=.7, s=1, cmap='Set2')

    # pred links
    preds = pd.read_csv("outputs/mixed_raw.csv")
    preds['smiles'] = preds['smiles'].apply(lambda x: x.rstrip("\n"))

    corrects = []
    for i, pocket in enumerate(pocket_list):
        scores = preds.loc[
            (preds['decoys'] == 'chembl') & (preds['pocket_id'] == pocket) & (preds['smiles'].isin(smiles_to_ind))]
        scores['rank'] = scores['mixed'].rank(method='average', ascending=True, pct=True)
        counts = scores['smiles'].value_counts()
        print(scores)
        try:
            for row in scores.itertuples():
                if row.rank < 0.995:
                    continue
                lig_ind = smiles_to_ind[row.smiles]
                if row.is_active:
                    ax.plot([X_embedded_pocket[i][0], X_embedded_lig[lig_ind][0]],
                            [X_embedded_pocket[i][1], X_embedded_lig[lig_ind][1] + offset],
                            linestyle='-', color='green', lw=1, alpha=.7)
                    corrects.append(i)

                else:
                    print("YO")
                    ax.plot([X_embedded_pocket[i][0], X_embedded_lig[lig_ind][0]],
                            [X_embedded_pocket[i][1], X_embedded_lig[lig_ind][1] + offset],
                            linestyle='-', color='grey', lw=1, alpha=0.3)
        except KeyError:
            print("MISSING ", i)
            continue

    # missing natives links
    print(corrects, len(pocket_list))
    for i, pocket in enumerate(pocket_list):
        if i in corrects:
            continue

        try:
            print("YO")
            # lig_ind = smiles_to_ind[df.loc[df['PDB_ID_POCKET'] == pocket]['LIGAND_SMILES'].iloc[0]]
            lig_ind = smiles_to_ind[df.loc[(df['pocket_id'] == pocket) & (df['is_active'] == 1)]['smiles'].iloc[0]]
            ax.plot([X_embedded_pocket[i][0], X_embedded_lig[lig_ind][0]],
                    [X_embedded_pocket[i][1], X_embedded_lig[lig_ind][1] + offset],
                    linestyle='-', color='red', lw=1, alpha=.7)
        except KeyError:
            print(i)
            continue

    # ax1.set_title("Ground Truth")
    # ax2.set_title("Prediction")
    ax.set_axis_off()
    # ax2.set_axis_off()
    plt.savefig("fig_2a.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    sims()
    # tsne()
