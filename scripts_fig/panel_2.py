"""
Plot relationship between dissimilarity from train set and performance
"""
import os
import sys

from collections import Counter
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from scipy.spatial.distance import squareform, cdist
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from sklearn import preprocessing
from seaborn.palettes import dark_palette, light_palette, blend_palette

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts_fig.plot_utils import group_df, get_rmscores, get_smooth_order, rotate_2D_coords, get_groups


def dock_correlation():
    """
    Docking vs rnamigos score correlation
    """
    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))
    df_paths = {'docknat': "outputs/docknat_grouped_42_raw.csv",
                'native': "outputs/native_42_raw.csv",
                'dock': "outputs/dock_42_raw.csv",
                }
    score_keys = {'docknat': 'docknat',
                  'native': 'raw_score_x',
                  'dock': 'raw_score_x'
                  }

    migos_to_use = 'docknat'
    dock_to_use = 'dock_pocket_norm' # raw_score_y

    dock_pred = pd.read_csv(df_paths[migos_to_use])
    dock = pd.read_csv("outputs/rdock_raw.csv")

    result = dock_pred.merge(dock, on=['pocket_id', 'decoys', 'smiles'])
    result = result.loc[(result['decoys'] == 'chembl') & (result['pocket_id'].isin(grouped_test.keys()))]
    print(result)
    if migos_to_use == 'docknat':
        result['dock_pocket_norm'] = 1 - result.groupby("pocket_id")['raw_score'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        result['dock_pocket_norm'] = 1 - result.groupby("pocket_id")['raw_score_y'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    if migos_to_use == 'dock':
        result[score_keys[migos_to_use]] = 1 - result[score_keys[migos_to_use]]

    result[score_keys[migos_to_use]] = result.groupby("pocket_id")[score_keys[migos_to_use]].transform(lambda x:(x - x.min()) / (x.max() - x.min()))

    actives = result.loc[result['is_active_x'] == 1.0]
    decoys = result.loc[result['is_active_x'] == 0.0]
    g = sns.regplot(data=result, x=dock_to_use, y=score_keys[migos_to_use], color=".3", ci=99, scatter_kws={"alpha": 0.3, "s": 2}, line_kws={"color": "red"})
    sns.scatterplot(data=actives, x=dock_to_use, y=score_keys[migos_to_use], color="blue", ax=g)

    r, p = spearmanr(result[dock_to_use], result[score_keys[migos_to_use]])
    plt.text(x=np.min(result[dock_to_use]), y=np.max(result[score_keys[migos_to_use]]) - 0.59,
         s=f"$\\rho$ = {r:.2f}",
         color='red', fontweight='bold')
    handles = [
        matplotlib.lines.Line2D([], [], marker='o', color='blue', linestyle='none', markersize=10, label='Native'),
        matplotlib.lines.Line2D([], [], marker='o', color='grey', linestyle='none', markersize=10, label='Decoy'),
    ]

    #plt.axvline(x=decoys[dock_to_use].mean(), color='grey', linestyle='--')
    #plt.axvline(x=actives[dock_to_use].mean(), color='blue', linestyle='--')

    """

    plt.text(x=-.05, y=decoys[score_keys[migos_to_use]].mean() + 0.02,
         s=f"$\\mu$ = {decoys[score_keys[migos_to_use]].mean():.2f}",
         color='grey', fontweight='bold')

    plt.text(x=-.05, y=actives[score_keys[migos_to_use]].mean() + 0.02,
         s=f"$\\mu$ = {actives[score_keys[migos_to_use]].mean():.2f}",
         color='blue', fontweight='bold')


    plt.text(x=actives[dock_to_use].mean(), y=0.10,
         s=f"$\\mu$ = {actives[dock_to_use].mean():.2f}",
         color='blue', fontweight='bold')

    plt.text(x=decoys[dock_to_use].mean() + 0.02, y=0.10,
         s=f"$\\mu$ = {decoys[dock_to_use].mean():.2f}",
         color='grey', fontweight='bold')

    """
    plt.axhline(y=decoys[score_keys[migos_to_use]].mean(), color='grey', linestyle='--')
    plt.axhline(y=actives[score_keys[migos_to_use]].mean(), color='blue', linestyle='--')

    plt.ylim([0, 1.1])


    # plt.axhline(y=decoys['mixed'].mean(), color='grey', linestyle='--')
    # plt.axhline(y=actives['mixed'].mean(), color='blue', linestyle='--')

    plt.legend(handles=handles, loc='lower right')

    plt.xlabel("Normalized rDock")
    plt.ylabel("MIXED")
    plt.savefig(f"figs/dock_corr_{migos_to_use}.pdf", format="pdf")
    plt.savefig(f"figs/dock_corr_{migos_to_use}.png", format="png")
    plt.show()
    pass


def barcodes(grouped=True):
    # TEST SET
    name_runs = {
        r"COMP": "native_42.csv",
        r"AFF": "dock_42.csv",
        r"rDock": "rdock.csv",
        r"MIXED": "mixed_grouped_42.csv",
    }
    rows = []
    prev_pockets = None
    for csv_name in name_runs.values():
        # print(m)
        df = pd.read_csv(f"outputs/{csv_name}")
        if grouped:
            df = group_df(df)
        row = df[df['decoys'] == 'chembl'].sort_values(by='pocket_id')
        all_pockets = row['pocket_id'].values
        if prev_pockets is None:
            prev_pockets = all_pockets
        else:
            assert (prev_pockets == all_pockets).all(), print(prev_pockets, all_pockets)
        rows.append(row['score'])

    # FIND SMOOTHER PERMUTED VERSION
    order = get_smooth_order(prev_pockets)
    for i in range(len(rows)):
        new_row = rows[i].values[order]
        rows[i] = new_row

    # sns.heatmap(rows, cmap='binary_r')
    # cmap = sns.color_palette("vlag_r", as_cmap=True)
    # cmap = sns.diverging_palette(0, 245, s=100, l=50, as_cmap=True)
    # cmap = custom_diverging_palette(0, 245, s_neg=100, l_neg=50, s_pos=90, l_pos=80, as_cmap=True)
    red_pal = sns.light_palette('#CF403E', reverse=True, n_colors=128 - 10)
    # blue_pal = sns.light_palette('#5c67ff', n_colors=30)[:10] # too grey/violet
    # blue_pal = sns.light_palette('#9dabe1', n_colors=10) # a bit violet and also lot of color
    # blue_pal = sns.light_palette('#a5b0d9', n_colors=10) # close
    # blue_pal = sns.light_palette('#7689d5', n_colors=10) # nice blue but a bit dense
    # blue_pal = sns.light_palette('#ccd6ff', n_colors=10) # brighter less blue
    # blue_pal = sns.light_palette('#d6ecff', n_colors=10) # almost white
    # blue_pal = sns.light_palette('#ebf5ff', n_colors=10) # whiter
    blue_pal = sns.light_palette('#fff', n_colors=10)  # white
    # blue_pal = sns.color_palette("light:b", n_colors=10) # hardcode blue
    cmap = blend_palette(np.concatenate([red_pal, blue_pal]), 1, as_cmap=True)

    # Handle spine
    ax = sns.heatmap(rows, cmap=cmap)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('grey')

    # Handle ticks
    xticks = np.arange(0, len(rows[0]), 10)
    xticks_labels = xticks + 1
    plt.xticks(xticks, xticks_labels, va="center")
    plt.tick_params(axis='x', bottom=False, labelbottom=True)
    plt.yticks(np.arange(len(name_runs)) + 0.5, [name for name in name_runs.keys()], rotation=0, va="center")
    plt.tick_params(axis='y', left=False, right=False, labelleft=True)

    # plotis is probably useless
    # selected_pockets = set(pockets)
    # test_index = np.array([name in selected_pockets for name in rmscores_labels])
    # test_rmscores_labels = rmscores_labels[test_index]
    # test_rmscores_values = rmscores_valu
    plt.xlabel(r"Pocket")
    plt.ylabel(r"Method")
    plt.savefig("figs/barcode.pdf", bbox_inches='tight')
    plt.show()
    pass


def train_sim_perf_plot(grouped=True):
    """
    Make the scatter plot of performance as a function of similarity to train set
    """
    get_groups()
    rmscores = get_rmscores()
    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))
    mixed_res = pd.read_csv(f'outputs/docknat_grouped_42.csv')

    fig, ax1 = plt.subplots()

    ax2 = ax1.twiny()

    ### include PDB natives 
    native_smiles = pd.read_csv("data/csvs/fp_data.csv")
    native_smiles['lig_ids'] = native_smiles['PDB_ID_POCKET'].apply(lambda x: x.split("_")[2])
    lig_counts = Counter(list(native_smiles['lig_ids']))
    mixed_res['native_count'] = mixed_res['pocket_id'].apply(lambda x: lig_counts[x.split("_")[2]])
    mixed_res['prevalence'] = mixed_res['native_count'] / (sum(lig_counts.values()))
    mixed_res['score'] = mixed_res['score'] + np.random.normal(0, 0.003, size=len(mixed_res))

    if grouped:
        mixed_res = group_df(mixed_res)
    mixed_res['train_sim_max'] = mixed_res['pocket_id'].apply(
        lambda x: rmscores.loc[x, rmscores.index.isin(names_train)].max())

    print(mixed_res)
    plt.axhline(y=mixed_res['score'].mean(), color='black', linestyle='-', linewidth=2)
    plt.axhline(y=mixed_res['score'].median(), color='black', linestyle='--', linewidth=2)
    sns.scatterplot(data=mixed_res, y='score', x='train_sim_max', alpha=0.5, ax=ax2, color='green', s=70)
    sns.scatterplot(data=mixed_res, y='score', x='prevalence', marker='^', alpha=0.5, ax=ax1, color='blue', s=70)
    ax1.legend()
    ax2.legend()
    ax1.tick_params(axis='x', colors='blue')
    ax2.tick_params(axis='x', colors='green')

    # plt.legend()
    # sns.despine()

    handles = [
        matplotlib.lines.Line2D([], [], marker='^', color='blue', linestyle='none', markersize=10, label='Ligands'),
        matplotlib.lines.Line2D([], [], marker='o', color='green', linestyle='none', markersize=10, label='Pockets'),
        matplotlib.lines.Line2D([], [], color='black', linestyle='--', markersize=10, label='Median'),
        matplotlib.lines.Line2D([], [], color='black', linestyle='-', markersize=10, label='Mean'),
    ]

    plt.legend(handles=handles, loc='lower center')

    plt.xlabel("Max RMscore to train set")
    # plt.ylabel("AuROC")
    plt.savefig("figs/train_max_perf.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def get_predictions_pocket(pocket, scores, ref_ligs=None, percentile=0.01):
    pocket_scores = scores.loc[scores['pocket_id'] == pocket]
    pocket_lig_scores = pocket_scores.loc[pocket_scores['smiles'].isin(ref_ligs)] if ref_ligs is not None \
        else pocket_scores
    scores, ligs = pocket_lig_scores[['docknat', 'smiles']].values.T
    sm_sorted = sorted(zip(scores, ligs), key=lambda x: x[0], reverse=True)
    top_K = int(len(ligs) * percentile)
    sm_keep = set([sm for _, sm in sm_sorted[:top_K]])
    return sm_keep


def compute_pred_distances(big_df_raw,
                           out_name="outputs/pred_mixed_overlap_vincent.csv",
                           percentile=0.01,
                           recompute=False,
                           plot_facet=False):
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

    rows = []
    for p1, p2 in (pocket_pairs):
        sm_keep_1 = pocket_preds[p1]
        sm_keep_2 = pocket_preds[p2]
        overlap = len(sm_keep_1 & sm_keep_2) / len(sm_keep_1 | sm_keep_2)
        rows.append({'pocket_1': p1, 'pocket_2': p2, 'overlap': overlap})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_name)

    if plot_facet:
        all_pred = Counter()
        for pred in pocket_preds.values():
            all_pred += Counter(pred)

        # Factorize the 'Category' column
        codes, uniques = pd.factorize(big_df_raw['smiles'])
        # Update the 'Category' column with the encoded integers
        big_df_raw['lig_id'] = codes

        orders = {sm: i for i, sm in enumerate(all_pred.keys())}
        df_ridge = big_df_raw.loc[big_df_raw['smiles'].isin(all_pred)]
        df_ridge['rank'] = df_ridge.groupby('pocket_id')['docknat'].rank(pct=True, ascending=True)
        df_ridge['order'] = df_ridge['smiles'].apply(lambda x: orders[x])
        df_ridge = df_ridge.sort_values(by='order')
        print(df_ridge)

        sns.set(font_scale=.5)

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


def double_heatmap(corr1, corr2=None,
                   kwargs1={},
                   kwargs2={}):
    default_kwargs_1 = {'cmap': sns.light_palette('forestgreen', as_cmap=True),
                        'linewidths': 0,
                        'square': True,
                        'cbar_kws': {"shrink": .95}}
    default_kwargs_2 = {'cmap': sns.light_palette('royalblue', as_cmap=True),
                        'linewidths': 0,
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
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_42_raw.csv')
    big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'docknat']]
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles', 'is_active'])

    test_pockets = sorted(big_df_raw['pocket_id'].unique())
    for p in test_pockets:
        print(p)
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
    palette_lig = sns.light_palette('yellowgreen', as_cmap=True)
    ax = double_heatmap(corr1=square_corrs, corr2=square_tani, kwargs2={'cmap': palette_lig})
    plt.savefig("figs/tanimotos_preds.pdf")
    plt.show()

    ax = double_heatmap(corr1=square_tani,
                        corr2=square_rms,
                        kwargs1={'cmap': palette_lig},
                        kwargs2={'cmap': sns.light_palette('royalblue'), 'vmax': 0.7, 'vmin': 0.2}, )
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
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_42_raw.csv')
    big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'docknat']]
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles', 'is_active'])
    test_pockets = set(big_df_raw['pocket_id'].unique())

    # Complement with train pockets for MDS
    (train_names, test_names, train_names_grouped, _) = pickle.load(open("data/train_test_75.p", 'rb'))
    if grouped:
        all_pockets = set(train_names_grouped.keys()).union(test_pockets)
    else:
        all_pockets = train_names.union(test_names)

    # Subset RMscores to get only train+test pockets
    recompute = False
    dump_pockets = 'temp_pockets.p'
    if not os.path.exists(dump_pockets) or recompute:
        rmscores = get_rmscores()
        rmscores_idx = [pocket in all_pockets for pocket in rmscores.columns]
        rmscores = rmscores.iloc[rmscores_idx, rmscores_idx]
        all_pockets = rmscores.columns
        dists = 1 - rmscores.values
        distance_symmetrized = (dists + dists.T) / 2
        # X_embedded_pocket = TSNE(n_components=2, init='random', metric='precomputed', learning_rate='auto',
        #                          ).fit_transform(distance_symmetrized)
        X_embedded_pocket = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distance_symmetrized)
        X_embedded_pocket = preprocessing.MinMaxScaler().fit_transform(X_embedded_pocket)
        pickle.dump((X_embedded_pocket, all_pockets, test_pockets), open(dump_pockets, 'wb'))
    X_embedded_pocket, all_pockets, test_pockets = pickle.load(open(dump_pockets, 'rb'))
    plt.scatter(X_embedded_pocket[:, 0], X_embedded_pocket[:, 1],
                # c=['royalblue' if pocket in test_pockets else 'navy' for pocket in all_pockets],
                c=['royalblue' for _ in all_pockets],
                s=[20 if pocket in test_pockets else 2 for pocket in all_pockets],
                alpha=.7)
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig("figs/tsne_pockets.pdf", format="pdf", bbox_inches='tight')
    plt.show()

    # GET LIGANDS SPACE EMBEDDINGS
    dump_ligs = 'temp_ligs.p'
    if not os.path.exists(dump_ligs) or recompute:
        smiles_list = sorted(big_df_raw['smiles'].unique())
        active_smiles = set(big_df_raw.loc[big_df_raw['is_active'] == 1]['smiles'].unique())
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
        tani = [DataStructs.TanimotoSimilarity(fp_1, fp_2) for fp_1, fp_2 in itertools.combinations(fps, 2)]
        dists = 1 - squareform(tani)
        distance_symmetrized = (dists + dists.T) / 2
        # X_embedded_lig = TSNE(n_components=2, learning_rate='auto', metric='precomputed', init='random').fit_transform(
        #     distance_symmetrized)
        X_embedded_lig = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distance_symmetrized)
        X_embedded_lig = preprocessing.MinMaxScaler().fit_transform(X_embedded_lig)
        pickle.dump((X_embedded_lig, smiles_list, active_smiles), open(dump_ligs, 'wb'))
    X_embedded_lig, smiles_list, active_smiles = pickle.load(open(dump_ligs, 'rb'))
    plt.scatter(X_embedded_lig[:, 0], X_embedded_lig[:, 1],
                c=['forestgreen' if sm in active_smiles else 'grey' for sm in smiles_list],
                # c=['forestgreen' for _ in smiles_list],
                s=[20 if sm in active_smiles else 1 for sm in smiles_list],
                alpha=.7)
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig("figs/tsne_ligands.pdf", format="pdf", bbox_inches='tight')
    plt.show()

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
        if i > 10: break
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
    best_angle = 10

    # PLOT 3D
    ax = plt.axes(projection='3d')
    X_embedded_pocket = rotate_2D_coords(X_embedded_pocket, angle=best_angle)
    ax.scatter(X_embedded_pocket[:, 0], X_embedded_pocket[:, 1], z_offset * np.ones(len(X_embedded_pocket)),
               c=['royalblue' if pocket in test_pockets else 'grey' for pocket in all_pockets],
               s=[20 if pocket in test_pockets else 0.5 for pocket in all_pockets],
               alpha=.9)
    ax.scatter(X_embedded_lig[:, 0], X_embedded_lig[:, 1], np.zeros(len(X_embedded_lig)),
               c=['forestgreen' if sm in active_smiles else 'grey' for sm in smiles_list],
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
    dock_correlation()
    # sims()
    # barcodes()
    tsne()
