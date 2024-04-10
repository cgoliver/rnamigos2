import os
import sys

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sns
from sklearn import metrics

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fig_scripts.plot_utils import get_groups, group_df


# Normalize
def normalize(scores):
    out_scores = (scores - scores.min()) / (scores.max() - scores.min())
    return out_scores


def get_ef_one(df, score):
    pockets = df['pocket_id'].unique()
    all_efs = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'], pocket_df[score], drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)
        all_efs.append(enrich)
    pocket_ef = np.mean(all_efs)
    return pocket_ef


def get_mix_pair(df, score1, score2, all_thresh, verbose=True):
    pockets = df['pocket_id'].unique()
    all_thresh_res = []
    for i, mixed in enumerate(all_thresh):
        all_efs = []
        for pi, p in enumerate(pockets):
            pocket_df = df.loc[df['pocket_id'] == p]
            pocket_df = pocket_df.reset_index(drop=True)
            docking_scores = pocket_df[score1]
            new_scores = pocket_df[score2]
            normalized_docking = normalize(docking_scores)
            normalized_new = normalize(new_scores)

            pocket_df['mixed'] = (mixed * normalized_docking + (1 - mixed) * normalized_new).values
            # pocket_df['mixed'] = -(mixed * np.exp(- normalized_docking / 3) +
            #                           (1 - mixed) * np.exp(-normalized_new / 3))
            # pocket_df['mixed'] = (mixed * normalized_docking ** 4 + (1 - mixed) * normalized_new ** 4).values
            fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'], pocket_df['mixed'],
                                                     drop_intermediate=True)
            enrich = metrics.auc(fpr, tpr)
            all_efs.append(enrich)
        pocket_ef = np.mean(all_efs)
        if verbose:
            print(i, pocket_ef)
        all_thresh_res.append(pocket_ef)
    return all_thresh_res


def mix_three(df, coeffs=(0.5, 0.5, 0), score1='dock', score2='fp', score3='native', return_dfs=False):
    """
    Mix three dataframes with specified scores and coeffs, and return either the enrichment factors or the
    actual detailed results
    """
    pockets = df['pocket_id'].unique()
    all_efs = []
    x, y, z = coeffs
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        pocket_df = pocket_df.reset_index(drop=True)
        docking_scores = pocket_df[score1]
        fp_scores = pocket_df[score2]
        native_scores = pocket_df[score3]
        normalized_docking = normalize(docking_scores)
        normalized_fp = normalize(fp_scores)
        normalized_native = normalize(native_scores)

        pocket_df['mixed'] = (x * normalized_docking
                              + y * normalized_fp
                              + z * normalized_native).values
        # pocket_df['mixed'] = -(x * np.exp(- normalized_docking / 3) +
        #                           y * np.exp(-normalized_fp / 3) +
        #                           z * np.exp(-normalized_native / 3))

        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'], pocket_df['mixed'], drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)

        if return_dfs:
            all_efs.append((pocket_df[['pocket_id', 'smiles', 'is_active', 'mixed']], p, enrich))
        else:
            all_efs.append(enrich)
    return all_efs


def mix_all(df):
    """
    Look into mixing of all three methods. The tricky part is sampling on the 2d simplex.
    :param df:
    :return:
    """
    # get all coefs, because of the simplex constraint it is not simple
    all_thresh = np.linspace(0, 1, 10)
    xs, ys, zs = np.meshgrid(all_thresh, all_thresh, all_thresh, indexing='ij')
    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()
    norm_coeffs = xs + ys + zs + 1e-5
    coeffs = np.stack((xs, ys, zs))
    coeffs = (coeffs / norm_coeffs).T

    # Not the most efficient : remove the first (full zeros) and then
    # go through the list and remove if close to existing one in the list
    to_remove = [0]
    import scipy.spatial.distance as scidist
    for i in range(1, len(coeffs)):
        min_dist = min(scidist.cdist(coeffs[:i], coeffs[i][None, ...]))
        if min_dist < 0.001:
            to_remove.append(i)
    coeffs = np.delete(coeffs, to_remove, axis=0)
    # print(f"Filtered coeffs results in {len(coeffs)} grid points")

    all_thresh_res = []
    for i, (x, y, z) in enumerate(coeffs):
        if not i % 20:
            print(f'Doing {i}/{len(coeffs)}')
        all_efs = mix_three(df=df, coeffs=(x, y, z), return_dfs=False)
        pocket_ef = np.mean(all_efs)
        # print(x, y, z, pocket_ef)
        all_thresh_res.append(pocket_ef)

    # Print best result
    zs = np.array(all_thresh_res)
    best_i = np.argmax(zs)
    print('Best results for ', coeffs[best_i], ' value of MAR : ', zs[best_i])

    # 3D scatter plot the results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = coeffs[:, 0].flatten()
    ys = coeffs[:, 1].flatten()
    ax.scatter(xs, ys, zs)
    plt.show()
    return coeffs[best_i]


def get_mix(df, coeffs, score1='dock', score2='fp', score3='native', outname_col='mixed', outname="default"):
    # Mixed df contains df_raw, pocket, enrichment
    mixed_df = mix_three(df, coeffs=coeffs, score1=score1, score2=score2, score3=score3, return_dfs=True)

    # Merge df and add decoys value
    mixed_df_raw = pd.concat([mixed[0] for mixed in mixed_df])
    mixed_df_raw = mixed_df_raw.rename(columns={'mixed': outname_col})
    dumb_decoy = [DECOY for _ in range(len(mixed_df_raw))]
    mixed_df_raw.insert(len(mixed_df_raw.columns), "decoys", dumb_decoy)
    mixed_df_raw.to_csv(f"outputs/{outname}_raw.csv")

    mixed_df = pd.DataFrame({"pocket_id": [mixed[1] for mixed in mixed_df],
                             'decoys': ['chembl' for _ in mixed_df],
                             'score': [mixed[2] for mixed in mixed_df]})
    mixed_df.to_csv(f"outputs/{outname}.csv")


def find_best_mix(runs, grouped=True, decoy='chembl', outname_csv=f'mixed'):
    """
    This will find the best coeff to merge dock, fp and native;
    print the best coeff and save the resulting efs in a csv.
    """
    raw_dfs = [pd.read_csv(f"outputs/{r}_raw.csv") for r in runs]
    raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
    if grouped:
        raw_dfs = [group_df(df) for df in raw_dfs]

    for df in raw_dfs:
        df['smiles'] = df['smiles'].str.strip()

    raw_dfs[0]['rdock'] = -raw_dfs[0]['raw_score'].values
    raw_dfs[1]['dock'] = -raw_dfs[1]['raw_score'].values
    raw_dfs[2]['fp'] = -raw_dfs[2]['raw_score'].values
    raw_dfs[3]['native'] = raw_dfs[3]['raw_score'].values

    big_df_raw = raw_dfs[1]
    big_df_raw = big_df_raw.merge(raw_dfs[2], on=['pocket_id', 'smiles', 'is_active'], how='outer')
    big_df_raw = big_df_raw.merge(raw_dfs[3], on=['pocket_id', 'smiles', 'is_active'], how='outer')
    big_df_raw = big_df_raw.merge(raw_dfs[0], on=['pocket_id', 'smiles', 'is_active'], how='inner')
    big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'rdock', 'dock', 'fp', 'native']]
    # rows_with_nan = big_df_raw[big_df_raw.isna().any(axis=1)]

    # Find the best mix, and then dump it
    # best_mix = [0.44, 0.39, 0.17]  # OLD
    # best_mix = [0.36841931, 0.26315665, 0.36841931]  # UNGROUPED
    # best_mix = [0.3529, 0.2353, 0.4118]  # UNGROUPED value of AuROC :  0.9827
    best_mix = mix_all(big_df_raw)
    # best_mix = [0.429, 0.190, 0.381]  # New models (seed 0) value 0.9878, balanced is 0.9878
    # best_mix = [0.388, 0.167, 0.444]  # New models (seed 1) value 0.9935, balanced is 0.9900
    # best_mix = [0.412, 0.118, 0.471]  # New models (seed 42) value 0.9840, balanced is 0.9805

    print("balanced perf", np.mean(mix_three(df=big_df_raw, coeffs=(0.3, 0.3, 0.3))))
    print("best perf", np.mean(mix_three(df=big_df_raw, coeffs=best_mix)))

    # Now dump this best mixed as a csv
    get_mix(big_df_raw, score1='dock', score2='fp', score3='native', coeffs=best_mix,
            outname_col='mixed', outname=outname_csv)

    # Get the best mixed and add it to the combined results df
    raw_df_mixed = pd.read_csv(f'outputs/{outname_csv}_raw.csv')
    big_df_raw = big_df_raw.merge(raw_df_mixed, on=['pocket_id', 'smiles', 'is_active'], how='inner')
    return big_df_raw


def get_table_mixing(df):
    all_methods = ['fp', 'native', 'dock', 'rdock']
    n_intervals = 10
    all_thresh = np.linspace(0, 1, n_intervals)
    all_res = {}
    # Do singletons
    for method in all_methods:
        result = get_ef_one(df, score=method)
        # print(method, result)
        all_res[method] = result
    # Do pairs
    for pair in itertools.combinations(all_methods, 2):
        all_results = get_mix_pair(df, score1=pair[0], score2=pair[1], all_thresh=all_thresh, verbose=False)
        best_idx = np.argmax(np.array(all_results))
        best_perf = all_results[best_idx]
        # print(pair, f"{best_idx}/{n_intervals}", best_perf)
        all_res[pair] = best_perf

    # Add mixed results
    result_mixed = get_ef_one(df, score='mixed')
    all_res['mixed'] = result_mixed

    pair = ('mixed', 'rdock')
    all_results = get_mix_pair(df, score1=pair[0], score2=pair[1], all_thresh=all_thresh, verbose=False)
    best_idx = np.argmax(np.array(all_results))
    best_perf = all_results[best_idx]
    # print(pair, f"{best_idx}/{n_intervals}", best_perf)
    all_res[pair] = best_perf

    for k, v in all_res.items():
        print(f"{k}, result: {v:.4f}")


def plot_pairs(df):
    score1 = 'rdock'
    # score1 = 'dock'
    # score1 = 'fp'
    # scores_2 = ['dock']
    # scores_2 = ['fp']
    # scores_2 = ['native']
    scores_2 = ['mixed']
    # scores_2 = ['native']
    # scores_2 = ['fp', 'native']
    # scores_2 = ['dock', 'fp', 'native']

    all_thresh = np.linspace(0, 1, 30)
    for score2 in scores_2:
        all_thresh_res = get_mix_pair(df, score1, score2, all_thresh=all_thresh)
        plt.plot(all_thresh, all_thresh_res, label=score2)

    # plt.ylim(0.98, 1)
    # plt.xlabel('Fraction of score 1')
    # plt.legend()
    # plt.show()
    return all_thresh_res


if __name__ == "__main__":
    # Fix groups
    # get_groups()

    # FIRST LET'S PARSE INFERENCE CSVS AND MIX THEM
    DECOY = 'chembl'
    # DECOY = 'pdb'
    GROUPED = True
    all_thresh = np.linspace(0, 1, 30)
    all_all_thresh_res = []
    for seed in 0, 1, 42:
        RUNS = ['rdock',
                f'dock_{seed}',
                f'fp_{seed}',
                f'native_{seed}',
                ]
        out_name = f'mixed{"_grouped" if GROUPED else ""}_{seed}'
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        # big_df_raw = find_best_mix(runs=RUNS, grouped=GROUPED, decoy=DECOY, outname_csv=out_name)
        # big_df_raw.to_csv(out_path_raw)

        # NOW WE HAVE THE BEST ENSEMBLE MODEL AS DATA, we can plot pairs and get the rdock+mixed
        big_df_raw = pd.read_csv(out_path_raw)
        # all_thresh_res = plot_pairs(big_df_raw)  # 0: 0.13-0.27, 1: 0.13-0.24; 42: 0.27-0.4
        # all_all_thresh_res.append(all_thresh_res)
        # get_table_mixing(big_df_raw)
    # all_all_thresh_res = np.mean(np.asarray(all_all_thresh_res), axis=0)
    # plt.plot(all_thresh, all_all_thresh_res, label='mean')
    # plt.legend()
    # plt.show()

    # To dump rdock_combined
    for seed in 0, 1, 42:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)
        coeffs = (0.75, 0.25, 0.)
        get_mix(big_df_raw, score1='mixed', score2='rdock', coeffs=coeffs,
                outname_col='combined', outname=f'mixed_rdock{"_grouped" if GROUPED else ""}_{seed}')

