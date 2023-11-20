import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import seaborn as sns


# Normalize
def normalize(scores):
    out_scores = (scores - scores.min()) / (scores.max() - scores.min())
    return out_scores


def get_one(df, score):
    """
    """
    pockets = df['pocket_id'].unique()
    all_efs = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        pocket_df = pocket_df.reset_index(drop=True)
        sorted_df = pocket_df.sort_values(by=score)
        sorted_df = sorted_df.reset_index(drop=True)
        native_ind = sorted_df.loc[sorted_df['is_active'] == 1].index[0]
        enrich = native_ind / (len(sorted_df) - 1)
        all_efs.append(enrich)
    pocket_ef = np.mean(all_efs)
    return pocket_ef


def get_mix_pair(df, score1, score2, all_thresh, verbose=True):
    """
    """
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

            pocket_df['combined'] = (mixed * normalized_docking + (1 - mixed) * normalized_new).values
            # pocket_df['combined'] = -(mixed * np.exp(- normalized_docking / 3) +
            #                           (1 - mixed) * np.exp(-normalized_new / 3))
            # pocket_df['combined'] = (mixed * normalized_docking ** 4 + (1 - mixed) * normalized_new ** 4).values
            sorted_df = pocket_df.sort_values(by='combined')
            sorted_df = sorted_df.reset_index(drop=True)
            native_ind = sorted_df.loc[sorted_df['is_active'] == 1].index[0]
            enrich = native_ind / (len(sorted_df) - 1)
            all_efs.append(enrich)
        pocket_ef = np.mean(all_efs)
        if verbose:
            print(i, pocket_ef)
        all_thresh_res.append(pocket_ef)
    return all_thresh_res


def get_table_mixing(df):
    all_methods = ['fp', 'native', 'dock', 'rdock']
    n_intervals = 10
    all_thresh = np.linspace(0, 1, n_intervals)
    all_res = {}
    # Do singletons
    for method in all_methods:
        result = get_one(df, score=method)
        print(method, result)
        all_res[method] = result

    # Do pairs
    for pair in itertools.combinations(all_methods, 2):
        all_results = get_mix_pair(df, score1=pair[0], score2=pair[1], all_thresh=all_thresh, verbose=False)
        best_idx = np.argmax(np.array(all_results))
        best_perf = all_results[best_idx]
        print(pair, f"{best_idx}/{n_intervals}", best_perf)
        all_res[pair] = best_perf
    for k, v in all_res.items():
        print(k, v)


def plot_pairs(df):
    score1 = 'rdock'
    # score1 = 'dock'
    # score1 = 'fp'
    # scores_2 = ['dock']
    # scores_2 = ['fp']
    # scores_2 = ['native']
    scores_2 = ['combined']
    # scores_2 = ['native']
    # scores_2 = ['fp', 'native']
    # scores_2 = ['dock', 'fp', 'native']

    all_thresh = np.linspace(0, 1, 30)
    for score2 in scores_2:
        all_thresh_res = get_mix_pair(df, score1, score2, all_thresh=all_thresh)
        plt.plot(all_thresh, all_thresh_res, label=score2)

    # plt.ylim(0.98, 1)
    plt.legend()
    plt.show()
    return df


def mix_three(df, coeffs=(0.5, 0.5, 0), score1='dock', score2='fp', score3='native', dump_df=False):
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

        pocket_df['combined'] = (x * normalized_docking
                                 + y * normalized_fp
                                 + z * normalized_native).values
        # pocket_df['combined'] = -(x * np.exp(- normalized_docking / 3) +
        #                           y * np.exp(-normalized_fp / 3) +
        #                           z * np.exp(-normalized_native / 3))
        sorted_df = pocket_df.sort_values(by='combined')
        sorted_df = sorted_df.reset_index(drop=True)
        native_ind = sorted_df.loc[sorted_df['is_active'] == 1].index[0]
        enrich = native_ind / (len(sorted_df) - 1)
        if dump_df:
            return all_efs.append((pocket_df[['pocket_id', 'smiles', 'is_active', 'combined']], p, enrich))
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
    print(f"Filtered coeffs results in {len(coeffs)} grid points")

    all_thresh_res = []
    for x, y, z in coeffs:
        all_efs = mix_three(df=df, coeffs=coeffs, dump_df=False)
        pocket_ef = np.mean(all_efs)
        print(x, y, z, pocket_ef)
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


def get_mix(df, coeffs, score1='dock', score2='fp', score3='native', outname_col='combined', outname="default"):
    # Mixed df contains df_raw, pocket, enrichment
    mixed_df = mix_three(df, coeffs=coeffs, score1=score1, score2=score2, score3=score3, dump_df=True)

    # Merge df and add decoys value
    mixed_df_raw = pd.concat([mixed[0] for mixed in mixed_df])
    mixed_df_raw[['decoys']] = ['chembl' for _ in range(len(mixed_df_raw))]
    mixed_df = pd.DataFrame({"pocket_id": [mixed[1] for mixed in mixed_df],
                             'decoys': ['chembl' for _ in mixed_df],
                             'score': [mixed[2] for mixed in mixed_df]})
    mixed_df_raw.rename({'combined': outname_col})
    mixed_df_raw.to_csv(f"../outputs/{outname}_raw.csv")
    mixed_df.to_csv(f"../outputs/{outname}.csv")


if __name__ == "__main__":
    # FIRST LET'S PARSE INFERENCE CSVS AND MIX THEM
    runs = ['rdock',
            'paper_dock',
            'paper_fp',
            'paper_native',
            ]
    decoy = 'chembl'
    raw_dfs = [pd.read_csv(f"../outputs/{r}_raw.csv") for r in runs]
    raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
    raw_dfs = [df.sort_values(by=['pocket_id', 'smiles', 'is_active']) for df in raw_dfs]
    big_df_raw = raw_dfs[0][['pocket_id', 'smiles', 'is_active']]

    # Now add score and flip docking scores, dock scores and distances for which low is better
    big_df_raw['rdock'] = -raw_dfs[0]['raw_score'].values
    big_df_raw['dock'] = -raw_dfs[1]['raw_score'].values
    big_df_raw['fp'] = -raw_dfs[2]['raw_score'].values
    big_df_raw['native'] = raw_dfs[3]['raw_score'].values

    # Find the best mix, and then dump it
    # best_mix = mix_all(big_df_raw)
    best_mix = [(0.44, 0.39, 0.17)]
    get_mix(big_df_raw, score1='dock', score2='fp', score3='native', coeffs=best_mix,
            outname_col='combined', outname='mixed')

    raw_df_combined = pd.read_csv('../outputs/mixed_raw.csv').sort_values(by=['pocket_id', 'smiles', 'is_active'])
    big_df_raw['combined'] = raw_df_combined[4]['combined'].values
    big_df_raw.to_csv('outputs/big_df_raw.csv')

    # NOW WE HAVE THE BEST ENSEMBLE MODEL AS DATA, we can plot pairs and get the rdock+mixed
    # big_df_raw= pd.read_csv("../outputs/big_df_raw.csv")

    plot_pairs(big_df_raw)
    get_table_mixing(big_df_raw)

    # # To dump rdock_combined
    coeffs = [(0.5, 0.5, 0.)]
    get_mix(big_df_raw, score1='combined', score2='rdock', coeffs=coeffs,
            outname_col='combined', outname='mixed_rdock')
