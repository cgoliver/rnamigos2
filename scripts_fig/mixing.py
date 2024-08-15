import os
import sys

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts_fig.plot_utils import get_groups, group_df

"""
The main two functions are:
- mix two scores than can explore many combinations
- mix three scores and optionally dump a results dataframe

Then we also add 
- mix_simplex: a function to explore many combinations with 3 weights
- compute_mix_csvs: creates the original big csv, adding docknat and mixed (potentially with simplex grid search)
- 

"""


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


def mix_two_scores(df, score1, score2, all_thresh, verbose=True):
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
    all_thresh_res = np.asarray(all_thresh_res)
    return all_thresh_res


def get_best_thresh_perf(all_thresh, all_results):
    best_idx = np.argmax(np.array(all_results))
    best_perf = all_results[best_idx]
    best_thresh = all_thresh[best_idx]
    return best_idx, best_perf, best_thresh


def mix_three_scores(df, coeffs=(0.5, 0.5, 0), score1='dock', score2='fp', score3='native',
                     outname=None, outname_col='mixed'):
    """
    Mix three dataframes with specified scores and coeffs, and return either the enrichment factors
    Optionally dump a dataframe
    """
    pockets = df['pocket_id'].unique()
    all_efs = []
    all_pockets = []
    all_dfs = []
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

        all_efs.append(enrich)
        all_pockets.append(p)
        all_dfs.append(pocket_df[['pocket_id', 'smiles', 'is_active', 'mixed']])
    if outname is not None:
        # Merge df and add decoys value
        mixed_df_raw = pd.concat(all_dfs)
        mixed_df_raw = mixed_df_raw.rename(columns={'mixed': outname_col})
        dumb_decoy = [DECOY for _ in range(len(mixed_df_raw))]
        mixed_df_raw.insert(len(mixed_df_raw.columns), "decoys", dumb_decoy)
        mixed_df_raw.to_csv(f"outputs/{outname}_raw.csv")

        mixed_df = pd.DataFrame({"pocket_id": all_pockets,
                                 'decoys': ['chembl' for _ in all_pockets],
                                 'score': all_efs})
        mixed_df.to_csv(f"outputs/{outname}.csv")
    return all_efs


def mix_simplex(df):
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
        all_efs = mix_three_scores(df=df, coeffs=(x, y, z))
        mean_ef = np.mean(all_efs)
        # print(x, y, z, pocket_ef)
        all_thresh_res.append(mean_ef)

    # Print best result
    zs = np.array(all_thresh_res)
    best_i = np.argmax(zs)
    print('Best results for ', [f"{coeff:.3f}" for coeff in coeffs[best_i]], ' value of AuROC : ', zs[best_i])

    # 3D scatter plot the results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = coeffs[:, 0].flatten()
    ys = coeffs[:, 1].flatten()
    ax.scatter(xs, ys, zs)
    plt.show()
    return coeffs[best_i]


def compute_mix_csvs(balanced=True):
    def find_best_mix(runs, grouped=True, decoy='chembl', outname_csv=f'mixed', balanced=True):
        """
        This will find the best coeff to merge dock, fp and native;
        print the best coeff and save the resulting efs in a csv.
        """
        raw_dfs = [pd.read_csv(f"outputs/{r}_raw.csv") for r in runs]
        raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
        raw_dfs = [df[['pocket_id', 'smiles', 'is_active', 'raw_score']] for df in raw_dfs]
        if grouped:
            raw_dfs = [group_df(df) for df in raw_dfs]

        for df in raw_dfs:
            df['smiles'] = df['smiles'].str.strip()

        raw_dfs[0]['rdock'] = -raw_dfs[0]['raw_score'].values
        raw_dfs[1]['dock'] = -raw_dfs[1]['raw_score'].values
        raw_dfs[2]['fp'] = -raw_dfs[2]['raw_score'].values
        raw_dfs[3]['native'] = raw_dfs[3]['raw_score'].values
        raw_dfs = [df.drop('raw_score', axis=1) for df in raw_dfs]


        big_df_raw = raw_dfs[1]
        big_df_raw = big_df_raw.merge(raw_dfs[2], on=['pocket_id', 'smiles', 'is_active'], how='outer')
        big_df_raw = big_df_raw.merge(raw_dfs[3], on=['pocket_id', 'smiles', 'is_active'], how='outer')
        big_df_raw = big_df_raw.merge(raw_dfs[0], on=['pocket_id', 'smiles', 'is_active'], how='inner')
        big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'rdock', 'dock', 'fp', 'native']]
        # rows_with_nan = big_df_raw[big_df_raw.isna().any(axis=1)]

        # Find the best mix, and then dump it
        if balanced:
            best_mix = (0.3, 0.3, 0.3)
        else:
            best_mix = mix_simplex(big_df_raw)
        # best_mix = [0.391 0.304 0.304]  # New models (seed 0) value 0.9959, balanced is 0.9952
        # best_mix = [0.388, 0.222, 0.388]  # New models (seed 1) value 0.9931, balanced is 0.9904
        # best_mix = [0.375, 0.292, 0.333]  # New models (seed 42) value 0.9904, balanced is 0.9898

        # print("balanced perf", np.mean(mix_three(df=big_df_raw, coeffs=(0.3, 0.3, 0.3))))
        # print("best perf", np.mean(mix_three(df=big_df_raw, coeffs=best_mix)))

        # Now dump this best mixed as a csv
        mix_three_scores(big_df_raw, score1='dock', score2='fp', score3='native', coeffs=best_mix,
                         outname_col='mixed', outname=outname_csv)

        # Get the best mixed and add it to the combined results df
        raw_df_mixed = pd.read_csv(f'outputs/{outname_csv}_raw.csv')[['pocket_id', 'smiles', 'is_active', 'mixed']]
        big_df_raw = big_df_raw.merge(raw_df_mixed, on=['pocket_id', 'smiles', 'is_active'], how='outer')

        # Add docknat
        if balanced:
            best_mix_docknat = (0.5, 0., 0.5)
        else:
            all_thresh = np.linspace(0, 1, 30)
            all_results = mix_two_scores(big_df_raw, score1='dock', score2='native', all_thresh=all_thresh)
            best_idx, best_perf, best_thresh = get_best_thresh_perf(all_thresh, all_results)
            best_mix_docknat = (best_thresh, 0, 1 - best_thresh)

        outname_csv = outname_csv.replace('mixed', 'docknat')
        mix_three_scores(big_df_raw, score1='dock', score2='fp', score3='native', coeffs=best_mix_docknat,
                         outname_col='docknat', outname=outname_csv)
        raw_df_mixed = pd.read_csv(f'outputs/{outname_csv}_raw.csv')[['pocket_id', 'smiles', 'is_active', 'docknat']]
        big_df_raw = big_df_raw.merge(raw_df_mixed, on=['pocket_id', 'smiles', 'is_active'], how='outer')
        return big_df_raw

    for seed in SEEDS:
        RUNS = ['rdock',
                f'dock_{seed}',
                f'fp_{seed}',
                f'native_{seed}',
                ]
        out_name = f'mixed{"_grouped" if GROUPED else ""}_{seed}'
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = find_best_mix(runs=RUNS, grouped=GROUPED, decoy=DECOY, outname_csv=out_name, balanced=balanced)
        big_df_raw.to_csv(out_path_raw)


def mix_two_dfs(df_1, df_2, score, all_thresh):
    """
    Instead of mixing one df on two scores, we have two dfs with one score...
    """
    df_1 = df_1[['pocket_id', 'smiles', 'is_active', score]]
    renamed_score = score + '_copy_2'
    df_2[renamed_score] = df_2[score]
    df_2 = df_2[['pocket_id', 'smiles', 'is_active', renamed_score]]
    df_to_use = df_1.merge(df_2, on=['pocket_id', 'smiles', 'is_active'], how='outer')
    all_results = mix_two_scores(df_to_use, score1=score, score2=renamed_score, all_thresh=all_thresh, verbose=False)
    best_idx, best_perf, best_thresh = get_best_thresh_perf(all_thresh, all_results)
    # print(pair, f"{best_idx}/{n_intervals}", best_perf)
    return best_perf


def compute_all_self_mix(balanced=True):
    for i in range(len(SEEDS)):
        to_compare = i, (i + 1) % len(SEEDS)
        out_path_raw_1 = f'outputs/big_df{"_grouped" if GROUPED else ""}_{SEEDS[to_compare[0]]}_raw.csv'
        big_df_raw_1 = pd.read_csv(out_path_raw_1)
        out_path_raw_2 = f'outputs/big_df{"_grouped" if GROUPED else ""}_{SEEDS[to_compare[1]]}_raw.csv'
        big_df_raw_2 = pd.read_csv(out_path_raw_2)
        for score in ['fp', 'native', 'dock']:
            all_thresh = [0.5] if balanced else np.linspace(0, 1, 30)
            best_perf = mix_two_dfs(big_df_raw_1, big_df_raw_2, score, all_thresh)
            print(score, best_perf)


def plot_pairs(score1='rdock', score2='docknat'):
    all_thresh = np.linspace(0, 1, 30)
    all_all_thresh_res = []
    for seed in SEEDS:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)
        all_thresh_res = mix_two_scores(big_df_raw, score1, score2, all_thresh=all_thresh)
        all_all_thresh_res.append(all_thresh_res)
        plt.plot(all_thresh, all_thresh_res, label=f'mixed seed {seed}')
    all_all_thresh_res = np.mean(np.asarray(all_all_thresh_res), axis=0).flatten()
    plt.plot(all_thresh, all_all_thresh_res, label='mean')
    plt.legend()
    plt.show()


def get_one_mixing_table(df, balanced=True):
    all_methods = ['fp', 'native', 'dock', 'rdock']

    n_intervals = 1 if balanced else 10
    all_thresh = [0.5] if balanced else np.linspace(0, 1, n_intervals)
    all_res = {}
    # Do singletons
    for method in all_methods:
        result = get_ef_one(df, score=method)
        # print(method, result)
        all_res[method] = result
    # Do pairs
    for pair in itertools.combinations(all_methods, 2):
        all_results = mix_two_scores(df, score1=pair[0], score2=pair[1], all_thresh=all_thresh, verbose=False)
        best_idx, best_perf, best_thresh = get_best_thresh_perf(all_thresh, all_results)
        # print(pair, f"{best_idx}/{n_intervals}", best_perf)
        all_res[pair] = best_perf

    # Add mixed results
    result_mixed = get_ef_one(df, score='mixed')
    all_res['mixed'] = result_mixed

    pair = ('mixed', 'rdock')
    all_results = mix_two_scores(df, score1=pair[0], score2=pair[1], all_thresh=all_thresh, verbose=False)
    best_idx, best_perf, best_thresh = get_best_thresh_perf(all_thresh, all_results)
    all_res[pair] = best_perf

    # Add docknat results
    result_mixed = get_ef_one(df, score='docknat')
    all_res['docknat'] = result_mixed

    pair = ('docknat', 'rdock')
    all_results = mix_two_scores(df, score1=pair[0], score2=pair[1], all_thresh=all_thresh, verbose=False)
    best_idx, best_perf, best_thresh = get_best_thresh_perf(all_thresh, all_results)
    all_res[pair] = best_perf

    for k, v in all_res.items():
        print(f"{k}, result: {v:.4f}")


def get_table_mixing(balanced=True):
    for seed in SEEDS:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)
        get_one_mixing_table(big_df_raw, balanced=balanced)


def combine_rdock():
    # To dump rdock_combined
    for seed in SEEDS:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)

        # Dump mixed+rdock
        coeffs = (0.75, 0.25, 0.)
        outname_mixed_combined = f'mixed_rdock{"_grouped" if GROUPED else ""}_{seed}'
        mix_three_scores(big_df_raw, score1='mixed', score2='rdock', coeffs=coeffs,
                         outname_col='combined', outname=outname_mixed_combined)
        mixed_df = pd.read_csv(f"outputs/{outname_mixed_combined}_raw.csv")
        print('mixed', get_ef_one(big_df_raw, score='mixed'))
        print('mixed+rdock', get_ef_one(mixed_df, score='combined'))

        # Dump docknat+rdock
        coeffs = (0.66, 0.33, 0.)
        outname_docknat_combined = f'docknat_rdock{"_grouped" if GROUPED else ""}_{seed}'
        mix_three_scores(big_df_raw, score1='docknat', score2='rdock', coeffs=coeffs,
                         outname_col='combined_docknat', outname=outname_docknat_combined)
        docknat_df = pd.read_csv(f"outputs/{outname_docknat_combined}_raw.csv")
        print('docknat', get_ef_one(big_df_raw, score='docknat'))
        print('docknat+rdock', get_ef_one(docknat_df, score='combined_docknat'))

        # Dump nat+rdock
        coeffs = (0.5, 0.5, 0.)
        outname_docknat_combined = f'nat_rdock{"_grouped" if GROUPED else ""}_{seed}'
        mix_three_scores(big_df_raw, score1='native', score2='rdock', coeffs=coeffs,
                         outname_col='combined_nat', outname=outname_docknat_combined)
        nat_df = pd.read_csv(f"outputs/{outname_docknat_combined}_raw.csv")
        print('nat', get_ef_one(big_df_raw, score='native'))
        print('nat+rdock', get_ef_one(nat_df, score='combined_nat'))
        print()


if __name__ == "__main__":
    # Fix groups
    # get_groups()

    DECOY = 'chembl'
    # DECOY = 'pdb'
    GROUPED = True
    SEEDS = [0, 1, 42]

    # FIRST LET'S PARSE INFERENCE CSVS AND MIX THEM
    compute_mix_csvs()

    # To compare to ensembling the same method with different seeds
    compute_all_self_mix()

    # NOW WE HAVE THE BEST ENSEMBLE MODEL AS DATA, we can plot pairs
    plot_pairs(score1='rdock', score2='native')

    # Get table with all mixing
    get_table_mixing()

    # Dump rdock_combined with mixed and docknat
    combine_rdock()
