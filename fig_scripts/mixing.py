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


def get_mix_pair(df, score1, score2, all_thresh):
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
            enrich = native_ind / len(sorted_df)
            all_efs.append(enrich)
        pocket_ef = np.mean(all_efs)
        print(i, pocket_ef)
        all_thresh_res.append(pocket_ef)
    return all_thresh_res


def mix_all(df):
    """
    """
    all_thresh = np.linspace(0.4, 0.6, 3)
    pockets = df['pocket_id'].unique()
    all_thresh_res = []
    xs, ys, zs = [],[],[]
    for i, mixed in enumerate(all_thresh):
        for j, mixed_2 in enumerate(all_thresh):
            all_efs = []
            for pi, p in enumerate(pockets):
                pocket_df = df.loc[df['pocket_id'] == p]
                pocket_df = pocket_df.reset_index(drop=True)
                docking_scores = pocket_df['dock']
                new_scores = pocket_df['fp']
                scores_3 = pocket_df['native']
                normalized_docking = normalize(docking_scores)
                normalized_new = normalize(new_scores)
                normalized_new_3 = normalize(scores_3)

                pocket_df['combined'] = (mixed * normalized_docking
                                         + (mixed_2) * normalized_new
                                         + (1 - mixed - mixed_2) * normalized_new_3).values
                # pocket_df['combined'] = -(mixed * np.exp(- normalized_docking / 3) +
                #                           (1 - mixed) * np.exp(-normalized_new / 3))
                # pocket_df['combined'] = (mixed * normalized_docking ** 4 + (1 - mixed) * normalized_new ** 4).values
                sorted_df = pocket_df.sort_values(by='combined')
                sorted_df = sorted_df.reset_index(drop=True)
                native_ind = sorted_df.loc[sorted_df['is_active'] == 1].index[0]
                enrich = native_ind / len(sorted_df)
                all_efs.append(enrich)
            pocket_ef = np.mean(all_efs)
            print(i, j, pocket_ef)
            xs.append(i)
            ys.append(j)
            zs.append(pocket_ef)
            all_thresh_res.append(pocket_ef)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    print(max(zs))
    plt.show()
    return all_thresh_res


def plot_pairs(df):
    score1 = 'rdock'
    score1 = 'dock'
    score1 = 'fp'
    # scores_2 = ['dock']
    # scores_2 = ['fp']
    # scores_2 = ['native']
    scores_2 = ['native']
    # scores_2 = ['fp', 'native']
    # scores_2 = ['dock', 'fp', 'native']

    all_thresh = np.linspace(0, 1, 10)
    for score2 in scores_2:
        all_thresh_res = get_mix_pair(df, score1, score2, all_thresh=all_thresh)
        plt.plot(all_thresh, all_thresh_res, label=score2)

    # plt.ylim(0.98, 1)
    plt.legend()
    plt.show()
    return df


if __name__ == "__main__":
    runs = ['rdock',
            'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0',
            'definitive_chembl_fp_dim64_simhungarian_prew0',
            'final_chembl_native_graphligs_dim64_optimol1'
            ]
    decoy = 'chembl'
    raw_dfs = [pd.read_csv(f"../outputs/{r}_newdecoys_raw.csv") for r in runs]
    raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
    raw_dfs = [df.sort_values(by=['pocket_id', 'smiles', 'is_active']) for df in raw_dfs]
    big_df_raw = raw_dfs[0][['pocket_id', 'smiles', 'is_active']]

    # Now add score and flip docking scores, dock scores and distances for which low is better
    big_df_raw['rdock'] = -raw_dfs[0]['raw_score'].values
    big_df_raw['dock'] = -raw_dfs[1]['raw_score'].values
    big_df_raw['fp'] = -raw_dfs[2]['raw_score'].values
    big_df_raw['native'] = raw_dfs[3]['raw_score'].values

    # big_df_raw= pd.read_csv("../outputs/big_df_raw.csv")
    # plot_pairs(big_df_raw)
    mix_all(big_df_raw)
