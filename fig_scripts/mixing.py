import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def vincent_ef_df():
    """
    """
    df = pd.read_csv("../outputs/big_df_raw.csv")
    pockets = df['PDB_POCKET'].unique()
    docking_scores = df['INTER_SCORE']
    # old_scores = df_old['PREDICTED_SCORE']
    new_scores = df['PREDICTED_SCORE_HUNG']

    # Normalize
    # docking_scores = - docking_scores / docking_scores.min()
    # old_scores = -old_scores / old_scores.min()
    # new_scores = -new_scores / new_scores.min()

    # Plot
    # plt.hist([docking_scores, old_scores, new_scores], label=['docking', 'old', 'new'])
    # plt.legend()
    # plt.show()
    # plt.scatter(df['INTER_SCORE'], df['PREDICTED_SCORE'], alpha=0.02)
    # plt.show()
    # oihgrz

    # all_thresh = np.linspace(0.95, 1, 30)
    all_thresh = np.linspace(0, 1, 10)
    all_thresh_res = []
    for i, mixed in enumerate(all_thresh):
        all_efs = []
        df['combined'] = -(mixed * docking_scores ** 4 + (1 - mixed) * new_scores ** 4)
        # df['combined'] = mixed * df['INTER_SCORE'] + (1 - mixed) * df['PREDICTED_SCORE']
        # df['combined'] = -(mixed * np.exp(- docking_scores / 3) +
        #                    (1 - mixed) * np.exp(-new_scores / 3))
        for pi, p in enumerate(pockets):
            try:
                pocket_df = df.loc[df['PDB_POCKET'] == p]
                pocket_df = pocket_df.reset_index(drop=True)
                native = pocket_df.iloc[0]['PDB_POCKET'].split("_")[2]
                score_column = 'combined'
                sorted_df = pocket_df.sort_values(by=score_column)
                sorted_df = sorted_df.reset_index(drop=True)
                native_ind = sorted_df.loc[sorted_df['LIG_NAME'] == native].index[0]
                enrich = 1 - (native_ind / len(sorted_df))
                all_efs.append(enrich)
            except Exception as e:
                pass
                # print(f"error {e} on {p}")
        pocket_ef = np.mean(all_efs)
        print(i, pocket_ef)
        all_thresh_res.append(pocket_ef)

    plt.plot(all_thresh, all_thresh_res)
    plt.show()
    return df


if __name__ == "__main__":
    df = vincent_ef_df()
