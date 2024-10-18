import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

"""
The main two functions are:
- mix two scores than can explore many combinations
- compute_mix_csvs: creates the original big csv, adding docknat and mixed (potentially with simplex grid search)
"""


# Normalize
def normalize(scores):
    out_scores = (scores - scores.min()) / (scores.max() - scores.min())
    return out_scores


def add_mixed_score(df, score1='dock', score2='native', out_col='mixed'):
    scores_1 = df[score1].values
    scores_2 = df[score2].values
    normalized_scores_1 = normalize(scores_1)
    normalized_scores_2 = normalize(scores_2)
    df[out_col] = (0.5 * normalized_scores_1 + 0.5 * normalized_scores_2)
    return df


def mix_two_scores(df, score1='dock', score2='native', outname=None, outname_col='mixed', add_decoy=True):
    """
    Mix two scores, and return raw, efs and mean efs. Optionally dump the dataframes.
    """
    pockets = df['pocket_id'].unique()
    all_efs = []
    all_pockets = []
    all_dfs = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        # Add temp_name in case one of the input is mixed. This could probably be removed
        pocket_df = pocket_df.reset_index(drop=True)
        pocket_df = add_mixed_score(pocket_df, score1, score2, out_col='temp_name')

        # Then compute aurocs and add to all results
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'],
                                                 pocket_df['temp_name'],
                                                 drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)
        all_efs.append(enrich)
        all_pockets.append(p)
        all_dfs.append(pocket_df[['pocket_id', 'smiles', 'is_active', 'temp_name']])

    if not 'DECOY' in globals():
        DECOY = 'pdb_chembl'

    # Merge raw df and add decoys value
    mixed_df_raw = pd.concat(all_dfs)
    mixed_df_raw = mixed_df_raw.rename(columns={'temp_name': outname_col})
    if "decoys" not in mixed_df_raw.columns and add_decoy:
        dumb_decoy = [DECOY for _ in range(len(mixed_df_raw))]
        mixed_df_raw.insert(len(mixed_df_raw.columns), "decoys", dumb_decoy)

    mixed_df = pd.DataFrame({"pocket_id": all_pockets,
                             'decoys': [DECOY for _ in all_pockets],
                             'score': all_efs})
    if outname is not None:
        mixed_df_raw.to_csv(f"outputs/{outname}_raw.csv")
        mixed_df.to_csv(f"outputs/{outname}.csv")
    return all_efs, mixed_df, mixed_df_raw


def get_mix_score(df, score1='dock', score2='native'):
    all_efs, _, _ = mix_two_scores(df, score1=score1, score2=score2)
    return np.mean(all_efs)


def mix_two_dfs(df_1, df_2, score_1, score_2=None, outname=None, outname_col='mixed'):
    """
    Instead of mixing one df on two scores, we have two dfs with one score...
    """
    score_2 = score_1 if score_2 is None else score_2
    df_1 = df_1[['pocket_id', 'smiles', 'is_active', score_1]]
    renamed_score = score_2 + '_copy_2'
    df_2 = df_2.copy()
    df_2[renamed_score] = df_2[score_2]
    df_2 = df_2[['pocket_id', 'smiles', 'is_active', renamed_score]]
    df_to_use = df_1.merge(df_2, on=['pocket_id', 'smiles', 'is_active'], how='outer')
    all_efs, mixed_df, mixed_df_raw = mix_two_scores(df_to_use,
                                                     score_1,
                                                     renamed_score,
                                                     outname=outname,
                                                     outname_col=outname_col)
    return all_efs, mixed_df, mixed_df_raw
