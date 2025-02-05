import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn import metrics

from rnamigos.utils.virtual_screen import get_auroc

"""
The main two functions are:
- mix two scores than can explore many combinations
- compute_mix_csvs: creates the original big csv, adding docknat and mixed (potentially with simplex grid search)
"""


# Normalize
def normalize(scores):
    out_scores = (scores - scores.min()) / (scores.max() - scores.min())
    return out_scores


def add_mixed_score(df, score1='dock', score2='native', out_col='mixed', use_max=True):
    scores_1 = df[score1].values
    scores_2 = df[score2].values
    if use_max:
        import scipy
        ranks_1 = scipy.stats.rankdata(scores_1, method='max')
        ranks_2 = scipy.stats.rankdata(scores_2, method='max')
        out_ranks = np.maximum(ranks_1, ranks_2)
        df[out_col] = out_ranks / np.max(out_ranks)
    else:
        normalized_scores_1 = normalize(scores_1)
        normalized_scores_2 = normalize(scores_2)
        df[out_col] = (0.5 * normalized_scores_1 + 0.5 * normalized_scores_2)
    return df


def mix_two_scores(df, score1='dock', score2='native', outname=None, outname_col='mixed', add_decoy=True,
                   use_max=True):
    """
    Mix two scores, and return raw, aurocs and mean aurocs. Optionally dump the dataframes.
    """
    pockets = df['pocket_id'].unique()
    all_aurocs = []
    all_pockets = []
    all_dfs = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        # Add temp_name in case one of the input is mixed. This could probably be removed
        pocket_df = pocket_df.reset_index(drop=True)
        pocket_df = add_mixed_score(pocket_df, score1, score2, out_col='temp_name', use_max=use_max)

        # Then compute aurocs and add to all results
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'],
                                                 pocket_df['temp_name'],
                                                 drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)
        all_aurocs.append(enrich)
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

    mixed_df_aurocs = pd.DataFrame({"pocket_id": all_pockets,
                                    'decoys': [DECOY for _ in all_pockets],
                                    'score': all_aurocs})
    if outname is not None:
        mixed_df_raw.to_csv(f"outputs/{outname}_raw.csv")
        mixed_df_aurocs.to_csv(f"outputs/{outname}.csv")
    return all_aurocs, mixed_df_aurocs, mixed_df_raw


def get_mix_score(df, score1='dock', score2='native'):
    all_aurocs, _, _ = mix_two_scores(df, score1=score1, score2=score2)
    return np.mean(all_aurocs)


def mix_two_dfs(df_1, df_2, score_1, score_2=None, outname=None, outname_col='mixed', use_max=True):
    """
    Instead of mixing one df on two scores, we have two dfs with one score...
    """
    score_2 = score_1 if score_2 is None else score_2
    df_1 = df_1[['pocket_id', 'smiles', 'is_active', score_1]]
    renamed_score = score_2 + '_copy_2'
    df_2 = df_2.copy()
    df_2[renamed_score] = df_2[score_2]
    df_2 = df_2[['pocket_id', 'smiles', 'is_active', renamed_score]]
    df_to_use = df_1.merge(df_2, on=['pocket_id', 'smiles', 'is_active'], how='inner')
    all_aurocs, mixed_df_aurocs, mixed_df_raw = mix_two_scores(df_to_use,
                                                               score_1,
                                                               renamed_score,
                                                               outname=outname,
                                                               outname_col=outname_col,
                                                               use_max=use_max)
    return all_aurocs, mixed_df_aurocs, mixed_df_raw


def mix_all(res_dir, pairs, recompute=False, score="raw_score", use_max=True):
    """
    Used to go from raw dfs to mixed raw_dfs
    :param res_dir:
    :param pairs:
    :param recompute:
    :return:
    """
    for pair, outname in pairs.items():
        outpath = os.path.join(res_dir, f"{outname}.csv")
        outpath_raw = os.path.join(res_dir, f"{outname}_raw.csv")
        if not recompute and os.path.exists(outpath) and os.path.exists(outpath_raw):
            continue

        path1 = os.path.join(res_dir, f"{pair[0]}_raw.csv")
        path2 = os.path.join(res_dir, f"{pair[1]}_raw.csv")
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        _, _, robin_raw_dfs = mix_two_dfs(df1, df2, score_1=score, use_max=use_max)
        robin_raw_dfs = robin_raw_dfs[["pocket_id", "smiles", "is_active", "mixed"]]
        robin_raw_dfs = robin_raw_dfs.rename(columns={"mixed": "raw_score"})
        robin_raw_dfs.to_csv(outpath_raw, index=False)


def unmix(mixed_df, score, decoys=('pdb', 'chembl', 'pdb_chembl'), outpath=None):
    pockets = mixed_df['pocket_id'].unique()
    if not isinstance(decoys, Iterable):
        decoys = [decoys]
    all_rows = []
    for decoy_mode in decoys:
        mixed_df_decoy = mixed_df[mixed_df['decoys'] == decoy_mode]
        for pi, p in enumerate(pockets):
            pocket_df = mixed_df_decoy.loc[mixed_df_decoy['pocket_id'] == p]
            try:
                enrich = get_auroc(pocket_df[score], pocket_df['is_active'])
            except:
                continue
            all_rows.append({"score": enrich, "metric": "MAR", "decoys": decoy_mode, "pocket_id": p})
    df = pd.DataFrame(all_rows)
    if outpath is not None:
        df.to_csv(outpath)
    return df
