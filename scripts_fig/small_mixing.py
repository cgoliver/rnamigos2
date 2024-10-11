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


def mix_two_scores(df, score1='dock', score2='native', outname=None, outname_col='mixed'):
    """
    Mix two scores, and return raw, efs and mean efs. Optionally dump the dataframes.
    """
    pockets = df['pocket_id'].unique()
    all_efs = []
    all_pockets = []
    all_dfs = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        pocket_df = pocket_df.reset_index(drop=True)
        scores_1 = pocket_df[score1]
        scores_2 = pocket_df[score2]
        normalized_scores_1 = normalize(scores_1)
        normalized_scores_2 = normalize(scores_2)

        pocket_df['temp_name'] = (0.5 * normalized_scores_1
                                  + 0.5 * normalized_scores_2).values
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'],
                                                 pocket_df['temp_name'],
                                                 drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)

        all_efs.append(enrich)
        all_pockets.append(p)
        all_dfs.append(pocket_df[['pocket_id', 'smiles', 'is_active', 'temp_name']])
    # Merge df and add decoys value
    mixed_df_raw = pd.concat(all_dfs)
    mixed_df_raw = mixed_df_raw.rename(columns={'temp_name': outname_col})
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


def compute_mix_csvs():
    def merge_csvs(runs, grouped=True, decoy=DECOY):
        """
        Aggregate rdock, native and dock results for a given decoy + add mixing strategies
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
        raw_dfs[2]['native'] = raw_dfs[2]['raw_score'].values
        raw_dfs = [df.drop('raw_score', axis=1) for df in raw_dfs]

        big_df_raw = raw_dfs[1]
        big_df_raw = big_df_raw.merge(raw_dfs[2], on=['pocket_id', 'smiles', 'is_active'], how='outer')
        big_df_raw = big_df_raw.merge(raw_dfs[0], on=['pocket_id', 'smiles', 'is_active'], how='inner')
        big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'rdock', 'dock', 'native']]

        _, _, raw_df_docknat = mix_two_scores(big_df_raw, score1='dock', score2='native', outname_col='docknat')
        big_df_raw = big_df_raw.merge(raw_df_docknat, on=['pocket_id', 'smiles', 'is_active'], how='outer')

        _, _, raw_df_rdocknat = mix_two_scores(big_df_raw, score1='rdock', score2='native', outname_col='rdocknat')
        big_df_raw = big_df_raw.merge(raw_df_rdocknat, on=['pocket_id', 'smiles', 'is_active'], how='outer')

        _, _, raw_df_combined = mix_two_scores(big_df_raw, score1='docknat', score2='rdock', outname_col='combined')
        big_df_raw = big_df_raw.merge(raw_df_combined, on=['pocket_id', 'smiles', 'is_active'], how='outer')
        return big_df_raw

    for seed in SEEDS:
        RUNS = ["rdock",
                "dock_new_pdbchembl_rnafm",
                "native_pretrain_new_pdbchembl_rnafm",
                ]
        # RUNS = ['rdock',
        #         f'dock_{seed}',
        #         f'native_{seed}',
        #         ]
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = merge_csvs(runs=RUNS, grouped=GROUPED, decoy=DECOY)
        big_df_raw.to_csv(out_path_raw)


def compute_all_self_mix():
    def mix_two_dfs(df_1, df_2, score_1, score_2=None):
        """
        Instead of mixing one df on two scores, we have two dfs with one score...
        """
        score_2 = score_1 if score_2 is None else score_2
        df_1 = df_1[['pocket_id', 'smiles', 'is_active', score_1]]
        renamed_score = score_2 + '_copy_2'
        df_2[renamed_score] = df_2[score_2]
        df_2 = df_2[['pocket_id', 'smiles', 'is_active', renamed_score]]
        df_to_use = df_1.merge(df_2, on=['pocket_id', 'smiles', 'is_active'], how='outer')
        mean_ef = get_mix_score(df_to_use, score1=score_2, score2=renamed_score)
        return mean_ef

    for i in range(len(SEEDS)):
        to_compare = i, (i + 1) % len(SEEDS)
        out_path_raw_1 = f'outputs/big_df{"_grouped" if GROUPED else ""}_{SEEDS[to_compare[0]]}_raw.csv'
        big_df_raw_1 = pd.read_csv(out_path_raw_1)
        out_path_raw_2 = f'outputs/big_df{"_grouped" if GROUPED else ""}_{SEEDS[to_compare[1]]}_raw.csv'
        big_df_raw_2 = pd.read_csv(out_path_raw_2)
        for score in ['native', 'dock']:
            best_perf = mix_two_dfs(big_df_raw_1, big_df_raw_2, score)
            print(score, best_perf)


def get_ef_one(df, score, outname=None):
    pockets = df['pocket_id'].unique()
    all_efs = []
    rows = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'], pocket_df[score], drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)
        all_efs.append(enrich)
        rows.append({'pocket_id': p, "score": enrich, "decoys": DECOY, "metric": "MAR"})
    if outname is not None:
        df = pd.DataFrame(rows)
        df.to_csv(outname)
    pocket_ef = np.mean(all_efs)
    return pocket_ef


def get_one_mixing_table(df, seed=42):
    all_methods = ['native', 'dock', 'rdock']
    all_res = {}
    # Do singletons
    for method in all_methods:
        outname = f'outputs/{method}_{seed}.csv'
        result = get_ef_one(df, score=method, outname=outname)
        all_res[method] = result
    # Do pairs
    # for pair in itertools.combinations(all_methods, 2):
    #     mean_ef = get_mix_score(df, score1=pair[0], score2=pair[1])
    #     all_res[pair] = mean_ef
    mean_ef = get_mix_score(df, score1="dock", score2="rdock")
    all_res['dock/rdock'] = mean_ef

    result_mixed = get_ef_one(df, score='docknat', outname=f'outputs/docknat_{seed}.csv')
    all_res['docknat'] = result_mixed
    result_mixed = get_ef_one(df, score='rdocknat', outname=f'outputs/rdocknat_{seed}.csv')
    all_res['rdocknat'] = result_mixed
    result_mixed = get_ef_one(df, score='combined', outname=f'outputs/combined_{seed}.csv')
    all_res['combined'] = result_mixed

    for k, v in all_res.items():
        print(f"{k} \t: {v:.4f}")


def get_table_mixing():
    for seed in SEEDS:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)
        get_one_mixing_table(big_df_raw)


if __name__ == "__main__":
    # Fix groups
    # get_groups()

    DECOY = 'pdb_chembl'
    # DECOY = 'chembl'
    GROUPED = True
    SEEDS = [0]
    # SEEDS = [0, 1, 42]

    # FIRST LET'S PARSE INFERENCE CSVS AND MIX THEM
    compute_mix_csvs()

    # To compare to ensembling the same method with different seeds
    # compute_all_self_mix()

    # Get table with all mixing
    get_table_mixing()
