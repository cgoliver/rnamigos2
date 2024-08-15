import glob
from collections import defaultdict
from pathlib import Path
import pandas as pd



def merge_splits(csvs):
    split_dict = defaultdict(list)
    for c in csvs:
        for i in range(10):
            split_dict[c].append(f"outputs/{c}_{i}.csv")
    merged_dfs = {}
    for n, paths in split_dict.items():
        merged_dfs[n] = pd.concat([pd.read_csv(p) for p in paths])
    return merged_dfs

if __name__ == "__main__":
    runs = ['rnamigos1_repro_real',
            'rnamigos2_dim16',
            'rnamigos2_dim64_simR_1_prew0',
            'rnamigos2_dim64_simhungarian_prew0'
            ]
    #csvs = glob.glob("outputs/rnamigos*.csv")
    dfs = merge_splits(runs)
    rows = []
    for name, df in dfs.items():
        print(name, df)
        df = df.groupby('decoys')['score'].mean().reset_index()
        for res in df.itertuples():
            rows.append({'name': name, 'decoy_mode': res.decoys, 'score': res.score})

    res_df = pd.DataFrame(rows)
    res_df = res_df.pivot(columns='decoy_mode', index='name', values='score')
    print(res_df.to_latex())
    print(res_df.to_markdown())
