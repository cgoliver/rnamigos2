import glob
from collections import defaultdict
from pathlib import Path
import pandas as pd



def merge_splits(csvs):
    split_dict = defaultdict(list)
    for c in csvs:
        name = "_".join(Path(c).stem.split("_")[:-1])
        split_dict[name].append(c)
    merged_dfs = {}
    for n, paths in split_dict.items():
        merged_dfs[n] = pd.concat([pd.read_csv(p) for p in paths])
    return merged_dfs

if __name__ == "__main__":
    csvs = glob.glob("outputs/*.csv")
    dfs = merge_splits(csvs)
    rows = []
    for name, df in dfs.items():
        print(name, df)
        df = df.groupby('decoys')['score'].mean().reset_index()
        for res in df.itertuples():
            rows.append({'name': name, 'decoy_mode': res.decoys, 'score': res.score})

    res_df = pd.DataFrame(rows)
    print(res_df)
    res_df = res_df.pivot_table(columns='decoy_mode', index='name', values='score')
    print(res_df.to_latex())
    print(res_df.to_markdown())
