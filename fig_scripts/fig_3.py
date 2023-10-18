import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

runs = [
        'definitive_chembl_dock_dim64_simhungarian_prew0.csv',
        'definitive_chembl_native_dim64_simhungarian_prew0.csv',
        'definitive_chembl_fp_dim64_simhungarian_prew0.csv',
        ]

names = ['affinity', 'is_native', 'fp']


if __name__ == "__main__":
    decoy_mode = 'pdb'

    dfs = (pd.read_csv(f"outputs/{f}") for f in runs)
    #dfs = (df.loc[df['decoys'] == decoy_mode] for df in dfs)
    dfs = (df.assign(name=names[i]) for i, df in enumerate(dfs))
    big_df = pd.concat(dfs)


    means = big_df.groupby(by=['decoys', 'name'])['score'].mean()
    print(means.to_markdown())

    g = sns.catplot(
                data=big_df, x="name", y="score", col="decoys",
                kind="bar", height=4, aspect=.6,
    )
    plt.show()

    sns.violinplot(data=big_df, inner='point', x='name', y='score')
    plt.show()

    bp = big_df.boxplot(column='score', by='name', grid=False)
    for i, (name, group) in enumerate(big_df.groupby(by='name')):
        y = group['score']
        plt.plot(x, y, 'r.', alpha=0.2)
    plt.show()
