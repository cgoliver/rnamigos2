import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



runs = [
        'final_chembl_dock_graphligs_dim64_optimol1_quant_stretch0.csv',
        'definitive_chembl_native_dim64_simhungarian_prew0.csv',
        'definitive_chembl_fp_dim64_simhungarian_prew0.csv',
        ]

names = ['affinity', 'is_native', 'fp']

allruns = [r for r in glob.glob("outputs/final*.csv") if '_raw' not in r]
allruns += [r for r in glob.glob("outputs/definitive*.csv") if '_raw' not in r]
allruns += ['outputs/rdock.csv']


if __name__ == "__main__":
    decoy_mode = 'pdb'

    dfs = (pd.read_csv(f"{f}") for f in allruns)
    #dfs = (df.loc[df['decoys'] == decoy_mode] for df in dfs)
    dfs = (df.assign(name=Path(allruns[i]).stem) for i, df in enumerate(dfs))
    #dfs = (df.assign(model_type=allruns[i].split('_')[2]) for i, df in enumerate(dfs))
    big_df = pd.concat(dfs)

    big_df = big_df.loc[big_df['decoys'] == 'pdb'].sort_values(by='score')


    means = big_df.groupby(by=['name', 'decoys'])['score'].mean()
    means = means.sort_values(ascending=False)
    print(means.to_markdown())
    
    """
    print(means.loc[means['decoys'] == 'pdb'].to_markdown())
    print(means.loc[means['decoys'] == 'pdb_chembl'].to_markdown())
    print(means.loc[means['decoys'] == 'decoy_finder'].to_markdown())
    """

    sns.violinplot(data=big_df,  x='score', y='name', order=big_df['name'].unique(), orient='h')
    plt.show()

    g = sns.catplot(
                data=big_df, x="score", y="name", col="decoys",
                kind="violin", height=4, aspect=.6,orient='h'
    )
    plt.show()

    sns.violinplot(data=big_df, inner='point', x='name', y='score')
    plt.show()

    bp = big_df.boxplot(column='score', by='name', grid=False)
    for i, (name, group) in enumerate(big_df.groupby(by='name')):
        y = group['score']
        plt.plot(x, y, 'r.', alpha=0.2)
    plt.show()
