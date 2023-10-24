import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



runs = ['rdock.csv',
        'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0.csv',
        'final_chembl_fp_dim64_simhungarian_prew0.csv',
        'final_chembl_native_graphligs_dim64_optimol1.csv'
       ]

runs = [
        'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin.csv',
        'final_chembl_fp_dim64_simhungarian_prew0_robin.csv',
        'final_chembl_native_graphligs_dim64_optimol1_robin.csv'
       ]



names = [r'\textrm{RDOCK 6.0}', r'\textrm{Dock}', r'\textrm{Fingerpint}', r'\textrm{Native}']
names = [r'\textrm{Dock}', r'\textrm{Fingerpint}', r'\textrm{Native}']

"""

allruns = [r for r in glob.glob("outputs/final*.csv") if '_raw' not in r]
allruns += [r for r in glob.glob("outputs/definitive*.csv") if '_raw' not in r]
allruns += ['outputs/rdock.csv']
"""


if __name__ == "__main__":
    decoy_mode = 'robin'

    dfs = (pd.read_csv(f"outputs/{f}") for f in runs)
    #dfs = (df.loc[df['decoys'] == decoy_mode] for df in dfs)
    dfs = (df.assign(name=names[i]) for i, df in enumerate(dfs))
    #dfs = (df.assign(model_type=allruns[i].split('_')[2]) for i, df in enumerate(dfs))
    big_df = pd.concat(dfs)

    big_df = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by='score')

    table = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by=['pocket_id', 'name'])
    print(table.to_latex(index=False, columns=['pocket_id', 'name', 'score']))


    means = big_df.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
    stds = list(big_df.groupby(by=['name', 'decoys'])['score'].std().reset_index()['score'])
    means['std'] = stds
    means['Mean Active Rank'] = means['score'].map('{:,.3f}'.format) + r' $\pm$ ' + means['std'].map('{:,.3f}'.format)
    means = means.sort_values(by='score', ascending=False)

    print(means.to_latex(index=False, columns=['name', 'Mean Active Rank'], float_format="%.2f"))
    
    """
    print(means.loc[means['decoys'] == 'pdb'].to_markdown())
    print(means.loc[means['decoys'] == 'pdb_chembl'].to_markdown())
    print(means.loc[means['decoys'] == 'decoy_finder'].to_markdown())
    """
    sns.stripplot(x = "name",
                  y = "score",
                  size=6,
                  color = 'red',
                  log_scale=False,
                  alpha=0.6,
                  data = big_df)

    sns.boxplot(x = "name",
                y = "score",
                data = big_df,
                width=.5,
                fill=False,
                linecolor='blue',
                color="blue", 
                fliersize=0,
                log_scale=False,
                )

    plt.xlabel("")
    plt.ylabel("Mean Active Rank (MAR)")
    plt.ylabel("Enrichment Factor @ 1\%")

    """
    sns.violinplot(x = "name",
                y = "score",
                data = big_df,
                width=.5,
                fill=False,
                linecolor='red',
                color="blue", 
                log_scale=True,
                cut=0
                )
    """
    sns.despine()

    plt.show()
