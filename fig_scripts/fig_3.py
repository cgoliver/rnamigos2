import glob
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_utils import PALETTE, CustomScale

# TEST SET
runs = [
    'definitive_chembl_fp_dim64_simhungarian_prew0_newdecoys.csv',
    'final_chembl_native_graphligs_dim64_optimol1_newdecoys.csv',
    'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_newdecoys.csv',
    'rdock_newdecoys.csv',
]
names = [
    r'\texttt{fp}',
    r'\texttt{native}',
    r'\texttt{dock}',
    r'\texttt{rDock}',
]
decoy_mode = 'chembl'

# ROBIN
# runs = [
#         'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin.csv',
#         'final_chembl_fp_dim64_simhungarian_prew0_robin.csv',
#         'final_chembl_native_graphligs_dim64_optimol1_robin.csv'
#        ]

# Parse ef data for the runs and gather them in a big database
dfs = (pd.read_csv(f"../outputs/{f}") for f in runs)
dfs = (df.assign(name=names[i]) for i, df in enumerate(dfs))
big_df = pd.concat(dfs)
big_df = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by='score')

# For a detailed score per pocket
# table = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by=['pocket_id', 'name'])
# print(table.to_latex(index=False, columns=['pocket_id', 'name', 'score']))

# Gather means and std in another df
means = big_df.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
stds = list(big_df.groupby(by=['name', 'decoys'])['score'].std().reset_index()['score'])
means['std'] = stds
means['Mean Active Rank'] = means['score'].map('{:,.3f}'.format) + r' $\pm$ ' + means['std'].map('{:,.3f}'.format)
means = means.sort_values(by='score', ascending=False)
print(means.to_latex(index=False, columns=['name', 'Mean Active Rank'], float_format="%.2f"))
sorterIndex = dict(zip(names, range(len(names))))
means['name_rank'] = means['name'].map(sorterIndex)
means = means.sort_values(['name_rank'], ascending=[True])

main_palette = PALETTE
violin_palette = PALETTE

plt.gca().set_yscale('custom')
# yticks= np.arange(0.6, 1)
yticks = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
plt.gca().set_yticks(yticks)

sns.stripplot(x="name",
              y="score",
              hue="name",
              order=names,
              legend=False,
              jitter=0,
              size=5,
              # color='black',
              palette=main_palette,
              marker="D",
              edgecolor='black',
              linewidth=0.6,
              # alpha=0.5,
              data=means)

sns.boxplot(x="name",
            y="score",
            hue="name",
            order=names,
            legend=False,
            data=big_df,
            width=.5,
            fill=False,
            palette=main_palette,
            fliersize=0,
            log_scale=False,
            meanline=True
            # showmeans=True
            )
big_df[['score']] = big_df[['score']].clip(lower=0.6)
sns.stripplot(x="name",
              y="score",
              hue="name",
              order=names,
              legend=False,
              jitter=0.07,
              size=5,
              palette=main_palette,
              log_scale=False,
              alpha=0.6,
              data=big_df)


def patch_violinplot(palette):
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        violins[i].set_edgecolor(palette[i])


violin_alpha = 0.3
sns.violinplot(x="name",
               y="score",
               hue="name",
               legend=False,
               data=big_df,
               width=.6,
               palette=violin_palette,
               cut=0,
               inner=None,
               alpha=violin_alpha,
               )
patch_violinplot(violin_palette)

plt.ylim(0.58, 1.001)
plt.xlabel("")
plt.ylabel("Mean Active Rank (MAR)")
# sns.despine()
plt.grid(True, which='both', axis='y')
plt.savefig("../outputs/violins.pdf", bbox_inches='tight')
plt.show()
