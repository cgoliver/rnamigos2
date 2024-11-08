import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts_fig.plot_utils import PALETTE_DICT, group_df

# TEST SET
name_runs = {
    r"COMPAT": "native_42.csv",
    r"AFF": "dock_42.csv",
    r"rDock": "rdock.csv",
    r"MIXED": "docknat_42.csv",
    r"MIXED+rDock": "combined_42.csv",
    # r"COMPAT+rDock": "rdocknat_42.csv",
}

main_palette = [
    # PALETTE_DICT['fp'],
    PALETTE_DICT['native'],
    PALETTE_DICT['dock'],
    PALETTE_DICT['rdock'],
    PALETTE_DICT['mixed'],
    PALETTE_DICT['mixed+rdock'],
    PALETTE_DICT['mixed+rdock'],
]
# violin_palette = PALETTE + PALETTE
names = list(name_runs.keys())
runs = list(name_runs.values())

# decoy_mode = 'pdb'
decoy_mode = 'chembl'
# decoy_mode = 'pdb_chembl'
grouped = True

# Parse ef data for the runs and gather them in a big database
dfs = [pd.read_csv(f"outputs/pockets/{f}") for f in runs]
dfs = [df.assign(name=names[i]) for i, df in enumerate(dfs)]
big_df = pd.concat(dfs)
big_df = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by='score')

# Get Rognan
rognan_dfs = [pd.read_csv(f"outputs/pockets/{f.replace('.csv', '_rognan.csv')}") for f in runs]
rognan_dfs = [df.assign(name=names[i]) for i, df in enumerate(rognan_dfs)]
rognan_dfs = [group_df(df) for df in rognan_dfs]
rognan_dfs = pd.concat(rognan_dfs)
rognan_dfs = rognan_dfs.loc[rognan_dfs['decoys'] == decoy_mode].sort_values(by='score')
rognan_means = rognan_dfs.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
print(rognan_means)

# This is to assess mean difference incurred by different groupings
# means_ungrouped = big_df.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
# import pickle
# script_dir = os.path.dirname(__file__)
# splits_file = os.path.join(script_dir, '../data/train_test_75.p')
# _, _, train_names_grouped, test_names_grouped = pickle.load(open(splits_file, 'rb'))
# groups = {**train_names_grouped, **test_names_grouped}
# centroids = big_df[big_df['pocket_id'].isin(groups)]
# means_centroids = centroids.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
# big_df_grouped = group_df(big_df)
# means_grouped = big_df_grouped.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
# print('Ungrouped')
# print(means_ungrouped)
# print('Group reps')
# print(means_centroids)
# print('Grouped')
# print(means_grouped)

if grouped:
    big_df = group_df(big_df)

# Compute pvalue for rev2
from scipy import stats

mixed_big = big_df[big_df['name'] == 'MIXED+rDock']['score'].values
rdock_big = big_df[big_df['name'] == 'rDock']['score'].values
# res = stats.ttest_ind(mixed_big, rdock_big)
res = stats.ttest_rel(mixed_big, rdock_big)
res_wil = stats.wilcoxon(mixed_big, rdock_big)
print(res, res_wil)

# For a detailed score per pocket
# table = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by=['pocket_id', 'name'])
# print(table.to_latex(index=False, columns=['pocket_id', 'name', 'score']))

# Gather means and std in another df
means = big_df.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
medians = big_df.groupby(by=['name', 'decoys'])['score'].median().reset_index()
stds = list(big_df.groupby(by=['name', 'decoys'])['score'].std().reset_index()['score'])
means['std'] = stds
means['Mean Active Rank'] = means['score'].map('{:,.3f}'.format) + r' $\pm$ ' + means['std'].map('{:,.3f}'.format)
means = means.sort_values(by='score', ascending=False)
sorterIndex = dict(zip(names, range(len(names))))
means['name_rank'] = means['name'].map(sorterIndex)
means = means.sort_values(['name_rank'], ascending=[True])
print(means.to_latex(index=False, columns=['name', 'Mean Active Rank'], float_format="%.2f"))
# sys.exit()

if decoy_mode == 'chembl':
    plt.gca().set_yscale('custom')
    lower = 0.45
    # yticks = np.arange(0.6, 1)
    # yticks = [0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    plt.gca().set_yticks(yticks)

# ADD WHISKERS
sns.boxplot(x="name",
            y="score",
            order=names,
            data=big_df,
            width=.5,
            fill=False,
            palette=main_palette,
            fliersize=0,
            log_scale=False,
            meanline=True
            )

# ADD POINTS
big_df[['score']] = big_df[['score']].clip(lower=lower)
sns.stripplot(x="name",
              y="score",
              order=names,
              jitter=0.07,
              size=5,
              palette=main_palette,
              log_scale=False,
              alpha=0.6,
              data=big_df)

# ADD DISTRIBUTION
violin_alpha = 0.4
sns.violinplot(x="name",
               y="score",
               order=names,
               data=big_df,
               width=.6,
               palette=main_palette,
               cut=0,
               inner=None,
               alpha=violin_alpha,
               )


def patch_violinplot(palette):
    """
    Correct the border to have it in the same color as whisker plot
    """
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    violins = [art for art in ax.get_children() if isinstance(art, PolyCollection)]
    for i in range(len(violins)):
        violins[i].set_edgecolor(palette[i])


patch_violinplot(main_palette)

# ADD MEANS
sns.stripplot(x="name",
              y="score",
              order=names,
              jitter=0,
              size=6,
              palette=main_palette,
              marker="D",
              edgecolor='black',
              linewidth=1.5,
              data=means)

# ADD ROGNAN MEANS
sns.stripplot(x="name",
              y="score",
              order=names,
              jitter=0,
              size=10,
              palette=main_palette,
              marker="*",
              edgecolor='firebrick',
              # edgecolor='black',
              linewidth=1.5,
              data=rognan_means)

plt.ylim(lower - 0.02, 1.001)
plt.xlabel("")
plt.ylabel("AuROC")
plt.grid(True, which='both', axis='y')
# Add vline to separate mixed from docking.
plt.vlines(len(runs) - 2.5, 0.45, 1, colors='grey', linestyles=(0, (5, 10)))
fig_name = f"figs/violins{'_chembl' if decoy_mode == 'chembl' else ''}.pdf"
plt.savefig(fig_name, bbox_inches='tight')
plt.show()
