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
    # r"\texttt{fp_old}": "paper_fp.csv",
    # r"\texttt{fp_0}": "fp_split_grouped0.csv",
    # r"\texttt{fp_pre}": "fp_split_grouped1.csv",
    # r"\texttt{fp}": "fp_0.csv",
    # r"\texttt{fp}": "fp_1.csv",
    # r"RECO": "fp_42.csv",
    # r"\texttt{recons}": "fp_42.csv",
    # r"\texttt{fp}": "fp_1_1_2.csv",
    # r"\texttt{fp_2}": "fp_split_grouped2.csv",
    # r"\texttt{native_old}": "paper_native.csv",
    # r"\texttt{native0}": "native_split_grouped0.csv", # r"\texttt{native_pre}": "native_split_grouped1.csv",
    # r"\texttt{native}": "native_0.csv",
    # r"\texttt{toto}": "toto_0.csv",
    # r"\texttt{native}": "native_1.csv",
    r"COMPAT": "native_42.csv",
    # r"\texttt{native}": "native_1.csv",
    # r"\texttt{native2}": "native_split_grouped2.csv",
    # r"\texttt{dock_old}": "paper_dock.csv",
    # r"\texttt{dock0}": "dock_split_grouped0.csv",
    # r"\texttt{dock_pre}": "dock_split_grouped1.csv",
    # r"\texttt{dock}": "dock_0.csv",
    # r"\texttt{dock}": "dock_1.csv",
    r"AFF": "dock_42.csv",
    # r"\texttt{dock2}": "dock_split_grouped2.csv",
    r"rDock": "rdock.csv",
    # r"\texttt{rDock\newline TOTAL}": "rdock_total.csv",
    r"MIXED": "docknat_grouped_42.csv",
    r"MIXED+rDock": "docknat_rdock_grouped_42.csv",
    # r"\texttt{rDock\newline TOTAL}": "rdock_total.csv",
    # r"\texttt{mixed}": "mixed_grouped_42.csv",
    # r"\texttt{mixed\newline+ rDock}": "mixed_rdock_grouped_42.csv",
}

# Difference Mean EF / print is because group_df() does not use group reps but subsamples.
# 0
#                   \textbackslash texttt\{fp\} & 0.874 \$\textbackslash pm\$ 0.220 \\ ~perf at logging epoch : 0.91
#               \textbackslash texttt\{native\} & 0.912 \$\textbackslash pm\$ 0.241 \\ Mean EF : 0.90436 \\ ~perf at logging epoch : 0.96
#                 \textbackslash texttt\{dock\} & 0.928 \$\textbackslash pm\$ 0.160 \\
# 1
#                   \textbackslash texttt\{fp\} & 0.868 \$\textbackslash pm\$ 0.211 \\ ~perf at logging epoch : 0.905
#               \textbackslash texttt\{native\} & 0.960 \$\textbackslash pm\$ 0.138 \\ Mean EF : 0.95541
#                 \textbackslash texttt\{dock\} & 0.945 \$\textbackslash pm\$ 0.141 \\
# 42
#                   \textbackslash texttt\{fp\} & 0.859 \$\textbackslash pm\$ 0.212 \\ ~perf at logging epoch : 0.92
#               \textbackslash texttt\{native\} & 0.944 \$\textbackslash pm\$ 0.168 \\
#                 \textbackslash texttt\{dock\} & 0.935 \$\textbackslash pm\$ 0.157 \\

main_palette = [
    # PALETTE_DICT['fp'],
    PALETTE_DICT['native'],
    PALETTE_DICT['dock'],
    PALETTE_DICT['rdock'],
    PALETTE_DICT['mixed'],
    PALETTE_DICT['mixed+rdock']]
# violin_palette = PALETTE + PALETTE
names = list(name_runs.keys())
runs = list(name_runs.values())

# decoy_mode = 'pdb'
decoy_mode = 'chembl'
# decoy_mode = 'pdb_chembl'

grouped = True

# Parse ef data for the runs and gather them in a big database
dfs = [pd.read_csv(f"outputs/{f}") for f in runs]
dfs = [df.assign(name=names[i]) for i, df in enumerate(dfs)]

big_df = pd.concat(dfs)
big_df = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by='score')

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

mixed_big = big_df[big_df['name'] == 'MIXED']['score'].values
rdock_big = big_df[big_df['name'] == 'rDock']['score'].values
# res = stats.ttest_ind(mixed_big, rdock_big)
res = stats.ttest_rel(mixed_big, rdock_big)
res_wil = stats.wilcoxon(mixed_big, rdock_big)

means = big_df.groupby(by=['name', 'decoys'])['score'].mean().reset_index()
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


plt.gca().set_yscale('custom')
yticks = np.arange(0.6, 1)
lower = 0.6
yticks = [0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
# lower = 0.
# yticks = [0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
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

plt.ylim(lower - 0.02, 1.001)
plt.xlabel("")
plt.ylabel("AuROC")
plt.grid(True, which='both', axis='y')
# Add vline to separate mixed from docking.
plt.vlines(len(runs) - 2.5, 0.65, 1, colors='grey', linestyles=(0, (5, 10)))
# plt.savefig("../outputs/violins.pdf", bbox_inches='tight')
plt.savefig("figs/violins_mixed.pdf", bbox_inches='tight')
plt.show()
