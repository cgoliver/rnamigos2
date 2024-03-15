import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fig_scripts.plot_utils import PALETTE, group_df



# TEST SET
name_runs = {
    # r"\texttt{fp_old}": "paper_fp.csv",
    # r"\texttt{fp_0}": "fp_split_grouped0.csv",
    r"\texttt{fp}": "fp_split_grouped1.csv",
    # r"\texttt{fp_2}": "fp_split_grouped2.csv",
    # r"\texttt{native_old}": "paper_native.csv",
    # r"\texttt{native0}": "native_split_grouped0.csv",
    r"\texttt{native}": "native_split_grouped1.csv",
    # r"\texttt{native2}": "native_split_grouped2.csv",
    # r"\texttt{dock_old}": "paper_dock.csv",
    # r"\texttt{dock0}": "dock_split_grouped0.csv",
    r"\texttt{dock}": "dock_split_grouped1.csv",
    # r"\texttt{dock2}": "dock_split_grouped2.csv",
    r"\texttt{rDock}": "rdock.csv",
    # r"\texttt{rDock\newline TOTAL}": "rdock_total.csv",
    r"\texttt{mixed}": "mixed.csv",
    r"\texttt{mixed\newline+ rDock}": "mixed_rdock.csv",
}

names = list(name_runs.keys())
runs = list(name_runs.values())

# decoy_mode = 'pdb'
decoy_mode = 'chembl'
# decoy_mode = 'pdb_chembl'

grouped = True

# Parse ef data for the runs and gather them in a big database
dfs = (pd.read_csv(f"outputs/{f}") for f in runs)
dfs = (df.assign(name=names[i]) for i, df in enumerate(dfs))
big_df = pd.concat(dfs)
big_df = big_df.loc[big_df['decoys'] == decoy_mode].sort_values(by='score')

if grouped:
    big_df = group_df(big_df)

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
# print(means.to_latex(index=False, columns=['name', 'Mean Active Rank'], float_format="%.2f"))

main_palette = PALETTE + PALETTE # useful extra colors for debug plotting more items
violin_palette = PALETTE + PALETTE

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
plt.vlines(3.5, 0.65, 1, colors='grey', linestyles=(0, (5, 10)))
# plt.savefig("../outputs/violins.pdf", bbox_inches='tight')
plt.savefig("fig_scripts/violins_mixed.pdf", bbox_inches='tight')
plt.show()
