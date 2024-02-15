import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import PALETTE

# TEST SET
runs = [
    'paper_fp.csv',
    'paper_native.csv',
    'paper_dock.csv',
    'rdock.csv',
    # 'rdock_total.csv',
    'mixed.csv',
    'mixed_rdock.csv',
]
names = [
    r'\texttt{fp}',
    r'\texttt{native}',
    r'\texttt{dock}',
    r'\texttt{rDock\newline INTER}',
    # r'\texttt{rDock\newline TOTAL}',
    r'\texttt{mixed}',
    r'\texttt{mixed\newline+ rDock}',
]
decoy_mode = 'chembl'
# decoy_mode = 'pdb_chembl'

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
yticks = [0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
plt.gca().set_yticks(yticks)

# ADD WHISKERS
sns.boxplot(x="name",
            y="score",
            order=names,
            # hue="name",
            # legend=False,
            data=big_df,
            width=.5,
            fill=False,
            palette=main_palette,
            fliersize=0,
            log_scale=False,
            meanline=True
            )

# ADD POINTS
big_df[['score']] = big_df[['score']].clip(lower=0.6)
sns.stripplot(x="name",
              y="score",
              order=names,
              # hue="name",
              # legend=False,
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
               # hue="name",
               # legend=False,
               data=big_df,
               width=.6,
               palette=main_palette,
               cut=0,
               inner=None,
               alpha=violin_alpha,
               )


def patch_violinplot(palette):
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
              # hue="name",
              # legend=False,
              jitter=0,
              size=6,
              palette=main_palette,
              marker="D",
              edgecolor='black',
              linewidth=1.5,
              data=means)

plt.ylim(0.58, 1.001)
plt.xlabel("")
plt.ylabel("Mean Active Rank")
plt.grid(True, which='both', axis='y')
plt.vlines(3.5, 0.65, 1, colors='grey', linestyles=(0, (5, 10)))
# plt.savefig("../outputs/violins.pdf", bbox_inches='tight')
plt.savefig("violins_mixed.pdf", bbox_inches='tight')
plt.show()
