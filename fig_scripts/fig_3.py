import glob
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TEST SET
runs = [
    'definitive_chembl_fp_dim64_simhungarian_prew0_newdecoys.csv',
    'final_chembl_native_graphligs_dim64_optimol1_newdecoys.csv',
    'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_newdecoys.csv',
    'rdock_newdecoys.csv',
]
# names = [
#     r'\textrm{Dock}',
#     r'\textrm{Fingerpint}',
#     r'\textrm{Native}',
#     r'\textrm{RDOCK 6.0}',
# ]
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
# names = [r'\textrm{Dock}', r'\textrm{Fingerpint}', r'\textrm{Native}']


# runs = [r for r in glob.glob("outputs/final*.csv") if '_raw' not in r]
# runs += [r for r in glob.glob("outputs/definitive*.csv") if '_raw' not in r]
# runs += ['outputs/rdock.csv']


if __name__ == "__main__":
    pass

    # Parse ef data for the runs and gather them in a big database
    dfs = (pd.read_csv(f"../outputs/{f}") for f in runs)
    dfs = (df.assign(name=names[i]) for i, df in enumerate(dfs))
    # dfs = (df.assign(model_type=allruns[i].split('_')[2]) for i, df in enumerate(dfs))
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

    # Choose palette to get a different color for us and rdock
    # palette_ours = sns.color_palette("Blues", n_colors=3)
    # palette_ours = sns.color_palette("crest", n_colors=3)
    # palette_rdock = sns.color_palette("Reds", n_colors=2)
    # palette = palette_ours + palette_rdock[1:]

    raw_hex = ["#61C6E7", "#4F7BF0", "#6183E7", "#FA4828"]
    raw_hex = ["#3180e0", "#2ba9ff", "#2957d8", "#FA4828"]
    hex_plt = [f"{raw}" for raw in raw_hex]
    palette = sns.color_palette(hex_plt)

    plt.rcParams['text.usetex'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.rc('grid', color='grey', alpha=0.2)

    # main_palette = palette[1::2]
    # violin_palette = palette[0::2]
    main_palette = palette
    violin_palette = palette

    sns.boxplot(x="name",
                y="score",
                hue="name",
                order=names,
                legend=False,
                data=big_df,
                width=.5,
                fill=False,
                # linecolor='blue',
                # color="blue",
                palette=main_palette,
                fliersize=0,
                log_scale=False,
                meanline=True
                # showmeans=True
                )
    big_df[['score']] = big_df[['score']].clip(lower=0.6)
    sns.stripplot(x="name",
                  y="score",
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


    # violin_palette = palette[1::2]
    violin_alpha = 0.3
    sns.violinplot(x="name",
                   y="score",
                   data=big_df,
                   width=.6,
                   # fill=False,
                   # palette=palette,
                   palette=violin_palette,
                   # split=True,
                   # linecolor='red',
                   # color="blue",
                   # log_scale=True,
                   cut=0,
                   inner=None,
                   alpha=violin_alpha,
                   )
    patch_violinplot(violin_palette)

    plt.ylim(0.58, 1.02)
    plt.xlabel("")
    plt.ylabel("Mean Active Rank (MAR)")
    # plt.ylabel("Enrichment Factor @ 1\%")
    # sns.despine()
    plt.grid(True, which='both', axis='y')
    plt.savefig("../outputs/violins.pdf")
    plt.show()
