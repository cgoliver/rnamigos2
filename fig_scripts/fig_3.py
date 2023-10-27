import glob
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
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

# SETUP PLOT
raw_hex = ["#61C6E7", "#4F7BF0", "#6183E7", "#FA4828"]
raw_hex = ["#3180e0", "#2ba9ff", "#2957d8", "#FA4828"]
hex_plt = [f"{raw}" for raw in raw_hex]
palette = sns.color_palette(hex_plt)
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams.update({'font.size': 16})
plt.rc('font', size=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
plt.rc('grid', color='grey', alpha=0.2)

main_palette = palette
violin_palette = palette


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis):
        mscale.ScaleBase.__init__(self, axis=axis)
        self.offset = 0.03
        self.thresh = None

    def get_transform(self):
        return self.CustomTransform(thresh=self.thresh, offset=self.offset)

    def set_default_locators_and_formatters(self, axis):
        pass

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh
            self.offset = offset

        def transform_non_affine(self, a):
            return - np.log(1 + self.offset - a)

        def inverted(self):
            return CustomScale.InvertedCustomTransform(thresh=self.thresh, offset=self.offset)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, offset, thresh):
            mtransforms.Transform.__init__(self)
            self.offset = offset
            self.thresh = thresh

        def transform_non_affine(self, a):
            return 1 - np.exp(-a) + self.offset

        def inverted(self):
            return CustomScale.CustomTransform(offset=self.offset, thresh=self.thresh)


mscale.register_scale(CustomScale)
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
sns.despine()
plt.grid(True, which='both', axis='y')
plt.savefig("../outputs/violins.pdf", bbox_inches='tight')
plt.show()
