from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import glob
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def virtual_screen(df, sort_up_to=0, score_column='rdock'):
    df = df.reset_index(drop=True)
    sort_up_to = int(sort_up_to)
    df[:sort_up_to] = df[:sort_up_to].sort_values(score_column, ascending=False).values
    native_ind = df.loc[df['is_active'] == 1].index[0]
    enrich = 1 - (native_ind / len(df))
    return enrich


def build_ef_df():
    runs = ['rdock',
            'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0',
            'definitive_chembl_fp_dim64_simhungarian_prew0',
            'final_chembl_native_graphligs_dim64_optimol1'
            ]
    decoy = 'chembl'
    raw_dfs = [pd.read_csv(f"../outputs/{r}_newdecoys_raw.csv") for r in runs]
    raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
    raw_dfs = [df.sort_values(by=['pocket_id', 'smiles', 'is_active']) for df in raw_dfs]
    big_df_raw = raw_dfs[0][['pocket_id', 'is_active']]

    # Now add score and flip docking scores, dock scores and distances for which low is better
    big_df_raw['rdock'] = -raw_dfs[0]['raw_score'].values
    big_df_raw['dock'] = -raw_dfs[1]['raw_score'].values
    big_df_raw['fp'] = -raw_dfs[2]['raw_score'].values
    big_df_raw['native'] = raw_dfs[3]['raw_score'].values

    pockets = big_df_raw['pocket_id'].unique()
    ef_df_rows = []
    nsteps = 20
    for pi, pocket in enumerate(pockets):
        if not pi % 20:
            print(f"Doing pocket {pi}/{len(pockets)}")
        pocket_df = big_df_raw.loc[big_df_raw['pocket_id'] == pocket]
        for n in range(10):
            # Shuffle
            np.random.seed(n)
            pocket_df = pocket_df.sample(frac=1)
            for sort_up_to in np.linspace(0, len(pocket_df), nsteps).astype(int):
                s = virtual_screen(pocket_df, sort_up_to, score_column='rdock')
                res = {'sort_up_to': sort_up_to,
                       'pocket': pocket,
                       'ef': s,
                       'model': 'rdock',
                       'seed': n}
                ef_df_rows.append(res)
        for sort_col in ['dock', 'fp', 'native']:
            pocket_df = pocket_df.sort_values(by=sort_col, ascending=False)
            for sort_up_to in np.linspace(0, len(pocket_df), nsteps).astype(int):
                s = virtual_screen(pocket_df, sort_up_to, score_column='rdock')
                res = {'sort_up_to': sort_up_to,
                       'pocket': pocket,
                       'ef': s,
                       'model': sort_col,
                       'seed': 0}
                ef_df_rows.append(res)
    df = pd.DataFrame(ef_df_rows)
    df.to_csv("time_ef.csv")
    return df


def line_plot(df):
    window_size = 3
    # df = df[~df['model'].isin(['Combined Score', 'Combined Score + RNAmigos Pre-sort'])]
    df['smoothed_ef'] = savgol_filter(df['ef'], window_size, 2)
    df = df.sort_values(by='time_limit')
    rdock = df[df['model'] == "RDOCK"]
    ours = df[df['model'] == "Mixed"]
    ours_gb = ours.groupby(['time_limit'], as_index=False)
    ours_means = ours_gb.mean()[['smoothed_ef']].values.squeeze()
    ours_stds = np.square(ours_gb.std()[['smoothed_ef']].values.squeeze())

    rdock_gb = rdock.groupby(['time_limit'], as_index=False)
    rdock_means = rdock_gb.mean()[['smoothed_ef']].values.squeeze()
    rdock_stds = np.square(rdock_gb.std()[['smoothed_ef']].values.squeeze())
    times = rdock_gb.std()[['time_limit']].values.squeeze() / 3600

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.2)
    plt.grid(True)

    ax = plt.gca()
    ax.plot(times, 0.867 * np.ones_like(rdock_means), color='red', linewidth=2, linestyle='-', label=r'Ours')
    ax.plot(times, rdock_means, label=r'rDock', linewidth=2)
    ax.fill_between(times, rdock_means - rdock_stds, rdock_means + rdock_stds, alpha=0.2)
    ax.plot(times, ours_means, label=r'\texttt{mixed}', linewidth=2)
    ax.fill_between(times, ours_means - ours_stds, ours_means + ours_stds, alpha=0.2)
    plt.axhline(y=0.933, color='grey', linestyle='--', alpha=0.7)

    # sns.lineplot(data=df, x='time_limit', y='ef', hue='model')
    plt.ylabel(r"Mean Active Rank (\texttt{MAR})")
    plt.xlabel(r"Time Limit (hours)")
    plt.legend(loc='lower right')
    plt.savefig("line.pdf", format="pdf")
    plt.show()
    pass


def vax_plot(df):
    print(df.columns)
    ref = df.loc[df['model'] == 'RDOCK'].groupby(['pocket', 'seed']).apply(lambda group: np.trapz(group['ef']))
    ref_mean = ref.groupby('pocket').mean().reset_index()
    ref_std = ref.groupby('pocket').std().reset_index()
    ref_aucs = {p: {'mean': m, 'std': st} for p, m, st in zip(ref_mean['pocket'], ref_mean[0], ref_std[0])}
    efficiency_df = df.groupby(['pocket', 'model', 'seed']).apply(
        lambda group: np.trapz(group['ef']) / ref_aucs[group.name[0]]['mean']).reset_index().rename(
        columns={0: 'efficiency'})
    # efficiency_df_agg = efficiency_df.groupby(['model', 'pocket']).mean().reset_index().rename(columns={0: 'efficiency_mean'})
    # efficiency_df_agg['efficiency_std'] = efficiency_df.groupby(['model', 'pocket']).std().reset_index().rename(columns={0: 'efficiency_std'})['efficiency_std']
    efficiency_df['pdbid'] = efficiency_df['pocket'].apply(lambda x: x.split("_")[0])
    efficiency_df['ligand'] = efficiency_df['pocket'].apply(lambda x: x.split("_")[2])

    efficiency_df['efficiency'] = efficiency_df['efficiency'] - 1.0
    efficiency_df['efficiency'] *= 100

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.2)

    # strategy = 'RNAmigos2.0'
    strategy = 'Mixed'
    # strategy = 'Combined Score'
    # strategy = 'Combined Score'
    plot_df = efficiency_df.loc[efficiency_df['model'] == strategy]
    plot_df = plot_df.sort_values(by='efficiency', ascending=False).reset_index()

    # sns.set(style="whitegrid")  # Optional: Set the style of the plot
    ax = sns.pointplot(data=plot_df, y='pdbid', x='efficiency', join=False, errorbar='sd', color='lightsteelblue',
                       scale=0.5)
    ax.axvline(x=0.0, color='red', linestyle='--', label="No effect")
    ax.legend(loc="lower right")
    # sns.despine()
    # ax.axvline(x=np.median(efficiency_df['efficiency']), color='grey', linestyle='--')

    # Plot point and thick line for standard deviation

    # sns.pointplot(x='efficiency', y='pdbid', data=plot_df, dodge=True, markers='_', scale=0.5, color='black', ax=ax, orient='h')  # Adjust orient='h' for horizontal orientation

    # for i, group_name in enumerate(plot_df['pdbid'].unique()):
    # group_values = plot_df.loc[plot_df['pdbid'] == group_name, 'efficiency']
    # std_value = group_values.std()
    # ax.plot([group_values.mean(), group_values.mean()], [i - 0.2, i + 0.2], color='black', linewidth=3)  # Swap x and y values and adjust coordinates for horizontal orientation
    # ax.plot([group_values.mean() - std_value, group_values.mean() + std_value], [i, i], color='black', linewidth=3)  # Swap x and y values and adjust coordinates for horizontal orientation

    # Set plot title and labels
    for path in ax.collections:
        path.set(color='steelblue', zorder=10)
    # ax.set_title(f"{strategy}")
    ax.set_ylabel(r"Pocket")
    ax.set_xlabel(r"Efficiency Gain (\%)")
    ax.set_xlim([-100, 100])
    ax.set_yticks([])
    ax.grid(True)
    plt.savefig("box.pdf", format="pdf")
    plt.show()
    pass


if __name__ == "__main__":
    df = build_ef_df()
    # df = pd.read_csv("time_ef.csv")
    # line_plot(df)
    # vax_plot(df)
    pass
