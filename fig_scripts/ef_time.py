from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import matplotlib

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'


def ef(df, score_column):
    native = df.iloc[0]['PDB_POCKET'].split("_")[2]
    sorted_df = df.sort_values(by=score_column)
    sorted_df = sorted_df.reset_index(drop=True)
    native_ind = sorted_df.loc[sorted_df['LIG_NAME'] == native].index[0]
    enrich = 1 - (native_ind / len(sorted_df))
    return enrich


def virtual_screen(df, score_column, time_column, time_limit, pocket_id, sort_col=None):
    if sort_col is None:
        pocket_df = df.loc[df['PDB_POCKET'] == pocket_id].sample(frac=1)
    else:
        pocket_df = df.loc[df['PDB_POCKET'] == pocket_id].sort_values(by=sort_col)

    pocket_df = pocket_df.reset_index(drop=True)

    elapsed_time = 0
    obtained_scores = []
    computed = 0
    for ind, row in pocket_df.iterrows():
        elapsed_time += row[time_column]
        if elapsed_time > time_limit:
            if not sort_col is None:
                obtained_scores.append(row[sort_col])
            else:
                obtained_scores.append(10000.)
        else:
            obtained_scores.append(row[score_column])
            computed += 1

    pocket_df['obtained_score'] = obtained_scores

    enrich = ef(pocket_df, 'obtained_score')

    return enrich


def launch_pockets(df, param, pockets, models, i):
    rows = []
    for pi, p in enumerate(pockets):
        print(p)
        print(f"{pi} of {len(pockets)}")
        for t in np.linspace(500, 70000, 20):
            scores = []
            # repeats screen and take average EF
            for n in range(10):
                np.random.seed(n)
                try:
                    s = virtual_screen(df,
                                       param['score_column'],
                                       param['time_column'],
                                       t,
                                       p,
                                       sort_col=param['sort_col'])
                    rows.append({'time_limit': t,
                                 'pocket': p,
                                 'ef': s,
                                 'model': models[i],
                                 'seed': n
                                 }
                                )
                except Exception as e:
                    print(f"error {e} on {p}")
    return rows


def build_ef_df(csv=None):
    """
    Index(['index', 'lig_id', 'docking_time', 'sort_time', 'report_time',
       'PDB_POCKET', 'LIG_NAME', 'POCKET_ID', 'INTER_SCORE', 'PREDICTED_SCORE',
       'INTER_SCORE_TRANS', 'PREDICTED_SCORE_TRANS', 'ELAPSED_TIME',
       'ELAPSED_TIME_2', 'RDOCK_TIME', 'RNAMIGOS_PREDICTION_TIME'],
      dtype='object')
    """

    df = pd.read_csv("predictions_docking_results_time_test.csv")
    df['combined'] = df['INTER_SCORE'] + df['PREDICTED_SCORE']
    pockets = df['PDB_POCKET'].unique()

    models = ['RDOCK', 'RNAmigos2.0', 'RNAmigos2.0 pre-sort + RDOCK', 'Combined Score',
              'Combined Score + RNAmigos Pre-sort', 'Combined Score + Combined sort']
    models = ['RDOCK', 'RNAmigos2.0 pre-sort + RDOCK']

    params = [{'score_column': 'INTER_SCORE', 'time_column': 'docking_time', 'sort_col': None},
              # {'score_column': 'PREDICTED_SCORE', 'time_column': 'ELAPSED_TIME_2', 'sort_col': None},
              {'score_column': 'INTER_SCORE', 'time_column': 'docking_time', 'sort_col': 'PREDICTED_SCORE'},
              # {'score_column': 'combined', 'time_column': 'docking_time', 'sort_col': None},
              # {'score_column': 'combined', 'time_column': 'docking_time', 'sort_col': 'PREDICTED_SCORE'},
              ]

    ef_df_rows = []
    for i, param in enumerate(params):
        print(param)
        """
        parallel = Parallel(n_jobs=16, return_as="generator")
        for rows in parallel(delayed(launch_pockets)(df, param, p, models, i) for p in pockets):
            ef_df_rows.extend(rows)
        """
        for pi, p in enumerate(pockets):
            print(p)
            print(f"{pi} of {len(pockets)}")
            for t in np.linspace(500, 70000, 20):
                scores = []
                # repeats screen and take average EF
                for n in range(10):
                    np.random.seed(n)
                    try:
                        s = virtual_screen(df,
                                           param['score_column'],
                                           param['time_column'],
                                           t,
                                           p,
                                           sort_col=param['sort_col'])
                        ef_df_rows.append({'time_limit': t,
                                           'pocket': p,
                                           'ef': s,
                                           'model': models[i],
                                           'seed': n
                                           }
                                          )
                    except Exception as e:
                        print(f"error {e} on {p}")

    df = pd.DataFrame(ef_df_rows)
    df.to_csv("time_ef_keep.csv")
    return df


def line_plot(df):
    window_size = 3
    df = df[~df['model'].isin(['Combined Score', 'Combined Score + RNAmigos Pre-sort'])]
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
    # df = build_ef_df()
    df = pd.read_csv("time_ef_keep.csv")
    df = df.replace({'RNAmigos2.0 pre-sort + RDOCK': 'Mixed'})
    line_plot(df)
    # vax_plot(df)
    pass
