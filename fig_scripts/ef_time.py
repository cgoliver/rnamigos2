import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_utils import PALETTE, CustomScale


def virtual_screen(df, sort_up_to=0, score_column='rdock'):
    df = df.reset_index(drop=True)
    sort_up_to = int(sort_up_to)
    df[:sort_up_to] = df[:sort_up_to].sort_values(score_column, ascending=False).values
    native_ind = df.loc[df['is_active'] == 1].index[0]
    enrich = 1 - (native_ind / len(df))
    return enrich


def build_ef_df():
    runs = ['rdock',
            'paper_dock',
            'paper_native',
            'paper_fp'
            'mixed'
            'mixed_rdock'
            ]
    decoy = 'chembl'
    raw_dfs = [pd.read_csv(f"../outputs/{r}_raw.csv") for r in runs]
    raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
    raw_dfs = [df.sort_values(by=['pocket_id', 'smiles', 'is_active']) for df in raw_dfs]
    big_df_raw = raw_dfs[0][['pocket_id', 'is_active']]

    # Now add score and flip docking scores, dock scores and distances for which low is better
    big_df_raw['rdock'] = -raw_dfs[0]['raw_score'].values
    big_df_raw['dock'] = -raw_dfs[1]['raw_score'].values
    big_df_raw['fp'] = -raw_dfs[2]['raw_score'].values
    big_df_raw['native'] = raw_dfs[3]['raw_score'].values
    big_df_raw['mixed'] = raw_dfs[4]['combined'].values
    big_df_raw['mixed_rdock'] = raw_dfs[5]['combined'].values

    pockets = big_df_raw['pocket_id'].unique()
    ef_df_rows = []
    nsteps = 20
    for pi, pocket in enumerate(pockets):
        if not pi % 20:
            print(f"Doing pocket {pi}/{len(pockets)}")
        # RDOCK alone
        pocket_df = big_df_raw.loc[big_df_raw['pocket_id'] == pocket]
        for n in range(10):
            # Shuffle
            np.random.seed(n)
            pocket_df = pocket_df.sample(frac=1)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                s = virtual_screen(pocket_df, sort_up_to, score_column='rdock')
                res = {'sort_up_to': i,
                       'pocket': pocket,
                       'ef': s,
                       'model': 'rdock',
                       'seed': n}
                ef_df_rows.append(res)

        # Presort
        for sort_col in ['dock', 'fp', 'native']:
            pocket_df = pocket_df.sort_values(by=sort_col, ascending=False)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                s = virtual_screen(pocket_df, sort_up_to, score_column='rdock')
                res = {'sort_up_to': i,
                       'pocket': pocket,
                       'ef': s,
                       'model': sort_col,
                       'seed': 0}
                ef_df_rows.append(res)

        # Best
        pocket_df = pocket_df.sort_values(by='mixed', ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            s = virtual_screen(pocket_df, sort_up_to, score_column='mixed_rdock')
            res = {'sort_up_to': i,
                   'pocket': pocket,
                   'ef': s,
                   'model': "mixed",
                   'seed': 0}
            ef_df_rows.append(res)
    df = pd.DataFrame(ef_df_rows)
    df.to_csv("time_ef.csv")
    return df


def get_means_stds(df, model):
    model_df = df[df['model'] == model]
    model_df_gb = model_df.groupby(['sort_up_to'], as_index=False)
    model_df_means = model_df_gb.mean()[['ef']].values.squeeze()
    model_df_stds = np.square(model_df_gb.std()[['ef']].values.squeeze())
    return model_df_means, model_df_stds


def plot_mean_std(ax, times, means, stds, label, color):
    ax.plot(times, means, label=label, linewidth=2, color=color)
    ax.fill_between(times, means - stds, means + stds, alpha=0.2, color=color)


def line_plot(df):
    # all_models = ['dock', 'fp', 'native', 'rdock']
    # names = [r'\texttt{fp}', r'\texttt{native}', r'\texttt{dock}', r'\texttt{rDock}', ]
    all_models = ['rdock', 'mixed']
    names = [r'\texttt{rDock}', r'\texttt{mixed+rDock}']
    model_res = []
    for model in all_models:
        means, stds = get_means_stds(df, model)
        model_res.append((means, stds))

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_yscale('custom')

    x_cross = 0.65
    xticks = [0, x_cross, 2, 4, 6, 8]
    xticks_labels = ["0", x_cross, "2", "4", "6", "8"]
    plt.gca().set_xticks(ticks= xticks, labels=xticks_labels)

    yticks = [0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    plt.gca().set_yticks(yticks)

    times = np.linspace(0, 8.3, 20)
    # palette = PALETTE
    palette = [PALETTE[3], PALETTE[-1]]
    for (means, stds), name, color in zip(model_res, names, palette):
        plot_mean_std(ax=ax, times=times, means=means, stds=stds, label=name, color=color)

    plt.axhline(y=0.99, color='grey', linestyle='--', alpha=0.7)
    plt.axvline(x=x_cross, color='grey', linestyle='--', alpha=0.7)

    # sns.lineplot(data=df, x='time_limit', y='ef', hue='model')
    plt.ylabel(r"Mean Active Rank")
    plt.xlabel(r"Time Limit (hours)")
    plt.legend(loc='center left')
    plt.ylim(0.45, 1.001)
    plt.savefig("line.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    pass


def vax_plot(df):
    ref = df.loc[df['model'] == 'rdock'].groupby(['pocket', 'seed']).apply(lambda group: np.trapz(group['ef']))
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
    strategy = 'mixed'
    # strategy = 'fp'
    # strategy = 'Combined Score'
    # strategy = 'Combined Score'
    plot_df = efficiency_df.loc[efficiency_df['model'] == strategy]
    plot_df = plot_df.sort_values(by='efficiency', ascending=False).reset_index()

    # sns.set(style="whitegrid")  # Optional: Set the style of the plot
    ax = sns.pointplot(data=plot_df, y='pdbid', x='efficiency', linestyle='none',
                       errorbar='sd', color=PALETTE[2],
                       linewidth=1.4,
                       # alpha=0.9,
                       # scale=0.5,
                       )
    ax.axvline(x=0.0, color='red', linestyle='--', label="No effect", linewidth=2)
    ax.legend(loc="center left")
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
    plt.savefig("box.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    pass


if __name__ == "__main__":
    # Build the time df for making the figures, this can be commented then
    # build_ef_df()

    df = pd.read_csv("time_ef.csv")
    line_plot(df)
    # vax_plot(df)
    pass
