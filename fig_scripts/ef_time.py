import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import pickle
import random

from plot_utils import PALETTE, CustomScale, group_df


def virtual_screen(df, sort_up_to=0, score_column='rdock'):
    df = df.reset_index(drop=True)
    sort_up_to = int(sort_up_to)
    df[:sort_up_to] = df[:sort_up_to].sort_values(score_column, ascending=False).values
    fpr, tpr, thresholds = metrics.roc_curve(df['is_active'], 1 - np.linspace(0, 1, num=len(df)),
                                             drop_intermediate=True)
    enrich = metrics.auc(fpr, tpr)
    return enrich


def build_ef_df(out_csv='fig_script/time_ef.csv', grouped=True):
    decoy = 'chembl'
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_raw.csv')
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles', 'is_active'])
    # Combined is not present in big_df_raw
    combined = pd.read_csv(f'outputs/mixed_rdock{"_grouped" if grouped else ""}_raw.csv')
    combined = combined.sort_values(by=['pocket_id', 'smiles', 'is_active'])
    big_df_raw['combined'] = combined['combined'].values
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
        for sort_col in ['dock', 'fp', 'native', 'mixed']:
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
            s = virtual_screen(pocket_df, sort_up_to, score_column='combined')
            res = {'sort_up_to': i,
                   'pocket': pocket,
                   'ef': s,
                   'model': "combined",
                   'seed': 0}
            ef_df_rows.append(res)
    df = pd.DataFrame(ef_df_rows)
    df.to_csv(out_csv)
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
    # Get results
    names = [r'\texttt{rDock}', r'\texttt{mixed+rDock}']
    palette = [PALETTE[3], PALETTE[-1]]
    model_res = []
    all_models = ['rdock', 'combined']
    for model in all_models:
        means, stds = get_means_stds(df, model)
        model_res.append((means, stds))

    # Set plot hparams
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['text.usetex'] = True
    plt.rc('grid', color='grey', alpha=0.2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_yscale('custom')

    times = np.linspace(0, 8.3, 20)
    # Add sole mixed performance
    mixed_means = [0.983] * 20
    ax.plot(times, mixed_means, label=r'\texttt{mixed}', linewidth=2, color=PALETTE[-2], linestyle='--')

    for (means, stds), name, color in zip(model_res, names, palette):
        plot_mean_std(ax=ax, times=times, means=means, stds=stds, label=name, color=color)

    # Manage ticks
    # xticks = [0, x_cross, 2, 4, 6, 8]
    # xticks_labels = ["0", x_cross, "2", "4", "6", "8"]
    xticks = [0, 2, 4, 6, 8]
    xticks_labels = ["0", "2", "4", "6", "8"]
    plt.gca().set_xticks(ticks=xticks, labels=xticks_labels)

    # Possible plot: Set y_lim to 0.99 and CustomScale to: offset=0.03, sup_lim=1
    # This shows how fast we go from mixed to mixed+rdock performance
    yticks = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    plt.gca().set_yticks(yticks)
    plt.ylim(0.4, 1)

    plt.ylabel(r"AuROC")
    plt.xlabel(r"Time Limit (hours)")
    plt.legend(loc='center left')
    plt.savefig("figs/efficiency_line.pdf", format="pdf", bbox_inches='tight')
    # plt.savefig("figs/efficiency_line_ylim.pdf", format="pdf", bbox_inches='tight')
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
    plt.savefig("figs/efficiency_vax.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    pass


if __name__ == "__main__":
    # Build the time df for making the figures, this can be commented then
    out_csv = 'fig_scripts/time_ef.csv'
    # build_ef_df(out_csv=out_csv)

    df = pd.read_csv(out_csv)
    line_plot(df)
    # vax_plot(df)
    pass
