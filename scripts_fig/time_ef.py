import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import pickle
import random

from plot_utils import PALETTE_DICT, CustomScale, group_df


def virtual_screen(df, sort_up_to=0, score_column='rdock'):
    df = df.reset_index(drop=True)
    sort_up_to = int(sort_up_to)
    df[:sort_up_to] = df[:sort_up_to].sort_values(score_column, ascending=False).values
    fpr, tpr, thresholds = metrics.roc_curve(df['is_active'], 1 - np.linspace(0, 1, num=len(df)),
                                             drop_intermediate=True)
    enrich = metrics.auc(fpr, tpr)
    return enrich


def build_ef_df(out_csv='fig_script/time_ef_grouped.csv', grouped=True):
    big_df_raw = pd.read_csv(f'outputs/big_df{"_grouped" if grouped else ""}_42_raw.csv')
    big_df_raw = big_df_raw.sort_values(by=['pocket_id', 'smiles'])

    # Combined and combined_docknat are not present in big_df_raw
    combined_mixed = pd.read_csv(f'outputs/mixed_rdock{"_grouped" if grouped else ""}_42_raw.csv')
    combined_mixed = combined_mixed[['pocket_id', 'smiles', 'is_active', 'combined']]
    big_df_raw = big_df_raw.merge(combined_mixed, on=['pocket_id', 'smiles', 'is_active'], how='outer')

    combined_docknat = pd.read_csv(f'outputs/docknat_rdock{"_grouped" if grouped else ""}_42_raw.csv')
    combined_docknat = combined_docknat[['pocket_id', 'smiles', 'is_active', 'combined_docknat']]
    big_df_raw = big_df_raw.merge(combined_docknat, on=['pocket_id', 'smiles', 'is_active'], how='outer')

    combined_nat = pd.read_csv(f'outputs/nat_rdock{"_grouped" if grouped else ""}_42_raw.csv')
    combined_nat = combined_nat[['pocket_id', 'smiles', 'is_active', 'combined_nat']]
    big_df_raw = big_df_raw.merge(combined_nat, on=['pocket_id', 'smiles', 'is_active'], how='outer')

    # Now iterate
    pockets = big_df_raw['pocket_id'].unique()
    ef_df_rows = []
    nsteps = 20
    nshuffles = 10
    for pi, pocket in enumerate(pockets):
        # if not pocket in ['6QIS_H_J48_101', '6XRQ_A_V8A_103']:
        #     continue
        if not pi % 20:
            print(f"Doing pocket {pi}/{len(pockets)}")

        # RDOCK alone
        pocket_df = big_df_raw.loc[big_df_raw['pocket_id'] == pocket]
        for n in range(nshuffles):
            # Shuffle
            pocket_df = pocket_df.sample(frac=1, random_state=n)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                ef = virtual_screen(pocket_df, sort_up_to, score_column='rdock')
                res = {'sort_up_to': i,
                       'pocket': pocket,
                       'ef': ef,
                       'model': 'rdock',
                       'seed': n}
                ef_df_rows.append(res)

        # Presort
        for sort_col in ['dock', 'fp', 'native', 'mixed', 'docknat']:
            pocket_df = pocket_df.sort_values(by=sort_col, ascending=False)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                ef = virtual_screen(pocket_df, sort_up_to, score_column='rdock')
                res = {'sort_up_to': i,
                       'pocket': pocket,
                       'ef': ef,
                       'model': sort_col,
                       'seed': 0}
                ef_df_rows.append(res)

        # mixed+combined
        pocket_df = pocket_df.sort_values(by='mixed', ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            ef = virtual_screen(pocket_df, sort_up_to, score_column='combined')
            res = {'sort_up_to': i,
                   'pocket': pocket,
                   'ef': ef,
                   'model': "combined",
                   'seed': 0}
            ef_df_rows.append(res)

        # docknat+combined_docknat
        pocket_df = pocket_df.sort_values(by='docknat', ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            s = virtual_screen(pocket_df, sort_up_to, score_column='combined_docknat')
            res = {'sort_up_to': i,
                   'pocket': pocket,
                   'ef': s,
                   'model': "combined_docknat",
                   'seed': 0}
            ef_df_rows.append(res)

        # docknat+combined_nat
        pocket_df = pocket_df.sort_values(by='docknat', ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            s = virtual_screen(pocket_df, sort_up_to, score_column='combined_nat')
            res = {'sort_up_to': i,
                   'pocket': pocket,
                   'ef': s,
                   'model': "combined_nat",
                   'seed': 0}
            ef_df_rows.append(res)
    df = pd.DataFrame(ef_df_rows)
    df.to_csv(out_csv)
    return df


def get_means_stds(df, model):
    model_df = df[df['model'] == model]

    # byhand
    # all_means = list()
    # all_stds = list()
    # for step in model_df['sort_up_to'].unique():
    #     subset = model_df[model_df['sort_up_to'] == step]
    #     mean = subset['ef'].mean()
    #     std = subset['ef'].std()
    #     all_means.append(mean)
    #     all_stds.append(std)

    model_df_gb = model_df.groupby(['sort_up_to'], as_index=False)
    model_df_means = model_df_gb.mean()[['ef']].values.squeeze()
    model_df_stds = model_df_gb.std()[['ef']].values.squeeze()
    n_pockets = len(model_df['pocket'].unique())
    model_df_stds = model_df_stds / np.sqrt(n_pockets)
    # model_df_stds = np.square(model_df_gb.std()[['ef']].values.squeeze())
    return model_df_means, model_df_stds


def plot_mean_std(ax, times, means, stds, label, color):
    means_low = np.clip(means - stds, 0, 1)
    means_high = np.clip(means + stds, 0, 1)
    ax.plot(times, means, label=label, linewidth=2, color=color)
    ax.fill_between(times, means_low, means_high, alpha=0.2, color=color)


def line_plot(df, mixed_model='combined'):
    # Get results
    names = [r'\texttt{rDock}', r'\texttt{RNAmigos++}']
    palette = [PALETTE_DICT['rdock'], PALETTE_DICT['mixed+rdock']]
    model_res = []
    assert mixed_model in {'combined', 'combined_docknat', 'combined_nat'}
    all_models = ['rdock', mixed_model]

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
    if mixed_model == 'combined':
        mixed_means = [0.9898] * 20
    elif mixed_model == 'combined_docknat':
        mixed_means = [0.9848] * 20
    elif mixed_model == 'combined_nat':
        mixed_means = [0.9848] * 20
    else:
        raise ValueError
    ax.plot(times, mixed_means, label=r'\texttt{RNAmigos2}', linewidth=2, color=PALETTE_DICT['mixed'], linestyle='--')

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


def vax_plot(df, mixed_model='combined'):
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
    plot_df = efficiency_df.loc[efficiency_df['model'] == mixed_model]
    plot_df = plot_df.sort_values(by='efficiency', ascending=False).reset_index()

    # sns.set(style="whitegrid")  # Optional: Set the style of the plot
    ax = sns.pointplot(data=plot_df, y='pdbid', x='efficiency', linestyle='none',
                       errorbar='sd', color=PALETTE_DICT['mixed+rdock'],
                       linewidth=1.4,
                       # alpha=0.9,
                       # scale=0.5,
                       )
    ax.axvline(x=0.0, color='red', linestyle='--', label="No effect", linewidth=2)
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
    ax.set_xlim([-20, 100])
    ax.set_yticks([])
    ax.grid(True)
    plt.savefig("figs/efficiency_vax.pdf", format="pdf", bbox_inches='tight')
    plt.show()
    pass


if __name__ == "__main__":
    # Build the time df for making the figures, this can be commented then
    out_csv = 'scripts_fig/time_ef.csv'
    # build_ef_df(out_csv=out_csv)

    df = pd.read_csv(out_csv, index_col=0)
    # mixed_model = 'combined'
    # mixed_model = 'combined_docknat'
    mixed_model = 'combined_nat'
    line_plot(df, mixed_model=mixed_model)
    # vax_plot(df, mixed_model=mixed_model)
    pass
