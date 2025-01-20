import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from plot_utils import PALETTE_DICT, CustomScale

import matplotlib as mpl

# Set font to Arial or Helvetica, which are commonly used in Nature journals
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]  # Use Arial or fallback options
mpl.rcParams["mathtext.fontset"] = "stixsans"  # Sans-serif font for math


def partial_virtual_screen(df, sort_up_to=0, score_column="rdock"):
    df = df.reset_index(drop=True)
    sort_up_to = int(sort_up_to)
    if sort_up_to > 0:
        # Get the first values, sort them and reassign them back to the original DataFrame
        df[:sort_up_to] = df[:sort_up_to].sort_values(score_column, ascending=False).values
    fpr, tpr, thresholds = metrics.roc_curve(
        df["is_active"], 1 - np.linspace(0, 1, num=len(df)), drop_intermediate=True
    )
    enrich = metrics.auc(fpr, tpr)
    return enrich


def build_auroc_df(out_csv="fig_script/time_auroc.csv", decoy="pdb_chembl", recompute=False):
    if not recompute and os.path.exists(out_csv):
        return
    big_df_raw = pd.read_csv("outputs/pockets/big_df_grouped_42_raw.csv")
    big_df_raw = big_df_raw.loc[big_df_raw["decoys"] == decoy]

    big_df_raw = big_df_raw.sort_values(by=["pocket_id", "smiles"])

    # Now iterate
    pockets = big_df_raw["pocket_id"].unique()
    df_aurocs_rows = []
    nsteps = 20
    nshuffles = 10
    for pi, pocket in enumerate(pockets):
        # if not pocket in ['6QIS_H_J48_101', '6XRQ_A_V8A_103']:
        #     continue
        if not pi % 20:
            print(f"Doing pocket {pi}/{len(pockets)}")
        pocket_df = big_df_raw.loc[big_df_raw["pocket_id"] == pocket]

        # RDOCK alone
        for n in range(nshuffles):
            # Shuffle
            pocket_df = pocket_df.sample(frac=1, random_state=n)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                aurocs = partial_virtual_screen(pocket_df, sort_up_to, score_column="rdock")
                res = {"sort_up_to": i, "pocket": pocket, "auroc": aurocs, "model": "rdock", "seed": n}
                df_aurocs_rows.append(res)

        # Presort
        for sort_col in ["dock", "native", "docknat"]:
            pocket_df = pocket_df.sort_values(by=sort_col, ascending=False)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                aurocs = partial_virtual_screen(pocket_df, sort_up_to, score_column="rdock")
                res = {"sort_up_to": i, "pocket": pocket, "auroc": aurocs, "model": sort_col, "seed": 0}
                df_aurocs_rows.append(res)

        # docknat+rdocknat
        pocket_df = pocket_df.sort_values(by="docknat", ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            auroc = partial_virtual_screen(pocket_df, sort_up_to, score_column="combined")
            res = {"sort_up_to": i, "pocket": pocket, "auroc": auroc, "model": "rdocknat", "seed": 0}
            df_aurocs_rows.append(res)
    df = pd.DataFrame(df_aurocs_rows)
    df.to_csv(out_csv)
    return df


def build_auroc_df_robin(out_csv="fig_script/time_auroc_robin.csv", recompute=False):
    if not recompute and os.path.exists(out_csv):
        return
    big_df = pd.read_csv(f"outputs/robin/big_df_raw.csv")
    big_df = big_df.sort_values(by=["pocket_id", "smiles"])

    big_df["rank_native"] = big_df.groupby("pocket_id")["native_42"].rank(ascending=True, pct=True)
    big_df["rank_dock"] = big_df.groupby("pocket_id")["dock_42"].rank(ascending=True, pct=True)

    def maxmin(column):
        return (column - column.min()) / (column.max() - column.min())

    big_df["scaled_native"] = big_df.groupby("pocket_id")["native_42"].transform(maxmin)
    big_df["scaled_dock"] = big_df.groupby("pocket_id")["dock_42"].transform(maxmin)

    big_df["maxmerge_42"] = big_df[["rank_native", "rank_dock"]].max(axis=1)
    big_df["maxmerge_42"] = big_df.groupby("pocket_id")["maxmerge_42"].rank(ascending=True, pct=True)
    big_df["rank_rdock"] = big_df.groupby("pocket_id")["rdock"].rank(ascending=True, pct=True)
    big_df["rank_rnamigos"] = big_df.groupby("pocket_id")["maxmerge_42"].rank(ascending=True, pct=True)
    big_df["combined_42_max"] = big_df[["rank_rnamigos", "rank_rdock"]].max(axis=1)
    big_df["combined_42_max"] = big_df.groupby("pocket_id")["combined_42_max"].rank(ascending=True, pct=True)

    # Now iterate
    pockets = big_df["pocket_id"].unique()
    df_auroc_rows = []
    nsteps = 20
    nshuffles = 10
    for pi, pocket in enumerate(pockets):
        print(f"Doing pocket {pi + 1}/{len(pockets)}")
        pocket_df = big_df.loc[big_df["pocket_id"] == pocket]

        # RDOCK alone
        for n in range(nshuffles):
            # Shuffle
            pocket_df = pocket_df.sample(frac=1, random_state=n)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                auroc = partial_virtual_screen(pocket_df, sort_up_to, score_column="rdock")
                res = {"sort_up_to": i, "pocket": pocket, "auroc": auroc, "model": "rdock", "seed": n}
                df_auroc_rows.append(res)

        # Presort
        for sort_col in ["rnamigos_42", "dock_42", "native_42"]:
            pocket_df = pocket_df.sort_values(by=sort_col, ascending=False)
            for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
                auroc = partial_virtual_screen(pocket_df, sort_up_to, score_column="rdock")
                res = {"sort_up_to": i, "pocket": pocket, "auroc": auroc, "model": sort_col, "seed": 0}
                df_auroc_rows.append(res)

        # docknat+rdocknat
        pocket_df = pocket_df.sort_values(by="maxmerge_42", ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            auroc = partial_virtual_screen(pocket_df, sort_up_to, score_column="combined_42_max")
            res = {"sort_up_to": i, "pocket": pocket, "auroc": auroc, "model": "combined_42_max", "seed": 0}
            df_auroc_rows.append(res)

        """
        pocket_df = pocket_df.sort_values(by="updated_rnamigos", ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            auroc = partial_virtual_screen(pocket_df, sort_up_to, score_column="updated_rdocknat")
            res = {"sort_up_to": i, "pocket": pocket, "auroc": auroc, "model": "updated_rdocknat", "seed": 0}
            df_auroc_rows.append(res)

        pocket_df = pocket_df.sort_values(by="updated_rnamigos", ascending=False)
        for i, sort_up_to in enumerate(np.linspace(0, len(pocket_df), nsteps).astype(int)):
            auroc = partial_virtual_screen(pocket_df, sort_up_to, score_column="updated_combined")
            res = {"sort_up_to": i, "pocket": pocket, "auroc": auroc, "model": "updated_combined", "seed": 0}
            df_auroc_rows.append(res)
        """

    df = pd.DataFrame(df_auroc_rows)
    df.to_csv(out_csv)
    return df


def get_means_stds(df, model, pocket_id=None):
    if pocket_id is None:
        model_df = df[df["model"] == model]
    else:
        model_df = df.loc[(df["model"] == model) & (df["pocket"] == pocket_id)]

    # byhand
    # all_means = list()
    # all_stds = list()
    # for step in model_df['sort_up_to'].unique():
    #     subset = model_df[model_df['sort_up_to'] == step]
    #     mean = subset['auroc'].mean()
    #     std = subset['auroc'].std()
    #     all_means.append(mean)
    #     all_stds.append(std)

    model_df_gb = model_df.groupby(["sort_up_to"], as_index=False)
    model_df_gb = model_df.groupby(["sort_up_to"])
    model_df_means = model_df_gb[["auroc"]].mean().values.squeeze()
    model_df_stds = model_df_gb[["auroc"]].std().values.squeeze()
    n_pockets = len(model_df["pocket"].unique())
    model_df_stds = model_df_stds / np.sqrt(n_pockets)
    # model_df_stds = np.square(model_df_gb.std()[['auroc']].values.squeeze())
    return model_df_means, model_df_stds


def plot_mean_std(ax, times, means, stds, label, color):
    means_low = np.clip(means - stds, 0, 1)
    means_high = np.clip(means + stds, 0, 1)
    ax.plot(times, means, label=label, linewidth=2, color=color)
    ax.fill_between(times, means_low, means_high, alpha=0.2, color=color)


def line_plot_per_pocket(df, mixed_model="combined", robin=False):
    # Get results
    names = [r"\texttt{rDock}", f"{mixed_model}"]
    palette = [PALETTE_DICT["rdock"], PALETTE_DICT["mixed+rdock"]]
    # assert mixed_model in {'combined', 'combined_docknat', 'combined_nat'}
    all_models = ["rdock", mixed_model]

    for pocket in df["pocket"].unique():
        model_res = []
        for model in all_models:
            means, stds = get_means_stds(df, model, pocket_id=pocket)
            model_res.append((means, stds))

        # Set plot hparams
        plt.rcParams.update({"font.size": 16})
        # plt.rcParams["text.usetex"] = True
        # plt.rc("grid", color="grey", alpha=0.2)
        # plt.grid(True)
        ax = plt.gca()
        ax.set_yscale("custom")

        times = np.linspace(0, 200, 20)
        # # Add sole mixed performance
        # CHEMBL results
        # if mixed_model == 'combined':
        #     mixed_means = [0.9898] * 20
        # elif mixed_model == 'combined_docknat':
        #     mixed_means = [0.9848] * 20
        # elif mixed_model == 'combined_nat':
        #     mixed_means = [0.9848] * 20
        # else:
        #     raise ValueError

        # PDB CHEMBL results
        # Add sole mixed performance
        if mixed_model == "combined":
            mixed_means = [0.9848] * 20
        elif mixed_model == "rdocknat":
            mixed_means = [0.896] * 20
        else:
            mixed_means = [0.850] * 20
            print("Unexpected model, dashed line is confused")
        if not robin:
            ax.plot(
                times,
                mixed_means,
                label=r"\texttt{RNAmigos2}",
                linewidth=2,
                color=PALETTE_DICT["mixed"],
                linestyle="--",
            )

        for (means, stds), name, color in zip(model_res, names, palette):
            plot_mean_std(ax=ax, times=times, means=means, stds=stds, label=name, color=color)

        # Manage ticks
        # xticks = [0, x_cross, 2, 4, 6, 8]
        # xticks_labels = ["0", x_cross, "2", "4", "6", "8"]
        # xticks = [0, 2, 4, 6, 8]
        # xticks_labels = ["0", "2", "4", "6", "8"]
        # plt.gca().set_xticks(ticks=xticks, labels=xticks_labels)

        # Possible plot: Set y_lim to 0.99 and CustomScale to: offset=0.03, sup_lim=1
        # This shows how fast we go from mixed to mixed+rdock performance
        yticks = [0.5, 0.7, 0.8, 0.9, 0.925, 0.94]
        # yticks = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
        # plt.gca().set_yticks(yticks)
        if not robin:
            plt.ylim(0.4, 0.94)

        plt.ylabel(r"AuROC")
        plt.xlabel(r"Time Limit (CPU hours)")
        plt.legend(loc="lower right")
        plt.title(pocket)
        sns.despine()
        plt.savefig(f"figs/efficiency_line_{pocket}.pdf", format="pdf", bbox_inches="tight")
        # plt.savefig("figs/efficiency_line_ylim.pdf", format="pdf", bbox_inches='tight')
        plt.show()
        pass


def line_plot(df, mixed_model="combined", robin=False):
    print(df)
    # Get results
    names = [r"\texttt{rDock}", f"{mixed_model}"]
    palette = [PALETTE_DICT["rdock"], PALETTE_DICT["mixed+rdock"]]


def line_plot(df, mixed_model="combined", robin=False, decoy_mode="pdb_chembl"):
    # Get results
    # names = [r'\texttt{rDock}', f'{mixed_model}']
    names = [r"\texttt{rDock}", r"\texttt{RNAmigos++}"]
    palette = [PALETTE_DICT["rdock"], PALETTE_DICT["mixed+rdock"]]

    plt.rcParams.update({"font.size": 16})
    plt.rcParams["text.usetex"] = True
    plt.rc("grid", color="grey", alpha=0.2)
    plt.grid(True)
    ax = plt.gca()
    ax.set_yscale("custom")

    times = np.linspace(0, 8.3, 20)
    # # Add sole mixed performance
    # CHEMBL results
    # if mixed_model == 'combined':
    #     mixed_means = [0.9898] * 20
    # elif mixed_model == 'combined_docknat':
    #     mixed_means = [0.9848] * 20
    # elif mixed_model == 'combined_nat':
    #     mixed_means = [0.9848] * 20
    # else:
    #     raise ValueError

    # PDB CHEMBL results
    # Add sole mixed performance
    if mixed_model == "combined":
        mixed_means = [0.9848] * 20
    elif mixed_model == "rdocknat":
        mixed_means = [0.896] * 20
    else:
        mixed_means = [0.850] * 20
        print("Unexpected model, dashed line is confused")
    if not robin:
        ax.plot(
            times, mixed_means, label=r"\texttt{RNAmigos2}", linewidth=2, color=PALETTE_DICT["mixed"], linestyle="--"
        )

    times = np.linspace(0, 8.3, 20)
    if not robin:
        if decoy_mode == "pdb_chembl":
            mixed_means = [0.896] * 20
        else:
            mixed_means = [0.954] * 20
        ax.plot(
            times, mixed_means, label=r"\texttt{RNAmigos2}", linewidth=2, color=PALETTE_DICT["mixed"], linestyle="--"
        )
    yticks = [0.5, 0.7, 0.8, 0.9, 0.925, 0.94]
    # yticks = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    # plt.gca().set_yticks(yticks)
    if not robin:
        ax.set_yscale("custom")
    if decoy_mode == "pdb_chembl":
        yticks = [0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94]
        plt.ylim(0.4, 0.94)
    else:
        yticks = [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
        plt.ylim(0.45, 1.0)
    plt.gca().set_yticks(yticks)

    plt.ylabel(r"AuROC")
    plt.xlabel(r"Time Limit (hours)")
    plt.legend(loc="lower right")
    fig_name = f"figs/efficiency_line{'_chembl' if decoy_mode == 'chembl' else ''}.pdf"
    plt.savefig(fig_name, format="pdf", bbox_inches="tight")
    plt.show()


def vax_plot(df, mixed_model="combined", decoy_mode="pdb_chembl"):
    ref = df.loc[df["model"] == "rdock"].groupby(["pocket", "seed"]).apply(lambda group: np.trapz(group["auroc"]))
    ref_mean = ref.groupby("pocket").mean().reset_index()
    ref_std = ref.groupby("pocket").std().reset_index()
    ref_aucs = {p: {"mean": m, "std": st} for p, m, st in zip(ref_mean["pocket"], ref_mean[0], ref_std[0])}
    efficiency_df = (
        df.groupby(["pocket", "model", "seed"])
        .apply(lambda group: np.trapz(group["auroc"]) / ref_aucs[group.name[0]]["mean"])
        .reset_index()
        .rename(columns={0: "efficiency"})
    )
    # efficiency_df_agg = efficiency_df.groupby(['model', 'pocket']).mean().reset_index().rename(columns={0: 'efficiency_mean'})
    # efficiency_df_agg['efficiency_std'] = efficiency_df.groupby(['model', 'pocket']).std().reset_index().rename(columns={0: 'efficiency_std'})['efficiency_std']
    efficiency_df["pdbid"] = efficiency_df["pocket"].apply(lambda x: x.split("_")[0])
    efficiency_df["ligand"] = efficiency_df["pocket"].apply(lambda x: x.split("_")[2])

    efficiency_df["efficiency"] = efficiency_df["efficiency"] - 1.0
    efficiency_df["efficiency"] *= 100

    plt.rcParams.update({"font.size": 16})
    plt.rcParams["text.usetex"] = True
    plt.rc("grid", color="grey", alpha=0.2)

    # strategy = 'RNAmigos2.0'
    plot_df = efficiency_df.loc[efficiency_df["model"] == mixed_model]
    plot_df = plot_df.sort_values(by="efficiency", ascending=False).reset_index()

    # sns.set(style="whitegrid")  # Optional: Set the style of the plot
    ax = sns.pointplot(
        data=plot_df,
        y="pdbid",
        x="efficiency",
        linestyle="none",
        errorbar="sd",
        color=PALETTE_DICT["mixed+rdock"],
        linewidth=1.4,
        # alpha=0.9,
        # scale=0.5,
    )
    ax.axvline(x=0.0, color="red", linestyle="--", label="No effect", linewidth=2)
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
        path.set(color="steelblue", zorder=10)
    # ax.set_title(f"{strategy}")
    ax.set_ylabel(r"Pocket")
    ax.set_xlabel(r"Efficiency Gain (\%)")
    ax.set_xlim([-20, 100])
    ax.set_xlim([-20, 50])
    ax.set_yticks([])
    ax.grid(True)
    fig_name = f"figs/efficiency_vax{'_chembl' if decoy_mode == 'chembl' else ''}.pdf"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.show()
    pass


if __name__ == "__main__":
    # Build the time df for making the figures
    # recompute = False
    recompute = False
    decoy_mode = "chembl"
    # decoy_mode = 'pdb_chembl'
    # FOR A NICE PLOT, one should also choose the right scale in plot_utils
    # out_csv = f'scripts_fig/time_auroc{"_chembl" if decoy_mode == "chembl" else ""}.csv'
    # build_auroc_df(out_csv=out_csv, recompute=recompute, decoy=decoy_mode)

    out_csv_robin = "scripts_fig/time_auroc_robin.csv"
    # recompute = False
    recompute = True
    build_auroc_df_robin(out_csv=out_csv_robin, recompute=recompute)

    # Then make plots
    df = pd.read_csv(out_csv_robin, index_col=0)
    # mixed_model = "rnamigos_42"
    # mixed_model = "maxmerge_42"
    mixed_model = "combined_42_max"
    # mixed_model = "combined"
    # mixed_model = 'dock'
    # line_plot(df, mixed_model=mixed_model, decoy_mode=decoy_mode)
    line_plot_per_pocket(df, mixed_model=mixed_model, robin=True)
    vax_plot(df, mixed_model=mixed_model, decoy_mode=decoy_mode)

    # df = pd.read_csv(out_csv_robin, index_col=0)
    # mixed_model = 'dock_rnafm_3'
    # mixed_model = 'native_validation'
    # mixed_model = 'updated_rnamigos'
    # mixed_model = 'updated_rdocknat'
    # mixed_model = 'updated_combined'
    # line_plot(df, mixed_model=mixed_model, robin=True)
    # vax_plot(df, mixed_model=mixed_model)
