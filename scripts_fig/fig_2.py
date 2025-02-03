import os
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn.palettes import dark_palette, light_palette, blend_palette
from scipy.stats import spearmanr

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts_fig.plot_utils import group_df, get_smooth_order, PALETTE_DICT


def fig_2ab(root="figs"):
    """
    Docking vs rnamigos score correlation
    """
    names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", "rb"))
    big_df = pd.read_csv("outputs/pockets/big_df_42_raw.csv")
    big_df = big_df.loc[big_df["decoys"] == "chembl"]
    big_df["docknat"] = (big_df["native"] + big_df["dock"]) / 2

    dock_to_use = "dock_pocket_norm"

    migos_scores = ["docknat", "dock"]
    letters = ["a", "b"]
    score_to_name = {"docknat": "AFF+COMPAT", "native": "COMPAT", "dock": "AFF"}

    for letter, migos in zip(letters, migos_scores):
        # keep only test pockets outputs
        big_df = big_df.loc[big_df["pocket_id"].isin(grouped_test.keys())]
        # normalize rDock scores

        big_df["dock_pocket_norm"] = big_df.groupby("pocket_id")["rdock"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        # Create a figure with a grid for the main plot and marginal plots (KDEs)
        fig, ax = plt.subplots(figsize=(8, 6))

        # pocket-wise norm of migos scores
        big_df[migos] = big_df.groupby("pocket_id")[migos].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        actives = big_df.loc[big_df["is_active"] == 1.0]
        decoys = big_df.loc[big_df["is_active"] == 0.0]
        g = sns.regplot(
            data=big_df,
            x=dock_to_use,
            y=migos,
            color=".3",
            ci=99,
            scatter_kws={"alpha": 0.3, "s": 2},
            line_kws={"color": "red"},
            ax=ax,
        )
        sns.scatterplot(data=actives, x=dock_to_use, y=migos, color="blue", ax=ax)

        r, p = spearmanr(big_df[dock_to_use], big_df[migos])
        plt.text(
            x=np.min(big_df[dock_to_use]),
            y=np.max(big_df[migos]) - 0.59,
            s=f"$\\rho$ = {r:.2f}",
            color="red",
            fontweight="bold",
        )

        handles = [
            matplotlib.lines.Line2D([], [], marker="o", color="blue", linestyle="none", markersize=10, label="Active"),
            matplotlib.lines.Line2D([], [], marker="o", color="grey", linestyle="none", markersize=10, label="Decoy"),
        ]

        plt.axhline(y=decoys[migos].mean(), color="grey", linestyle="--")
        plt.axhline(y=actives[migos].mean(), color="blue", linestyle="--")

        plt.ylim([0, 1.1])

        plt.legend(handles=handles, loc="lower right")

        plt.xlabel("Normalized rDock")
        plt.ylabel(score_to_name[migos])
        plt.savefig(f"{root}/fig_2{letter}.pdf", format="pdf")
        plt.savefig(f"{root}/fig_2{letter}.png", format="png")
        plt.clf()
        pass


def fig_2c(root="figs"):
    # TEST SET
    name_runs = {
        r"COMPAT": "native_42.csv",
        r"AFF": "dock_42.csv",
        r"rDock": "rdock.csv",
        r"MIXED": "docknat_42.csv",
        r"MIXED+rDock": "combined_42.csv",
        # r"COMPAT+rDock": "rdocknat_42.csv",
    }

    main_palette = [
        # PALETTE_DICT['fp'],
        PALETTE_DICT["native"],
        PALETTE_DICT["dock"],
        PALETTE_DICT["rdock"],
        PALETTE_DICT["mixed"],
        PALETTE_DICT["mixed+rdock"],
        PALETTE_DICT["mixed+rdock"],
    ]
    # violin_palette = PALETTE + PALETTE
    names = list(name_runs.keys())
    runs = list(name_runs.values())

    # decoy_mode = 'pdb'
    decoy_mode = "chembl"
    # decoy_mode = 'pdb_chembl'
    grouped = True

    # Parse ef data for the runs and gather them in a big database
    dfs = [pd.read_csv(f"outputs/pockets/{f}") for f in runs]
    dfs = [df.assign(name=names[i]) for i, df in enumerate(dfs)]
    big_df = pd.concat(dfs)
    big_df = big_df.loc[big_df["decoys"] == decoy_mode].sort_values(by="score")

    # Get Rognan
    rognan_dfs = [pd.read_csv(f"outputs/pockets/{f.replace('.csv', '_rognan.csv')}") for f in runs]
    rognan_dfs = [df.assign(name=names[i]) for i, df in enumerate(rognan_dfs)]
    rognan_dfs = [group_df(df) for df in rognan_dfs]
    rognan_dfs = pd.concat(rognan_dfs)
    rognan_dfs = rognan_dfs.loc[rognan_dfs["decoys"] == decoy_mode].sort_values(by="score")
    rognan_means = rognan_dfs.groupby(by=["name", "decoys"])["score"].mean().reset_index()

    if grouped:
        big_df = group_df(big_df)

    # Compute pvalue for rev2
    from scipy import stats

    mixed_big = big_df[big_df["name"] == "MIXED+rDock"]["score"].values
    rdock_big = big_df[big_df["name"] == "rDock"]["score"].values
    # res = stats.ttest_ind(mixed_big, rdock_big)
    res = stats.ttest_rel(mixed_big, rdock_big)
    res_wil = stats.wilcoxon(mixed_big, rdock_big)

    # Gather means and std in another df
    means = big_df.groupby(by=["name", "decoys"])["score"].mean().reset_index()
    medians = big_df.groupby(by=["name", "decoys"])["score"].median().reset_index()
    stds = list(big_df.groupby(by=["name", "decoys"])["score"].std().reset_index()["score"])
    means["std"] = stds
    means["Mean Active Rank"] = means["score"].map("{:,.3f}".format) + r" $\pm$ " + means["std"].map("{:,.3f}".format)
    means = means.sort_values(by="score", ascending=False)
    sorterIndex = dict(zip(names, range(len(names))))
    means["name_rank"] = means["name"].map(sorterIndex)
    means = means.sort_values(["name_rank"], ascending=[True])

    if decoy_mode == "chembl":
        plt.gca().set_yscale("custom")
        lower = 0.45
        yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
        plt.gca().set_yticks(yticks)

    # ADD WHISKERS
    sns.boxplot(
        x="name",
        y="score",
        order=names,
        data=big_df,
        width=0.5,
        fill=False,
        palette=main_palette,
        fliersize=0,
        log_scale=False,
        meanline=True,
    )

    # ADD POINTS
    big_df[["score"]] = big_df[["score"]].clip(lower=lower)
    sns.stripplot(
        x="name",
        y="score",
        order=names,
        jitter=0.07,
        size=5,
        palette=main_palette,
        log_scale=False,
        alpha=0.6,
        data=big_df,
    )

    # ADD DISTRIBUTION
    violin_alpha = 0.4
    sns.violinplot(
        x="name",
        y="score",
        order=names,
        data=big_df,
        width=0.6,
        palette=main_palette,
        cut=0,
        inner=None,
        alpha=violin_alpha,
    )

    plt.savefig(f"{root}/fig_2c.pdf", format="pdf")
    plt.clf()
    pass


def fig_2d(root="figs", transposed=False, decoy_mode="chembl"):
    """Heatmap of performance per pocket for all sub-models"""
    name_runs = {
        "COMPAT": "native_42.csv",
        "AFF": "dock_42.csv",
        "rDock": "rdock.csv",
        "MIXED": "docknat_42.csv",
    }

    rows = []
    prev_pockets = None
    for csv_name in name_runs.values():
        df = pd.read_csv(f"outputs/pockets/{csv_name}")
        df = group_df(df)
        row = df[df["decoys"] == decoy_mode].sort_values(by="pocket_id")
        all_pockets = row["pocket_id"].values
        if prev_pockets is None:
            prev_pockets = all_pockets
        else:
            assert (prev_pockets == all_pockets).all(), print(prev_pockets, all_pockets)
        rows.append(row["score"])

    # FIND SMOOTHER PERMUTED VERSION
    # smooth = True
    smooth = False
    if smooth:
        order = get_smooth_order(prev_pockets)
        for i in range(len(rows)):
            new_row = rows[i].values[order]
            rows[i] = new_row

    n_over = 15
    blue_pal = sns.light_palette("#fff", n_colors=n_over)  # white
    red_pal = sns.light_palette("#CF403E", reverse=True, n_colors=128 - n_over)
    cmap = blend_palette(np.concatenate([red_pal, blue_pal]), 1, as_cmap=True)

    if not transposed:
        ax = sns.heatmap(rows, cmap=cmap)
    else:
        ax = sns.heatmap(np.array(rows).T, cmap=cmap)

    # Handle spine
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")

    # Handle ticks
    if not transposed:
        xticks = np.arange(0, len(rows[0]), 10)
        xticks_labels = xticks + 1
        plt.xticks(xticks, xticks_labels, va="center")
        plt.tick_params(axis="x", bottom=False, labelbottom=True)
        plt.yticks(np.arange(len(name_runs)) + 0.5, [name for name in name_runs.keys()], rotation=0, va="center")
        plt.tick_params(axis="y", left=False, right=False, labelleft=True)
    else:
        plt.xticks(np.arange(len(name_runs)) + 0.5, [name for name in name_runs.keys()], rotation=0, va="center")
        plt.tick_params(axis="x", bottom=False, labelbottom=True)
        yticks = np.arange(0, len(rows[0]), 10)
        yticks_labels = yticks + 1
        # In heatmaps, id0 is at the top, it looks weird
        yticks_positions = len(rows[0]) - yticks - 1
        plt.yticks(yticks_positions, yticks_labels, va="center")
        plt.tick_params(axis="y", left=False, labelleft=True)

    if not transposed:
        plt.xlabel(r"Pocket")
        plt.ylabel(r"Method")
    else:
        plt.xlabel(r"Method")
        plt.ylabel(r"Pocket")

    fig_name = f"{root}/fig_2d.pdf"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    root = "figs_repro"
    fig_2ab(root)
    fig_2c(root)
    fig_2d(root)
    pass
