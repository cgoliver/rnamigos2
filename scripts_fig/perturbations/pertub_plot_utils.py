import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from scripts_fig.plot_utils import PALETTE_DICT


def add_delta(df, df_ref):
    if df_ref is None:
        raise ValueError("You must provide a df_ref (DF_UNPERT) to get deltas ")
    df = df.merge(df_ref, how="left", on="pocket_id")
    df["delta"] = df["score"] - df["unpert_score"]
    return df


def add_pert_magnitude(df):
    # pert_magn = (df['extra'].values) / df['pocket_size'].values
    # pert_magn = (df['missing'].values) / df['pocket_size'].values
    # pert_magn = df['missing'].values
    pert_magn = (df["extra"].values + df["missing"].values) / df["pocket_size"].values
    # jaccard
    pert_magn = (df["pocket_size"].values - df["missing"].values) / (df["pocket_size"].values + df["extra"].values)
    df["magnitude"] = pert_magn
    return df


def plot_overlap(df, df_ref=None, filter_good=True, **kwargs):
    df = add_pert_magnitude(df)
    df = add_delta(df, df_ref)
    if filter_good:
        df = filter_on_good_pockets(df)
    plt.scatter(df["magnitude"], df["delta"], **kwargs)


def get_low_high(df, fractions, to_plot="score", filter_good=True, good_pockets=None, error_bar=True, metric="ef"):
    if not isinstance(fractions, (list, tuple)):
        fractions = [fractions]
    # df = df[df['replicate'].isin([str(x) for x in (0, 1)])]
    if filter_good:
        df = filter_on_good_pockets(df, good_pockets=good_pockets)
    df = df[df["thresh"].isin([str(x) for x in fractions])]
    if metric != "ef":
        df[to_plot] = 100 * df[to_plot]
    means = df.groupby("thresh")[to_plot].mean().values
    if error_bar:
        stds = df.groupby("thresh").agg({to_plot: lambda x: x.std() / np.sqrt(len(x))}).values.flatten()
    else:
        stds = df.groupby("thresh")[to_plot].std().values
    means_low = means - stds
    means_high = means + stds
    return means, means_low, means_high


def plot_one(
    df,
    fractions,
    filter_good=False,
    good_pockets=None,
    plot_delta=True,
    color="blue",
    label="default_label",
    metric="ef",
    df_ref=None,
):
    if plot_delta:
        df = add_delta(df, df_ref)
        to_plot = "delta"
    else:
        to_plot = "score"
    means, means_low, means_high = get_low_high(
        df, fractions, to_plot=to_plot, filter_good=filter_good, good_pockets=good_pockets, metric=metric
    )
    plt.plot(fractions, means, linewidth=2, color=color, label=label)
    plt.fill_between(fractions, means_low, means_high, alpha=0.2, color=color)
    print(means)


def plot_list(dfs, fractions, colors="blue", title="default_label", **kwargs):
    for i, df in enumerate(dfs):
        plot_one(df, fractions, color=colors[i], label=rf"${i}$", **kwargs)
        # plot_one(df, fractions, color=colors[i], label=rf"$r={i}$", **kwargs)
    plt.title(title)


def end_plot(fractions, colors, fig_name=None):

    # Create handles with circles
    split_title = True
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=i + 1)
        for i, color in enumerate(colors[:4])
    ]
    if not split_title:
        handles = [Line2D([0], [0], linestyle="None", label=r"Fraction $r$", markersize=0)] + handles

    if split_title:
        plt.legend(
            handles=handles, loc="lower center", ncols=len(handles), title=r"Number of hops $h$", handletextpad=-0.5
        )
    else:
        plt.legend(handles=handles, loc="lower center", ncols=len(handles), handletextpad=0.0, columnspacing=0.1)
    # plt.legend(loc="lower center", ncols=4, title=r"$r$")

    # End of the plot + pretty plot
    # plt.hlines(y=0.984845, xmin=min(fractions), xmax=max(fractions),
    #            label=r'Original pockets', color=PALETTE_DICT['mixed'], linestyle='--')
    plt.hlines(
        y=98.1, xmin=min(fractions), xmax=max(fractions), linestyle="--", linewidth=2, color=PALETTE_DICT["mixed"]
    )
    t = r"\texttt{RNAmigos}"
    plt.text(1.1, 98.3, t, color=PALETTE_DICT["mixed"], usetex=True, fontsize=20)

    plt.hlines(y=95.5, xmin=min(fractions), xmax=max(fractions), linestyle="--", linewidth=2, color="black")
    t = "Pocket Swap"
    plt.text(1.1, 95.7, t, color="black", usetex=False, fontsize=16)

    plt.xlabel(r"Fraction of pocket")
    plt.ylabel(r"AuROC")
    plt.gca().set_xticks(ticks=fractions)
    plt.gca().set_yticks(ticks=[94 + 2 * i for i in range(4)] + [100])
    sns.despine(left=False, bottom=False)
    plt.ylim(94, 99)
    # plt.ylim(, 96)
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches="tight")
    plt.show()


def filter_on_good_pockets(df, good_pockets):
    return df[df["pocket_id"].isin(good_pockets)]
