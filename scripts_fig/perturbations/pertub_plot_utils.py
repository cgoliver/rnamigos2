import matplotlib.pyplot as plt
import numpy as np


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
    pert_magn = (df["pocket_size"].values - df["missing"].values) / (
            df["pocket_size"].values + df["extra"].values
    )
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
        stds = (
            df.groupby("thresh")
            .agg({to_plot: lambda x: x.std() / np.sqrt(len(x))})
            .values.flatten()
        )
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
    means, means_low, means_high = get_low_high(df, fractions,
                                                to_plot=to_plot,
                                                filter_good=filter_good,
                                                good_pockets=good_pockets,
                                                metric=metric)
    plt.plot(fractions, means, linewidth=2, color=color, label=label)
    plt.fill_between(fractions, means_low, means_high, alpha=0.2, color=color)


def plot_list(dfs, fractions, colors="blue", label="default_label", **kwargs):
    for i, df in enumerate(dfs):
        plot_one(df, fractions, color=colors[i], label=f"{label}: {i}", **kwargs)


def end_plot():
    # End of the plot + pretty plot
    # plt.hlines(y=0.934, xmin=min(fractions), xmax=max(fractions),  # dock
    # plt.hlines(y=0.951, xmin=min(fractions), xmax=max(fractions),  # native
    # plt.hlines(y=0.984845, xmin=min(fractions), xmax=max(fractions),
    #            label=r'Original pockets', color=PALETTE_DICT['mixed'], linestyle='--')
    # plt.hlines(y=0.9593, xmin=min(fractions), xmax=max(fractions),
    #            label=r'rDock', color=PALETTE_DICT['rdock'], linestyle='--')
    plt.legend(loc="lower right")
    plt.ylabel(r"mean AuROC over pockets")
    plt.xlabel(r"Fraction of nodes sampled")
    plt.show()


def filter_on_good_pockets(df, good_pockets):
    return df[df["pocket_id"].isin(good_pockets)]
