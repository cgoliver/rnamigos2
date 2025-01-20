""" Bar plot comparing VS methods on ChEMBL.
"""

import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

paths = {
    "RNAmigos1": "outputs/fp_split_grouped1_raw.csv",
    "RNAmigos2": "outputs/pockets/big_df_grouped_42_raw.csv",
    "RLDOCK": "outputs/rldock_docking_consolidate_all_terms.csv",
    "rDock": "outputs/rdock_raw.csv",
    "dock6": "outputs/dock6_results_13_10_2024.csv",
    "AnnapuRNA": "outputs/annapurna_results_consolidate.csv",
    "AutoDock-Vina": "outputs/vina_docking_consolidate.csv",
}

score_to_use = {
    "RNAmigos1": "raw_score",
    "RNAmigos2": "maxmerge_42",
    "RLDOCK": "Total_Energy",
    "AnnapuRNA": "score_RNA-Ligand",
    "AutoDock-Vina": "score",
    "rDock": "raw_score",
    "dock6": "GRID_SCORE",
}

names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", "rb"))


def merge_raw_dfs():
    """Combine all raw dfs into one df which is: pocket ID x method x AuROC"""
    dfs = []
    cols = ["raw_score", "pocket_id", "smiles", "is_active", "normed_score"]
    for method, path in paths.items():
        df = pd.read_csv(path)

        if method == "RNAmigos2":
            df["rank_native"] = df.groupby(["pocket_id", "decoys"])["native"].rank(ascending=True, pct=True)
            df["rank_dock"] = df.groupby(["pocket_id", "decoys"])["dock"].rank(ascending=True, pct=True)
            df["maxmerge_42"] = df[["rank_native", "rank_dock"]].max(axis=1)
            df["maxmerge_42"] = df.groupby(["pocket_id", "decoys"])["maxmerge_42"].rank(ascending=True, pct=True)

        if method in ["RNAmigos1", "RNAmigos2", "rDock"]:
            df = df.loc[df["decoys"] == "chembl"]
        if method == "RLDOCK":
            print(df.columns, "RL")
            df["raw_score"] = df["Total_Energy"] - (df["Self_energy_of_ligand"] + df["Self_energy_of_receptor"])
            df.loc[df["raw_score"] > 0, "raw_score"] = 0
        else:
            df["raw_score"] = df[score_to_use[method]]

        if method in ["RNAmigos2"]:
            df["normed_score"] = df.groupby(["pocket_id"])["raw_score"].rank(pct=True)
        else:
            df["normed_score"] = df.groupby(["pocket_id"])["raw_score"].rank(pct=True, ascending=False)

        df = df.loc[:, cols]
        df["method"] = method
        df["decoys"] = "chembl"
        dfs.append(df)

    big_df = pd.concat(dfs)
    return big_df


def plot(df):
    df = df.loc[df["pocket_id"].isin(grouped_test)]
    df = df.loc[df["is_active"] > 0]
    df = df.loc[df["decoys"] == "chembl"]

    custom_palette_bar = {
        method: "#e9e9f8" if method.startswith("RNAmigos") else "#d3d3d3" for method in df["method"].unique()
    }

    custom_palette_point = {
        method: "#b2b2ff" if method.startswith("RNAmigos") else "#a5a5a5" for method in df["method"].unique()
    }

    order = [
        "RLDOCK",
        "AutoDock-Vina",
        "AnnapuRNA",
        "dock6",
        "rDock",
        "RNAmigos1",
        "RNAmigos2",
    ]
    print(df)
    df = df.reset_index(drop=True)
    g = sns.barplot(
        df,
        x="method",
        y="normed_score",
        order=order,
        palette=custom_palette_bar,
        alpha=0.7,
    )
    sns.stripplot(
        df,
        x="method",
        y="normed_score",
        ax=g,
        order=order,
        alpha=0.6,
        palette=custom_palette_point,
    )
    sns.despine()
    plt.xticks(rotation=45)

    for patch in g.patches:
        # Get the x and y position of the top of the bar
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()

        # Plot a black dot on top of the bar
        plt.plot(x, y, "ko", color="black", markersize=5, markeredgewidth=1, zorder=3)

    for i, method in enumerate(order):
        mean = df.loc[df["method"] == method]["normed_score"].mean()
        std = df.loc[df["method"] == method]["normed_score"].std()
        g.text(
            i, 1.05, f"{mean:.2f}$\pm${std:.2f}", fontsize=8, ha="center", va="bottom", color="black"
        )  # Value of the mean with formatting

    plt.gca().set_xlabel("")
    plt.gca().set_ylabel("")
    plt.tight_layout()
    plt.savefig("figs/fig3a.pdf", format="pdf")
    plt.show()
    pass


if __name__ == "__main__":
    df = merge_raw_dfs()
    print(df)
    plot(df)
    pass
