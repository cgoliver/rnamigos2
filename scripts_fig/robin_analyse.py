import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rnamigos.utils.mixing_utils import normalize

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

POCKET_PATH = "data/json_pockets_expanded"

def plot_all():
    big_df = []
    models = [
        "rdock",
        "dock_42",
        "native_42",
        "rnamigos",
        "rdocknat",
        "combined",
    ]
    for model in models:
        out_csv = os.path.join(RES_DIR, f"{model}.csv")
        df = pd.read_csv(out_csv)
        df["name"] = model
        big_df.append(df)
    big_df = pd.concat(big_df)
    custom_palette = sns.color_palette()
    plt.rcParams["axes.grid"] = True

    g = sns.FacetGrid(
        big_df,
        col="pocket_id",
        hue="name",
        col_wrap=2,
        height=4,
        palette=custom_palette,
        sharey=False,
    )
    g.map(sns.lineplot, "frac", "score").add_legend()

    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_perturbed(model="pre_fm", group=True):
    big_df = []
    for swap in range(4):
        res_dir = "outputs/robin/" if swap == 0 else f"outputs/robin_swap_{swap}"
        out_csv = os.path.join(res_dir, f"{model}.csv")
        df = pd.read_csv(out_csv)
        df["name"] = f"{model}_swap{swap}"
        big_df.append(df)
    if group:
        other_scores = [df["score"].values for df in big_df[1:]]
        perturbed = big_df[1]
        perturbed["name"] = "perturbed"
        import numpy as np

        perturbed["score"] = np.mean(other_scores, axis=0)
        big_df = [big_df[0], perturbed]
    big_df = pd.concat(big_df)

    custom_palette = None
    plt.rcParams["axes.grid"] = True
    g = sns.FacetGrid(
        big_df,
        col="pocket_id",
        hue="name",
        col_wrap=2,
        height=4,
        palette=custom_palette,
        sharey=False,
    )
    g.map(sns.lineplot, "frac", "score").add_legend()

    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_distributions(score_to_use='native_validation', in_csv="outputs/robin/big_df_raw.csv"):
    merged = pd.read_csv(in_csv)
    colors = sns.color_palette(["#33ccff", "#00cccc", "#3366ff", "#9999ff"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, pocket_id in enumerate(merged["pocket_id"].unique()):
        merged_pocket = merged[merged["pocket_id"] == pocket_id].copy()
        merged_pocket[score_to_use] = normalize(merged_pocket[score_to_use])
        g = sns.kdeplot(
            data=merged_pocket,
            x=score_to_use,
            hue="is_active",
            palette={1: colors[i], 0: 'lightgrey'},
            fill=True,
            alpha=0.9,
            linewidth=0,
            common_norm=False,
            ax=axes[i],
        )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SWAP = 0
    RES_DIR = "outputs/robin/" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"

    plot_all()
    # PLOT PERTURBED VERSIONS
    # plot_perturbed(model="rnamigos++", group=True)

    # score_to_use = 'rdock'
    # score_to_use = 'dock_42'
    # score_to_use = 'native_42'
    # score_to_use = 'rnamigos'
    score_to_use = 'combined'
    plot_distributions(score_to_use=score_to_use)
