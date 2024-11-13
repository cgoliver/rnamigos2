import os

from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rnamigos.utils.mixing_utils import normalize
from rnamigos.utils.virtual_screen import enrichment_factor
from rnamigos.utils.virtual_screen import raw_df_to_mean_auroc, raw_df_to_efs

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

POCKET_PATH = "data/json_pockets_expanded"


def ef_lines(swap=0):
    df = pd.read_csv(f"{RES_DIR}/big_df_raw.csv")
    scores = ["rdock", "dock_42", "native_42", "rnamigos_42", "combined_42"]
    # scores = ["rdock", "dock_42", "native_42", "combined"]

    pockets = list(df["pocket_id"].unique())
    fracs = [0.01, 0.02, 0.05, 0.1, 0.2]

    ef_df_rows = []
    for score in scores:
        for pocket in pockets:
            for frac in fracs:
                out = df.loc[df["pocket_id"] == pocket]
                ef = enrichment_factor(out[score], out["is_active"], frac=frac)
                ef_df_rows.append({"pocket_id": pocket, "score": score, "frac": frac, "ef": ef})
                pass
            pass
        pass
    pass

    ef_df = pd.DataFrame(ef_df_rows)
    custom_palette = sns.color_palette()
    plt.rcParams["axes.grid"] = True
    print(ef_df)
    g = sns.FacetGrid(ef_df, col="pocket_id", hue="score", col_wrap=2, height=4)

    # Map the data to the grid as a line plot
    g.map(sns.lineplot, "frac", "ef")
    g.add_legend()
    g.set_axis_labels("Fraction", "Enrichment Factor (EF)")
    g.set_titles("{col_name}")
    plt.xscale("log")

    plt.tight_layout()
    plt.show()


def plot_all():
    big_df = []
    models = [
        # "rdock",
        # "dock_42",
        # "native_42",
        "rnamigos",
        # "rdocknat",
        # "combined",
        # "native_nonpocket",
        # "native_rognanpocket",
        "rnamigos_nonpocket",
        "rnamigos_rognanpocket",
        # "native_tune_bestrobin_val",
        # "native_tune_bestrobin_test",
        "rnamigos_nativetune_val",
        "rnamigos_nativetune_test",
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


def plot_distributions(score_to_use="native_validation", in_csv="outputs/robin/big_df_raw.csv"):
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
            palette={1: colors[i], 0: "lightgrey"},
            fill=True,
            alpha=0.9,
            linewidth=0,
            common_norm=False,
            ax=axes[i],
            clip=(0, 1.0),
        )

        t_stat, p_value = stats.ttest_ind(
            merged_pocket.loc[merged_pocket["is_active"] == 1][score_to_use],
            merged_pocket.loc[merged_pocket["is_active"] == 0][score_to_use],
        )

        # Print the results
        print(f"T-statistic: {t_stat}")
        print(f"P-value: {p_value}")
        axes[i].set_xlim(0.5, 1.1)
        axes[i].set_title(f"{pocket_id} (p={p_value:.2e})")

    plt.tight_layout()
    plt.show()


def compute_hopkins():
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from sklearn.neighbors import NearestNeighbors

    def hopkins_statistic(X, sample_size=0.1):
        n, d = X.shape

        # Determine the number of samples
        if isinstance(sample_size, float):
            n_samples = int(sample_size * n)
        else:
            n_samples = sample_size

        # Randomly sample `n_samples` points from X
        indices = np.random.choice(n, n_samples, replace=False)
        X_sample = X[indices]

        # Generate uniform samples within the feature space bounds of X
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
        # uniform_samples = np.random.uniform(X_min, X_max, (n_samples, d))
        uniform_samples = np.random.randint(X_min, X_max + 1, size=(n_samples, d))

        # Initialize nearest neighbor model
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)

        # Distance from each point in the real sample to its nearest neighbor in X
        u_distances, _ = nn.kneighbors(X_sample, return_distance=True)
        u_distances = u_distances[:, 1]

        # Distance from each uniform sample point to its nearest neighbor in X
        w_distances, _ = nn.kneighbors(uniform_samples, return_distance=True)
        w_distances = w_distances[:, 0]

        # Compute the Hopkins statistic
        hopkins_stat = np.sum(w_distances) / (np.sum(w_distances) + np.sum(u_distances))
        return f"{hopkins_stat:.3f}"

    def retain_top_x_percent(df, column, x=10):
        subset_df = df[['pocket_id', 'smiles', 'is_active', column]]
        all_tops = []
        for p in subset_df['pocket_id'].unique():
            df = subset_df[subset_df['pocket_id'] == p]
            n = int(len(df) * (x / 100))
            top_x_percent_df = df.nlargest(n, column)
            all_tops.append(top_x_percent_df)
        return all_tops

    def get_active_fps(df):
        active = df.loc[df['is_active'] == 1]
        smiles_list = active["smiles"].unique().tolist()
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = np.asarray([MACCSkeys.GenMACCSKeys(m) for m in mols])
        return fps

    def print_hopkins(df, column):
        top = retain_top_x_percent(df.copy(), column)
        fps = [get_active_fps(top_pocket) for top_pocket in top]
        hopkins_stat = [hopkins_statistic(fps, sample_size=1.0) for fps in fps]
        print(f"Hopkins Statistic {column} :", hopkins_stat)

    in_csv = "outputs/robin/big_df_raw.csv"
    in_df = pd.read_csv(in_csv)
    # print(in_df.columns)
    print("2GDI_Y_TPP_100,  5BTP_A_AMZ_106, 2QWY_B_SAM_300, 3FU2_A_PRF_101")
    print_hopkins(in_df, "dock_42")
    print_hopkins(in_df, "native_42")
    print_hopkins(in_df, "rnamigos_42")
    print_hopkins(in_df, "rdock")


if __name__ == "__main__":
    SWAP = 0
    RES_DIR = "outputs/robin/" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"

    # ef_lines()
    # plot_all()
    # PLOT PERTURBED VERSIONS
    # plot_perturbed(model="rnamigos_42", group=True)

    # score_to_use = "rdock"
    # score_to_use = "dock_42"
    # score_to_use = "native_42"
    # score_to_use = "rnamigos_42" if SWAP == 0 else "rnamigos"
    score_to_use = "combined_42" if SWAP == 0 else "combined"
    # score_to_use = "rnamigos_nativetune_val"
    # score_to_use = "rnamigos_rognanpocket"
    for swap in range(0, 4):
        # score_to_use = "dock_42" if swap == 0 else "dock"
        # score_to_use = "native_42"
        score_to_use = "rnamigos_42" if swap == 0 else "rnamigos"
        score_to_use = "rdock"
        print(score_to_use)
        print(f"SWAP : {swap}")
        RES_DIR = "outputs/robin/" if swap == 0 else f"outputs/robin_swap_{swap}"
        merged = pd.read_csv(f"{RES_DIR}/big_df_raw.csv")
        auroc = raw_df_to_mean_auroc(merged, score=score_to_use)
        print(auroc)
        # plot_distributions(score_to_use=score_to_use, in_csv=f"{RES_DIR}/big_df_raw.csv")
