"""
Optimal transport + diversity analysis of ROBIN and PDB ligands.
"""

import pickle
import itertools
from joblib import Parallel, delayed

import numpy as np
import ot
from scipy.spatial.distance import squareform
from scipy.spatial.distance import jaccard
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

pocket_names = [
    "2GDI_Y_TPP_100",
    "5BTP_A_AMZ_106",
    # "2QWY_A_SAM_100",
    "2QWY_B_SAM_300",
    # "3FU2_C_PRF_101",
    "3FU2_A_PRF_101",
]
ligand_names = [
    "TPP",
    "ZTP",
    "SAM_ll",
    "PreQ1",
]

robins = [
    "2GDI_Y_TPP_100",
    "5BTP_A_AMZ_106",
    # "2QWY_A_SAM_100",
    "2QWY_B_SAM_300",
    # "3FU2_C_PRF_101",
    "3FU2_A_PRF_101",
]
robin_ids = [
    "TPP",
    "ZTP",
    "SAM_ll",
    "PreQ1",
]

pocket_to_id = {p: l for p, l in zip(pocket_names, ligand_names)}


def compute_tanimoto(fp_1, fp_2):
    return jaccard(fp_1, fp_2)


def get_ot(fps1, fps2):
    N = len(fps1)
    M = len(fps2)
    print(N, M)

    # Step 3: Define uniform marginal distributions
    a = np.ones(N) / N  # Source distribution (uniform)
    b = np.ones(M) / M  # Target distribution (uniform)
    # Step 2: Create pairs of fingerprints
    M = np.zeros((len(fps1), len(fps2)))
    for i, fp_1 in enumerate(fps1):
        for j, fp_2 in enumerate(fps2):
            M[i][j] = jaccard(fp_1, fp_2)

    otp = ot.emd(a, b, M)
    total_cost = np.sum(otp * M)
    return total_cost


if __name__ == "__main__":

    big_df = pd.read_csv("outputs/robin/big_df_raw.csv")

    big_df["rank_rnamigos"] = big_df.groupby("pocket_id")["rnamigos_42"].rank(ascending=False, pct=True)
    big_df["rank_rdock"] = big_df.groupby("pocket_id")["rdock"].rank(ascending=True, pct=True)
    print(big_df)

    smiles_to_ind = pickle.load(open("smiles_to_ind.p", "rb"))
    ind_to_smiles = {i: sm for i, sm in smiles_to_ind.items()}
    fps = np.load("fps.npy")

    colors = sns.color_palette(["#33ccff", "#00cccc", "#3366ff", "#9999ff"])

    active_smiles_all = {}
    to_plot = []

    thresh = 0.97
    # for plot_decoys in [True, False]:  # cheap hack for putting active points on top

    spreads = {}
    rows = []
    costs = []
    for robin_1 in robins:
        row = []
        for robin_2 in robins:
            print(robin_1, robin_2)
            actives_1 = big_df.loc[(big_df["pocket_id"] == robin_1) & (big_df["is_active"] == 1)]
            actives_2 = big_df.loc[(big_df["pocket_id"] == robin_2) & (big_df["is_active"] == 1)]
            fps_active_1 = [fps[smiles_to_ind[s]] for s in actives_1["smiles"] if s in smiles_to_ind]
            fps_active_2 = [fps[smiles_to_ind[s]] for s in actives_2["smiles"] if s in smiles_to_ind]

            total_cost = get_ot(fps_active_1, fps_active_2)
            row.append(total_cost)

        costs.append(row)

    costs = np.array(costs)
    np.fill_diagonal(costs, np.nan)

    sns.heatmap(
        costs,
        annot=True,
        cmap="coolwarm",
        xticklabels=robins,
        yticklabels=robins,
        vmin=0.24,
        vmax=0.3,
        square=True,
    )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figs/EMD_robin_actives.pdf", format="pdf")
    plt.show()
