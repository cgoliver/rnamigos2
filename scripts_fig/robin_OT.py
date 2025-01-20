"""
Optimal transport + diversity analysis of ROBIN and PDB ligands.
"""

import pickle
import random
import itertools
from joblib import Parallel, delayed

import torch
import numpy as np
import ot
from scipy.spatial.distance import squareform
from scipy.spatial.distance import jaccard
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import QED
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


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg="max", device="cpu", p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i : i + y_gen.shape[1]] = np.maximum(agg_tanimoto[i : i + y_gen.shape[1]], jac.max(0))
            elif agg == "mean":
                agg_tanimoto[i : i + y_gen.shape[1]] += jac.sum(0)
                total[i : i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)


def internal_diversity(fps, n_jobs=1, device="cpu", fp_type="morgan", p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    return 1 - (average_agg_tanimoto(fps, fps, agg="mean", device=device, p=p)).mean()


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


def smiles_to_mol(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    clean_mols = []
    for mol, sm in zip(mols, smiles_list):
        if mol is None:
            continue
        clean_mols.append(mol)
    return clean_mols


def smiles_to_fp(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    clean_mols = []
    clean_smiles = []
    for mol, sm in zip(mols, smiles_list):
        if mol is None:
            continue
        clean_mols.append(mol)
        clean_smiles.append(sm)

    fps = np.array([MACCSkeys.GenMACCSKeys(m) for m in clean_mols])
    return fps


def all_QED(mols):
    return [QED.qed(mol) for mol in mols]


def spread(fps):
    costs = []
    for fp1 in fps:
        row = []
        for fp2 in fps:
            total_cost = jaccard(fp1, fp2)
            row.append(total_cost)
        costs.append(row)

    return np.mean(costs)


def cost_matrix(fps_1, fps_2, square=False, xticklabels=None, yticklabels=None, save=None):

    if xticklabels is None:
        xticklabels = range(len(fps_1))
    if yticklabels is None:
        yticklabels = range(len(fps_2))

    costs = np.zeros((len(fps_1), len(fps_2)))
    for i in range(len(fps_1)):
        row = []
        for j in range(i, len(fps_2)):
            total_cost = get_ot(fps_1[i], fps_2[j])
            costs[i][j] = total_cost
            costs[j][i] = total_cost

    if square:
        np.fill_diagonal(costs, np.nan)

    sns.heatmap(costs, annot=True, cmap="coolwarm", square=True, xticklabels=xticklabels, yticklabels=yticklabels)

    plt.tight_layout()
    if not save is None:
        plt.savefig(save, format="pdf")
    plt.show()
    pass


if __name__ == "__main__":

    # PDB & CHEMBL

    chembl_df = pd.read_csv("outputs/pockets/big_df_grouped_42_raw.csv")
    chembl_smiles = chembl_df.loc[chembl_df["decoys"] == "chembl"]["smiles"].unique()
    chembl_QED = all_QED(smiles_to_mol(chembl_smiles))
    print(f"CHEMBL QED: {np.mean(chembl_QED)}")
    chembl_fps = smiles_to_fp(chembl_smiles)[:200]

    pdb_mols = smiles_to_mol(pd.read_csv("data/csvs/fp_data.csv")["LIGAND_SMILES"].unique())
    print(f"PDB QED: {np.mean(all_QED(pdb_mols))}")
    thresh = 0.95

    pdb_fps = smiles_to_fp(pd.read_csv("data/csvs/fp_data.csv")["LIGAND_SMILES"].unique())
    pdb_div = internal_diversity(pdb_fps)
    print(f"PDB diversity: {pdb_div}")

    big_df = pd.read_csv("outputs/robin/big_df_raw.csv")

    big_df["rank_rnamigos"] = big_df.groupby("pocket_id")["rnamigos_42"].rank(ascending=True, pct=True)
    big_df["rank_native"] = big_df.groupby("pocket_id")["native_42"].rank(ascending=True, pct=True)
    big_df["rank_dock"] = big_df.groupby("pocket_id")["dock_42"].rank(ascending=True, pct=True)
    big_df["rank_rdock"] = big_df.groupby("pocket_id")["rdock"].rank(ascending=True, pct=True)

    big_df["maxmerge_42"] = big_df[["rank_native", "rank_dock"]].max(axis=1)
    big_df["maxmerge_42"] = big_df.groupby("pocket_id")["maxmerge_42"].rank(ascending=True, pct=True)

    for robin in robins:
        """
        actives = big_df.loc[(big_df["pocket_id"] == robin) & (big_df["is_active"] == 1)]["smiles"]
        inactives = big_df.loc[(big_df["pocket_id"] == robin) & (big_df["is_active"] == 0)]["smiles"]
        actives_mols = smiles_to_mol(actives)
        inactives_mols = smiles_to_mol(inactives)
        print(f"Active {robin} {np.mean(all_QED(actives_mols))}")
        print(f"Inactive {robin} {np.mean(all_QED(inactives_mols))}")
        """

        migos_select = big_df.loc[(big_df["pocket_id"] == robin) & (big_df["rank_rnamigos"] > thresh)]["smiles"]
        rdock_select = big_df.loc[(big_df["pocket_id"] == robin) & (big_df["rank_rdock"] > thresh)]["smiles"]

        migos_mols = smiles_to_mol(migos_select)
        rdock_mols = smiles_to_mol(rdock_select)

        # print(f"Migos actives {robin} {np.mean(all_QED(migos_mols))}")
        # print(f"rDock actives {robin} {np.mean(all_QED(rdock_mols))}")
    # ROBIN

    robin_actives_GT = [
        smiles_to_fp(big_df.loc[(big_df["pocket_id"] == robin) & (big_df["is_active"] == 1)]["smiles"])
        for robin in robins
    ]

    print("HELLO")
    for r, a in zip(robin_actives_GT, robins):
        print(a, len(r))

    rnamigos_actives = [
        smiles_to_fp(big_df.loc[(big_df["pocket_id"] == robin) & (big_df["rank_rnamigos"] > thresh)]["smiles"])
        for robin in robins
    ]
    merge_actives = [
        smiles_to_fp(
            big_df.loc[(big_df["pocket_id"] == robin) & (big_df["maxmerge_42"] > thresh) & (big_df["is_active"] == 1)][
                "smiles"
            ]
        )
        for robin in robins
    ]

    native_actives = [
        smiles_to_fp(big_df.loc[(big_df["pocket_id"] == robin) & (big_df["rank_native"] > thresh)]["smiles"])
        for robin in robins
    ]
    dock_actives = [
        smiles_to_fp(big_df.loc[(big_df["pocket_id"] == robin) & (big_df["rank_dock"] > thresh)]["smiles"])
        for robin in robins
    ]
    rdock_actives = [
        smiles_to_fp(
            big_df.loc[(big_df["pocket_id"] == robin) & (big_df["rank_rdock"] > thresh) & (big_df["is_active"] == 1)][
                "smiles"
            ]
        )
        for robin in robins
    ]

    # SPREADS
    """
    for i, rob in enumerate(robins):
        print(f"diversity rnamigos {rob}: {internal_diversity(rnamigos_actives[i]):.2f}")
        print(f"diversity merge {rob}: {internal_diversity(merge_actives[i]):.2f}")
        print(f"diversity dock {rob}: {internal_diversity(dock_actives[i]):.2f}")
        print(f"diversity native {rob}: {internal_diversity(native_actives[i]):.2f}")
        print(f"diversity rDock {rob}: {internal_diversity(rdock_actives[i]):.2f}")

    """
    # OTs

    # cost_matrix(robin_actives_GT, robin_actives_GT, square=True, xticklabels=robins, yticklabels=robins)
    """
    cost_matrix(rnamigos_actives, [chembl_fps] * len(robins))
    cost_matrix(robin_actives_GT, [chembl_fps] * len(robins))
    cost_matrix(rdock_actives, [chembl_fps] * len(robins))
    """
    cost_matrix(
        native_actives,
        dock_actives,
        xticklabels=robins,
        yticklabels=robins,
        save="figs/native_dock_ot.pdf",
    )
