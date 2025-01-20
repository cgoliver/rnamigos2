import os
import sys
from pathlib import Path

import torch
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem import MACCSkeys

from rnaglib.drawing import rna_draw
from rnaglib.utils import load_json

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts_fig.plot_utils import PALETTE_DICT, group_df
from rnamigos.utils.virtual_screen import get_auroc

import matplotlib as mpl

# Set font to Arial or Helvetica, which are commonly used in Nature journals
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]  # Use Arial or fallback options
mpl.rcParams["mathtext.fontset"] = "stixsans"  # Sans-serif font for math


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


def enrichment_factor(scores, is_active, frac=0.01):
    """
    Redefine the VS one to return thresholds
    """
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=True)]
    scores_sorted = [s for s, _ in sorted(zip(scores, is_active), reverse=True)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    ef = (n_actives_screened / n_screened) / (n_actives / len(scores))
    return ef, scores_sorted[n_screened]


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

pocket_to_id = {p: l for p, l in zip(pocket_names, ligand_names)}


def get_dfs_docking(ligand_name):
    out_dir = "outputs/robin"

    # Get relevant mapping smiles : normalized score
    docking_df = pd.read_csv(os.path.join(out_dir, "robin_docking_consolidated_v2.csv"))
    # docking_df = pd.read_csv(os.path.join(out_dir, "robin_targets_docking_consolidated.csv"))
    docking_df = docking_df[docking_df["TARGET"] == ligand_name]
    scores = -docking_df[["INTER"]].values.squeeze()

    # DEAL WITH NANS, ACTUALLY WHEN SORTING NANS, THEY GO THE END
    # count = np.count_nonzero(np.isnan(scores))
    # scores = np.sort(scores)
    # cropped_scores = np.concatenate((scores[:10], scores[-10:]))
    # print(cropped_scores)

    scores[scores < 0] = 0
    scores = np.nan_to_num(scores, nan=0)

    mi = np.nanmin(scores)
    ma = np.nanmax(scores)
    print(ma, mi)
    # normalized_scores = (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores))
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())

    # normalized_scores = scores
    mapping = {}
    for smiles, score in zip(docking_df[["SMILE"]].values, normalized_scores):
        mapping[smiles[0]] = score
    mapping = defaultdict(int, mapping)

    all_smiles = []
    # Use this mapping to create our actives/inactives distribution dataframe
    active_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "actives.txt")
    # out_path = os.path.join(out_dir, f"{pocket_name}_actives.txt")
    smiles_list = [s.lstrip().rstrip() for s in list(open(active_ligands_path).readlines())]
    all_smiles.extend(smiles_list)
    actives_df = pd.DataFrame({"rDock": [mapping[sm] for sm in smiles_list]})
    actives_df["is_active"] = 1

    scores = actives_df[["rDock"]].values.squeeze()
    ma = np.nanmax(scores)
    mi = np.nanmin(scores)
    count = np.count_nonzero(np.isnan(scores))
    print(
        f"actives max/min : {ma} {mi}, nancount : {count} "
        f"scores over 200 : {np.sum(scores > 200)} length : {len(scores)} "
    )

    inactives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "decoys.txt")
    # out_path = os.path.join(out_dir, f"{pocket_name}_inactives.txt")
    smiles_list = [s.lstrip().rstrip() for s in list(open(inactives_ligands_path).readlines())]
    all_smiles.extend(smiles_list)
    inactives_df = pd.DataFrame({"rDock": [mapping[sm] for sm in smiles_list]})
    inactives_df["is_active"] = 0

    scores = inactives_df[["rDock"]].values.squeeze()
    ma = np.nanmax(scores)
    mi = np.nanmin(scores)
    count = np.count_nonzero(np.isnan(scores))
    print(
        f"inactives max/min : {ma} {mi}, nancount : {count} "
        f"scores over 200 : {np.sum(scores > 200)} length : {len(scores)} "
    )

    merged = pd.concat([actives_df, inactives_df]).reset_index()
    merged["smiles"] = all_smiles
    return merged


def get_dfs_migos(df_path, pocket_name, swap=False, normalize=False, top_pct=None):
    df = pd.read_csv(df_path)
    df = df.loc[df["pocket_id"] == pocket_name]
    if top_pct:
        thresh = df["raw_score"].quantile(top_pct)
        df = df.loc[df["raw_score"] > thresh]
    return df


def get_dfs_migos_(pocket_name, swap=False, swap_on="merged", normalize=False):
    # names = ["smiles", "dock", "is_native", "native_fp", "merged"]
    out_dir = "outputs/robin_docknative_rev"
    df_path = "outputs/robin/rnamigos_rognanpocket_raw.csv"
    df_all = pd.read_csv(df_path)
    df = df_all.loc[df_all["pocket_id"] == pocket_name]
    # merged = pd.concat([actives_df, inactives_df])
    if swap:
        other_pocket = random.choice(list(set(pocket_names) - set([pocket_name])))
        # model_outs_other = pd.read_csv(os.path.join(out_dir, f"{other_pocket}_results.txt"))
        model_outs_other = df_all.loc[df_all["pocket_id"] == other_pocket]
        both = df.merge(model_outs_other, on="smiles", suffixes=("_orig", "_other"))
        if swap_on == "merged":
            both["merged"] = both["merged_other"]
            both["is_active"] = both["is_active_orig"]
        if swap_on == "split":
            both["is_active"] = both["is_active_other"]
            both["merged"] = both["merged_orig"]
        return both

    if normalize:
        df["is_native"] = df["is_native"] - df["is_native"].min() / (df["is_native"].max() - df["is_native"].min())
        df["dock"] = df["dock"] - df["dock"].min() / (df["dock"].max() - df["dock"].min())
    return df


def make_violins(df):
    df = df.drop(columns=["dock_42", "native_42", "rnamigos_42", "rdocknat_42", "combined_42"], axis=1)
    df = df.melt(
        id_vars=["pocket_id", "smiles", "is_active"],
        value_vars=["maxmerge_42", "combined_42_max", "rank_rdock"],
        var_name="method",
        value_name="score",
    )
    df = df.loc[df["is_active"] == 1]
    print(df)
    print(df.columns)

    # Create a FacetGrid with one subplot for each pocket
    g = sns.FacetGrid(df, col="pocket_id", height=5, aspect=1.5)

    # Map the histplot to the grid
    g.map(sns.histplot, "score", data=df, hue="method", multiple="dodge", legend=True)

    plt.legend()
    plt.show()

    pass


def make_fig_rocs(data, scores_to_use):
    print(big_df)
    pocket_ids = data["pocket_id"].unique()  # Get unique pocket_ids

    # Create subplots, one for each pocket_id
    fig, axes = plt.subplots(ncols=len(pocket_ids), figsize=(6 * len(pocket_ids), 6))

    for ax, pocket_id in zip(axes, pocket_ids):
        # Filter data for the current pocket_id
        pocket_data = data[data["pocket_id"] == pocket_id]

        for score_col in scores_to_use:
            # Compute ROC curve for each score column
            fpr, tpr, _ = roc_curve(pocket_data["is_active"], pocket_data[score_col])
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            ax.plot(fpr, tpr, label=f"{score_col} (AUC = {roc_auc:.2f})")

        # Customize the plot
        ax.plot([0, 1], [0, 1], "k--")  # Diagonal line (random classifier)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{pocket_id}")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("figs/robin_rocs.pdf", format="pdf")
    plt.show()

    pass


def make_fig(big_df, score_to_use, swap=False, normalize_migos=True, prefix="robin_fig"):
    # fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    # plt.gca().set_yscale('custom')

    all_efs = list()
    all_aurocs = list()

    rows = []

    for i, (pocket_name, ligand_name) in enumerate(zip(pocket_names, ligand_names)):
        # ax = axs[i]
        fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set_xscale('custom')
        df = big_df.loc[big_df["pocket_id"] == pocket_name]

        # GET EFS
        # fracs = [0.01, 0.05]
        fracs = [0.01, 0.02, 0.05, 0.1]
        linecolors = ["black", "grey", "lightgrey"]

        colors = sns.color_palette("Paired", 4)
        colors = sns.color_palette(["#149950", "#00c358", "#037938", "#149921"])

        colors = sns.color_palette(["#33ccff", "#00cccc", "#3366ff", "#9999ff"])

        default_frac = 0.01
        offset = 0
        legend_y = 0.9
        legend_x = 0.7
        curve_fill = True
        # sns.kdeplot(data=merged, x=score_to_use, fill=False, hue='split', alpha=.9,
        #             palette={'actives': colors[i], 'inactives': 'lightgrey'}, common_norm=False, ax=ax, log_scale=False)
        df = df.sort_values(by="is_active")
        g = sns.kdeplot(
            data=df.loc[df["is_active"] == 1],
            x=score_to_use,
            # hue="is_active",
            ax=ax,
            # palette={'actives': colors[i], 'inactives': 'lightgrey'},
            fill=False,
            # alpha=0.9,
            linewidth=0,
            bw_adjust=0.5,
            common_norm=False,
        )
        xx, yy = g.lines[0].get_data()
        # decoy_xx, decoy_yy = g.lines[1].get_data()
        # ax.fill_between(xx, 0, yy, color=colors[i], alpha=0.1)

        # ax.plot(xx, yy, color=colors[i], alpha=1, linewidth=1.5)
        # thresh = df[score_to_use].quantile(0.8)
        sns.histplot(
            data=df.loc[df["is_active"] == 1], stat="density", x=score_to_use, ax=ax, alpha=0.2, color=colors[i]
        )

        ax.set_xlim([0, 1])

        fpr, tpr, thresholds = metrics.roc_curve(df["is_active"], df[score_to_use])
        auroc = metrics.auc(fpr, tpr)
        t_stat, p_value = stats.mannwhitneyu(
            df[df["is_active"] == 1][score_to_use], df[df["is_active"] == 0][score_to_use]
        )
        print(p_value)

        thresh_select = df[score_to_use].quantile(0.98)
        selected_fps = smiles_to_fp(df.loc[df[score_to_use] > thresh_select]["smiles"])
        diversity = internal_diversity(selected_fps)

        efs = []
        for _, frac in enumerate(fracs):
            ef, thresh = enrichment_factor(scores=df[score_to_use], is_active=df["is_active"], frac=frac)
            efs.append(ef)

            """
            if curve_fill:
                xy_tail = [(x, y) for x, y in zip(xx, yy) if x > thresh]
                x_tail = [x for x, y in xy_tail]
                y_tail = [y for x, y in xy_tail]

                ax.fill_between(
                    x_tail,
                    0,
                    y_tail,
                    color=colors[i],
                    alpha=0.15,
                )

            else:
                ax.fill_between(
                    x_tail,
                    0,
                    1,
                    color=colors[i],
                    alpha=0.07,
                )

            arrow_to_y = legend_y - offset

            ax.text(
                legend_x,
                legend_y - offset,
                f"EF@{int(frac * 100)}%: {ef:.2f}",
                fontsize=10,
                transform=ax.transAxes,
            )

            offset += 0.05
            """

        for frac, ef in zip(fracs, efs):
            rows.append(
                {
                    "pocket": pocket_to_id[pocket_name],
                    f"EF@{frac * 100:.0f}%": ef,
                    "thresh": f"{frac * 100:.0f}",
                    "score": score_to_use,
                    "AuROC": auroc,
                    "p_value": p_value,
                    "Diversity": diversity,
                }
            )

        # ax.fill_between(decoy_xx, 0, decoy_yy, color="lightgrey", alpha=0.8)
        # ax.plot(decoy_xx, decoy_yy, color='white', alpha=0.9)
        ef, thresh = enrichment_factor(scores=df[score_to_use], is_active=df["is_active"], frac=default_frac)
        all_efs.append(ef)
        print(f"EF@{frac} : ", pocket_name, ef)
        # GET AUROC

        print(p_value)
        all_aurocs.append(auroc)

        ax.set_title(f"{pocket_name} AuROC: {auroc:.2f} \n p={p_value:.2e}")
        # g.legend().remove()
        xticks = ax.get_xticks()
        plt.yticks([])
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(np.linspace(80, 100, len(xticks)))
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"figs/panel_3_{prefix}_{pocket_to_id[pocket_name]}.pdf", format="pdf")
        # plt.show()

        # print('AuROC : ', pocket_name, auroc)
        # print()

        # auroc = get_auroc(scores, actives)
        # print(auroc)
    #     #ef = f"EF@1\% {list(ef_df.loc[ef_df['pocket_id'] == name]['score'])[0]:.3f}"
    #     axs[i][0].text(0, 0, f"{name} EF: {ef:.3} MAR: {mar:.3}")
    #     axs[i][0].axis("off")
    #     axs[i][1].axis("off")
    #     sns.despine()
    #
    print(np.mean(all_efs))
    print(np.mean(all_aurocs))
    # names = ['smiles', 'dock', 'is_native', 'native_fp', 'merged']
    # plt.show()
    return pd.DataFrame(rows)


def make_table(df):

    rows_with_nan = df[df.isna().any(axis=1)]
    print(rows_with_nan)
    print(rows_with_nan["score"].unique())

    df = df.replace("native_42", "Compat")
    df = df.replace("dock_42", "Aff")
    df = df.replace("rdock", "rDock")
    df = df.replace("maxmerge_42", "RNAmigos2")
    df = df.replace("combined_42", "RNAmigos++")
    df = df.replace("combined_42_max", "RNAmigos++max")
    df = df.replace("combined_42_mean", "RNAmigos++mean")
    print(df)

    table = pd.pivot_table(
        df,
        values=["AuROC", "EF@1%", "EF@2%", "EF@5%", "Diversity"],
        index=["pocket", "score"],
    )
    print(table.to_latex(float_format="{:.2f}".format, escape=True))

    def make_max_bold(group):
        # For each column, find the maximum and apply \textbf
        for col in group.columns:
            max_val = group[col].max()
            group[col] = group[col].apply(lambda x: f"\\textbf{{{x:.2f}}}" if x == max_val else f"{x:.2f}")
        return group

    # Apply the bold formatting to the table
    table_bold = table.groupby(["pocket"]).apply(make_max_bold)

    # Print the LaTeX output
    print(table_bold.to_latex(float_format="{:.2f}".format, escape=False))

    pass


def make_table_auroc(df):
    print(df)
    df = df.rename(columns={"auroc": "AuROC"})
    df = df.replace("rnamigos_42", "RNAmigos2")
    df = df.replace("native_42", "Compat")
    df = df.replace("dock_42", "Aff")
    df = df.replace("rdock", "rDock")
    df = df.replace("maxmerge_42", "RNAmigos2")
    df = df.replace("combined_42", "RNAmigos++")
    df = df.replace("combined_42_max", "RNAmigos++max")
    df = df.replace("combined_42_mean", "RNAmigos++mean")
    table = pd.pivot_table(
        df,
        values=["AuROC", "diversity"],
        index=["pocket", "score"],
    )
    print(table.to_latex(float_format="{:.2f}".format))

    filtered_df = df[df["score"] == "rnamigos_42"][["pocket", "ef thresh", "auroc", "diversity"]]

    # Pivot the table
    pivoted_df = filtered_df.pivot_table(
        index=["pocket", "auroc"], columns="ef thresh", values="diversity", aggfunc="first"
    ).reset_index()

    # Rename columns for clarity
    pivoted_df.columns.name = None
    pivoted_df.rename(columns={1: "Diversity (ef 1)", 2: "Diversity (ef 2)", 5: "Diversity (ef 5)"}, inplace=True)

    # Reorder columns
    final_df = pivoted_df[["pocket", "auroc", "Diversity (ef 1)", "Diversity (ef 2)", "Diversity (ef 5)"]]

    # Convert to LaTeX
    latex_table = final_df.to_latex(index=False, float_format="%.6f")

    pass


if __name__ == "__main__":

    # add merging score
    big_df = pd.read_csv("outputs/robin/big_df_raw.csv")

    big_df["rank_native"] = big_df.groupby("pocket_id")["native_42"].rank(ascending=True, pct=True)
    big_df["rank_dock"] = big_df.groupby("pocket_id")["dock_42"].rank(ascending=True, pct=True)
    big_df["rank_rdock"] = big_df.groupby("pocket_id")["rdock"].rank(ascending=True, pct=True)

    def maxmin(column):
        return (column - column.min()) / (column.max() - column.min())

    big_df["scaled_native"] = big_df.groupby("pocket_id")["native_42"].transform(maxmin)
    big_df["scaled_dock"] = big_df.groupby("pocket_id")["dock_42"].transform(maxmin)

    big_df["maxmerge_42"] = big_df[["rank_native", "rank_dock"]].max(axis=1)
    big_df["maxmerge_42"] = big_df.groupby("pocket_id")["maxmerge_42"].rank(ascending=True, pct=True)
    big_df["rank_rnamigos"] = big_df.groupby("pocket_id")["maxmerge_42"].rank(ascending=True, pct=True)
    big_df["combined_42_max"] = big_df[["rank_rnamigos", "rank_rdock"]].max(axis=1)
    big_df["combined_42_max"] = big_df.groupby("pocket_id")["combined_42_max"].rank(ascending=True, pct=True)
    big_df["combined_42_mean"] = (big_df["maxmerge_42"] + big_df["rdock"]) / 2

    # big_df["maxmerge_42"] = big_df[["rank_native", "rank_dock"]].max(axis=1)
    # big_df["maxmerge_42"] = big_df.groupby("pocket_id")["maxmerge_42"].rank(ascending=True, pct=True)

    """
    for pocket in pocket_names:
        df = big_df.loc[big_df["pocket_id"] == pocket]
        print(df["maxmerge_42"].min(), df["maxmerge_42"].max())
        sns.kdeplot(
            data=df,
            x="maxmerge_42",
            hue="is_active",
            # palette={'actives': colors[i], 'inactives': 'lightgrey'},
            fill=True,
            alpha=0.9,
            common_norm=False,
        )
        plt.show()
    big_df.to_csv("big_df_merge.csv")
    """
    scores = [
        "rdock",
        "native_42",
        "dock_42",
        "combined_42",
        "maxmerge_42",
        "combined_42_max",
        "combined_42_mean",
        "rank_rdock",
    ]

    scores_roc = [
        "rdock",
        "maxmerge_42",
        "combined_42_max",
    ]

    make_violins(big_df)
    make_fig_rocs(big_df, scores_roc)
    dfs = []

    for score_to_use in scores:
        dfs.append(make_fig(big_df, score_to_use, prefix=f"resubmit_{score_to_use}", normalize_migos=True))
    make_table(pd.concat(dfs))
    # make_table_auroc(pd.concat(dfs))
