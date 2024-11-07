import os
import sys
from pathlib import Path

from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn import metrics

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


def make_fig(model_output, swap=False, normalize_migos=True, prefix="robin_fig"):
    # fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    # plt.gca().set_yscale('custom')

    all_efs = list()
    all_aurocs = list()

    rows = []

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    for i, (pocket_name, ligand_name) in enumerate(zip(pocket_names, ligand_names)):
        # ax = axs[i]
        fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set_xscale('custom')

        # FOR DOCKING
        # merged_rdock = get_dfs_docking(ligand_name=ligand_name)
        # score_to_use = 'docking_score'

        # FOR MIGOS
        df = get_dfs_migos(model_output, pocket_name=pocket_name, swap=swap, normalize=normalize_migos)
        print(df)
        # merged_migos["dock_nat"] = (normalize(merged_migos["is_active"]) + normalize(merged_migos["dock"])) / 2
        # print(merged_rdock)

        # merged = pd.merge(merged_migos, merged_rdock, on=["smiles", "is_active"], how="outer")

        # merged = merged.fillna(0)
        # merged["RNAmigos2++"] = (merged["dock_nat"] + merged["rDock"]) / 2

        # scores = merged["score"]

        # GET EFS
        # fracs = [0.01, 0.05]
        fracs = [0.01, 0.02, 0.05]
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
            data=df,
            x="raw_score",
            hue="is_active",
            ax=ax,
            # palette={'actives': colors[i], 'inactives': 'lightgrey'},
            # fill=True,
            # alpha=0.9,
            linewidth=0,
            common_norm=False,
        )
        xx, yy = g.lines[0].get_data()
        decoy_xx, decoy_yy = g.lines[1].get_data()
        # ax.fill_between(xx, 0, yy, color=colors[i], alpha=0.1)
        ax.plot(xx, yy, color=colors[i], alpha=1, linewidth=1.5)
        thresh = df["raw_score"].quantile(0.8)
        ax.set_xlim([thresh, 1])

        fpr, tpr, thresholds = metrics.roc_curve(df["is_active"], df["raw_score"])
        auroc = metrics.auc(fpr, tpr)
        t_stat, p_value = stats.ttest_ind(df[df["is_active"] == 1]["raw_score"], df[df["is_active"] == 0]["raw_score"])
        print(p_value)

        for _, frac in enumerate(fracs):
            ef, thresh = enrichment_factor(scores=df["raw_score"], is_active=df["is_active"], frac=frac)
            rows.append(
                {
                    "pocket": pocket_to_id[pocket_name],
                    "ef": ef,
                    "thresh": f"{frac * 100:.0f}",
                    "score": Path(model_output).stem,
                    "auroc": auroc,
                    "p_value": p_value,
                }
            )

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

        ax.fill_between(decoy_xx, 0, decoy_yy, color="lightgrey", alpha=0.8)
        # ax.plot(decoy_xx, decoy_yy, color='white', alpha=0.9)
        ef, thresh = enrichment_factor(scores=df["raw_score"], is_active=df["is_active"], frac=default_frac)
        all_efs.append(ef)
        print(f"EF@{frac} : ", pocket_name, ef)
        # GET AUROC

        print(p_value)
        all_aurocs.append(auroc)

        ax.set_title(f"{pocket_name} AuROC: {auroc:.2f} \n p={p_value:.2e}")
        g.legend().remove()
        xticks = ax.get_xticks()
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
    df = df.rename(columns={"ef": "Enrichment Factor", "thresh": "cutoff (%)"})
    df = df.replace("dock_nat", "RNAmigos2")
    df = df.replace("is_native", "Compat")
    df = df.replace("dock", "Aff")
    table = pd.pivot_table(
        df,
        values=["Enrichment Factor"],
        columns=["cutoff (%)"],
        index=["pocket", "score"],
    )
    print(table.to_latex(float_format="{:.2f}".format))

    pass


def make_table_auroc(df):
    df = df.rename(columns={"auroc": "AuROC"})
    df = df.replace("dock_nat", "RNAmigos2")
    df = df.replace("is_native", "Compat")
    df = df.replace("dock", "Aff")
    table = pd.pivot_table(
        df,
        values=["AuROC"],
        index=["pocket", "score"],
    )
    print(table.to_latex(float_format="{:.2f}".format))
    pass


if __name__ == "__main__":

    dfs = []
    # for score in ["dock_nat", "is_native", "dock", "rDock", "RNAmigos2++"]:
    #    dfs.append(make_fig(score, prefix=f"repro_{score}", normalize_migos=True))
    for model in ["rnamigos_42", "rdock"]:
        df_path = Path("outputs/robin") / f"{model}_raw.csv"
        dfs.append(make_fig(df_path, prefix=f"resubmit_{model}", normalize_migos=True))
    make_table(pd.concat(dfs))
    make_table_auroc(pd.concat(dfs))
