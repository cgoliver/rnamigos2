"""
Examples of command lines to train a model are available in scripts_run/train.sh
"""

import os
import sys

from pathlib import Path

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import torch
import seaborn as sns

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.utils.virtual_screen import enrichment_factor
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.learning.models import get_model_from_dirpath
from scripts_run.robin_inference import robin_inference

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

POCKET_PATH = "data/json_pockets_expanded"


def one_robin(pocket_id, ligand_name, model=None, use_rna_fm=False, do_mixing=False):
    dgl_pocket_graph, _ = load_rna_graph(
        POCKET_PATH / Path(pocket_id).with_suffix(".json"),
        use_rnafm=use_rna_fm,
    )
    final_df = robin_inference(
        ligand_name=ligand_name,
        dgl_pocket_graph=dgl_pocket_graph,
        model=model,
        use_ligand_cache=True,
        ligand_cache="data/ligands/robin_lig_graphs.p",
        do_mixing=do_mixing,
        debug=False
    )
    final_df["pocket_id"] = pocket_id
    rows = []
    for frac in (0.01, 0.02, 0.05):
        ef = enrichment_factor(final_df["model"],
                               final_df["is_active"],
                               lower_is_better=False,
                               frac=frac,
                               )
        rows.append({"pocket_id": pocket_id, "score": ef, "frac": frac})
    return pd.DataFrame(rows), pd.DataFrame(final_df)


def get_all_preds(model, use_rna_fm):
    robin_dfs = [df for df in Parallel(n_jobs=4)(delayed(one_robin)(pocket_id, ligand_name, model, use_rna_fm)
                                                 for ligand_name, pocket_id in ROBIN_POCKETS.items())]
    robin_efs, robin_raw_dfs = list(map(list, zip(*robin_dfs)))
    robin_ef_df = pd.concat(robin_efs)
    robin_raw_df = pd.concat(robin_raw_dfs)
    return robin_ef_df, robin_raw_df


def get_all_csvs(recompute=False):
    model_dir = "results/trained_models/"
    os.makedirs(RES_DIR, exist_ok=True)
    for model, model_path in MODELS.items():
        out_csv = os.path.join(RES_DIR, f"{model}.csv")
        out_csv_raw = os.path.join(RES_DIR, f"{model}_raw.csv")
        if os.path.exists(out_csv) and not recompute:
            continue
        full_model_path = os.path.join(model_dir, model_path)
        rnafm = model_path.endswith('rnafm')
        model = get_model_from_dirpath(full_model_path)
        df_ef, df_raw = get_all_preds(model, use_rna_fm=rnafm)
        df_ef.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)


def mix(df1, df2, outpath=None):
    def normalize(scores):
        out_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return out_scores

    norm_score1 = normalize(df1["model"])
    norm_score2 = normalize(df2["model"])
    mixed_scores = 0.5 * (norm_score1 + norm_score2)
    out_df = df1.copy()
    out_df['mixed_score'] = mixed_scores
    out_df = out_df.drop(columns=['model'])
    if outpath is not None:
        out_df.to_csv(outpath, index=False)
    return out_df


def mix_all():
    for pair, outname in PAIRS.items():
        path1 = os.path.join(RES_DIR, f"{pair[0]}_raw.csv")
        path2 = os.path.join(RES_DIR, f"{pair[1]}_raw.csv")
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        robin_efs, robin_raw_dfs = [], []
        for pocket_id in ROBIN_POCKETS.values():
            rows = []
            df1_lig = df1[df1["pocket_id"] == pocket_id]
            df2_lig = df2[df2["pocket_id"] == pocket_id]
            mixed_df_lig = mix(df1_lig, df2_lig)
            robin_raw_dfs.append(mixed_df_lig)
            for frac in (0.01, 0.02, 0.05):
                ef = enrichment_factor(mixed_df_lig["mixed_score"],
                                       mixed_df_lig["is_active"],
                                       lower_is_better=False,
                                       frac=frac)
                robin_efs.append({"pocket_id": pocket_id, "score": ef, "frac": frac})
        robin_efs = pd.DataFrame(robin_efs)
        robin_raw_dfs = pd.concat(robin_raw_dfs)
        outpath = os.path.join(RES_DIR, f"{outname}.csv")
        outpath_raw = os.path.join(RES_DIR, f"{outname}_raw.csv")
        robin_efs.to_csv(outpath, index=False)
        robin_raw_dfs.to_csv(outpath_raw, index=False)


def plot_all():
    big_df = []
    models = MODELS
    models = list(MODELS) + list(PAIRS.values())
    for model in models:
        out_csv = os.path.join(RES_DIR, f"{model}.csv")
        df = pd.read_csv(out_csv)
        df['name'] = model
        big_df.append(df)
    big_df = pd.concat(big_df)

    custom_palette = {
        'native': '#1f77b4',  # blue
        'native_rnafm': '#ff7f0e',  # orange (distinct for rnafm)
        'native_pre': '#2ca02c',  # green
        'native_pre_rnafm': '#d62728',  # red (distinct for rnafm)
        'dock': '#9467bd',  # purple
        'dock_rnafm': '#8c564b'  # brown (distinct for rnafm)
    }
    custom_palette = sns.color_palette("Paired")

    # custom_palette = sns.color_palette(palette=custom_palette)
    # sns.set_palette(custom_palette)

    plt.rcParams['axes.grid'] = True

    g = sns.FacetGrid(big_df, col="pocket_id", hue="name", col_wrap=2, height=4, palette=custom_palette, sharey=False)
    g.map(sns.lineplot, "frac", "score").add_legend()

    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.show()

    # axs = plt.subplots(2,2)
    # for i, pockets in enumerate(ROBIN_POCKETS.values()):
    #
    # sns.lineplot(data=big_df, x="", y="passengers", hue="month")


if __name__ == "__main__":
    RES_DIR = "outputs/robin/"
    MODELS = {
        "native": "is_native/native_nopre_new_pdbchembl",
        "native_rnafm": "is_native/native_nopre_new_pdbchembl_rnafm",
        "native_pre": "is_native/native_pretrain_new_pdbchembl",
        "native_pre_rnafm": "is_native/native_pretrain_new_pdbchembl_rnafm",
        "dock": "dock/dock_new_pdbchembl",
        "dock_rnafm": "dock/dock_new_pdbchembl_rnafm",
    }

    PAIRS = {("native", "dock"): "vanilla",
             ("native_rnafm", "dock_rnafm"): "vanilla_fm",
             ("native_pre", "dock"): "pre",
             ("native_pre_rnafm", "dock_rnafm"): "pre_fm"}

    # pocket_id = "TPP"
    # lig_name = "2GDI_Y_TPP_100"
    # model_dir = "results/trained_models/"
    # model_path = "is_native/native_nopre_new_pdbchembl"
    # full_model_path = os.path.join(model_dir, model_path)
    # model = get_model_from_dirpath(full_model_path)
    # one_robin(pocket_id, lig_name, model, use_rna_fm=False)

    # get_all_csvs(recompute=True)
    # mix_all()
    plot_all()
