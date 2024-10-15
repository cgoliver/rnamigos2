"""
Examples of command lines to train a model are available in scripts_run/train.sh
"""

import os
import sys

from pathlib import Path

from joblib import Parallel, delayed
from collections import defaultdict
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.utils.virtual_screen import enrichment_factor
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataloader import get_vs_loader
from rnamigos.utils.virtual_screen import get_efs
from scripts_fig.plot_utils import group_df

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

POCKET_PATH = "data/json_pockets_expanded"



def plot_all():
    big_df = []
    models = MODELS
    models = list(MODELS) + list(PAIRS.values()) + ["rdock"]
    for model in models:
        out_csv = os.path.join(RES_DIR, f"{model}.csv")
        df = pd.read_csv(out_csv)
        df["name"] = model
        big_df.append(df)
    big_df = pd.concat(big_df)

    custom_palette = {
        "native": "#1f77b4",  # blue
        "native_rnafm": "#ff7f0e",  # orange (distinct for rnafm)
        "native_pre": "#2ca02c",  # green
        "native_pre_rnafm": "#d62728",  # red (distinct for rnafm)
        "dock": "#9467bd",  # purple
        "dock_rnafm": "#8c564b",  # brown (distinct for rnafm)
        "rdock": "black",
    }
    custom_palette = sns.color_palette("Paired")

    # custom_palette = sns.color_palette(palette=custom_palette)
    # sns.set_palette(custom_palette)

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

    # custom_palette = sns.color_palette("Paired")
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


def pdb_eval(cfg, model):
    # Final VS validation on each decoy set
    logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
    logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")

    test_systems = get_systems_from_cfg(cfg, return_test=True)
    model = model.to("cpu")
    rows, raw_rows = [], []
    decoys = ["chembl", "pdb", "pdb_chembl", "decoy_finder"]
    for decoy_mode in decoys:
        dataloader = get_vs_loader(systems=test_systems, decoy_mode=decoy_mode, cfg=cfg)
        decoy_rows, decoys_raw_rows = get_efs(
            model=model,
            dataloader=dataloader,
            decoy_mode=decoy_mode,
            cfg=cfg,
            verbose=True,
        )
        rows += decoy_rows
        raw_rows += decoys_raw_rows

    # Make it a df
    df = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows)

    # Dump csvs
    d = Path(cfg.result_dir, parents=True, exist_ok=True)
    base_name = Path(cfg.name).stem
    # df.to_csv(d / (base_name + ".csv"))
    # df_raw.to_csv(d / (base_name + "_raw.csv"))

    df_chembl = df.loc[df["decoys"] == "chembl"]
    print(f"{cfg.name} Mean MAR on chembl: {np.mean(df_chembl['score'].values)}")
    df_chembl = group_df(df_chembl)
    print(
        f"{cfg.name} Mean grouped MAR on chembl: {np.mean(df_chembl['score'].values)}"
    )

    df_pdbchembl = df.loc[df["decoys"] == "pdb_chembl"]
    print(f"{cfg.name} Mean MAR on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")
    df_pdbchembl = group_df(df_pdbchembl)
    print(
        f"{cfg.name} Mean grouped MAR on pdbchembl: {np.mean(df_pdbchembl['score'].values)}"
    )
    pass


if __name__ == "__main__":
    SWAP = 0
    RES_DIR = "outputs/robin/" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"
    MODELS = {
        # "native": "is_native/native_nopre_new_pdbchembl",
        # "native_rnafm": "is_native/native_nopre_new_pdbchembl_rnafm",
        # "native_pre": "is_native/native_pretrain_new_pdbchembl",
        "native_pre_rnafm": "is_native/native_pretrain_new_pdbchembl_rnafm",
        # "is_native_old": "is_native/native_42",
        # "native_pre_rnafm_tune": "is_native/native_pretrain_new_pdbchembl_rnafm_159_best",
        # "dock": "dock/dock_new_pdbchembl",
        "dock_rnafm": "dock/dock_new_pdbchembl_rnafm",
    }

    PAIRS = {
        # ("native", "dock"): "vanilla",
        # ("native_rnafm", "dock_rnafm"): "vanilla_fm",
        # ("native_pre", "dock"): "pre",
        # ("native_pre_rnafm_tune", "dock_rnafm"): "pre_fm",
        ("native_pre_rnafm", "dock_rnafm"): "native_dock_pre_fm",
        ("native_dock_pre_fm", "rdock"): "rnamigos++",
    }

    plot_all()
    # COMPUTE PERTURBED VERSIONS
    for i in range(1, 4):
        SWAP = i
        RES_DIR = "outputs/robin/" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"
        get_all_csvs(recompute=True)
        mix_all()
    plot_perturbed(model="rnamigos++", group=True)
