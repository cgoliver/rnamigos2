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

# from rnamigos.utils.virtual_screen import enrichment_factor
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataloader import get_loader, get_vs_loader
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.virtual_screen import get_efs
from scripts_fig.plot_utils import group_df
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


def enrichment_factor(scores, is_active, lower_is_better=True, frac=0.01):
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [
        a
        for _, a in sorted(
            zip(scores, is_active), key=lambda x: x[0], reverse=not lower_is_better
        )
    ]
    scores_sorted = [
        s
        for s, _ in sorted(
            zip(scores, is_active), key=lambda x: x[0], reverse=not lower_is_better
        )
    ]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    ef = (n_actives_screened / n_screened) / (n_actives / len(scores))
    return ef, scores_sorted[n_screened]


def one_robin(pocket_id, ligand_name, model=None, use_rna_fm=False, do_mixing=False):
    dgl_pocket_graph, _ = load_rna_graph(
        POCKET_PATH / Path(pocket_id).with_suffix(".json"),
        use_rnafm=use_rna_fm,
    )
    final_df = robin_inference(
        ligand_name=ligand_name,
        dgl_pocket_graph=dgl_pocket_graph,
        model=model,
        use_ligand_cache=False,
        ligand_cache="data/ligands/robin_lig_graphs.p",
        do_mixing=do_mixing,
        debug=False,
    )
    final_df["pocket_id"] = pocket_id
    rows = []
    # sns.kdeplot(final_df, x="score", hue="is_active", common_norm=False)
    # plt.show()
    for frac in (0.01, 0.02, 0.05):
        ef, _ = enrichment_factor(
            final_df["score"],
            final_df["is_active"],
            lower_is_better=False,
            frac=frac,
        )
        rows.append({"pocket_id": pocket_id, "score": ef, "frac": frac})
    return pd.DataFrame(rows), pd.DataFrame(final_df)


def get_all_preds(model, use_rna_fm):
    robin_ligs = list(ROBIN_POCKETS.keys())
    robin_pockets = list(ROBIN_POCKETS.values())

    # Associate the ligands with the wrong pockets.
    robin_lig_pocket_dict = {
        lig: robin_pockets[(i + SWAP) % len(robin_ligs)]
        for i, lig in enumerate(robin_ligs)
    }

    robin_dfs = [
        df
        for df in Parallel(n_jobs=4)(
            delayed(one_robin)(pocket_id, ligand_name, model, use_rna_fm)
            for ligand_name, pocket_id in robin_lig_pocket_dict.items()
        )
    ]
    robin_efs, robin_raw_dfs = list(map(list, zip(*robin_dfs)))
    robin_ef_df = pd.concat(robin_efs)
    robin_raw_df = pd.concat(robin_raw_dfs)

    # The naming is based on the pocket. So to keep ligands swap consistent, we need to change that
    old_to_new = {
        robin_pockets[i]: robin_pockets[(i + SWAP) % len(robin_ligs)]
        for i in range(len(robin_ligs))
    }
    new_to_old = {v: k for k, v in old_to_new.items()}
    robin_ef_df["pocket_id"] = robin_ef_df["pocket_id"].map(new_to_old)
    robin_raw_df["pocket_id"] = robin_raw_df["pocket_id"].map(new_to_old)
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
        rnafm = model_path.endswith("rnafm")
        rnafm = "rnafm" in model_path
        model = get_model_from_dirpath(full_model_path)
        df_ef, df_raw = get_all_preds(model, use_rna_fm=rnafm)
        df_ef.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)


def get_dfs_docking():
    """
    Go from columns:
    TARGET,_TITLE1,SMILE,TOTAL,INTER,INTRA,RESTR,VDW,TYPE

    To columns:
    raw: score,smiles,is_active,pocket_id
    efs: pocket_id,score,frac
    """

    ref_raw_df = pd.read_csv("outputs/robin/dock_raw.csv")
    docking_df = pd.read_csv("data/robin_docking_consolidated_v2.csv")
    # For each pocket, get relevant mapping smiles : normalized score,
    # then use it to create the appropriate raw, and clean csvs
    all_raws, all_dfs = [], []
    for ligand_name, pocket_id in ROBIN_POCKETS.items():
        docking_df_lig = docking_df[docking_df["TARGET"] == ligand_name]
        ref_raw_df_lig = ref_raw_df[ref_raw_df["pocket_id"] == pocket_id]
        scores = -pd.to_numeric(
            docking_df_lig["INTER"], errors="coerce"
        ).values.squeeze()
        scores[scores < 0] = 0
        scores = np.nan_to_num(scores, nan=0)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        mapping = {}
        for smiles, score in zip(docking_df_lig[["SMILE"]].values, normalized_scores):
            mapping[smiles[0]] = score
        mapping = defaultdict(int, mapping)
        docking_scores = [mapping[smile] for smile in ref_raw_df_lig["smiles"].values]
        ref_raw_df_lig["score"] = docking_scores

        # Go from RAW to EF
        rows = []
        for frac in (0.01, 0.02, 0.05):
            ef, _ = enrichment_factor(
                ref_raw_df_lig["score"],
                ref_raw_df_lig["is_active"],
                lower_is_better=False,
                frac=frac,
            )
            rows.append({"pocket_id": pocket_id, "score": ef, "frac": frac})
        all_raws.append(ref_raw_df_lig.copy())
        all_dfs.append(pd.DataFrame(rows))
    all_raws = pd.concat(all_raws)
    all_dfs = pd.concat(all_dfs)
    all_raws.to_csv(f"{RES_DIR}/rdock_raw.csv")
    all_dfs.to_csv(f"{RES_DIR}/rdock.csv")


def mix(df1, df2, outpath=None):
    def normalize(scores):
        out_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return out_scores

    norm_score1 = normalize(df1["score"])
    norm_score2 = normalize(df2["score"])
    mixed_scores = 0.5 * (norm_score1 + norm_score2)
    out_df = df1.copy()
    out_df["mixed_score"] = mixed_scores
    out_df = out_df.drop(columns=["score"])
    out_df = out_df.rename(columns={"mixed_score": "score"})
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
                ef, _ = enrichment_factor(
                    mixed_df_lig["score"],
                    mixed_df_lig["is_active"],
                    lower_is_better=False,
                    frac=frac,
                )
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
    # RES_DIR = "outputs/robin/" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"
    RES_DIR = "outputs/robin_bk/" if SWAP == 0 else f"outputs/robin_bk_swap_{SWAP}"
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

    # TEST ONE INFERENCE
    # pocket_id = "TPP"
    # lig_name = "2GDI_Y_TPP_100"
    # model_dir = "results/trained_models/"
    # model_path = "is_native/native_nopre_new_pdbchembl"
    # full_model_path = os.path.join(model_dir, model_path)
    # model = get_model_from_dirpath(full_model_path)
    # one_robin(pocket_id, lig_name, model, use_rna_fm=False)

    """
    model_dir = "results/trained_models/"
    for model_name, model_path in MODELS.items():
        p = os.path.join(model_dir, model_path)
        model, cfg = get_model_from_dirpath(p, return_cfg=True)
        pdb_eval(cfg, model)
    """

    # GET ALL CSVs for the models and plot them
    get_all_csvs(recompute=True)
    get_dfs_docking()
    mix_all()
    plot_all()
    # COMPUTE PERTURBED VERSIONS
    for i in range(1, 4):
        SWAP = i
        RES_DIR = "outputs/robin/" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"
        get_all_csvs(recompute=True)
        mix_all()
    plot_perturbed(model="rnamigos++", group=True)
