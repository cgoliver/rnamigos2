""" Run a virtual screen with a trained model
"""

from dgl import DGLGraph
from loguru import logger
import numpy as np
import pandas as pd
from sklearn import metrics
import torch


def get_auroc(scores, is_active):
    fpr, tpr, thresholds = metrics.roc_curve(is_active, scores, drop_intermediate=True)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def enrichment_factor(scores, is_active, frac=0.01):
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), key=lambda x: x[0], reverse=True)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    return (n_actives_screened / n_screened) / (n_actives / len(scores))


def run_virtual_screen(model, dataloader, metric=get_auroc, lower_is_better=False, verbose=True):
    """run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataloader: Loader of VirtualScreenDataset object
    :param metric: function that takes a list of prediction and an is_active indicator and returns a score
    :param lower_is_better: set to true for dock and native_fp models
    """
    aurocs, all_scores, status, all_smiles, pocket_names = [], [], [], [], []
    logger.debug(f"Doing VS on {len(dataloader)} pockets.")
    failed_set = set()
    failed = 0
    for i, (pocket_name, pocket_graph, ligands, is_active, smiles) in enumerate(dataloader):
        if pocket_graph is None:
            failed_set.add(pocket_graph)
            logger.trace(pocket_graph)
            logger.debug(f"VS fail")
            failed += 1
            continue
        if not i % 20:
            if verbose:
                logger.info(f"Done {i}/{len(dataloader)}")
        if (isinstance(ligands, torch.Tensor) and len(ligands) < 10) or (
            isinstance(ligands, DGLGraph) and ligands.batch_size < 10
        ):
            logger.warning(f"Skipping pocket{i}, not enough decoys")
            continue
        scores = model.predict_ligands(pocket_graph, ligands)[:, 0].numpy()
        if lower_is_better:
            scores = -scores
        is_active = is_active.numpy()
        aurocs.append(metric(scores, is_active))
        all_scores.append(list(scores))
        status.append(list(is_active))
        pocket_names.append(pocket_name)
        all_smiles.append(smiles)
    logger.debug(f"VS failed on {failed_set}")
    return aurocs, all_scores, status, pocket_names, all_smiles


def run_results_to_raw_df(scores, status, pocket_names, all_smiles, decoy_mode):
    raw_rows = list()
    for pocket_id, score_list, status_list, smiles_list in zip(pocket_names, scores, status, all_smiles):
        for score, status, smiles in zip(score_list, status_list, smiles_list):
            raw_rows.append(
                {
                    "raw_score": score,
                    "is_active": status,
                    "pocket_id": pocket_id,
                    "smiles": smiles,
                    "decoys": decoy_mode,
                }
            )
    return pd.DataFrame(raw_rows)


def run_results_to_auroc_df(aurocs, scores, pocket_names, decoy_mode):
    rows = list()
    for auroc, score, pocket_id in zip(aurocs, scores, pocket_names):
        rows.append(
            {
                "score": auroc,
                "metric": "EF" if decoy_mode == "robin" else "AuROC",
                "decoys": decoy_mode,
                "pocket_id": pocket_id,
            }
        )
    return pd.DataFrame(rows)


def get_results_dfs(model, dataloader, decoy_mode, cfg, verbose=False):
    lower_is_better = cfg.train.target in ["dock", "native_fp"]
    metric = enrichment_factor if decoy_mode == "robin" else get_auroc
    if verbose:
        print(f"DOING: {cfg.name}, LOWER IS BETTER: {lower_is_better}")
    aurocs, scores, status, pocket_names, all_smiles = run_virtual_screen(
        model, dataloader, metric=metric, lower_is_better=lower_is_better, verbose=verbose
    )
    auroc_df = run_results_to_auroc_df(aurocs, scores, pocket_names, decoy_mode)
    raw_df = run_results_to_raw_df(scores, status, pocket_names, all_smiles, decoy_mode)
    if verbose:
        print(f"Mean AuROC for {decoy_mode} {cfg.name}:", np.mean(aurocs))
    return auroc_df, raw_df


def raw_df_to_aurocs(raw_df, score="raw_score"):
    """
    df_raw => MAR
    :param raw_df:
    :param score:
    :return:
    """
    pockets = raw_df["pocket_id"].unique()
    all_aurocs = []
    for pi, pocket in enumerate(pockets):
        pocket_df = raw_df.loc[raw_df["pocket_id"] == pocket]
        auroc = get_auroc(pocket_df[score], pocket_df["is_active"])
        all_aurocs.append({"score": auroc, "pocket_id": pocket})
    df_auroc = pd.DataFrame(all_aurocs)
    print(df_auroc)
    return df_auroc


def raw_df_to_mean_auroc(raw_df, score="raw_score"):
    """
    df_raw => MAR
    :param raw_df:
    :param score:
    :return:
    """
    df_auroc = raw_df_to_aurocs(raw_df, score)
    return np.mean(df_auroc["score"].values)


def raw_df_to_efs(raw_df, score="raw_score", fracs=(0.01, 0.02, 0.05)):
    """
    df_raw => efs
    :param df:
    :param score:
    :param outname:
    :return:
    """
    ef_rows = []
    for frac in fracs:
        for pocket, group in raw_df.groupby("pocket_id"):
            ef_frac = enrichment_factor(group[score], group["is_active"], frac=frac)
            ef_rows.append({"score": ef_frac, "pocket_id": pocket, "frac": frac})
    df_ef = pd.DataFrame(ef_rows)
    return df_ef
