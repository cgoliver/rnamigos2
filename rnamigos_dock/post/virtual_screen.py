""" Run a virtual screen with a trained model
"""
from dgl import DGLGraph
from loguru import logger
import numpy as np
import pandas as pd
from sklearn import metrics
import torch


def mean_active_rank(scores, is_active, lower_is_better=True, **kwargs):
    """ Compute the average rank of actives in the scored ligand set

    Arguments
    ----------
    scores (list): list of scalar scores for each ligand in the library
    is_active (list): binary vector with 1 if ligand is active or 0 else, one for each of the scores
    lower_is_better (bool): True if a lower score means higher binding likelihood, False v.v.

    Returns
    ---------
    int
        Mean rank of the active ligand [0, 1], 1 is the best score.
        

    >>> mean_active_rank([-1, -5, 1], [0, 1, 0], lower_is_better=True)
    >>> 1.0

    """
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    if lower_is_better:
        scores = 1 - scores
    fpr, tpr, thresholds = metrics.roc_curve(is_active, scores, drop_intermediate=True)
    auroc = metrics.auc(fpr, tpr)
    # is_active_sorted = sorted(zip(scores, is_active))
    # mar = (np.mean([rank for rank, (_, is_active) in enumerate(is_active_sorted) if is_active]) + 1) / len(scores)
    return auroc


def enrichment_factor(scores, is_active, lower_is_better=True, frac=0.01):
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    return (n_actives_screened / n_screened) / (n_actives / len(scores))


def run_virtual_screen(model, dataloader, metric=mean_active_rank, **kwargs):
    """run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataloader: Loader of VirtualScreenDataset object
    :param metric: function that takes a list of prediction and an is_active indicator and returns a score 
    :param return_model_outputs: whether to return the scores given by the model.

    :returns scores: list of scores, one for each graph in the dataset 
    :returns inds: list of indices in the dataloader for which the score computation was successful
    """
    efs, all_scores, status, all_smiles, pocket_names = [], [], [], [], []
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
            logger.info(f"Done {i}/{len(dataloader)}")
        if ((isinstance(ligands, torch.Tensor) and len(ligands) < 10)
                or (isinstance(ligands, DGLGraph) and ligands.batch_size < 10)):
            logger.warning(f"Skipping pocket{i}, not enough decoys")
            continue
        scores = model.predict_ligands(pocket_graph, ligands)[:, 0].numpy()
        is_active = is_active.numpy()
        efs.append(metric(scores, is_active, **kwargs))
        all_scores.append(list(scores))
        status.append(list(is_active))
        pocket_names.append(pocket_name)
        all_smiles.append(smiles)
    logger.debug(f"VS failed on {failed_set}")
    # print(failed)
    # print(efs)
    return efs, all_scores, status, pocket_names, all_smiles
