""" Run a virtual screen with a trained model
"""
import numpy as np
from loguru import logger
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
    is_active_sorted = sorted(zip(scores, is_active), reverse=lower_is_better)
    return (np.mean([rank for rank, (score, is_active) in enumerate(is_active_sorted) if is_active]) + 1) / len(scores)




def enrichment_factor(scores, is_active, lower_is_better=True, **kwargs):
    n_actives = np.sum(is_active)
    n_screened = int(kwargs['frac'] * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    return (n_actives_screened / n_screened) / (n_actives / len(scores))


def run_virtual_screen(model, dataloader, metric=mean_active_rank, use_embedding_distance=True, **kwargs):
    """run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataloader: Loader of VirtualScreenDataset object
    :param metric: function that takes a list of prediction and an is_active indicator and returns a score 
    :param return_model_outputs: whether to return the scores given by the model.

    :returns scores: list of scores, one for each graph in the dataset 
    :returns inds: list of indices in the dataloader for which the score computation was successful
    """
    efs, inds, all_scores, pocket_ids = [],[],[],[]
    logger.debug(f"Doing VS on {len(dataloader)} pockets.")
    failed_set = set()
    for i, (pocket_graph, ligands, is_active) in enumerate(dataloader):
        pocket_id = dataloader.dataset.all_pockets_id[i]
        if pocket_graph is None:
            failed_set.add(pocket_graph)
            logger.trace(pocket_graph)
            logger.debug(f"VS fail")
            continue
        if not i % 20:
            logger.info(f"Done {i}/{len(dataloader)}")

        model = model.to('cpu')
        scores = list(model.predict_ligands(pocket_graph, 
                                            ligands, 
                                            ).squeeze().cpu().numpy())
        all_scores.append(scores)
        efs.append(metric(scores, is_active, **kwargs))
        inds.append(i)
        pocket_ids.append(pocket_id)
    logger.debug(f"VS failed on {failed_set}")
    print(efs)
    return efs, inds, all_scores, pocket_ids
