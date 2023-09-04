""" Run a virtual screen with a trained model
"""
import numpy as np

def mean_active_rank(scores, is_active, lower_is_better=True, **kwargs):
    is_active_sorted = sorted(zip(scores, is_active), reverse=lower_is_better)
    return 1 - (np.mean([rank for rank, (score, is_active)  in enumerate(is_active_sorted) if is_active]) / len(scores))

def enrichment_factor(scores, is_active, lower_is_better=True, **kwargs):
    n_actives = np.sum(is_active)
    n_screened = int(kwargs['frac'] * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    return (n_actives_screened / n_screened) / (n_actives / len(scores))

def run_virtual_screen(model, dataset, metric=mean_active_rank, **kwargs):
    """run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataset: VirtualScreenDataset object
    :param metric: function that takes a list of prediction and an is_active indicator and returns a score 
    """
    efs = []
    print(len(dataset))
    for pocket_graph, ligands, is_active in dataset:
        scores = [model(pocket_graph, ligand.unsqueeze(dim=0)) for ligand in ligands]
        efs.append(metric(scores, is_active, **kwargs))
    return efs
