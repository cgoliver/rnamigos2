"""
    Functions for getting decoys.
"""
import os
import pickle
import random
import numpy as np
from scipy.spatial.distance import jaccard

def get_decoys(mode='pdb', annots_dir='../data/annotated/pockets_nx'):
    """
    Build decoys set for validation.
    """
    if mode == 'pdb-whole':
        fp_dict = pickle.load(open('data/all_ligs_pdb_maccs.p', 'rb'))
        return fp_dict

    if mode=='pdb':
        fp_dict = {}
        for g in os.listdir(annots_dir):
            try:
                lig_id = g.split(":")[2]
            except Exception as e:
                print(f"failed on {g}, {e}")
                continue
            _,_,_,fp = pickle.load(open(os.path.join(annots_dir, g), 'rb'))
            fp_dict[lig_id] = fp
        #fp_dict = {'lig_id': [fp], ...}
        #ligand's decoys are all others except for the active
        decoy_dict = {k:(v, [f for lig,f in fp_dict.items() if lig != k]) for k,v in fp_dict.items()}
        return decoy_dict
    pass

def distance_rank(active, pred, decoys, dist_func=jaccard):
    """
        Get rank of prediction in `decoys` given a known active ligand.
    """

    pred_dist = dist_func(active, pred)
    rank = 0
    for lig in decoys:
        d = dist_func(active, lig)
        if d < pred_dist:
            rank += 1
    return 1- (rank / (len(decoys) + 1))

def decoy_test(fp_pred, true_id, decoys,
                    shuffle=False):
    """
        Check performance against decoy set.
        decoys --> {'ligand_id', ('expected_FP', [decoy_fps])}
        test_set --> [annot_graph_path,]
        :model trained model
        :test_set inputs for model to test (RNA graphs)
        :decoys dictionary with list of decoys for each input to test.
        :test_graphlist list of graph names to use in the test.

        :return: enrichment score
    """
    ranks = []
    sims = []

    ligs = list(decoys.keys())
    try:
        decoys[true_id]
    except KeyError:
        print("missing fp", true_id)
    fp_pred = fp_pred.detach().numpy() > 0.5
    fp_pred = fp_pred.astype(int)
    if shuffle:
        #pick a random ligand to be the true one
        orig = true_id
        true_id = np.random.choice(ligs, replace=False)
    active = decoys[true_id][0]
    decs = decoys[true_id][1]
    rank = distance_rank(active, fp_pred, decs)
    sim = jaccard(active, fp_pred)
    return rank, sim
def decoy_test_(fp_pred, true_fp, decoys,
                    shuffle=False):
    """
        Check performance against decoy set.
        :decoys list of decoys 

        :return: enrichment score
    """

    fp_pred = fp_pred.detach().numpy() > 0.5
    fp_pred = fp_pred.astype(int)
    print(fp_pred, true_fp)
    rank = distance_rank(true_fp, fp_pred, decoys)
    sim = jaccard(true_fp, fp_pred)
    return rank, sim
