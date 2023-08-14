"""
    Swap the fingerprints around from an annotated folder.
"""
import os, pickle
import random
import numpy as np


def scramble_fingerprints(annot_dir, dump_dir):
    """
        Assign a fingerprint to each graph chosen at random from all other
        fingerprints in annot_dir.
    """

    datas = os.listdir(annot_dir)
    indices = range(len(datas))
    fps = []
    for i,g in enumerate(datas):
        _,_,_,fp = pickle.load(open(os.path.join(annot_dir, g), 'rb'))
        fps.append(fp)
    for i,g in enumerate(datas):
        G,tree,ring,fp = pickle.load(open(os.path.join(annot_dir, g), 'rb'))
        new_fp = fps[random.choice(indices)]
        print(f"old fp {fp}")
        print(f"new fp {new_fp}")
        pickle.dump((G,tree,ring,new_fp), open(os.path.join(dump_dir, g), 'wb'))
    pass

def random_fingerprints(annot_dir, dump_dir):
    """
        Assign a fingerprint to each graph chosen at random from all other
        fingerprints in annot_dir.
    """

    datas = os.listdir(annot_dir)
    for i,g in enumerate(datas):
        G,tree,ring,_ = pickle.load(open(os.path.join(annot_dir, g), 'rb'))
        rand_fp = np.random.randint(2, size=166)
        print(rand_fp)
        pickle.dump((G,tree,ring,rand_fp), open(os.path.join(dump_dir, g), 'wb'))
    pass

if __name__ == "__main__":
    scramble_fingerprints('../data/annotated/pockets_nx_symmetric', '../data/annotated/pockets_nx_symmetric_scramble')
    # random_fingerprints('../data/annotated/pockets_nx_symmetric', '../data/annotated/pockets_nx_symmetric_random')
    pass
