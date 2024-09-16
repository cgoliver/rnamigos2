import os

import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from rnaglib.utils import load_json
from rnaglib.transforms import RNAFMTransform
from rnaglib.data_loading import RNADataset

# grab the RNAs
pockets_dir = "data/json_pockets_expanded"
out_dir = "data/pocket_embeddings"
os.makedirs(out_dir, exist_ok=True)

dset = RNADataset(redundancy="all", in_memory=False)
t = RNAFMTransform()

emb_cache = {}
for pocket in tqdm(os.listdir(pockets_dir)):
    outname = os.path.join(out_dir, f"{Path(pocket).stem}.npz")
    if os.path.exists(outname):
        continue
    g = load_json(Path(pockets_dir) / pocket)
    pocket_embs = {}
    embs_list, missing_nodes = [], []
    for node in g.nodes():
        pdbid, chain, pos = node.split(".")

        # try to load embs from cache if not from disk
        try:
            g = emb_cache[pdbid]
        except KeyError:
            g = dset.get_pdbid(pdbid)
            try:
                emb_cache[pdbid] = t(g)
            except IndexError:
                break

        # try to get a node's embedding and remember if it was mising
        try:
            z = g["rna"].nodes[node]["rnafm"]
        except KeyError:
            missing_nodes.append(node)

        pocket_embs[node] = z
        embs_list.append(z)
        pass

    # fill in missing nodes with mean embedding
    if missing_nodes:
        mean_emb = torch.mean(torch.stack(embs_list), dim=0)
        for node in missing_nodes:
            pocket_embs[node] = mean_emb
    # finally dump
    np.savez(outname, **pocket_embs)
