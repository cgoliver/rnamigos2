import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.learning.dataset import InferenceDataset

POCKETS_DIR = Path("data/json_pockets_expanded")
MODEL_DIR = Path("results/trained_models/")
MODE = "is_native"
MODEL_NAME = "native_rnafm_dout5_4"
USE_RNAFM = True

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f"figs/embedding_logs/{MODEL_NAME}_all")
    model = get_model_from_dirpath(MODEL_DIR / MODE / MODEL_NAME)
    # use a dummy ligand since we just care about pocket
    smiles_path = "data/ligand_db/4LF7_A_PAR_1818/pdb_chembl/decoys.txt"
    inactives_smiles_set = set([s.lstrip().rstrip() for s in list(open(smiles_path).readlines())])
    smiles_list = list(inactives_smiles_set)
    ligs = InferenceDataset(smiles_list)

    all_g_embs, all_lig_embs = [], []
    for i, pocket_id in enumerate(os.listdir(POCKETS_DIR)):
        dgl_pocket_graph, _ = load_rna_graph(
            POCKETS_DIR / Path(pocket_id).with_suffix(".json"),
            use_rnafm=USE_RNAFM,
        )
        pred, (g_emb, lig_emb) = model(dgl_pocket_graph, ligs[0][0])
        all_g_embs.append(g_emb.squeeze())
        if i < 1:
            for j, lig in enumerate(ligs):
                try:
                    pred, (g_emb, lig_emb) = model(dgl_pocket_graph, lig[0])
                    # print(embs.shape, dgl_pocket_graph)
                    all_lig_embs.append(lig_emb.squeeze())
                except:
                    print(f"failed on lig {j}")

    P = torch.stack(all_g_embs)
    P_norm = F.normalize(P, p=2, dim=1)
    cos_sim = torch.mm(P, P.t()).flatten().tolist()
    sns.displot(cos_sim)
    plt.show()

    L = torch.stack(all_lig_embs)
    writer.add_embedding(P, global_step=0, tag="pockets")
    writer.add_embedding(L, global_step=0, tag="ligands")
    writer.close()
    pass
