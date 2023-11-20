import os
import sys

import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos_dock.learning.loader import get_systems, InferenceDataset
from rnamigos_dock.tools.graph_utils import get_dgl_graph
from rnamigos_dock.learning.models import get_model_from_dirpath


def col(x):
    return x[0]


def do_inference(cif_path, residue_list, ligands_path, out_path):
    # Get dgl graph with node expansion BFS
    dgl_graph = get_dgl_graph(cif_path, residue_list)
    smiles_list = [s.lstrip().rstrip() for s in list(open(ligands_path).readlines())]
    # Loader is asynchronous
    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 0,
                   'collate_fn': lambda x: x[0]
                   }

    ### MODEL LODADING
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(__file__)
    models = {
        'dock': os.path.join(script_dir, '../saved_models/paper_dock'),
        'is_native': os.path.join(script_dir, '../saved_models/paper_native'),
        'native_fp': os.path.join(script_dir, '../saved_models/paper_fp')
    }
    results = {}
    for model_name, model_path in models.items():
        model = get_model_from_dirpath(model_path)
        # model = model.to(device) TODO
        model = model.to('cpu')
        dataset = InferenceDataset(dgl_graph,
                                   smiles_list,
                                   use_graphligs=model.use_graphligs,
                                   )
        dataloader = GraphDataLoader(dataset=dataset, **loader_args)
        all_scores = []
        for pocket_graph, ligands in dataloader:
            scores = list(model.predict_ligands(pocket_graph,
                                                ligands,
                                                ).squeeze().cpu().numpy())
            all_scores.extend(scores)
        all_scores = np.asarray(all_scores)
        # Flip raw scores as lower is better for those models
        if model_name in {'dock', 'native_fp'}:
            all_scores = -all_scores
        results[model_name] = np.asarray(all_scores)

    # MIX : best mix = 0.44, 0.39, 0.17
    def normalize(scores):
        out_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return out_scores

    results = {model_name: normalize(result) for model_name, result in results.items()}
    mixed_scores = 0.44 * results['dock'] + 0.39 * results['native_fp'] + 0.17 * results['is_native']

    with open(out_path, 'w') as out:
        for smiles, score in zip(smiles_list, mixed_scores):
            out.write(f"{smiles} {score}\n")


@hydra.main(version_base=None, config_path="../conf", config_name="inference")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    do_inference(cif_path=cfg.cif_path,
                 residue_list=cfg.residue_list,
                 ligands_path=cfg.ligands_path,
                 out_path=cfg.out_path)


if __name__ == "__main__":
    pass
    main()
    # do_inference(cif_path="sample_files/3ox0.cif",
    #              residue_list="A.25,A.26,A.7,A.7".split(','),
    #              ligands_path="sample_files/test_smiles.txt",
    #              out_path="test.out")
