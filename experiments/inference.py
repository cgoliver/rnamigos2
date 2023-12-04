import os
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import copy

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos_dock.learning.loader import get_systems, InferenceDataset
from rnamigos_dock.tools.graph_utils import get_dgl_graph
from rnamigos_dock.learning.models import get_model_from_dirpath


def do_inference(cif_path, residue_list, ligands_path, out_path, dump_all=False):
    ### MODEL LODADING
    script_dir = os.path.dirname(__file__)
    models_path = {
        'dock': os.path.join(script_dir, '../saved_models/paper_dock'),
        'is_native': os.path.join(script_dir, '../saved_models/paper_native'),
        'native_fp': os.path.join(script_dir, '../saved_models/paper_fp')
    }
    models = {model_name: get_model_from_dirpath(model_path) for model_name, model_path in models_path.items()}

    # Get dgl graph with node expansion BFS
    dgl_graph = get_dgl_graph(cif_path, residue_list)
    smiles_list = [s.lstrip().rstrip() for s in list(open(ligands_path).readlines())]

    # Get ready to loop through ligands
    dataset = InferenceDataset(smiles_list)
    loader_args = {'shuffle': False,
                   'batch_size': 64,
                   'num_workers': 0,
                   'collate_fn': dataset.collate
                   }
    dataloader = DataLoader(dataset=dataset, **loader_args)

    results = {model_name: [] for model_name in models.keys()}
    for ligands_graph, ligands_vector in dataloader:
        for model_name, model in models.items():
            if model_name == 'native_fp':
                scores = model.predict_ligands(dgl_graph, ligands_vector)
            else:
                scores = model.predict_ligands(dgl_graph, ligands_graph)
            results[model_name].extend(list(scores.squeeze().cpu().numpy()))

    for model_name, all_scores in results.items():
        all_scores = np.asarray(all_scores)
        # Flip raw scores as lower is better for those models
        if model_name in {'dock', 'native_fp'}:
            all_scores = -all_scores
        results[model_name] = all_scores

    # MIX : best mix = 0.44, 0.39, 0.17
    def normalize(scores):
        out_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return out_scores

    results = {model_name: normalize(result) for model_name, result in results.items()}
    # print(results['native_fp'])
    mixed_scores = 0.44 * results['dock'] + 0.39 * results['native_fp'] + 0.17 * results['is_native']

    with open(out_path, 'w') as out:
        if not dump_all:
            for smiles, score in zip(smiles_list, mixed_scores):
                out.write(f"{smiles} {score}\n")
        else:
            for smiles, dock_score, native_score, fp_score, mixed_score in zip(smiles_list,
                                                                               results['dock'],
                                                                               results['is_native'],
                                                                               results['native_fp'],
                                                                               mixed_scores):
                out.write(f"{smiles} {dock_score} {native_score} {fp_score} {mixed_score}\n")


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
    # main()
    do_inference(cif_path="sample_files/3ox0.cif",
                 residue_list="A.25,A.26,A.7,A.8".split(','),
                 ligands_path="sample_files/test_smiles.txt",
                 out_path="test.out",
                 dump_all=True)
