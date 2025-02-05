import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import time
from torch.utils.data import DataLoader
import pandas as pd

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.dataset import InferenceDataset
from rnamigos.utils.graph_utils import get_dgl_graph
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.mixing_utils import add_mixed_score


def inference_raw(
        dgl_graph,
        smiles_list,
        models,
        ligand_cache=None,
        use_ligand_cache=False,
):
    """
    Run inference from python objects
    """
    # Get ready to loop through ligands
    dataset = InferenceDataset(smiles_list, ligand_cache=ligand_cache, use_ligand_cache=use_ligand_cache)
    batch_size = 64
    loader_args = {
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": 0,
        "collate_fn": dataset.collate,
    }
    dataloader = DataLoader(dataset=dataset, **loader_args)

    results = {model_name: [] for model_name in models.keys()}
    t0 = time.time()
    for i, (ligands_graph, ligands_vector) in enumerate(dataloader):
        for model_name, model in models.items():
            if model.target == "native_fp":
                scores = model.predict_ligands(dgl_graph, ligands_vector)
            else:
                scores = model.predict_ligands(dgl_graph, ligands_graph)
            scores = list(scores[:, 0].numpy())
            results[model_name].extend(scores)
        if not i % 50 and i > 0:
            print(f"Done {i * batch_size}/{len(dataset)} in {time.time() - t0}")

    results = {model_name: np.asarray(all_scores) for model_name, all_scores in results.items()}

    # Post-process raw scores to get a consistent 'higher is better' numpy array score
    for model_name, all_scores in results.items():
        # Flip raw scores as for those models, lower is better
        if models[model_name].target in {"dock", "native_fp"}:
            all_scores = -1 * all_scores
        # print(f"{model_name} {models[model_name].target} {all_scores[:10]}")
        results[model_name] = all_scores
    results = pd.DataFrame(results)
    results["smiles"] = smiles_list
    return results


def get_models(models_path=None, model=None):
    # Load models
    script_dir = os.path.dirname(__file__)
    if model is not None:
        if isinstance(model, dict):
            return model
        return {"raw_score": model}

    if models_path is None:
        models_path = {
            "is_native": "results/trained_models/is_native/native_42",
            "dock": "results/trained_models/dock/dock_42",
        }
        models_path = {model_name: os.path.join(script_dir, "..", model_path)
                       for model_name, model_path in models_path.items()}
    models = {model_name: get_model_from_dirpath(model_path)
              for model_name, model_path in models_path.items()}
    return models


def do_inference(cif_path,
                 residue_list,
                 ligands_path,
                 out_path,
                 model=None,
                 models_path=None,
                 ligand_cache=None,
                 use_ligand_cache=False,
                 do_mixing=True,
                 dump_all=False):
    """
    Run inference from files
    """
    models = get_models(models_path=models_path, model=model)

    # Do we need to have rna-fm embeddings with the required models ?
    # Assert the answer is the same for all queried models
    need_rna_fm = [model.encoder.in_dim == 644 for model in models.values()]
    assert len(set(need_rna_fm)) == 1

    # Get dgl graph with node expansion BFS, load smiles and models
    dgl_graph = get_dgl_graph(cif_path, residue_list, use_rnafm=need_rna_fm[0])
    print("Successfully built the graph")
    smiles_list = [s.lstrip().rstrip() for s in list(open(ligands_path).readlines())]

    # Get raw results df
    results_df = inference_raw(
        dgl_graph=dgl_graph,
        smiles_list=smiles_list,
        models=models,
        ligand_cache=ligand_cache,
        use_ligand_cache=use_ligand_cache)

    if do_mixing:
        names = list(models.keys())
        results_df = add_mixed_score(df=results_df, score1=names[0], score2=names[1], use_max=True)
        if not dump_all:
            results_df = results_df[['smiles', 'mixed']]
    if out_path is not None:
        results_df.to_csv(out_path)
    return results_df


@hydra.main(version_base=None, config_path="conf", config_name="inference")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Done importing")
    do_inference(
        cif_path=cfg.cif_path,
        residue_list=cfg.residue_list,
        ligands_path=cfg.ligands_path,
        out_path=cfg.out_path,
    )


if __name__ == "__main__":
    main()
    # do_inference(cif_path="sample_files/3ox0.cif",
    #              residue_list="A.25,A.26,A.7,A.8".split(','),
    #              ligands_path="sample_files/test_smiles.txt",
    #              out_path="test.out",
    #              dump_all=True)
