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


def inference(
    dgl_graph,
    smiles_list,
    out_path="rnamigos_out.csv",
    mixing_coeffs=(0.5, 0.0, 0.5),
    model=None,
    models_path=None,
    dump_all=False,
    ligand_cache=None,
    use_ligand_cache=False,
    do_mixing=True,
):
    """
    Run inference from python objects
    """
    # Load models
    script_dir = os.path.dirname(__file__)
    if models_path is None:
        models_path = {
            "dock": os.path.join(script_dir, "../results/trained_models/dock/dock_42"),
            "native_fp": os.path.join(
                script_dir, "../results/trained_models/native_fp/fp_42"
            ),
            "is_native": os.path.join(
                script_dir, "../results/trained_models/is_native/native_42"
            ),
        }
    if model is None:
        models = {
            model_name: get_model_from_dirpath(model_path)
            for model_name, model_path in models_path.items()
        }
    else:
        models = {"score": model}

    # Get ready to loop through ligands
    dataset = InferenceDataset(
        smiles_list,
        ligand_cache=ligand_cache,
        use_ligand_cache=use_ligand_cache,
    )
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
            if model_name == "native_fp":
                scores = model.predict_ligands(dgl_graph, ligands_vector)
            else:
                scores = model.predict_ligands(dgl_graph, ligands_graph)
            scores = list(scores[:, 0].numpy())
            results[model_name].extend(scores)
        if not i % 10 and i > 0:
            print(f"Done {i * batch_size}/{len(dataset)} in {time.time() - t0}")

    # Post-process raw scores to get a consistent 'higher is better' numpy array score
    for model_name, all_scores in results.items():
        all_scores = np.asarray(all_scores)
        # Flip raw scores as lower is better for those models
        if model_name in {"dock", "native_fp"}:
            all_scores = -all_scores
        results[model_name] = all_scores

    df = pd.DataFrame(results)
    df["smiles"] = smiles_list
    if do_mixing:
        # Normalize each methods outputs and mix methods together : best mix = 0.44, 0.39, 0.17
        def normalize(scores):
            out_scores = (scores - scores.min()) / (scores.max() - scores.min())
            return out_scores

        normalized_results = {
            model_name: normalize(result) for model_name, result in results.items()
        }
        mixed_scores = (
            mixing_coeffs[0] * normalized_results["dock"]
            + mixing_coeffs[1] * normalized_results["native_fp"]
            + mixing_coeffs[2] * normalized_results["is_native"]
        )
        df["mixed_score"] = mixed_scores
        if not dump_all:
            df = df[["smiles", "mixed_score"]]
    if out_path is not None:
        df.to_csv(out_path)
    return df


def do_inference(cif_path, residue_list, ligands_path, out_path, dump_all=False):
    """
    Run inference from files
    """
    # Get dgl graph with node expansion BFS
    dgl_graph = get_dgl_graph(cif_path, residue_list)
    print("Successfully built the graph")
    smiles_list = [s.lstrip().rstrip() for s in list(open(ligands_path).readlines())]
    print("Successfully parsed ligands, ready for inference")
    inference(
        dgl_graph=dgl_graph,
        smiles_list=smiles_list,
        out_path=out_path,
        dump_all=dump_all,
    )


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
