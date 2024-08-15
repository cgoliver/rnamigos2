import os
import sys

import argparse
from pathlib import Path
from yaml import safe_load
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from rdkit import Chem
from omegaconf import OmegaConf

import torch

from rnaglib.utils import graph_io

from rnamigos.learning.ligand_encoding import MolGraphEncoder
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.graph_utils import load_rna_graph


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="results/trained_models/dock/dock_42")
    parser.add_argument("--ligand-cam", action="store_true", default=False,
                        help="Pass to get the CAM on the ligand side, else returns pocket CAM.")
    parser.add_argument("--layer", default=-1, help="Index of layer in conv model to use.")
    parser.add_argument("--pocket-dir", default="data/json_pockets_expanded")
    parser.add_argument("--pocket-id", default="1AJU_A_ARG_47", help="Pocket to analyze")
    parser.add_argument("--smiles", default=None, help="SMILES string to analyze")
    parser.add_argument("--mol2-path", default=None, help="Path to Mol2 file of ligand in PDB")
    parser.add_argument("--raw-values", action="store_true", default=False,
                        help="Use raw CAM values, othewrise max/min scaled.")
    parser.add_argument("--outpath-pocket", type=str, default="./pocket.cxc",
                        help="Path to write Chimera command file.")
    parser.add_argument("--outpath-lig", type=str, default="./lig.cxc", help="Path to write Chimera command file.")
    return parser.parse_args()


def grad_CAM(model, layer, pocket, lig, scaled=True, ligand_cam=True, save_csv=None):
    inner_layer_output = {}

    def get_inner_layer_output(module, input, output):
        inner_layer_output["output"] = output

    if ligand_cam:
        model.lig_encoder.encoder.layers[layer].register_forward_hook(get_inner_layer_output)
    else:
        model.encoder.layers[layer].register_forward_hook(get_inner_layer_output)

    score, embs = model(pocket, lig)

    model_out = score[0][0]
    conv_out = inner_layer_output['output']

    grads = torch.autograd.grad(model_out, conv_out, retain_graph=True, allow_unused=False)[0]
    grads = (conv_out > 0).float() * (grads > 0).float() * grads

    weights = grads.mean(dim=0)
    # importance for each node
    cam = (conv_out.detach() * weights).sum(dim=1)
    if scaled:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    if not save_csv is None:
        if ligand_cam:
            graph = pocket_graph
            suffix = "_ligand"
        else:
            graph = ligand_graph
            suffix = "_pocket"

        rows = []
        for node, val in zip(sorted(graph.nodes()), cam):
            rows.append({"id": node.item(), "cam": val.item()})

        pd.DataFrame(rows).to_csv(save_csv)
    return cam


def highlight_pdb_pocket(pocket_path, cam, outpath):
    """ Generate chimera command file to place a colormapped surface mesh around
    atoms and residues using the CAM values.
    """
    pocket_graph = graph_io.load_json(pocket_path)
    lig_chain, lig_name, lig_pos = os.path.basename(pocket_path).rstrip(".json").split("_")[1:]
    colormap = matplotlib.colormaps['bwr']
    keep_ligs = []
    with open(outpath, "w") as cmds:
        for node, val in zip(sorted(pocket_graph.nodes()), cam):
            color = colors.to_hex(colormap(val.item()))
            pdbid, chain, pos = node.split(".")
            cmds.write(f"color /{chain}:{pos} {color}\n")
            keep_ligs.append(f"{chain}:{pos}")

        cmds.write(f"delete ~/{','.join(keep_ligs)},{lig_chain}:{lig_pos}\n")
        cmds.write("nucleotides tube/slab shape box \n")
        cmds.write("show cartoons \n")
        cmds.write("set bgColor white \n")
        cmds.write("view")


def highlight_pdb_ligand(lig_graph, cam, outpath):
    """ Generate chimera command file to place a colormapped surface mesh around
    atoms and residues using the CAM values.
    """
    lig_chain, lig_name, lig_pos = os.path.basename(pocket_path).rstrip(".json").split("_")[1:]
    colormap = matplotlib.colormaps['bwr']
    keep_ligs = []
    with open(outpath, "w") as cmds:
        for (node, data), val in zip(sorted(lig_graph.nodes(data=True)), cam):
            color = colors.to_hex(colormap(val.item()))
            cmds.write(f"color /{lig_chain}:{lig_pos}@{data['atom_name']} {color}\n")
        cmds.write("hb ligand restrict cross color yellow radius 0.2")


if __name__ == "__main__":
    args = cline()

    with open(Path(args.saved_model_dir, 'config.yaml'), 'r') as f:
        params = safe_load(f)
        cfg_load = OmegaConf.create(params)
    model = get_model_from_dirpath(args.saved_model_dir)

    pocket_path = os.path.join(args.pocket_dir, f"{args.pocket_id}.json")
    pocket_graph, _ = load_rna_graph(rna_path=pocket_path,
                                     undirected=False,
                                     use_rings=False)

    if args.smiles:
        ligand_graph = MolGraphEncoder().smiles_to_graph_one(args.smiles)
    else:
        ligand_graph, ligand_graph_nx = MolGraphEncoder().mol2_to_graph_one(args.mol2_path)

    cam_pocket = grad_CAM(model, args.layer, pocket_graph, ligand_graph, ligand_cam=False, scaled=not args.raw_values,
                          save_csv=f"outputs/CAM/{args.pocket_id}_pocket_cam.csv")
    cam_ligand = grad_CAM(model, args.layer, pocket_graph, ligand_graph, ligand_cam=True, scaled=not args.raw_values,
                          save_csv=f"outputs/CAM/{args.pocket_id}_ligand_cam.csv")

    highlight_pdb_pocket(pocket_path, cam_pocket, args.outpath_pocket)
    highlight_pdb_ligand(ligand_graph_nx, cam_ligand, args.outpath_lig)
    pass
