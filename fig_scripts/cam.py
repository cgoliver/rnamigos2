import argparse
import os
import sys
import torch
from pathlib import Path
from yaml import safe_load
from dgl.dataloading import GraphDataLoader
from torchvision.models.feature_extraction import create_feature_extractor

from rnamigos_dock.learning.ligand_encoding import MolGraphEncoder
from rnamigos_dock.learning.models import Embedder, LigandGraphEncoder, LigandEncoder, Decoder, RNAmigosModel
from rnamigos_dock.tools.graph_utils import load_rna_graph

def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="saved_models/paper_dock")
    parser.add_argument("--ligand-cam", action="store_true", default=False, help="Pass to get the CAM on the ligand side, else returns pocket CAM.")
    parser.add_argument("--layer", default=-1, type=int, help="Index of layer to analyze.")
    parser.add_argument("--pocket_dir", default="data/json_pockets_expanded")
    parser.add_argument("--pocket_id", default="1AJU_A_ARG_47", help="Pocket to analyze")
    parser.add_argument("--smiles", default="OC(=O)[C@H](CCCNC(=[NH2+])N)N", help="SMILES string to analyze")
    parser.add_argument("--raw-values", action="store_true", default=False, help="Use raw CAM values, othewrise max/min scaled.")
    return parser.parse_args()

def load_params(saved_model_dir):
    with open(Path(saved_model_dir, 'config.yaml'), 'r') as f:
        params = safe_load(f)
    return params


def load_model(params, saved_model_dir):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    rna_encoder = Embedder(in_dim=params['model']['encoder']['in_dim'],
                           hidden_dim=params['model']['encoder']['hidden_dim'],
                           num_hidden_layers=params['model']['encoder']['num_layers'],
                           batch_norm=params['model']['batch_norm'],
                           dropout=params['model']['dropout'],
                           num_bases=params['model']['encoder']['num_bases']
                           )

    print(params)
    if params['model']['use_graphligs']:
        graphlig_cfg = params['model']['graphlig_encoder']
        # For optimol compatibility.
        if graphlig_cfg['use_pretrained']:
            lig_encoder = LigandGraphEncoder(features_dim=16,
                                             l_size=56,
                                             num_rels=4,
                                             gcn_hdim=32,
                                             gcn_layers=3,
                                             batch_norm=False,
                                             cut_embeddings=True)
        else:
            lig_encoder = LigandGraphEncoder(features_dim=graphlig_cfg['features_dim'],
                                             l_size=graphlig_cfg['l_size'],
                                             gcn_hdim=graphlig_cfg['gcn_hdim'],
                                             gcn_layers=graphlig_cfg['gcn_layers'],
                                             batch_norm=params['model']['batch_norm'])

    else:
        lig_encoder = LigandEncoder(in_dim=params['model']['lig_encoder']['in_dim'],
                                    hidden_dim=params['model']['lig_encoder']['hidden_dim'],
                                    num_hidden_layers=params['model']['lig_encoder']['num_layers'],
                                    batch_norm=params['model']['batch_norm'],
                                    dropout=params['model']['dropout'])

    decoder = Decoder(dropout=params['model']['dropout'],
                      batch_norm=params['model']['batch_norm'],
                      **params['model']['decoder']
                      )

    model = RNAmigosModel(encoder=rna_encoder,
                          decoder=decoder,
                          lig_encoder=lig_encoder if params['train']['target'] in ['dock', 'is_native'] else None,
                          pool=params['model']['pool'],
                          pool_dim=params['model']['encoder']['hidden_dim']
                          )



    # load model
    state_dict = torch.load(Path(saved_model_dir, 'model.pth'), map_location='cpu')['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    for name,p in model.named_parameters():
        print(name, p.shape)


    return model


def grad_CAM(model, layer, pocket, lig, scaled=True, ligand_cam=True):

    inner_layer_output = {}
    def get_inner_layer_output(module, input, output):
        inner_layer_output["output"] = output

    if ligand_cam:
        model.lig_encoder.encoder.layers[-1].register_forward_hook(get_inner_layer_output)
    else:
        model.encoder.layers[-1].register_forward_hook(get_inner_layer_output)

    score, embs = model(pocket, lig)

    print(inner_layer_output)
    model_out = score[0][0]
    model_out.backward(retain_graph=True)

    conv_out = inner_layer_output['output']

    grads = torch.autograd.grad(model_out, conv_out, retain_graph=True, allow_unused=False)[0]
    grads = (conv_out > 0).float() * (grads > 0).float() * grads

    weights = grads.mean(dim=0)
    # importance for each node
    cam = (conv_out.detach() * weights).sum(dim=1)
    if scaled:
        cam = (cam - cam.min())/(cam.max() - cam.min())

    return cam

if __name__ == "__main__":
    args = cline()
    params = load_params(args.model_dir)
    model = load_model(params, args.model_dir)
    pocket_graph,_ = load_rna_graph(rna_path=os.path.join(args.pocket_dir, f"{args.pocket_id}.json"),
                                                   undirected=False,
                                                   use_rings=False)

    ligand_graph = MolGraphEncoder().smiles_to_graph_one(args.smiles)
    grad_CAM(model, args.layer, pocket_graph, ligand_graph, ligand_cam=args.ligand_cam, scaled=not args.raw_values)
    pass
