from pathlib import Path

import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader
from yaml import safe_load

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from rnaglib.data import graph_from_pdbid
from rnaglib.prepare_data import fr3d_to_graph 

from rnamigos_dock.learning.loader import VirtualScreenDataset, get_systems
from rnamigos_dock.learning.models import Embedder, LigandGraphEncoder, LigandEncoder, Decoder, RNAmigosModel


@hydra.main(version_base=None, config_path="../conf", config_name="inference")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    '''
    Hardware settings
    '''

    ### MODEL LODADING

    with open(Path(cfg.saved_model_dir, 'config.yaml'), 'r') as f:
        params = safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rna_encoder = Embedder(in_dim=params['model']['encoder']['in_dim'],
                           hidden_dim=params['model']['encoder']['hidden_dim'],
                           num_hidden_layers=params['model']['encoder']['num_layers'],
                           batch_norm=params['model']['batch_norm'],
                           dropout=params['model']['dropout'],
                           num_bases=params['model']['encoder']['num_bases']
                           )

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

    state_dict = torch.load(Path(cfg.saved_model_dir, 'model.pth'), map_location=device)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    model = model.to(device)

    ### DATA PREP

    if cfg.cif_path is None:
        # load prebuilt graph and use cfg.residue_list
        test_systems = pd.DataFrame({'PDB_ID_POCKET': [Path(cfg.pdbid)]})
        graph = graph_from_pdbid(cfg.pdbid)
        graph = graph.subgraph([f"{cfg.pdbid.lower()}.{res}" for res in cfg.reslist.split(";")]).copy()
    else:
        # convert cif to graph
        test_systems = pd.DataFrame({'PDB_ID_POCKET': [Path(cfg.cif_path).stem]})
        graph = fr3d_to_graph(cfg.cif_path)
        if cfg.residue_list is not None:
            # subset cif with given reslist
            graph = graph.subgraph([f"{cfg.pdbid.lower()}.{res}" for res in cfg.reslist.split(";")]).copy()
            pass

    
    # Loader is asynchronous
    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }

    smiles_list =  list(open(cfg.ligands_path).readlines())
    dataset = InferenceDataset(graph,
                               smiles_list,
                               systems=test_systems,
                               use_graphligs=params['model']['use_graphligs'],
                               )

    dataloader = GraphDataLoader(dataset=dataset, **loader_args)

    for pocket_graph, ligands in dataloader:
        model = model.to('cpu')
        scores = list(model.predict_ligands(pocket_graph,
                                            ligands,
                                            ).squeeze().cpu().numpy())

    with open(cfg.out_path, 'w') as out:
        for score, smiles in zip(scores, smiles_list):
            out.write(f"{smiles} {score}\n")
if __name__ == "__main__":
    main()
