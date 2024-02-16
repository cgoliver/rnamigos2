import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dgl.dataloading import GraphDataLoader
from yaml import safe_load

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from rnamigos_dock.learning.loader import VirtualScreenDataset, get_systems
from rnamigos_dock.learning.models import Embedder, LigandGraphEncoder, LigandEncoder, Decoder, RNAmigosModel
from rnamigos_dock.post.virtual_screen import mean_active_rank, enrichment_factor, run_virtual_screen


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    '''
    Hardware settings
    '''

    with open(Path(cfg.saved_model_dir, 'config.yaml'), 'r') as f:
        params = safe_load(f)
        print(params['train'])

    # torch.multiprocessing.set_sharing_strategy('file_system')
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

    if cfg.custom_dir:
        test_systems = pd.DataFrame({'PDB_ID_POCKET': [Path(g).stem for g in os.listdir(cfg.data.pocket_graphs)]})
    else:
        test_systems = get_systems(target=params['train']['target'],
                                   rnamigos1_split=params['train']['rnamigos1_split'],
                                   use_rnamigos1_train=params['train']['use_rnamigos1_train'],
                                   use_rnamigos1_ligands=False,
                                   return_test=True)
    # Loader is asynchronous
    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }

    rows, raw_rows = [], []
    if cfg.decoys == 'all':
        decoys = ['chembl', 'pdb', 'pdb_chembl', 'decoy_finder']
    else:
        decoys = [cfg.decoys]

    for decoy_mode in decoys:
        pocket_path = cfg.data.pocket_graphs if cfg.custom_dir else params['data']['pocket_graphs']
        dataset = VirtualScreenDataset(pocket_path,
                                       cache_graphs=False,
                                       ligands_path=params['data']['ligand_db'],
                                       systems=test_systems,
                                       decoy_mode=decoy_mode,
                                       fp_type='MACCS',
                                       use_graphligs=params['model']['use_graphligs'],
                                       rognan=cfg.rognan,
                                       group_ligands=True)

        dataloader = GraphDataLoader(dataset=dataset, **loader_args)

        print('Created data loader')

        '''
        Experiment Setup
        '''
        lower_is_better = params['train']['target'] in ['dock', 'native_fp']
        efs, scores, status, pocket_names, all_smiles = run_virtual_screen(model,
                                                                           dataloader,
                                                                           metric=enrichment_factor if decoy_mode == 'robin' else mean_active_rank,
                                                                           lower_is_better=lower_is_better,
                                                                           )
        for pocket_id, score_list, status_list, smiles_list in zip(pocket_names, scores, status, all_smiles):
            for score, status, smiles in zip(score_list, status_list, smiles_list):
                raw_rows.append({'raw_score': score, 'is_active': status, 'pocket_id': pocket_id, 'smiles': smiles,
                                 'decoys': decoy_mode})

        for ef, score, pocket_id in zip(efs, scores, pocket_names):
            rows.append({
                'score': ef,
                'metric': 'EF' if decoy_mode == 'robin' else 'MAR',
                'decoys': decoy_mode,
                'pocket_id': pocket_id})
        print('Mean EF :', np.mean(efs))

    df = pd.DataFrame(rows)
    d = Path(cfg.result_dir, parents=True, exist_ok=True)
    df.to_csv(d / cfg.csv_name)

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(d / Path(cfg.csv_name.split(".")[0] + "_raw.csv"))


if __name__ == "__main__":
    main()
