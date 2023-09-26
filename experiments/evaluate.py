import os
import sys
import time
from pathlib import Path

import numpy as np
from dgl.dataloading import GraphDataLoader
from yaml import safe_load 

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from rnamigos_dock.learning.loader import VirtualScreenDataset, get_systems
from rnamigos_dock.learning.models import Embedder, LigandEncoder, Decoder, RNAmigosModel
from rnamigos_dock.post.virtual_screen import mean_active_rank, run_virtual_screen


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    '''
    Hardware settings
    '''

    with open(Path(cfg.saved_model_dir, 'config.yaml'), 'r') as f:
        params = safe_load(f)

    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_rnamigos1_train = False
    use_rnamigos1_ligands = False
    rnamigos1_split = 0
    test_systems = get_systems(target=cfg.train.target,
                               rnamigos1_split=cfg.train.rnamigos1_split,
                               use_rnamigos1_train=cfg.train.use_rnamigos1_train,
                               use_rnamigos1_ligands=cfg.train.use_rnamigos1_ligands,
                               return_test=True)

    dataset = VirtualScreenDataset(pockets_path=cfg.data.pocket_graphs,
                                   ligands_path=cfg.data.ligand_db,
                                   systems=test_systems,
                                   decoy_mode='pdb',
                                   fp_type='MACCS')


    # Loader is asynchronous
    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }
    dataloader = GraphDataLoader(dataset=dataset, **loader_args)

    print('Created data loader')

    '''
    Model loading
    '''
    
    rna_encoder = Embedder(in_dim=params['model']['encoder']['in_dim'],
                           hidden_dim=params['model']['encoder']['hidden_dim'],
                           num_hidden_layers=params['model']['encoder']['num_layers'],
                           batch_norm=params['model']['batch_norm'],
                           dropout=params['model']['dropout'],
                           num_bases=params['model']['encoder']['num_bases']
                           )

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


    print(params)
    state_dict = torch.load(Path(cfg.saved_model_dir, 'model.pth'))['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    model = model.to(device)

    print(f'Using {model.__class__} as model')

    '''
    Experiment Setup
    '''
    import time
    t0 = time.perf_counter()
    lower_is_better = params['train']['target'] in ['dock', 'native_fp']
    efs, inds = run_virtual_screen(model, dataloader, metric=mean_active_rank, lower_is_better=lower_is_better)
    print(efs)
    print('Mean EF :', np.mean(efs))
    print('Time :', time.perf_counter() - t0)


if __name__ == "__main__":
    main()
