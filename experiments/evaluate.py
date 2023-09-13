import os
import sys
import time

import numpy as np
from dgl.dataloading import GraphDataLoader

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

    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_systems = get_systems(target='is_native', split='TEST')
    dataset = VirtualScreenDataset(pockets_path=cfg.data.pocket_graphs,
                                   ligands_path=cfg.data.ligand_db,
                                   systems=test_systems,
                                   edge_types=cfg.tokens.edge_types,
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
    rna_encoder = Embedder(in_dim=cfg.model.encoder.in_dim,
                           hidden_dim=cfg.model.encoder.hidden_dim,
                           num_hidden_layers=cfg.model.encoder.num_layers)

    lig_encoder = LigandEncoder(in_dim=cfg.model.lig_encoder.in_dim,
                                hidden_dim=cfg.model.lig_encoder.hidden_dim,
                                num_hidden_layers=cfg.model.lig_encoder.num_layers)

    decoder = Decoder(in_dim=cfg.model.decoder.in_dim,
                      out_dim=cfg.model.decoder.out_dim,
                      hidden_dim=cfg.model.decoder.hidden_dim,
                      num_layers=cfg.model.decoder.num_layers)

    model = RNAmigosModel(encoder=rna_encoder,
                          decoder=decoder,
                          lig_encoder=lig_encoder if cfg.train.target in ['dock', 'is_native'] else None,
                          pool=cfg.model.pool)

    if cfg.model.use_pretrained:
        model.from_pretrained(cfg.model.pretrained_path)

    model = model.to(device)

    print(f'Using {model.__class__} as model')

    '''
    Experiment Setup
    '''
    import time
    t0 = time.perf_counter()
    lower_is_better = cfg.train.target in ['dock', 'native_fp']
    efs = run_virtual_screen(model, dataloader, metric=mean_active_rank, lower_is_better=lower_is_better)
    print(efs)
    print('Mean EF :', np.mean(efs))
    print('Time :', time.perf_counter() - t0)


if __name__ == "__main__":
    main()
