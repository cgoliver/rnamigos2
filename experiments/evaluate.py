import os, sys

from omegaconf import DictConfig, OmegaConf
import hydra
import torch

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem



from rnamigos_dock.learning.loader import VirtualScreenDataset 
from rnamigos_dock.learning.loader import Loader
from rnamigos_dock.learning.models import Embedder, LigandEncoder, Decoder, Model
from rnamigos_dock.learning.utils import mkdirs
from rnamigos_dock.post.virtual_screen import mean_active_rank 
from rnamigos_dock.post.virtual_screen import enrichment_factor 
from rnamigos_dock.post.virtual_screen import run_virtual_screen


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    '''
    Hardware settings
    '''

    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = VirtualScreenDataset(cfg.data.test_graphs, 
                                   cfg.data.ligand_db,
                                   nuc_types=cfg.tokens.nuc_types,
                                   edge_types=cfg.tokens.edge_types,
                                   )

    print('Created data loader')

    '''
    Model loading
    '''

    print("Loaded data")

    print("creating model")
    rna_encoder = Embedder(in_dim=cfg.model.encoder.in_dim,
                             hidden_dim=cfg.model.encoder.hidden_dim,
                             num_hidden_layers=cfg.model.encoder.num_layers,
                             )

    lig_encoder = LigandEncoder(in_dim=cfg.model.lig_encoder.in_dim,
                                hidden_dim=cfg.model.lig_encoder.hidden_dim,
                                num_hidden_layers=cfg.model.lig_encoder.num_layers)

    decoder = Decoder(in_dim=cfg.model.decoder.in_dim,
                      out_dim=cfg.model.decoder.out_dim,
                      hidden_dim=cfg.model.decoder.hidden_dim,
                      num_layers=cfg.model.decoder.num_layers)

    model = Model(encoder=rna_encoder,
                  decoder=decoder,
                  lig_encoder=lig_encoder,
                  pool=cfg.model.pool,
                  )

    if cfg.model.use_pretrained:
        model.from_pretrained(cfg.model.pretrained_path)

    model = model.to(device)

    print(f'Using {model.__class__} as model')

    '''
    Experiment Setup
    '''

    efs = run_virtual_screen(model, dataset, metric=mean_active_rank)
    print(efs)
    
        
if __name__ == "__main__":
    main()
