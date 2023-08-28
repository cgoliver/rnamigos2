import argparse
import os, sys
import pickle
import copy
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import hydra

import rnamigos_dock.learning.learn as learn
from rnamigos_dock.learning.loader import DockingDataset 
from rnamigos_dock.learning.loader import Loader
from rnamigos_dock.learning.models import RNAEncoder, LigandEncoder, Decoder, Model
from rnamigos_dock.learning.utils import mkdirs


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    if cfg.train.seed > 0:
        torch.manual_seed(cfg.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    '''
    Hardware settings
    '''

    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # This is to create an appropriate number of workers, but works too with cpu
    if cfg.train.parallel:
        used_gpus_count = torch.cuda.device_count()
    else:
        used_gpus_count = 1

    print(f'Using {used_gpus_count} GPUs')

    '''
    Dataloader creation
    '''


    dataset = DockingDataset(annotated_path=cfg.data.train_graphs,
                             shuffle=cfg.train.shuffle,
                             seed=cfg.train.seed,
                             nuc_types=cfg.tokens.nuc_types,
                             edge_types=cfg.tokens.edge_types,
                             target=cfg.train.target,
                             debug=cfg.debug
                             )

    loader = Loader(dataset,
                    batch_size=cfg.train.batch_size, 
                    num_workers=cfg.train.num_workers,
                    )

    print('Created data loader')

    '''
    Model loading
    '''

    print("Loading data...")

    train_loader, test_loader = loader.get_data()

    print("Loaded data")

    print("creating model")
    rna_encoder = RNAEncoder(in_dim=cfg.model.encoder.in_dim,
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
    Optimizer instanciation
    '''

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    '''
    Experiment Setup
    '''
    
    name = f"{cfg.name}"
    print(name)
    result_folder, save_path = mkdirs(name)
    print(save_path)
    writer = SummaryWriter(result_folder)
    print(f'Saving result in {result_folder}/{name}')


    
    all_graphs = np.array(test_loader.dataset.dataset.all_graphs)
    test_inds = test_loader.dataset.indices
    train_inds = train_loader.dataset.indices

    # pickle.dump(({'test': all_graphs[test_inds], 'train': all_graphs[train_inds]}),
    #                open(os.path.join(result_folder, f'splits_{k}.p'), 'wb'))

    '''
    Run
    '''
    num_epochs = cfg.train.num_epochs

    print("training...")
    learn.train_dock(model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     device=cfg.device,
                     train_loader=train_loader,
                     test_loader=test_loader,
                     save_path=save_path,
                     writer=writer,
                     num_epochs=num_epochs,
                     early_stop_threshold=cfg.train.early_stop)
        
if __name__ == "__main__":
    main()
