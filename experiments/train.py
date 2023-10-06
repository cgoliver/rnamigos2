from pathlib import Path

import numpy.random
from dgl.dataloading import GraphDataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from rnaglib.kernels.node_sim import SimFunctionNode
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

from rnamigos_dock.learning.loader import get_systems, DockingDataset, NativeSampler, RingCollater, VirtualScreenDataset
from rnamigos_dock.learning import learn
from rnamigos_dock.learning.models import Embedder, LigandEncoder, LigandGraphEncoder, Decoder, RNAmigosModel
from rnamigos_dock.post.virtual_screen import mean_active_rank, run_virtual_screen
from rnamigos_dock.learning.utils import mkdirs

torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')

    if cfg.train.seed > 0:
        numpy.random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    '''
    Hardware settings
    '''

    # torch.multiprocessing.set_sharing_strategy('file_system')
    if torch.cuda.is_available():
        if cfg.device != 'cpu':
            try:
                gpu_number = int(cfg.device)
            except:
                gpu_number = 0
            device = f'cuda:{gpu_number}'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
        print("No GPU found, running on the CPU")

    '''
    Dataloader creation
    '''

    train_systems = get_systems(target=cfg.train.target,
                                rnamigos1_split=cfg.train.rnamigos1_split,
                                use_rnamigos1_train=cfg.train.use_rnamigos1_train,
                                use_rnamigos1_ligands=cfg.train.use_rnamigos1_ligands)
    test_systems = get_systems(target=cfg.train.target,
                               rnamigos1_split=cfg.train.rnamigos1_split,
                               use_rnamigos1_train=cfg.train.use_rnamigos1_train,
                               use_rnamigos1_ligands=cfg.train.use_rnamigos1_ligands,
                               return_test=True)

    if cfg.train.simfunc not in {'R_iso', 'R_1', 'hungarian'}:
        node_simfunc = None
    else:
        node_simfunc = SimFunctionNode(cfg.train.simfunc, depth=cfg.train.simfunc_depth)

    dataset_args = {'pockets_path': cfg.data.pocket_graphs,
                    'target': cfg.train.target,
                    'shuffle': cfg.train.shuffle,
                    'seed': cfg.train.seed,
                    'debug': cfg.debug,
                    'use_graphligs': cfg.model.use_graphligs,
                    'undirected': cfg.data.undirected}

    train_systems, validation_systems = np.split(train_systems.sample(frac=1, random_state=42),
                                                 [int(.8 * len(train_systems))])
    train_dataset = DockingDataset(systems=train_systems, use_rings=node_simfunc is not None, **dataset_args)
    validation_dataset = DockingDataset(systems=validation_systems, use_rings=False, **dataset_args)
    # These one cannot be a shared object
    train_sampler = NativeSampler(train_systems) if cfg.train.target == 'is_native' else None
    validation_sampler = NativeSampler(validation_systems) if cfg.train.target == 'is_native' else None
    # Cannot collect empty rings...
    train_collater = RingCollater(node_simfunc=node_simfunc, max_size_kernel=cfg.train.max_kernel)
    val_collater = RingCollater(node_simfunc=None)
    loader_args = {'shuffle': train_sampler is None,
                   'batch_size': cfg.train.batch_size,
                   'num_workers': cfg.train.num_workers}

    train_loader = GraphDataLoader(dataset=train_dataset,
                                   sampler=train_sampler,
                                   collate_fn=train_collater.collate,
                                   **loader_args)
    val_loader = GraphDataLoader(dataset=validation_dataset,
                                 sampler=validation_sampler,
                                 collate_fn=val_collater.collate,
                                 **loader_args)

    print('Created data loader')

    '''
    Model loading
    '''

    print("creating model")
    rna_encoder = Embedder(in_dim=cfg.model.encoder.in_dim,
                           hidden_dim=cfg.model.encoder.hidden_dim,
                           num_hidden_layers=cfg.model.encoder.num_layers,
                           batch_norm=cfg.model.batch_norm,
                           dropout=cfg.model.dropout,
                           num_bases=cfg.model.encoder.num_bases
                           )
    if cfg.model.use_pretrained:
        print(">>> Using pretrained weights")
        rna_encoder.from_pretrained(cfg.model.pretrained_path)
    rna_encoder.subset_pocket_nodes = True

    if cfg.model.use_graphligs:
        graphlig_cfg = cfg.model.graphlig_encoder
        if graphlig_cfg.use_pretrained:
            lig_encoder = LigandGraphEncoder.from_pretrained("pretrained/optimol")
        else:
            lig_encoder = LigandGraphEncoder(features_dim=graphlig_cfg.features_dim,
                                             l_size=graphlig_cfg.l_size,
                                             gcn_hdim=graphlig_cfg.gcn_hdim,
                                             gcn_layers=graphlig_cfg.gcn_layers,
                                             batch_norm=cfg.model.batch_norm)
    else:
        lig_encoder = LigandEncoder(in_dim=cfg.model.lig_encoder.in_dim,
                                    hidden_dim=cfg.model.lig_encoder.hidden_dim,
                                    num_hidden_layers=cfg.model.lig_encoder.num_layers,
                                    batch_norm=cfg.model.batch_norm,
                                    dropout=cfg.model.dropout)

    decoder = Decoder(in_dim=cfg.model.decoder.in_dim,
                      out_dim=cfg.model.decoder.out_dim,
                      hidden_dim=cfg.model.decoder.hidden_dim,
                      num_layers=cfg.model.decoder.num_layers,
                      activation=cfg.model.decoder.activation,
                      batch_norm=cfg.model.batch_norm,
                      dropout=cfg.model.dropout)

    model = RNAmigosModel(encoder=rna_encoder,
                          decoder=decoder,
                          lig_encoder=lig_encoder if cfg.train.target in ['dock', 'is_native'] else None,
                          pool=cfg.model.pool,
                          pool_dim=cfg.model.encoder.hidden_dim
                          )

    model = model.to(device)

    print(f'Using {model.__class__} as model')

    '''
    Optimizer instanciation
    '''

    if cfg.train.loss == 'l2':
        criterion = torch.nn.MSELoss()
    if cfg.train.loss == 'l1':
        criterion = torch.nn.L1Loss()
    if cfg.train.loss == 'bce':
        criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    '''
    Experiment Setup
    '''

    name = f"{cfg.name}"
    print(name)
    result_folder, save_path = mkdirs(name, prefix=cfg.train.target)
    print(save_path)
    writer = SummaryWriter(result_folder)
    print(f'Saving result in {result_folder}/{name}')
    OmegaConf.save(cfg, Path(result_folder, "config.yaml"))

    '''
    Run
    '''
    num_epochs = cfg.train.num_epochs
    save_path = Path(result_folder, 'model.pth')
    # model = model.from_pretrained(save_path)
    # model.eval()

    print("training...")
    learn.train_dock(model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     device=device,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     save_path=save_path,
                     writer=writer,
                     num_epochs=num_epochs,
                     early_stop_threshold=cfg.train.early_stop,
                     pretrain_weight=cfg.train.pretrain_weight)

    logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
    logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")

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

    lower_is_better = cfg.train.target in ['dock', 'native_fp']
    efs, inds, scores, pocket_ids = run_virtual_screen(model, dataloader, metric=mean_active_rank,
                                                       lower_is_better=lower_is_better)

    df = pd.DataFrame({'ef': efs, 'inds': inds})
    df.to_csv(Path(result_folder, 'ef.csv'))
    logger.info(f"{cfg.name} mean EF {np.mean(efs)}")


if __name__ == "__main__":
    main()
