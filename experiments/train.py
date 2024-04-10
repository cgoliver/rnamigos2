"""
We have 3 options for training modes (`train.target`):

* `dock`: predict the docking INTER score (regression)
* `is_native`: predict whether the given ligand is the native for the given pocket (binary classification)
* `native_fp`: given only a pocket, predict the native ligand's fingerprint. This is the RNAmigos1.0 setting (multi-label classification)`

Make sure to set the correct `train.loss` given the target you chose. Options:

* `l1`: L2 loss
* `l2`: L1 loss
* `bce`: Binary crossentropy
"""
import os
import sys
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

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos_dock.learning.loader import (get_systems, DockingDataset, IsNativeSampler, NativeFPSampler,
                                           RingCollater, VirtualScreenDataset)
from rnamigos_dock.learning import learn
from rnamigos_dock.learning.models import Embedder, LigandEncoder, LigandGraphEncoder, Decoder, RNAmigosModel
from rnamigos_dock.post.virtual_screen import mean_active_rank, run_virtual_screen
from rnamigos_dock.learning.utils import mkdirs

torch.multiprocessing.set_sharing_strategy('file_system')
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

    train_val_systems = get_systems(target=cfg.train.target,
                                    rnamigos1_split=cfg.train.rnamigos1_split,
                                    use_rnamigos1_train=cfg.train.use_rnamigos1_train,
                                    use_rnamigos1_ligands=cfg.train.use_rnamigos1_ligands,
                                    group_pockets=cfg.train.group_pockets,
                                    filter_robin=cfg.train.filter_robin)
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
                    'use_normalized_score': cfg.train.use_normalized_score,
                    'stretch_scores': cfg.train.stretch_scores,
                    'undirected': cfg.data.undirected}

    def split(train, frac=0.8, system_based=True):
        if system_based:
            train_names = train['PDB_ID_POCKET'].unique()
            train_names = np.random.choice(train_names, size=int(frac * len(train_names)), replace=False)
            train_systems = train[train['PDB_ID_POCKET'].isin(train_names)]
            validation_systems = train[~train['PDB_ID_POCKET'].isin(train_names)]
        else:
            train_systems, validation_systems = np.split(train.sample(frac=1, random_state=42),
                                                         [int(frac * len(train))])
        return train_systems, validation_systems

    train_systems, validation_systems = split(train_val_systems.copy(), frac=0.8, system_based=False)
    # This avoids having too many pockets in the VS validation
    _, vs_validation_systems = split(train_val_systems.copy(), frac=0.8, system_based=True)
    train_dataset = DockingDataset(systems=train_systems, use_rings=node_simfunc is not None, **dataset_args)
    validation_dataset = DockingDataset(systems=validation_systems, use_rings=False, **dataset_args)
    # These one cannot be a shared object
    if cfg.train.target == 'is_native':
        train_sampler = IsNativeSampler(train_systems, group_sampling=cfg.train.group_sample)
        validation_sampler = IsNativeSampler(validation_systems, group_sampling=cfg.train.group_sample)
    elif cfg.train.target == 'native_fp':
        train_sampler = NativeFPSampler(train_systems, group_sampling=cfg.train.group_sample)
        validation_sampler = NativeFPSampler(validation_systems, group_sampling=cfg.train.group_sample)
    else:
        train_sampler, validation_sampler = None, None

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

    vs_loader_args = {'shuffle': False,
                      'batch_size': 1,
                      'num_workers': 4,
                      'collate_fn': lambda x: x[0]
                      }

    val_vs_dataset = VirtualScreenDataset(cfg.data.pocket_graphs,
                                          decoy_mode='chembl',
                                          ligands_path=cfg.data.ligand_db,
                                          systems=vs_validation_systems,
                                          use_graphligs=cfg.model.use_graphligs,
                                          group_ligands=True)
    val_vs_loader = GraphDataLoader(dataset=val_vs_dataset, **vs_loader_args)
    test_vs_dataset = VirtualScreenDataset(cfg.data.pocket_graphs,
                                           decoy_mode='chembl',
                                           ligands_path=cfg.data.ligand_db,
                                           systems=test_systems,
                                           use_graphligs=cfg.model.use_graphligs,
                                           group_ligands=True)
    test_vs_loader = GraphDataLoader(dataset=test_vs_dataset, **vs_loader_args)
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
    writer = SummaryWriter(result_folder)
    print(f'Saving result in {result_folder}/{name}')
    OmegaConf.save(cfg, Path(result_folder, "config.yaml"))

    '''
    Run
    '''
    num_epochs = cfg.train.num_epochs
    save_path = Path(result_folder, 'model.pth')

    print("training...")
    _, best_model = learn.train_dock(model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     device=device,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     val_vs_loader=val_vs_loader,
                                     test_vs_loader=test_vs_loader,
                                     save_path=save_path,
                                     writer=writer,
                                     num_epochs=num_epochs,
                                     early_stop_threshold=cfg.train.early_stop,
                                     pretrain_weight=cfg.train.pretrain_weight,
                                     cfg=cfg,
                                     )

    logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
    logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")

    # %%%%%%%%%%%%%%%%%%%%%%%
    model = model.to('cpu')
    rows, raw_rows = [], []
    decoys = ['chembl', 'pdb', 'pdb_chembl', 'decoy_finder']
    for decoy_mode in decoys:
        pocket_path = cfg.data.pocket_graphs
        final_vs_dataset = VirtualScreenDataset(pocket_path,
                                                cache_graphs=False,
                                                ligands_path=cfg.data.ligand_db,
                                                systems=test_systems,
                                                decoy_mode=decoy_mode,
                                                fp_type='MACCS',
                                                use_graphligs=cfg.model.use_graphligs,
                                                group_ligands=False)
        dataloader = GraphDataLoader(dataset=final_vs_dataset, **vs_loader_args)

        print('Created data loader')

        '''
        Experiment Setup
        '''
        lower_is_better = cfg.train.target in ['dock', 'native_fp']
        efs, scores, status, pocket_names, all_smiles = run_virtual_screen(model,
                                                                           dataloader,
                                                                           metric=mean_active_rank,
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
    logger.info(f"{cfg.name} mean EF {np.mean(efs)}")

    d = Path("outputs/all", parents=True, exist_ok=True)
    base_name = Path(cfg.name).stem

    df = pd.DataFrame(rows)
    df.to_csv(d / (base_name + '.csv'))

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(d / (base_name + "_raw.csv"))


if __name__ == "__main__":
    main()
