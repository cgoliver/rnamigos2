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

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from dgl.dataloading import GraphDataLoader
import numpy as np
import pathlib
import pandas as pd
from rnaglib.kernels.node_sim import SimFunctionNode
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos_dock.learning.dataset import get_systems_from_cfg
from rnamigos_dock.learning.dataset import DockingDataset, train_val_split
from rnamigos_dock.learning.dataloader import IsNativeSampler, NativeFPSampler, RingCollater, get_vs_loader

from rnamigos_dock.learning import learn
from rnamigos_dock.learning.models import cfg_to_model
from rnamigos_dock.learning.utils import mkdirs, setup_device, setup_seed
from rnamigos_dock.post.virtual_screen import get_efs

from fig_scripts.plot_utils import group_df

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    setup_seed(cfg.train.seed)
    device = setup_device(cfg.device)

    '''
    Dataloader creation
    '''
    train_val_systems = get_systems_from_cfg(cfg)
    test_systems = get_systems_from_cfg(cfg, return_test=True)

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

    train_systems, validation_systems = train_val_split(train_val_systems.copy(), frac=0.8, system_based=False)
    # This avoids having too many pockets in the VS validation
    _, vs_validation_systems = train_val_split(train_val_systems.copy(), frac=0.8, system_based=True)
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

    val_vs_loader = get_vs_loader(systems=vs_validation_systems,
                                  decoy_mode=cfg.decoy_mode,
                                  reps_only=True,
                                  cfg=cfg)
    test_vs_loader = get_vs_loader(systems=vs_validation_systems,
                                   decoy_mode=cfg.decoy_mode,
                                   reps_only=True,
                                   cfg=cfg)

    # Model loading
    model = cfg_to_model(cfg)
    model = model.to(device)

    # Optimizer instanciation
    if cfg.train.loss == 'l2':
        criterion = torch.nn.MSELoss()
    elif cfg.train.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif cfg.train.loss == 'bce':
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError(f'Unsupported loss function: {cfg.train.loss}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Experiment Setup
    name = f"{cfg.name}"
    save_path, save_name = mkdirs(name, prefix=cfg.train.target)
    writer = torch.utils.tensorboard.SummaryWriter(save_path)
    print(f'Saving result in {save_path}/{name}')
    OmegaConf.save(cfg, pathlib.Path(save_path, "config.yaml"))

    # Run
    print("Training...")
    _, best_model = learn.train_dock(model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     device=device,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     val_vs_loader=val_vs_loader,
                                     test_vs_loader=test_vs_loader,
                                     save_path=save_name,
                                     writer=writer,
                                     num_epochs=cfg.train.num_epochs,
                                     early_stop_threshold=cfg.train.early_stop,
                                     pretrain_weight=cfg.train.pretrain_weight,
                                     cfg=cfg)

    # Final VS validation + file dumping
    logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
    logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")
    best_model = best_model.to('cpu')
    rows, raw_rows = [], []
    decoys = ['chembl', 'pdb', 'pdb_chembl', 'decoy_finder']
    for decoy_mode in decoys:
        dataloader = get_vs_loader(systems=test_systems, decoy_mode=decoy_mode, cfg=cfg)

        # Experiment Setup
        decoy_rows, decoys_raw_rows = get_efs(model=best_model, dataloader=dataloader, decoy_mode=decoy_mode, cfg=cfg,
                                              verbose=True)
        rows += decoy_rows
        raw_rows += decoys_raw_rows

    # Make it a df
    df = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows)

    # Dump csvs
    d = pathlib.Path(cfg.result_dir, parents=True, exist_ok=True)
    base_name = pathlib.Path(cfg.name).stem
    df.to_csv(d / (base_name + '.csv'))
    df_raw.to_csv(d / (base_name + "_raw.csv"))

    df = df.loc[df['decoys'] == 'chembl']
    print(f"{cfg.name} Mean EF : {np.mean(df['score'].values)}")
    df = group_df(df)
    print(f"{cfg.name} Mean grouped EF : {np.mean(df['score'].values)}")


if __name__ == "__main__":
    main()
