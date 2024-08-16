"""
Examples of command lines to train a model are available in scripts_run/train.sh
"""
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

import numpy as np
import pathlib
import pandas as pd
import torch
from torch.utils import tensorboard

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataset import get_dataset, train_val_split
from rnamigos.learning.dataloader import get_loader, get_vs_loader

from rnamigos.learning import learn
from rnamigos.learning.models import cfg_to_model
from rnamigos.utils.learning_utils import mkdirs, setup_device, setup_seed
from rnamigos.utils.virtual_screen import get_efs

from scripts_fig.plot_utils import group_df

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    # General config
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    setup_seed(cfg.train.seed)
    device = setup_device(cfg.device)

    # Get model
    model = cfg_to_model(cfg)
    model = model.to(device)

    # Dataset/loaders creation and splitting.

    # Systems are basically lists of all pocket/pair/labels to consider. We then split the train_val systems.
    train_val_systems = get_systems_from_cfg(cfg, return_test=False)
    test_systems = get_systems_from_cfg(cfg, return_test=True)
    train_systems, validation_systems = train_val_split(train_val_systems.copy(), system_based=False)
    # We then create datasets, potentially additionally returning rings and dataloaders.
    # Dataloader creation is a bit tricky as it involves custom Samplers and Collaters that depend on the task at hand
    train_dataset = get_dataset(cfg, train_systems, training=True)
    validation_dataset = get_dataset(cfg, validation_systems, training=False)
    train_loader = get_loader(cfg, train_dataset, train_systems, training=True)
    val_loader = get_loader(cfg, validation_dataset, validation_systems, training=False)

    # In addition to those 'classical' loaders, we also create ones dedicated to VS validation.
    # Splitting for VS validation is based on systems, to avoid having too many pockets
    _, vs_validation_systems = train_val_split(train_val_systems.copy(), system_based=True)
    val_vs_loader = get_vs_loader(systems=vs_validation_systems,
                                  decoy_mode=cfg.train.vs_decoy_mode,
                                  reps_only=True,
                                  cfg=cfg)
    test_vs_loader = get_vs_loader(systems=test_systems,
                                   decoy_mode=cfg.train.vs_decoy_mode,
                                   reps_only=True,
                                   cfg=cfg)

    # Maybe monitor rognan's performance ?
    val_vs_loader_rognan = None
    if cfg.train.do_rognan:
        val_vs_loader_rognan = get_vs_loader(systems=vs_validation_systems,
                                             decoy_mode=cfg.train.vs_decoy_mode,
                                             reps_only=True,
                                             rognan=True,
                                             cfg=cfg)

    # Optimizer instantiation
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
    writer = tensorboard.SummaryWriter(save_path)
    print(f'Saving result in {save_path}/{name}')
    OmegaConf.save(cfg, pathlib.Path(save_path, "config.yaml"))

    # Training
    _, best_model = learn.train_dock(model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     device=device,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     val_vs_loader=val_vs_loader,
                                     val_vs_loader_rognan=val_vs_loader_rognan,
                                     test_vs_loader=test_vs_loader,
                                     save_path=save_name,
                                     writer=writer,
                                     num_epochs=cfg.train.num_epochs,
                                     early_stop_threshold=cfg.train.early_stop,
                                     pretrain_weight=cfg.train.pretrain_weight,
                                     cfg=cfg)

    # Final VS validation on each decoy set
    logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
    logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")
    best_model = best_model.to('cpu')
    rows, raw_rows = [], []
    decoys = ['chembl', 'pdb', 'pdb_chembl', 'decoy_finder']
    for decoy_mode in decoys:
        dataloader = get_vs_loader(systems=test_systems, decoy_mode=decoy_mode, cfg=cfg)
        decoy_rows, decoys_raw_rows = get_efs(model=best_model, dataloader=dataloader, decoy_mode=decoy_mode,
                                              cfg=cfg, verbose=True)
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

    df_chembl = df.loc[df['decoys'] == 'chembl']
    print(f"{cfg.name} Mean EF on chembl: {np.mean(df_chembl['score'].values)}")
    df_chembl = group_df(df_chembl)
    print(f"{cfg.name} Mean grouped EF on chembl: {np.mean(df_chembl['score'].values)}")

    df_pdbchembl = df.loc[df['decoys'] == 'pdb_chembl']
    print(f"{cfg.name} Mean EF on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")
    df_pdbchembl = group_df(df_pdbchembl)
    print(f"{cfg.name} Mean grouped EF on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")


if __name__ == "__main__":
    main()
