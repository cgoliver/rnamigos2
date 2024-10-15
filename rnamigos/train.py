"""
Examples of command lines to train a model are available in scripts_run/train.sh
"""

import os
import sys
import json
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from joblib import Parallel, delayed
from loguru import logger
import optuna

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
from rnamigos.utils.virtual_screen import get_efs, enrichment_factor
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.learning.models import get_model_from_dirpath

from scripts_run.robin_inference import robin_eval
from scripts_fig.plot_utils import group_df

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

POCKET_PATH = "data/json_pockets_expanded"




def get_loaders(cfg, tune=False, trial=None):
    # Systems are basically lists of all pocket/pair/labels to consider. We then split the train_val systems.
    train_val_systems = get_systems_from_cfg(cfg, return_test=False)
    test_systems = get_systems_from_cfg(cfg, return_test=True)
    train_systems, validation_systems = train_val_split(
        train_val_systems.copy(), system_based=False
    )
    # We then create datasets, potentially additionally returning rings and dataloaders.
    # Dataloader creation is a bit tricky as it involves custom Samplers and Collaters that depend on the task at hand
    train_dataset = get_dataset(cfg, train_systems, training=True)
    validation_dataset = get_dataset(cfg, validation_systems, training=False)
    train_loader = get_loader(
        cfg, train_dataset, train_systems, training=True, trial=trial, tune=tune
    )
    val_loader = get_loader(
        cfg,
        validation_dataset,
        validation_systems,
        training=False,
    )

    # In addition to those 'classical' loaders, we also create ones dedicated to VS validation.
    # Splitting for VS validation is based on systems, to avoid having too many pockets
    _, vs_validation_systems = train_val_split(
        train_val_systems.copy(), system_based=True
    )
    val_vs_loader = get_vs_loader(
        systems=vs_validation_systems,
        decoy_mode=cfg.train.vs_decoy_mode,
        reps_only=True,
        cfg=cfg,
    )
    test_vs_loader = get_vs_loader(
        systems=test_systems,
        decoy_mode=cfg.train.vs_decoy_mode,
        reps_only=True,
        cfg=cfg,
    )

    # Maybe monitor rognan's performance ?
    val_vs_loader_rognan = None
    if cfg.train.do_rognan:
        val_vs_loader_rognan = get_vs_loader(
            systems=vs_validation_systems,
            decoy_mode=cfg.train.vs_decoy_mode,
            reps_only=True,
            rognan=True,
            cfg=cfg,
        )

    return train_loader, val_loader, val_vs_loader, val_vs_loader_rognan, test_vs_loader


def objective(trial, cfg) -> float:
    setup_seed(cfg.train.seed)
    device = setup_device(cfg.device)

    # Get model
    model = cfg_to_model(cfg, trial=trial, tune=cfg.train.tune)
    model = model.to(device)

    train_loader, val_loader, val_vs_loader, val_vs_loader_rognan, test_vs_loader = (
        get_loaders(cfg, tune=cfg.train.tune, trial=trial)
    )

    # Dataset/loaders creation and splitting.
    # Optimizer instantiation
    if cfg.train.loss == "l2":
        criterion = torch.nn.MSELoss()
    elif cfg.train.loss == "l1":
        criterion = torch.nn.L1Loss()
    elif cfg.train.loss == "bce":
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {cfg.train.loss}")

    lr = cfg.train.learning_rate
    weight_decay = cfg.train.learning_rate

    if cfg.train.tune:
        lr = trial.suggest_float("train.learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("train.weight_decay", 1e-5, 1e-2, log=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Experiment Setup
    name = f"{cfg.name}"
    if cfg.train.tune:
        name += f"_{trial.number}"
    save_path, save_name = mkdirs(name, prefix=cfg.train.target)
    writer = tensorboard.SummaryWriter(save_path)
    print(f"Saving result in {save_path}/{name}")
    trial_params = [f"{k}={v}" for k, v in trial.params.items()]
    trial_params = OmegaConf.from_dotlist(trial_params)
    new_cfg = OmegaConf.merge(cfg, trial_params)
    OmegaConf.save(new_cfg, pathlib.Path(save_path, "config.yaml"))
    # also dump the trial params
    with open(pathlib.Path(save_path) / "trial_params.json", "w") as j:
        json.dump(trial.params, j)

    # Training
    val_loss, best_model = learn.train_dock(
        model=model,
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
        debug=cfg.debug,
        cfg=cfg,
    )

    with open(pathlib.Path(save_path) / f"trial_score.txt", "w") as t:
        t.write(str(val_loss))

    return val_loss


def pdb_eval(cfg, model):
    # Final VS validation on each decoy set
    logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
    logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")

    test_systems = get_systems_from_cfg(cfg, return_test=True)
    model = model.to("cpu")
    rows, raw_rows = [], []
    decoys = ["chembl", "pdb", "pdb_chembl", "decoy_finder"]
    for decoy_mode in decoys:
        dataloader = get_vs_loader(systems=test_systems, decoy_mode=decoy_mode, cfg=cfg)
        decoy_rows, decoys_raw_rows = get_efs(
            model=model,
            dataloader=dataloader,
            decoy_mode=decoy_mode,
            cfg=cfg,
            verbose=True,
        )
        rows += decoy_rows
        raw_rows += decoys_raw_rows

    # Make it a df
    df = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows)

    # Dump csvs
    d = pathlib.Path(cfg.result_dir, parents=True, exist_ok=True)
    base_name = pathlib.Path(cfg.name).stem
    df.to_csv(d / (base_name + ".csv"))
    df_raw.to_csv(d / (base_name + "_raw.csv"))

    df_chembl = df.loc[df["decoys"] == "chembl"]
    print(f"{cfg.name} Mean EF on chembl: {np.mean(df_chembl['score'].values)}")
    df_chembl = group_df(df_chembl)
    print(f"{cfg.name} Mean grouped EF on chembl: {np.mean(df_chembl['score'].values)}")

    df_pdbchembl = df.loc[df["decoys"] == "pdb_chembl"]
    print(f"{cfg.name} Mean EF on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")
    df_pdbchembl = group_df(df_pdbchembl)
    print(
        f"{cfg.name} Mean grouped EF on pdbchembl: {np.mean(df_pdbchembl['score'].values)}"
    )
    pass





@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    # General config
    print(OmegaConf.to_yaml(cfg))
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.train.n_trials)

    # load best model if tuned
    if cfg.train.tune:
        name = f"{cfg.name}_{study.best_trial.number}"
        save_path = os.path.join("results", "trained_models", cfg.train.target, name)
        trial_df = study.trials_dataframe()
        trial_df.to_csv(Path(save_path) / "trials.csv")

        model = get_model_from_dirpath(
            save_path, tune=cfg.train.tune, trial=study.best_trial
        )

        src_path_obj = Path(save_path)
        new_dir_name = src_path_obj.name + "_best"
        dest_dir_path = src_path_obj.with_name(new_dir_name)

        if dest_dir_path.exists():
            shutil.rmtree(dest_dir_path)

        shutil.copytree(src_path_obj, dest_dir_path)

    # else load chosen model
    else:
        save_path = os.path.join(
            "results", "trained_models", cfg.train.target, cfg.name
        )
        model = get_model_from_dirpath(save_path)

    pdb_eval(cfg, model)
    robin_eval(cfg, model)


if __name__ == "__main__":
    main()
