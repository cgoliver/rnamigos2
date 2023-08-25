import argparse
import os, sys
import pickle
import copy
import numpy as np

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import hydra

from rnaglib.learning import models, learn
from rnaglib.data_loading import graphloader
from rnaglib.benchmark import evaluate
from rnaglib.kernels import node_sim


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg: DictConfig):
    node_sim_func = node_sim.SimFunctionNode(method='R_graphlets', depth=2)
    node_features = ['nt_code']
    unsupervised_dataset = loader.UnsupervisedDataset(node_simfunc=node_sim_func,
                                                  node_features=node_features)

    embedder_model = models.Embedder(infeatures_dim=unsupervised_dataset.input_dim,
                                     dims=[64, 64])

    optimizer = optim.Adam(embedder_model.parameters())

    learn.pretrain_unsupervised(model=embedder_model,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                learning_routine=learn.LearningRoutine(num_epochs=cfg.train.epochs),
                                rec_params={"similarity": True, "normalize": False, "use_graph": True, "hops": 2})
    pass

if __name__ == "__main__":
    main()
