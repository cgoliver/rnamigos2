import os, sys

import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import hydra

from rnaglib.learning import models, learn
from rnaglib.data_loading import graphloader
from rnaglib.kernels import node_sim

@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg: DictConfig):
    node_sim_func = node_sim.SimFunctionNode(method=cfg.train.simfunc, depth=cfg.train.num_hops)
    node_features = ['nt_code']
    unsupervised_dataset = loader.UnsupervisedDataset(node_simfunc=node_sim_func,
                                                      node_features=node_features)

    embedder_model = models.Embedder(infeatures_dim=unsupervised_dataset.input_dim,
                                     dims=cfg.model.hidden_dims)

    optimizer = optim.Adam(embedder_model.parameters())

    learn.pretrain_unsupervised(model=embedder_model,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                learning_routine=learn.LearningRoutine(num_epochs=cfg.train.epochs),
                                rec_params={"similarity": True, "normalize": False, "use_graph": True, "hops": cfg.train.num_hops})
    
    torch.save(
                {'args': args, 'state_dict': self.state_dict()},
                osp.pah.join(cfg.paths.models, cfg.name)
                )

if __name__ == "__main__":
    main()
