import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch

from rnaglib.kernels import node_sim
from rnaglib.data_loading import rna_dataset, rna_loader
from rnaglib.representations import GraphRepresentation, RingRepresentation
from rnaglib.learning import learning_utils, learn
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.models import Embedder
from rnamigos.utils.graph_utils import to_undirected


@hydra.main(version_base=None, config_path="conf", config_name="pretrain")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Choose the data, features and targets to use from graphs and kernel for pretraining
    node_features = ['nt_code']
    node_simfunc = node_sim.SimFunctionNode(method=cfg.simfunc, depth=cfg.depth)
    edge_map = GRAPH_KEYS['edge_map'][TOOL]
    edge_map = to_undirected(edge_map) if cfg.data.undirected else edge_map
    graph_representation = GraphRepresentation(framework='dgl', edge_map=edge_map)
    ring_representation = RingRepresentation(node_simfunc=node_simfunc, max_size_kernel=50)
    unsupervised_dataset = rna_dataset.RNADataset(nt_features=node_features,
                                                  data_path=cfg.data.pretrain_graphs,
                                                  representations=[ring_representation, graph_representation])
    train_loader = rna_loader.get_loader(dataset=unsupervised_dataset, split=False, num_workers=cfg.num_workers)

    model = Embedder(in_dim=cfg.model.encoder.in_dim,
                     hidden_dim=cfg.model.encoder.hidden_dim,
                     num_hidden_layers=cfg.model.encoder.num_layers,
                     batch_norm=cfg.model.batch_norm,
                     dropout=cfg.model.dropout,
                     subset_pocket_nodes=False
                     )
    optimizer = torch.optim.Adam(model.parameters())

    learn.pretrain_unsupervised(model=model,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                learning_routine=learning_utils.LearningRoutine(num_epochs=cfg.epochs),
                                rec_params={"similarity": True, "normalize": False, "use_graph": False,
                                            "hops": cfg.depth})
    model_dir = Path(cfg.paths.pretrain_save, cfg.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(model_dir, 'model.pth'))
    OmegaConf.save(cfg, Path(model_dir, "config.yaml"))


if __name__ == "__main__":
    main()
