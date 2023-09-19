from pathlib import Path

import torch

from omegaconf import DictConfig, OmegaConf
import hydra

from rnaglib.kernels import node_sim
from rnaglib.data_loading import rna_dataset, rna_loader
from rnaglib.representations import GraphRepresentation, RingRepresentation
from rnaglib.learning import models, learning_utils, learn

from rnamigos_dock.learning.models import Embedder
from rnamigos_dock.tools import to_undirected


class PossiblyUndirectedGraphRepresentation(GraphRepresentation):
    def __init__(self, undirected=False, **kwargs):
        super().__init__(**kwargs)
        self.undirected = undirected
        if undirected:
            self.edge_map = to_undirected(self.edge_map)

    def to_nx(self, graph, features_dict):
        # Get Edge Labels
        if self.undirected:
            graph = graph.to_undirected()
        return super().to_nx(graph, features_dict)


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Choose the data, features and targets to use
    node_features = ['nt_code']

    ###### Unsupervised phase : ######
    # Choose the data and kernel to use for pretraining
    print('Starting to pretrain the network')
    node_simfunc = node_sim.SimFunctionNode(method=cfg.simfunc, depth=cfg.depth)
    graph_representation = PossiblyUndirectedGraphRepresentation(framework='dgl', undirected=cfg.data.undirected)
    ring_representation = RingRepresentation(node_simfunc=node_simfunc, max_size_kernel=50)
    unsupervised_dataset = rna_dataset.RNADataset(nt_features=node_features,
                                                  data_path=cfg.data.pretrain_graphs,
                                                  representations=[ring_representation, graph_representation])
    train_loader = rna_loader.get_loader(dataset=unsupervised_dataset, split=False, num_workers=4)

    model = Embedder(in_dim=cfg.model.encoder.in_dim,
                     hidden_dim=cfg.model.encoder.hidden_dim,
                     num_hidden_layers=cfg.model.encoder.num_layers,
                     )

    optimizer = torch.optim.Adam(model.parameters())
    learn.pretrain_unsupervised(model=model,
                                optimizer=optimizer,
                                train_loader=train_loader,
                                learning_routine=learning_utils.LearningRoutine(num_epochs=cfg.epochs),
                                rec_params={"similarity": True, "normalize": False, "use_graph": True,
                                            "hops": cfg.depth, "device": cfg.device})
    model_dir = Path(cfg.paths.pretrain_save, cfg.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(model_dir, 'model.pth'))
    OmegaConf.save(cfg, Path(model_dir, "config.yaml"))


if __name__ == "__main__":
    main()
