"""
Script for RGCN model.

"""

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling, GlobalAttentionPooling
from dgl import mean_nodes
from dgl.nn.pytorch.conv import RelGraphConv


class Decoder(nn.Module):
    """
        NN which makes a prediction (fp or binding/non binding) from a pooled
        graph embedding.

        Linear/ReLu layers with Sigmoid in output since fingerprints between 0 and 1.
    """

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, dropout=0.2, batch_norm=True, activation=None):
        super(Decoder, self).__init__()
        # self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        if activation == 'softmax':
            self.activation = nn.Softmax()
        
        # create layers
        self.layers, self.batch_norms = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()
        batch_norms = nn.ModuleList()
        layers.append(nn.Linear(self.in_dim, self.hidden_dim))
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # hidden to output
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        batch_norms.append(nn.BatchNorm1d(self.out_dim))

        return layers, batch_norms

    def forward(self, x):
        output = x
        for layer in range(self.num_layers):
            output = self.layers[layer](output)
            output = self.batch_norms[layer](output)
            if layer == self.num_layers - 1:
                output = F.dropout(output, self.dropout, training=self.training)
            else:
                output = F.dropout(F.relu(output), self.dropout, training=self.training)

        return self.activation(output)


class LigandEncoder(nn.Module):
    """
        Model for producing node embeddings.
    """

    def __init__(self, in_dim, hidden_dim, num_hidden_layers, batch_norm=True, dropout=0.2, num_rels=19):
        super(LigandEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.layers, self.batch_norms  = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()
        batch_norms = nn.ModuleList()

        # input feature is just node degree
        i2h = self.build_hidden_layer(self.in_dim, self.hidden_dim)
        layers.append(i2h)
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        for i in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(self.hidden_dim, self.hidden_dim)
            layers.append(h2h)
            batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        # hidden to output
        h2o = self.build_output_layer(self.hidden_dim, self.hidden_dim)
        layers.append(h2o)
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        return layers, batch_norms

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def build_hidden_layer(self, in_dim, out_dim):
        return torch.nn.Linear(in_dim, out_dim)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return torch.nn.Linear(in_dim, out_dim)

    def forward(self, fp):
        h = fp.float()
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.batch_norms[i](h)

            if i < self.num_hidden_layers:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
        return h


class Embedder(nn.Module):
    """
        Model for producing node embeddings.
    """

    def __init__(self, in_dim, hidden_dim, num_hidden_layers, batch_norm=True, num_rels=20, dropout=0.2, num_bases=-1):
        super(Embedder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.layers, self.batch_norms = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()
        batch_norms = nn.ModuleList()

        # input feature is just node degree
        i2h = self.build_hidden_layer(self.in_dim, self.hidden_dim)
        layers.append(i2h)
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        for i in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(self.hidden_dim, self.hidden_dim)
            layers.append(h2h)
            batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        # hidden to output
        h2o = self.build_output_layer(self.hidden_dim, self.hidden_dim)
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(h2o)
        return layers, batch_norms

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def build_hidden_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels,
                            num_bases=self.num_bases,
                            activation=F.relu)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels, num_bases=self.num_bases,
                            activation=None)

    def forward(self, g):
        h = g.ndata['nt_features']
        for i, layer in enumerate(self.layers):
            h = layer(g, h, g.edata['edge_type'])
            h = self.batch_norms[i](h)

            if i < self.num_hidden_layers:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

        g.ndata['h'] = h

        # This tedious step is necessary, otherwise subgraphing looses track of the batch
        graphs = dgl.unbatch(g)
        all_subgraphs = []
        all_embs = []
        for graph in graphs:
            subgraph = dgl.node_subgraph(graph, graph.ndata['in_pocket'])
            embeddings = subgraph.ndata.pop('h')
            all_subgraphs.append(subgraph)
            all_embs.append(embeddings)
        subgraph = dgl.batch(all_subgraphs)
        embeddings = torch.cat(all_embs, dim=0)
        return subgraph, embeddings


###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~
class RNAmigosModel(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 lig_encoder=None,
                 pool='att',
                 pool_dim=32
                 ):
        """

        :param dims: the embeddings dimensions
        :param attributor_dims: the number of motifs to look for
        :param num_rels: the number of possible edge types
        :param num_bases: technical rGCN option
        :param rec: the constant in front of reconstruction loss
        :param mot: the constant in front of motif detection loss
        :param orth: the constant in front of dictionnary orthogonality loss
        :param attribute: Wether we want the network to use the attribution module
        """
        super(RNAmigosModel, self).__init__()

        if pool == 'att':
            pooling_gate_nn = nn.Linear(pool_dim, 1)
            self.pool = GlobalAttentionPooling(pooling_gate_nn)
        else:
            self.pool = SumPooling()

        self.encoder = encoder
        self.decoder = decoder
        self.lig_encoder = lig_encoder

    def predict_ligands(self, g, ligands):
        with torch.no_grad():
            g, embeddings = self.encoder(g)
            graph_pred = self.pool(g, embeddings)

            # Batch ligands together, encode them
            if not self.lig_encoder is None:
                lig_h = self.lig_encoder(ligands)
                graph_pred = graph_pred.expand(len(lig_h), -1)
                pred = torch.cat((graph_pred, lig_h), dim=1)
                pred = self.decoder(pred)
                return pred
            # Do FP prediction and use cdist
            ligand_pred = self.decoder(graph_pred)
            distances = torch.cdist(ligand_pred, ligands.float())
            return distances

    def forward(self, g, lig_fp):
        g, embeddings = self.encoder(g)
        pred = self.pool(g, embeddings)
        if not self.lig_encoder is None:
            lig_h = self.lig_encoder(lig_fp)
            pred = torch.cat((pred, lig_h), dim=1)
        pred = self.decoder(pred)
        return pred, embeddings

    def from_pretrained(self, model_path, verbose=False):
        state_dict = torch.load(model_path)
        if verbose:
            for n, p in self.named_parameters():
                try:
                    self.state_dict()[n][:] = p
                except KeyError:
                    continue
                else:
                    logger.info(f"Loaded parameter {n} from pretrained model")
        else:
            self.load_state_dict(state_dict, strict=False)
