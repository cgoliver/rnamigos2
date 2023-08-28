"""
Script for RGCN model.

"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling,GlobalAttentionPooling
from dgl import mean_nodes
from dgl.nn.pytorch.conv import RelGraphConv

class Decoder(nn.Module):
    """
        NN which makes a prediction (fp or binding/non binding) from a pooled
        graph embedding.

        Linear/ReLu layers with Sigmoid in output since fingerprints between 0 and 1.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        # self.num_nodes = num_nodes
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers

        # create layers
        self.build_model()

    def build_model(self):
        layers = []
        layers.append(nn.Linear(self.in_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
        
        # hidden to output
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LigandEncoder(nn.Module):
    """
        Model for producing node embeddings.
    """
    def __init__(self, in_dim, hidden_dim, num_hidden_layers, num_rels=19, num_bases=-1):
        super(LigandEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers 

        self.layers = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()

        # input feature is just node degree
        i2h = self.build_hidden_layer(self.in_dim, self.hidden_dim)
        layers.append(i2h)

        for i in range(self.num_hidden_layers - 1): 
            h2h = self.build_hidden_layer(self.hidden_dim, self.hidden_dim)
            layers.append(torch.nn.ReLU())
            layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(self.hidden_dim, self.hidden_dim)
        layers.append(h2o)
        return layers

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
        for layer in self.layers:
            h = layer(h)
        return h 

class Embedder(nn.Module):
    """
        Model for producing node embeddings.
    """
    def __init__(self, in_dim, hidden_dim, num_hidden_layers, num_rels=19, num_bases=-1):
        super(Embedder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers 
        self.num_rels = num_rels
        self.num_bases = num_bases

        self.layers = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()

        # input feature is just node degree
        i2h = self.build_hidden_layer(1, self.hidden_dim)
        layers.append(i2h)

        for i in range(self.num_hidden_layers - 1): 
            h2h = self.build_hidden_layer(self.hidden_dim, self.hidden_dim)
            layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(self.hidden_dim, self.hidden_dim)
        layers.append(h2o)
        return layers

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
        h = g.ndata['one_hot'].reshape(-1,1)
        for layer in self.layers:
            h = layer(g, h, g.edata['one_hot'])
        g.ndata['h'] = h
        embeddings = g.ndata.pop('h')
        return embeddings
###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~
class Model(nn.Module):
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
        super(Model, self).__init__()

        if pool == 'att':
            pooling_gate_nn = nn.Linear(pool_dim, 1)
            self.pool = GlobalAttentionPooling(pooling_gate_nn)
        else:
            self.pool = SumPooling()

        self.encoder = encoder 
        self.decoder = decoder 
        self.lig_encoder = lig_encoder

    def forward(self, g, lig_fp):
        embeddings = self.encoder(g)
        g_h = self.pool(g, embeddings)
        if not self.lig_encoder is None:
            lig_h = self.lig_encoder(lig_fp)
            pred = torch.cat((g_h, lig_h), 1)
        pred = self.decoder(pred)
        return pred 

    def from_pretrained(self, model_path):
        self.encoder.load_state_dict(torch.load(model_path))
