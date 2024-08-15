"""
Script for RGCN model.

"""

import os
import sys

import json
from omegaconf import OmegaConf
from pathlib import Path
from yaml import safe_load

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, GlobalAttentionPooling
from dgl.nn.pytorch.conv import RelGraphConv


class RGCN(nn.Module):
    """ RGCN encoder with num_hidden_layers + 2 RGCN layers, and sum pooling. """

    def __init__(self, features_dim, h_dim, num_rels, num_layers, num_bases=-1, gcn_dropout=0, batch_norm=False,
                 self_loop=False, jumping_knowledge=True):
        super(RGCN, self).__init__()

        self.features_dim, self.h_dim = features_dim, h_dim
        self.num_layers = num_layers
        self.p = gcn_dropout

        self.self_loop = self_loop
        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(h_dim) for _ in range(num_layers)])
        self.pool = SumPooling()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, self_loop=self.self_loop,
                           activation=nn.ReLU(), dropout=self.p)
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_layers - 2):
            h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels,
                               self_loop=self.self_loop, activation=nn.ReLU(), dropout=self.p)
            self.layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(self.h_dim, self.h_dim, self.num_rels,
                           self_loop=self.self_loop, activation=nn.ReLU(), dropout=self.p)
        self.layers.append(h2o)

    def forward(self, g):
        sequence = []
        for i, layer in enumerate(self.layers):
            # Node update
            g.ndata['h'] = layer(g, g.ndata['h'], g.edata['edge_type'])
            # Jumping knowledge connexion
            sequence.append(g.ndata['h'])
            if self.batch_norm:
                g.ndata['h'] = self.batch_norm_layers[i](g.ndata['h'])
        # Concatenation :
        g.ndata['h'] = torch.cat(sequence, dim=1)  # Num_nodes * (h_dim*num_layers)
        out = self.pool(g, g.ndata['h'])
        return out


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
        elif activation == 'softmax':
            self.activation = nn.Softmax()
        else:
            self.activation = None

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
            if self.batch_norm:
                output = self.batch_norms[layer](output)
            if layer == self.num_layers - 1:
                output = F.dropout(output, self.dropout, training=self.training)
            else:
                output = F.dropout(F.relu(output), self.dropout, training=self.training)
        if self.activation is not None:
            output = self.activation(output)
        return output


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
            if self.batch_norm:
                h = self.batch_norms[i](h)

            if i < self.num_hidden_layers:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
        return h


class LigandGraphEncoder(nn.Module):
    def __init__(self,
                 l_size=56,
                 gcn_hdim=32,
                 gcn_layers=3,
                 features_dim=22,
                 num_rels=4,
                 batch_norm=False,
                 # cat_mu_v=True,
                 cut_embeddings=False
                 ):
        super(LigandGraphEncoder, self).__init__()
        self.features_dim = features_dim
        self.gcn_hdim = gcn_hdim
        self.gcn_layers = gcn_layers
        self.num_rels = num_rels
        # self.cat_mu_v = cat_mu_v
        # To use on optimol's embeddings
        self.cut_embeddings = cut_embeddings
        # Bottleneck
        self.l_size = l_size
        # layers:
        self.encoder = RGCN(self.features_dim, self.gcn_hdim, self.num_rels, self.gcn_layers, num_bases=-1,
                            batch_norm=batch_norm)

        self.encoder_mean = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)
        self.encoder_logv = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def from_pretrained(trained_dir):
        # Loads trained model weights, with or without the affinity predictor
        params = json.load(open(os.path.join(trained_dir, 'params.json'), 'r'))
        weight_path = os.path.join(trained_dir, 'weights.pth')
        ligraph_encoder = LigandGraphEncoder(**params, cut_embeddings=True)
        whole_state_dict = torch.load(weight_path)
        filtered_state_dict = {}
        for (k, v) in whole_state_dict.items():
            if 'encoder' in k:
                if k.startswith('encoder.layers'):
                    filtered_state_dict[k.replace('weight', 'linear_r.W')] = v
                else:
                    filtered_state_dict[k] = v
        ligraph_encoder.load_state_dict(filtered_state_dict)
        return ligraph_encoder

    def forward(self, g):
        g.ndata['h'] = g.ndata['node_features']
        # Weird optimol pretrained_model
        if self.cut_embeddings:
            g.ndata['h'] = g.ndata['h'][:, :-6]
        e_out = self.encoder(g)
        mu = self.encoder_mean(e_out)
        # if self.cat_mu_v:
        #     logv = self.encoder_logv(e_out)
        #     mu = torch.cat((mu, logv), dim=-1)
        return mu


class Embedder(nn.Module):
    """
        Model for producing node embeddings.
    """

    def __init__(self, in_dim, hidden_dim, num_hidden_layers, subset_pocket_nodes=True,
                 batch_norm=True, num_rels=20, dropout=0.2, num_bases=-1):
        super(Embedder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        num_bases = num_rels if num_bases == -1 else num_bases
        self.num_bases = num_bases
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.subset_pocket_nodes = subset_pocket_nodes

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
                            regularizer='basis' if self.num_rels > 0 else None,
                            num_bases=self.num_bases,
                            activation=None)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels, num_bases=self.num_bases,
                            regularizer='basis' if self.num_rels > 0 else None,
                            activation=None)

    def forward(self, g):
        h = g.ndata['nt_features']
        for i, layer in enumerate(self.layers):
            h = layer(g, h, g.edata['edge_type'])
            if self.batch_norm:
                h = self.batch_norms[i](h)

            if i < self.num_hidden_layers:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

        g.ndata['h'] = h

        graphs, embeddings = g, h
        if self.subset_pocket_nodes:
            # This tedious step is necessary, otherwise subgraphing looses track of the batch
            graphs = dgl.unbatch(g)
            all_subgraphs = []
            all_embs = []
            for graph in graphs:
                subgraph = dgl.node_subgraph(graph, graph.ndata['in_pocket'])
                embeddings = subgraph.ndata.pop('h')
                all_subgraphs.append(subgraph)
                all_embs.append(embeddings)
            graphs = dgl.batch(all_subgraphs)
            embeddings = torch.cat(all_embs, dim=0)
        return graphs, embeddings

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
        return self


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

    @property
    def use_graphligs(self):
        return self.lig_encoder is not None and isinstance(self.lig_encoder, LigandGraphEncoder)

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
            distances = torch.cdist(ligands.float(), ligand_pred)
            return distances

    def forward(self, g, lig_fp):
        g, embeddings = self.encoder(g)
        pred = self.pool(g, embeddings)
        if not self.lig_encoder is None:
            lig_h = self.lig_encoder(lig_fp)
            pred = torch.cat((pred, lig_h), dim=1)
        pred = self.decoder(pred)
        return pred, embeddings

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.load_state_dict(state_dict)
        return self


def cfg_to_model(cfg, for_loading=False):
    """
    for_loading skips pretrained network subparts since it will be used for loading a fully pretrained model
    """
    rna_encoder = Embedder(in_dim=cfg.model.encoder.in_dim,
                           hidden_dim=cfg.model.encoder.hidden_dim,
                           num_hidden_layers=cfg.model.encoder.num_layers,
                           batch_norm=cfg.model.batch_norm,
                           dropout=cfg.model.dropout,
                           num_bases=cfg.model.encoder.num_bases
                           )
    if cfg.model.use_pretrained and not for_loading:
        print(">>> Using pretrained weights")
        rna_encoder.from_pretrained(cfg.model.pretrained_path)
    rna_encoder.subset_pocket_nodes = True

    if cfg.model.use_graphligs:
        graphlig_cfg = cfg.model.graphlig_encoder
        if graphlig_cfg.use_pretrained:
            if for_loading:
                lig_encoder = LigandGraphEncoder(features_dim=16,
                                                 l_size=56,
                                                 num_rels=4,
                                                 gcn_hdim=32,
                                                 gcn_layers=3,
                                                 batch_norm=False,
                                                 cut_embeddings=True)
            else:
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
    return model


def get_model_from_dirpath(saved_model_dir):
    # Create the right model with the right params
    with open(Path(saved_model_dir, 'config.yaml'), 'r') as f:
        params = safe_load(f)
    cfg = OmegaConf.create(params)
    model = cfg_to_model(cfg, for_loading=True)

    # Load params and use eval()
    state_dict = torch.load(Path(saved_model_dir, 'model.pth'), map_location='cpu')['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model
