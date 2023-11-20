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

class Attributor(nn.Module):
    """
        NN which makes a prediction (fp or binding/non binding) from a pooled
        graph embedding.

        Linear/ReLu layers with Sigmoid in output since fingerprints between 0 and 1.
    """
    def __init__(self, dims, clustered=False):
        super(Attributor, self).__init__()
        # self.num_nodes = num_nodes
        self.dims = dims
        self.clustered = clustered

        # create layers
        self.build_model()

    def build_model(self):
        layers = []

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        for dim_in, dim_out in zip(short, short[1:]):
            # print('in',dim_in, dim_out)
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        # hidden to output
        layers.append(nn.Linear(last_hidden, last))
        #predict one class
        if not self.clustered:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Embedder(nn.Module):
    """
        Model for producing node embeddings.
    """
    def __init__(self, dims, num_rels=19, num_bases=-1):
        super(Embedder, self).__init__()
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases

        self.layers = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        # input feature is just node degree
        i2h = self.build_hidden_layer(1, self.dims[0])
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            # print('in',dim_in, dim_out)
            h2h = self.build_hidden_layer(dim_in, dim_out)
            layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(last_hidden, last)
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
        h = g.in_degrees().view(-1, 1).float().to(self.current_device)
        for layer in self.layers:
            # layer(g)
            h = layer(g, h, g.edata['one_hot'])
        g.ndata['h'] = h
        del h
        embeddings = g.ndata.pop('h')
        return embeddings
###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~
class Model(nn.Module):
    def __init__(self, dims, device, attributor_dims, num_rels, pool='att', num_bases=-1,
                             pos_weight=0, nucs=True, clustered=False, num_clusts=8):
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
        # self.num_nodes = num_nodes
        self.dims = dims
        self.attributor_dims = attributor_dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.pos_weight = pos_weight
        self.device = device
        self.nucs = nucs
        self.clustered = clustered
        self.num_clusts = num_clusts

        if pool == 'att':
            pooling_gate_nn = nn.Linear(attributor_dims[0], 1)
            self.pool = GlobalAttentionPooling(pooling_gate_nn)
        else:
            self.pool = SumPooling()

        self.embedder = Embedder(dims=dims, num_rels=num_rels, num_bases=num_bases)

        self.attributor = Attributor(attributor_dims, clustered=clustered)

    def forward(self, g):
        embeddings = self.embedder(g)
        if self.nucs:
            nucs = g.ndata['one_hot'].reshape(-1,1)
            embeddings = torch.cat((embeddings, g.ndata['one_hot'].view(-1,1)),1)
        fp = self.pool(g, embeddings)
        fp = self.attributor(fp)
        return fp, embeddings

    def rec_loss(self, embeddings, target_K, similarity=True):
        """
        :param embeddings: The node embeddings
        :param target_K: The similarity matrix
        :param similarity: Boolean, if true, the similarity is used with the cosine, otherwise the distance is used
        :return:
        """
        if similarity:
            K_predict = self.matrix_cosine(embeddings, embeddings)

        else:
            K_predict = self.matrix_dist(embeddings)
            target_K = torch.ones(target_K.shape, device=target_K.device) - target_K

        reconstruction_loss = torch.nn.MSELoss()(K_predict, target_K)
        # self.draw_rec(target_K, K_predict)
        return reconstruction_loss
    # Below are loss computation function related to this model
    @staticmethod
    def matrix_dist(a, plus_one=False):
        """
        Pairwise dist of a set of a vector of size b
        returns a matrix of size (a,a)
        :param plus_one: if we want to get positive values
        """
        if plus_one:
            return torch.norm(a[:, None] - a, dim=2, p=2) + 1
        return torch.norm(a[:, None] - a, dim=2, p=2)

    @staticmethod
    def matrix_cosine(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def fp_loss(self, target_fp, pred_fp):
        if self.clustered:
            loss = torch.nn.CrossEntropyLoss()(pred_fp, target_fp)
        else:
            # loss = torch.nn.MSELoss()(pred_fp, target_fp)
            loss = torch.nn.BCELoss()(pred_fp, target_fp)
        return loss
    def compute_loss(self, target_fp, pred_fp, embeddings, target_K, 
                                               rec_lam=1, 
                                               fp_lam=1,
                                               similarity=False):
        """
        Compute the total loss of the model.
        Includes the reconstruction loss with optional similarity/distance boolean switch
        If the attributions are not None, adds the motif regularisation (orth term) as well as a motif finding
        loss with an optional scaling switch.
        :param embeddings: The first tensor produced by a forward call
        :param attributions: The second one
        :param target_K:
        :param similarity:
        :param scaled:
        :return:
        """
        loss = fp_lam * self.fp_loss(target_fp, pred_fp)\
               + rec_lam * self.rec_loss(embeddings, target_K, similarity=similarity)
        return loss

    def draw_rec(self, true_K, predicted_K, title=""):
        """
        A way to assess the reconstruction task visually
        :param true_K:
        :param predicted_K:
        :param loss_value: python float
        :return:
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        kws = {'cbar_kws': {'shrink': 0.5},
               'square':True,
               'vmin': 0,
               'vmax': 1
               }

        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(true_K.clone().detach().numpy(), ax=ax1, **kws)
        sns.heatmap(predicted_K.clone().detach().numpy(),  ax=ax2, **kws)
        ax1.set_title("Ground Truth")
        ax2.set_title("GCN")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

