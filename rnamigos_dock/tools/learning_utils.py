import os
import configparser
from ast import literal_eval
import pickle
from tqdm import tqdm

import torch
import numpy as np
import networkx as nx

from learning.loader import Loader, InferenceLoader
from learning.learn import send_graph_to_device
from learning.rgcn import Model


def remove(name):
    """
    delete an experiment results
    :param name:
    :return:
    """
    import shutil

    script_dir = os.path.dirname(__file__)
    logdir = os.path.join(script_dir, f'../results/logs/{name}')
    weights_dir = os.path.join(script_dir, f'../results/trained_models/{name}')
    experiment = os.path.join(script_dir, f'../results/experiments/{name}.exp')
    shutil.rmtree(logdir)
    shutil.rmtree(weights_dir)
    os.remove(experiment)
    return True


def setup():
    """
    Create all relevant directories to setup the learning procedure
    :return:
    """
    script_dir = os.path.dirname(__file__)
    resdir = os.path.join(script_dir, f'../results/')
    logdir = os.path.join(script_dir, f'../results/logs/')
    weights_dir = os.path.join(script_dir, f'../results/trained_models/')
    experiment = os.path.join(script_dir, f'../results/experiments/')
    os.mkdir(resdir)
    os.mkdir(logdir)
    os.mkdir(weights_dir)
    os.mkdir(experiment)


def mkdirs_learning(name, permissive=True):
    """
    Try to make the logs folder for each experiment
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    from tools.utils import makedir
    log_path = os.path.join('results/logs', name)
    save_path = os.path.join('results/trained_models', name)
    makedir(log_path, permissive)
    makedir(save_path, permissive)
    save_name = os.path.join(save_path, name + '.pth')
    return log_path, save_name


def load_model(run):
    """
        Load full trained model with id `run`

    """

    meta = pickle.load(open(f'./results/trained_models/{run}/meta.p', 'rb'))

    edge_map = meta['edge_map']
    num_edge_types = len(edge_map)

    model_dict = torch.load(f'./results/trained_models/{run}/{run}.pth', map_location='cpu')
    model = Model(dims=meta['embedding_dims'], attributor_dims=meta['attributor_dims'], num_rels=num_edge_types,
                  num_bases=-1,
                  device='cpu',
                  pool=meta['pool'])
    model.load_state_dict(model_dict['model_state_dict'])
    return model, meta


def load_data(annotated_path, meta, get_sim_mat=True):
    """

        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
    """
    loader = Loader(
                    annotated_path=annotated_path,
                    batch_size=1, num_workers=1,
                    sim_function=meta['sim_function'],
                    get_sim_mat=get_sim_mat)

    train_loader, _, test_loader = loader.get_data()
    return train_loader, test_loader

def predict(model, loader, max_graphs=10, device='cpu'):
    all_graphs = loader.dataset.all_interactions
    Z = []
    fps = []
    g_inds = []

    model = model.to(device)
    with torch.no_grad():
        for i, (graph, K, fp, graph_index) in tqdm(enumerate(loader), total=len(loader)):

            graph = send_graph_to_device(graph, device)
            fp, z = model(graph)

            Z.append(z.cpu().numpy())
            fps.append(fp.cpu().numpy())
    Z = np.concatenate(Z)
    fps = np.array(fps)
    return fps, Z

def inference_on_dir(run, graph_dir, ini=True, max_graphs=10, get_sim_mat=False,
                     split_mode='test', attributions=False, device='cpu'):
    """
        Load model and get node embeddings.

        The results then need to be parsed as the order of the graphs is random and that the order of
        each node in the graph is the messed up one (sorted)

        Returns : embeddings and attributions, as well as 'g_inds':
        a dict (graph name, node_id in sorted g_nodes) : index in the embedding matrix

        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
        :max_graphs max number of graphs to get embeddings for
    """


    model, meta = meta_load_model(run)

    loader = InferenceLoader(graph_dir).get_data()

    return predict(model, loader, max_graphs=max_graphs,
                               device=device)

def meta_load_model(run):
    """
        Load full trained model with id `run`

    """

    meta = pickle.load(open(f'models/{run}/meta.p', 'rb'))
    print(meta)

    edge_map = meta['edge_map']
    num_edge_types = len(edge_map)

    model_dict = torch.load(f'models/{run}/{run}.pth', map_location='cpu')
    model = Model(dims=meta['embedding_dims'], attributor_dims=meta['attributor_dims'], num_rels=num_edge_types,
                  num_bases=-1, device='cpu')
    model.load_state_dict(model_dict['model_state_dict'])
    return model, meta

def model_from_hparams(hparams):
    """
        Load full trained model with id `run`

    """
    edge_map = hparams.get('edges', 'edge_map')
    num_edge_types = len(edge_map)
    run = hparams.get('argparse', 'name')
    model_dict = torch.load(f'../results/trained_models/{run}/{run}.pth', map_location='cpu')
    model = Model(dims=hparams.get('argparse', 'embedding_dims'),
                  attributor_dims=hparams.get('argparse', 'attributor_dims'),
                  num_rels=num_edge_types,
                  num_bases=-1,
                  hard_embed=hparams.get('argparse', 'hard_embed'))
    model.load_state_dict(model_dict['model_state_dict'])
    return model


def data_from_hparams(annotated_path, hparams, get_sim_mat=True):
    """
        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
    """
    dims = hparams.get('argparse', 'embedding_dims')

    loader = Loader(annotated_path=annotated_path,
                    batch_size=hparams.get('argparse', 'batch_size'),
                    num_workers=1,
                    sim_function=hparams.get('argparse', 'sim_function'),
                    depth=hparams.get('argparse', 'kernel_depth'),
                    hard_embed=hparams.get('argparse', 'hard_embed'),
                    hparams=hparams,
                    get_sim_mat=get_sim_mat)

    train_loader, _, test_loader = loader.get_data()
    return train_loader, test_loader


def get_rgcn_outputs(run, graph_dir, ini=False, max_graphs=100, nc_only=False, get_sim_mat=True):
    """
        Load model and get node embeddings.

        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
        :max_graphs max number of graphs to get embeddings for
    """

    from tools.graph_utils import dgl_to_nx
    if ini:
        hparams = ConfParser(default_path=os.path.join('../results/experiments', f'{run}.exp'))
        model = model_from_hparams(hparams)
        train_loader, test_loader = data_from_hparams(graph_dir, hparams, get_sim_mat=get_sim_mat)
        edge_map = hparams.get('edges', 'edge_map')
        similarity = hparams.get('argparse', 'similarity')
    else:
        model, meta = load_model(run)
        train_loader, test_loader = load_data(graph_dir, meta, get_sim_mat=get_sim_mat)
        edge_map = meta['edge_map']
        similarity = False

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])
    Z = []
    fp_mat = []
    nx_graphs = []
    KS = []

    # maps full nodeset index to graph and node index inside graph
    node_map = {}

    ind = 0
    offset = 0
    for i, (graph, K, graph_sizes) in enumerate(train_loader):
        if i > max_graphs - 1:
            break
        fp, z = model(graph)
        KS.append(K)
        fp_mat.append(np.array(fp.detach().numpy()))
        for j, emb  in enumerate(z.detach().numpy()):
            Z.append(np.array(emb))
            node_map[ind] = (i, j)
            ind += 1
        # nx_graphs.append(nx_graph)
        nx_g = dgl_to_nx(graph, edge_map)
        #assign unique id to graph nodes
        nx_g = nx.relabel_nodes(nx_g,{node:offset+k for k,node in enumerate(nx_g.nodes())})
        offset += len(nx_g.nodes())
        # print(z)
        # rna_draw(nx_g)
        nx_graphs.append(nx_g)
        pass

    Z = np.array(Z)
    fp_mat = np.array(fp_mat)
    return nx_graphs, Z, fp_mat, KS, node_map, similarity


class ConfParser:
    def __init__(self,
                 default_path='../default.ini',
                 path_to_conf=None,
                 argparse=None):

        self.default_path = default_path

        # Build the hparam object
        self.hparams = configparser.ConfigParser()

        # Add the default configurations, optionaly another .conf and an argparse object
        self.hparams.read(self.default_path)

        if path_to_conf is not None:
            print('confing')
            self.add_conf(path_to_conf)
        if argparse is not None:
            self.add_argparse(argparse)

    @staticmethod
    def merge_conf_into_default(default, new):
        for section in new.sections():
            for keys in new[section]:
                try:
                    default[section][keys]
                except KeyError:
                    raise KeyError(f'The provided value {section, keys} in the .conf are not present in the default, '
                                   f'thus not acceptable values')
                print(section, keys)
                default[section][keys] = new[section][keys]

    def add_conf(self, path_to_new):
        """
        Merge another conf parsing into self.hparams
        :param path_to_new:
        :return:
        """
        conf = configparser.ConfigParser()
        conf.read(path_to_new)
        print(f'confing using {path_to_new}')
        return self.merge_conf_into_default(self.hparams, conf)

    @staticmethod
    def merge_dict_into_default(default, new):
        """
        Same merge but for a dict of dicts
        :param default:
        :param new:
        :return:
        """
        for section in new.sections():
            for keys in new[section]:
                try:
                    default[section][keys]
                except KeyError:
                    raise KeyError(f'The provided value {section, keys} in the .conf are not present in the default, '
                                   f'thus not acceptable values')
                default[section][keys] = new[section][keys]
        return default

    def add_dict(self, section_name, dict_to_add):
        """
        Add a dictionnary as a section of the .conf. It needs to be turned into strings
        :param section_name: string to be the name of the section
        :param dict_to_add: any dictionnary
        :return:
        """

        new = {item: str(value) for item, value in dict_to_add.items()}

        try:
            self.hparams[section_name]
        # If it does not exist
        except KeyError:
            self.hparams[section_name] = new
            return

        for keys in new:
            self.hparams[section_name][keys] = new[keys]

    def add_argparse(self, argparse_obj):
        """
        Add the argparse object as a section of the .conf
        :param argparse_obj:
        :return:
        """
        self.add_dict('argparse', argparse_obj.__dict__)

    def get(self, section, key):
        """
        A get function that also does the casting into what is useful for model results
        :param section:
        :param key:
        :return:
        """
        try:
            return literal_eval(self.hparams[section][key])
        except ValueError:
            return self.hparams[section][key]

    def __str__(self):
        print(self.hparams.sections())
        for section in self.hparams.sections():
            print(section.upper())
            for keys in self.hparams[section]:
                print(keys)
            print('-' * 10)
        return ' '

    def dump(self, path):
        with open(path, 'w') as save_path:
            self.hparams.write(save_path)
