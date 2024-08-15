import os
import io
import numpy as np

import dgl
import networkx as nx
import torch


def dgl_to_nx(g_dgl, edge_map):
    hot_to_label = {v: k for k, v in edge_map.items()}
    hots = g_dgl.edata['one_hot'].detach().numpy()
    G = dgl.to_networkx(g_dgl)
    labels = {e: hot_to_label[i] for i, e in zip(hots, G.edges())}
    nx.set_edge_attributes(G, labels, 'label')
    G = nx.to_undirected(G)
    return G


def mkdirs(name, prefix='', permissive=True):
    """
    Try to make the logs folder
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    save_path = os.path.join('results', 'trained_models', prefix, name)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not permissive:
            raise ValueError('This name is already taken !')
    save_name = os.path.join(save_path, 'model.pth')
    return save_path, save_name


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape), o.size())
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


def setup_device(device):
    if torch.cuda.is_available():
        if device != 'cpu':
            try:
                gpu_number = int(device)
            except:
                gpu_number = 0
            device = f'cuda:{gpu_number}'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
        print("No GPU found, running on the CPU")
    return device


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    pass

    # for key, value in labels.items():
    #     tensor = torch.from_numpy(value)
    #     labels[key] = tensor
    #     tensor.requires_grad = False
