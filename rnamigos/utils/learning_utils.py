import os
import io
import numpy as np

import dgl
import networkx as nx
import torch


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


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    g = g.to(device)
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g


if __name__ == '__main__':
    pass
