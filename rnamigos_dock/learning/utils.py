import os
import io
import numpy as np

import dgl
import networkx as nx


def dgl_to_nx(g_dgl, edge_map):
    hot_to_label = {v:k for k,v in edge_map.items()}
    print(hot_to_label)
    hots = g_dgl.edata['one_hot'].detach().numpy()
    G = dgl.to_networkx(g_dgl) 
    labels = {e:hot_to_label[i] for i,e in zip(hots, G.edges())}
    print(labels)
    nx.set_edge_attributes(G,labels, 'label')
    print(G.nodes())
    G = nx.to_undirected(G)
    return G

def mkdirs(name, permissive=True):
    """
    Try to make the logs folder
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    save_path = os.path.join('results', 'trained_models', name)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        if not permissive:
            raise ValueError('This name is already taken !')
    save_name = os.path.join(save_path, name + '.pth')
    return save_path, save_name


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape), o.size())
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


if __name__ == '__main__':
    pass

    # for key, value in labels.items():
    #     tensor = torch.from_numpy(value)
    #     labels[key] = tensor
    #     tensor.requires_grad = False
