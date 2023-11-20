import os
import itertools
import pickle

import networkx as nx
import torch
import dgl
from tqdm import tqdm


def get_edge_map(graphs_dir):
    edge_labels = set()
    print("Collecting edge labels.")
    for g in tqdm(os.listdir(graphs_dir)):
        graph,_,_,_ = pickle.load(open(os.path.join(graphs_dir, g), 'rb'))
        edges = {e_dict['label'] for _, _, e_dict in graph.edges(data=True)}
        edge_labels = edge_labels.union(edges)

    return {label: i for i, label in enumerate(sorted(edge_labels))}


def nx_to_dgl(graph, edge_map, nucs=False):
    #adding the self edges
    # graph.add_edges_from([(n, n, {'label': 'X'}) for n in graph.nodes()])
    graph = nx.to_directed(graph)
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in
               (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

    node_attrs = None
    if nucs:
        nuc_map = {n:i for i,n in enumerate(['A', 'C', 'G', 'N', 'U'])}
        one_hot_nucs  = {node: torch.tensor(nuc_map[label.upper()], dtype=torch.float32) for node, label in
                   (nx.get_node_attributes(graph, 'nt')).items()}
    else:
        one_hot_nucs  = {node: torch.tensor(0, dtype=torch.float32) for node, label in
                   (nx.get_node_attributes(graph, 'nt')).items()}

    nx.set_node_attributes(graph, name='one_hot', values=one_hot_nucs)

    #g_dgl = dgl.DGLGraph()
    g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['one_hot'])
    #g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['one_hot'])

    return graph, g_dgl

def _nx_to_dgl(graph, edge_map, embed_dim):
    """
        Networkx graph to DGL.
    """

    one_hot = {edge: edge_map[label] for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    g_dgl = dgl.DGLGraph()
    g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
    n_nodes = len(g_dgl.nodes())
    g_dgl.ndata['h'] = torch.ones((n_nodes, embed_dim))

    return graph, g_dgl


def get_embeddings(model, dgl_graph):
    z, _ = model(dgl_graph)
    return z.detach().numpy()
