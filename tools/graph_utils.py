import pickle
import os
import itertools
from tqdm import tqdm
import networkx as nx
import torch
import dgl


def get_edge_map(graphs_dir):
    edge_labels = set()
    print("Collecting edge labels.")
    for g in tqdm(os.listdir(graphs_dir)):
        graph, _, _ = pickle.load(open(os.path.join(graphs_dir, g), 'rb'))
        edges = {e_dict['label'] for _, _, e_dict in graph.edges(data=True)}
        edge_labels = edge_labels.union(edges)

    return {label: i for i, label in enumerate(sorted(edge_labels))}


def nx_to_dgl_jacques(graph, edge_map):
    """
        Returns one training item at index `idx`.
    """
    #adding the self edges
    # graph.add_edges_from([(n, n, {'label': 'X'}) for n in graph.nodes()])
    graph = nx.to_undirected(graph)
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in
               (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

    g_dgl = dgl.DGLGraph()
    # g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['one_hot'])
    g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['angles', 'identity'])

    #JACQUES
    # Init node embeddings with nodes features
    floatid = g_dgl.ndata['identity'].float()
    g_dgl.ndata['h'] = torch.cat([g_dgl.ndata['angles'], floatid], dim = 1)

    print("HII")
    return graph, g_dgl

def nx_to_dgl_(graph, edge_map, embed_dim):
    """
        Networkx graph to DGL.
    """
    import torch
    import dgl

    graph, _, ring = pickle.load(open(graph, 'rb'))
    one_hot = {edge: edge_map[label] for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    g_dgl = dgl.DGLGraph()
    g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
    n_nodes = len(g_dgl.nodes())
    g_dgl.ndata['h'] = torch.ones((n_nodes, embed_dim))

    return graph, g_dgl


def dgl_to_nx(graph, edge_map):
    g = dgl.to_networkx(graph, edge_attrs=['one_hot'])
    edge_map_r = {v: k for k, v in edge_map.items()}
    nx.set_edge_attributes(g, {(n1, n2): edge_map_r[d['one_hot'].item()] for n1, n2, d in g.edges(data=True)}, 'label')
    return g


def bfs_expand(G, initial_nodes, depth=2):
    """
        Extend motif graph starting with motif_nodes.
        Returns list of nodes.
    """

    total_nodes = [list(initial_nodes)]
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
        total_nodes.append(depth_ring)
    return set(itertools.chain(*total_nodes))


def bfs(G, initial_node, depth=2):
    """
        Generator for bfs given graph and initial node.
        Yields nodes at next hop at each call.
    """

    total_nodes = [[initial_node]]
    visited = []
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            visited.append(n)
            for nei in G.neighbors(n):
                if nei not in visited:
                    depth_ring.append(nei)
        total_nodes.append(depth_ring)
        yield depth_ring


def graph_ablations(G, mode):
    """
        Remove edges with certain labels depending on the mode.

        :params
        
        :G Binding Site Graph
        :mode how to remove edges ('bb-only', 'wc-bb', 'wc-bb-nc', 'no-label')

        :returns: Copy of original graph with edges removed/relabeled.
    """

    H = nx.Graph()

    if mode == 'label-shuffle':
        # assign a random label from the same graph to each edge.
        labels = [d['label'] for _, _, d in G.edges(data=True)]
        shuffle(labels)
        for n1, n2, d in G.edges(data=True):
            H.add_edge(n1, n2, label=labels.pop())
        return H

    if mode == 'no-label':
        for n1, n2, d in G.edges(data=True):
            H.add_edge(n1, n2, label='X')
        return H
    if mode == 'wc-bb-nc':
        for n1, n2, d in G.edges(data=True):
            label = d['label']
            if d['label'] not in ['CWW', 'B53']:
                label = 'NC'
            H.add_edge(n1, n2, label=label)
        return H

    if mode == 'bb-only':
        valid_edges = ['B53']
    if mode == 'wc-bb':
        valid_edges = ['B53', 'CWW']

    for n1, n2, d in G.edges(data=True):
        if d['label'] in valid_edges:
            H.add_edge(n1, n2, label=d['label'])
    return H
