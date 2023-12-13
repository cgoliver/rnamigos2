import dgl
import itertools
import networkx as nx
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm
import torch

import rnaglib
from rnaglib.prepare_data import fr3d_to_graph
from rnaglib.utils import graph_io
from rnaglib.utils import NODE_FEATURE_MAP
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB


def get_edge_map(graphs_dir):
    edge_labels = set()
    print("Collecting edge labels.")
    for g in tqdm(os.listdir(graphs_dir)):
        graph, _, _ = pickle.load(open(os.path.join(graphs_dir, g), 'rb'))
        edges = {e_dict['label'] for _, _, e_dict in graph.edges(data=True)}
        edge_labels = edge_labels.union(edges)

    return {label: i for i, label in enumerate(sorted(edge_labels))}


def nx_to_dgl_(graph, edge_map, embed_dim):
    """
        Networkx graph to DGL.
    """
    import torch
    import dgl

    graph, _, ring = pickle.load(open(graph, 'rb'))
    one_hot = {edge: edge_map[label] for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
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


# Adapted from rglib
def to_undirected(edge_map):
    """
    Make edge labels symmetric for a graph.
    :param graph: Nx graph
    :return: Same graph but edges are now symmetric
    """
    remap = {}
    for old_label in edge_map.keys():
        new_label = old_label[0] + "".join(sorted(old_label[1:]))
        remap[old_label] = new_label
    new_map = {label: i for i, label in enumerate(sorted(set(remap.values())))}
    undirected_edge_map = {old_label: new_map[remap[old_label]] for old_label in edge_map.keys()}
    return undirected_edge_map


def load_rna_graph(rna_path, edge_map=EDGE_MAP_RGLIB, undirected=True, use_rings=False):
    """
    NetworkX Graph or path to a json => DGL graph
    """
    if isinstance(rna_path, str):
        pocket_graph = graph_io.load_json(rna_path)
    if isinstance(rna_path, nx.Graph):
        pocket_graph = rna_path
    # possibly undirected, just update the edge map to keep a DiGraph
    edge_map = to_undirected(edge_map) if undirected else edge_map
    edge_map = {key.upper(): value for key, value in edge_map.items()}
    one_hot = {edge: torch.tensor(edge_map[label.upper()]) for edge, label in
               (nx.get_edge_attributes(pocket_graph, 'LW')).items()}
    nx.set_edge_attributes(pocket_graph, name='edge_type', values=one_hot)

    # Needed for graph creation with fred, the key changed
    _, ndata = list(pocket_graph.nodes(data=True))[0]
    if 'nt' in ndata.keys():
        nx.set_node_attributes(pocket_graph, name='nt_code',
                               values={node: d['nt'] for node, d in pocket_graph.nodes(data=True)})
    one_hot_nucs = {node: NODE_FEATURE_MAP['nt_code'].encode(label) for node, label in
                    (nx.get_node_attributes(pocket_graph, 'nt_code')).items()}

    pocket_nodes = {node: label for node, label in (nx.get_node_attributes(pocket_graph, 'in_pocket')).items()}
    pocket_nodes = {node: True if node not in pocket_nodes else pocket_nodes[node] for node in pocket_graph.nodes()}
    nx.set_node_attributes(pocket_graph, name='nt_features', values=one_hot_nucs)
    nx.set_node_attributes(pocket_graph, name='in_pocket', values=pocket_nodes)
    pocket_graph_dgl = dgl.from_networkx(nx_graph=pocket_graph,
                                         edge_attrs=['edge_type'],
                                         node_attrs=['nt_features', 'in_pocket'],
                                         )
    rings = []
    if use_rings:
        for node, data in sorted(pocket_graph.nodes(data=True)):
            if data['in_pocket']:
                rings.append(data['edge_annots'])
        return pocket_graph_dgl, rings
    else:
        return pocket_graph_dgl, rings


def get_dgl_graph(cif_path, residue_list):
    """
    :param cif_path: toto/tata/1cqr.cif
    :param residue_list: list of strings "A.2","A.3",... ,"A.85" (missing pdb, useful for inference)
    :return:
    """
    ### DATA PREP
    # convert cif to graph and keep only relevant keys
    nx_graph = fr3d_to_graph(cif_path)
    # This is the pdbid used by fr3d
    pdbid = Path(cif_path).stem.lower()
    if residue_list is not None:
        # subset cif with given reslist
        reslist = [f"{pdbid}.{res}" for res in residue_list]
        expanded_reslist = rnaglib.utils.graph_utils.bfs(nx_graph, reslist, depth=4, label='LW')
        in_pocket = {node: node in reslist for node in expanded_reslist}
        expanded_graph = nx_graph.subgraph(expanded_reslist)
        nx.set_node_attributes(expanded_graph, name='in_pocket', values=in_pocket)
    else:
        expanded_graph = nx_graph
        in_pocket = {node: True for node in nx_graph.nodes}
        nx.set_node_attributes(expanded_graph, name='in_pocket', values=in_pocket)
    dgl_graph, _ = load_rna_graph(expanded_graph)
    return dgl_graph


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
        np.shuffle(labels)
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
