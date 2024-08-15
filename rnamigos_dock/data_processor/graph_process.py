"""
    Tools for processing pdb graphs.
"""

import itertools
from random import shuffle

import networkx as nx

faces = ['W', 'S', 'H']
orientations = ['C', 'T']
valid_edges = set(
    ['B53'] + [orient + "".join(sorted(e1 + e2)) for e1, e2 in itertools.product(faces, faces) for orient in
               orientations])


def remove_self_loops(G):
    selfs = list(nx.selfloop_edges(G))
    G.remove_edges_from(selfs)


def remove_non_standard_edges(G):
    remove = []
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in valid_edges:
            remove.append((n1, n2))
    G.remove_edges_from(remove)


def symmetric(label):
    """
        Returns symmetric version of label.
        e.g symmetric(CHW) = CHW, symmetric(CWH) = CHW
    """
    # if not a base pair keep the same
    if label[0] not in ['C', 'T']:
        return label
    else:
        return label[0] + "".join(sorted(label[1:]))


def to_orig(G):
    """
        Convert new networkx graphs (from vlad 2019) to original format.
        Keep only B53 and 12 base pair interactions.
        Annotate nodes with position, chain, and nucleotide ID attributes.

        Arguments:
            G (networkx graph): networkx graph to convert to original.
        Returns:
            networkx graph
    """
    remove_self_loops(G)
    # remove_non_standard_edges(G)
    H = nx.Graph()
    for n1, n2, d in G.edges(data=True):
        # label = symmetric(d['label'])
        label = symmetric(d['LW'])
        if label.upper() in valid_edges:
            # H.add_edge(n1, n2, label=label)
            H.add_edge(n1, n2, label=label.upper())

    # add pdb position and nt to node data
    d_orig = {n: d['nt_resnum'] for n, d in G.nodes(data=True)}
    nx.set_node_attributes(H, {n: d_orig[n] for n in H.nodes()}, 'pdb_pos')
    d_orig = {n: d['nt_code'] for n, d in G.nodes(data=True)}
    nx.set_node_attributes(H, {n: d_orig[n] for n in H.nodes()}, 'nt')
    d_orig = {n: n.split('.')[1] for n, d in G.nodes(data=True)}
    nx.set_node_attributes(H, {n: d_orig[n] for n in H.nodes()}, 'chain')
    return H


def kill_islands(G, min_size=4):
    """
        Kill components that are too small
    """
    remove = []
    for comp in nx.connected_components(G):
        if len(comp) < min_size:
            remove.extend(list(comp))
        pass
    G.remove_nodes_from(remove)


def graph_ablations(G, mode):
    """
        Remove edges with certain labels depending on the mode.

        :params
        
        :G Binding Site Graph
        :mode how to remove edges ('bb-only', 'wc-bb', 'no-stack')

        :returns: Copy of original graph with edges removed/relabeled.
    """

    H = nx.Graph()

    H.add_nodes_from(G.nodes(data=True))
    # nx.set_node_attributes(H, 'pdb_pos', {n:d['pdb_pos'] for n,d in G.nodes(data=True)})
    # nx.set_node_attributes(H, 'nt', {n:d['nt'] for n,d in G.nodes(data=True)})
    if mode == 'label-shuffle':
        # assign a random label from the same graph to each edge.
        labels = [d['label'] for _, _, d in G.edges(data=True)]
        shuffle(labels)
        for n1, n2, d in G.edges(data=True):
            H.add_edge(n1, n2, label=labels.pop())
        return H

        pass

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


"""
def find_node(graph, chain, pos):
    for n,d in graph.nodes(data=True):
        if (n[0] == chain) and (d['nucleotide'].pdb_pos == str(pos)):
            return n
    return None
"""


def find_node(graph, chain, pos):
    for n, d in graph.nodes(data=True):
        if (n.split('.')[1] == chain) and (str(d['nt_resnum']) == str(pos)):
            return n
    return None


def has_NC(G):
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in ['CWW', 'B53']:
            return True
    return False


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


def floaters(G):
    """
    Try to connect floating base pairs. (Single base pair not attached to backbone).
    Otherwise remove.
    """
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    floaters = []
    for u, v in G.edges():
        if deg_ok(G, u, v, 1):
            floaters.append((u, v))

    G.remove_edges_from(floaters)

    return G


def dangle_trim(G):
    """
    Recursively remove dangling nodes from graph.
    """
    # is_backbone = lambda n,G: sum([G[n][nei]['label'] != 'B53' for nei in G.neighbors(n)]) == 0
    is_backbone = lambda n, G: sum([G[n][nei]['LW'] != 'B53' for nei in G.neighbors(n)]) == 0
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        dangles = []
        for n in cur_G.nodes:
            # node_deg = degree(i, G, current_nodeset)
            # print(node_deg)
            # if node_deg == 2 and is_backbone(n, G):
            # if cur_G.degree(n) == 1 and is_backbone(n, cur_G) or cur_G.degree(n) == 0:
            if cur_G.degree(n) == 1 or cur_G.degree(n) == 0:
                dangles.append(n)
        if len(dangles) == 0:
            break
        else:
            cur_G.remove_nodes_from(dangles)
            cur_G = cur_G.copy()
    return cur_G


def stack_trim(G):
    """
    Remove stacks from graph.
    """
    is_ww = lambda n, G: 'CWW' in [info['label'] for node, info in G[n].items()]
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        stacks = []
        for n in cur_G.nodes:
            if cur_G.degree(n) == 2 and is_ww(n, cur_G):
                # potential stack opening
                partner = None
                stacker = None
                for node, info in cur_G[n].items():
                    if info['label'] == 'B53':
                        stacker = node
                    elif info['label'] == 'CWW':
                        partner = node
                    else:
                        pass
                if cur_G.degree(partner) > 3:
                    continue
                partner_2 = None
                stacker_2 = None
                for node, info in cur_G[partner].items():
                    if info['label'] == 'B53':
                        stacker_2 = node
                    elif info['label'] == 'CWW':
                        partner_2 = node
                try:
                    if cur_G[stacker][stacker_2]['label'] == 'CWW':
                        stacks.append(n)
                        stacks.append(partner)
                except KeyError:
                    continue
        if len(stacks) == 0:
            break
        else:
            cur_G.remove_nodes_from(stacks)
            cur_G = cur_G.copy()
    return cur_G


def in_stem(G, u, v):
    non_bb = lambda G, n: len([info['label'] for node, info in G[n].items() if info['label'] != 'B53'])
    is_ww = lambda G, u, v: G[u][v]['label'] == 'CWW'
    if is_ww(G, u, v) and (non_bb(G, u) in (1, 2)) and (non_bb(G, v) in (1, 2)):
        return True
    return False


if __name__ == "__main__":
    pass
