"""
Functions for comparing node similarity.
"""
import os
import pickle
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools

# if __name__ == '__main__':
    # import sys

    # sys.path.append('../')

# from src.rna_ged import ged
# from src.graphlets import *


# from learning.timed_learning import train_model
# from src.rna_ged import ged
# from src.graphlets import *
# from src.data_loader import load_graphset

def R_1(list1, list2):
    """
    Compute R function over lists of features:
    first attempt : count intersect and normalise by the number (Jacard?)
    :param list1: list of features
    :param list2: ''
    :return:
    """
    feat_1 = Counter(list1)
    feat_2 = Counter(list2)

    # sometimes ring is empty which throws division by zero error
    if len(feat_1) == 0 and len(feat_2) == 0:
        return 0

    diff = feat_1 & feat_2
    hist = feat_1 | feat_2
    x = sum(diff.values()) / sum(hist.values())
    return x


def R_IDF(list1, list2, IDF):
    """
    Compute IDF weighted R function over lists of features:
    frac{\sum_i^N w_i min(x_i, y_i)}{\sum_i^N w_i max(x_i, y_i)}
    first attempt : count intersect and normalise by the number (Jacard?)
    :param list1: list of features
    :param list2: ''
    :param IDF: Dictionary with IDF values for each label
    :return:
    """

    feat_1 = Counter(list1)
    feat_2 = Counter(list2)

    num = 0
    den = 0
    minmax = lambda x: (min(x), max(x))
    for e, w in IDF.items():
        mi, ma = minmax((feat_1[e], feat_2[e]))
        num += w * mi
        den += w * ma

    return num / den


def R_graphlet(G1, G2, list_1, list_2, memory):
    """
        Compare graphlet rings using GED and memoizing.
    """
    sum_dists = 0
    for i, n1 in enumerate(list_1):
        g1_s = graphlet_string(G1, n1)
        for n2 in list_2[i:]:
            g2_s = graphlet_string(G2, n2)
            try:
                d = memory[g1_s][g2_s]
            except:
                g1 = extract_graphlet(G1, n1)
                g2 = extract_graphlet(G2, n2)
                d = ged(g1, g2)
                memory[g1_s][g2_s] = d

            sum_dists += d

    return sum_dists / (1 / 2 * (len(list_1) * len(list_2))), memory


def compare_rings(rings1, rings2, K, IDF=None, decay=0.5, method='R_1'):
    res = 0
    for k in range(1, K):
        if method == 'R_1':
            res += decay ** (k - 1) * R_1(rings1[k], rings2[k])
        if method == 'R_IDF':
            if not IDF:
                raise ValueError("Missing IDF dictionary.")
            res += decay ** (k - 1) * R_IDF(rings1[k], rings2[k], IDF)
    return res


def compare_rings_graphlet(g1, g2, rings1, rings2, K=3, decay=0.5, memory=None):
    if not memory:
        memory = defaultdict(dict)
    res = 0
    for k in range(K):
        r, memory = R_graphlet(g1, g2, rings1[k], rings2[k], memory)
        res += decay ** k * r
    return res, memory


def graph_edge_freqs(graphs, stop=0):
    """
        Get IDF for each edge label over whole dataset.

        {'CWW': 110, 'TWW': 23}
    """
    graph_counts = Counter()
    # get document frequencies
    num_docs = 0
    for graph in graphs:
        labels = {e['label'] for _, _, e in graph.edges(data=True)}
        graph_counts.update(labels)
        num_docs += 1
        if num_docs > 100 and stop:
            break
    return {k: np.log(num_docs / graph_counts[k] + 1) for k in graph_counts}


def edge_freqs(G):
    """
        Get number of occurences within a graph for each label.
    """
    return Counter([e['label'] for _, _, e in G.edges(data=True)])


class SimFunctionNode():
    """
    Factory object to factor out the method choices from the function calls
    """

    def __init__(self, method, depth, decay=0.5, IDF=None):
        self.method = method
        self.depth = depth
        self.decay = decay
        self.IDF = IDF

    def compare(self, node1, node2):
        return compare_rings(node1, node2, K=self.depth, method=self.method, IDF=self.IDF)


def k_block(graph, trees1, rings1, graph2, trees2, rings2, node_sim):
    """
    Experimental : pass the right argument with the method... dangerous
    :param graph:
    :param trees1:
    :param rings1: dict {nodes : rings for this node} for the 1st graph
    :param graph2:
    :param trees2:
    :param rings2:
    :param node_sim:
    :return:
    """

    if node_sim.method == 'rings':

        nodes = list(rings1.values()) + list(rings2.values())
        block = np.zeros((len(nodes), len(nodes)))
        sims = [node_sim.compare(n1[1], n2[1])
                for i,(n1, n2) in enumerate(itertools.combinations(nodes, 2))]
        block[np.triu_indices(len(nodes), 1)] = sims
        block += block.T
        # for i, node1 in enumerate(nodes):
        # for j, node2 in enumerate(nodes):
        # edge_list1 = node1[1]
        # edge_list2 = node2[1]
        # speed this up since matrix symmetrical.
        # block[i, j] = node_sim.compare(edge_list1, edge_list2)
        return block
    else:
        raise ValueError('wrong method')

def k_block_list(rings, node_sim):
    """
    Defines the block creation using a list of rings (should also ultimately include trees)
    :param rings: alist of rings
    :param node_sim: the pairwise node comparison function
    :return:
    """
    rings_values = [list(ring.values()) for ring in rings]
    nodes = list(itertools.chain.from_iterable(rings_values))
    block = np.zeros((len(nodes), len(nodes)))
    sims = [node_sim.compare(n1[1], n2[1])
            for i, (n1, n2) in enumerate(itertools.combinations(nodes, 2))]
    block[np.triu_indices(len(nodes), 1)] = sims
    block += block.T

    # 1.825 is the value of a node to itself (1 + decay + decay**2)
    # should be geometric series of depth and decay.. (1-node_sim.decay^node_sim.depth) / (1-node_sim.decay)
    # block /= 1.825
    block /= (1 - (node_sim.decay ** node_sim.depth)) / (1 - node_sim.decay)
    block += np.eye(len(nodes))

    return block

def k_block_all(dest, graph_dir):
    """
        Build K for all pairs in graph dir.
    """
    node_sim = SimFunctionNode(method='rings', depth=4)
    graph_paths = os.listdir(graph_dir)
    for g1, g2 in combinations(graph_paths, 2):
        g1_id = g1.split(".")[0].split("_")[-1]
        g2_id = g2.split(".")[0].split("_")[-1]

        g1, t1, r1 = pickle.load(open(os.path.join(graph_dir, g1), 'rb'))
        g2, t2, r2 = pickle.load(open(os.path.join(graph_dir, g2), 'rb'))
        K = k_block((g1_id, g1), t1, r1, (g2_id, g2), t2, r2, node_sim=node_sim)
        pickle.dump(K, open(f"{dest}/K_{g1_id}_{g2_id}.p", 'rb'))

    pass


if __name__ == "__main__":
    k_block_all("../data/chunks_nx_annot", "../data/chunks_nx")
    sys.exit()
    import pickle
    import time

    L = pickle.load(open('../data/subset_annotated.p', 'rb'))
    node_sim = SimFunctionNode(method='rings', depth=4)
    block = 0
    time_block = []

    for i, (name1, graph1, trees1, rings1) in enumerate(L):
        for j, (name2, graph2, trees2, rings2) in enumerate(L):
            # TODO : This is the call to fix, since the method is list, pass list as arguments ?
            a = time.perf_counter()
            block = k_block(graph1, trees1, rings1, graph2, trees2, rings2, node_sim=node_sim)
            t = time.perf_counter() - a
            time_block.append(t)
            if i + j > 20:
                break
        break
    print(block)
    print(time_block)
    print(np.mean(time_block))

    """
    data = load_graphset()
    # idf = graph_edge_freqs(data, stop=100)
    g, name = next(data)
    # node_tree(g, list(g.nodes())[0])

    # l1, e1 = node_2_unordered_rings(g, list(g.nodes())[0])
    # l2, e2 = node_2_unordered_rings(g, list(g.nodes())[0])
    test_1 = [[None], ['CWW', 'B53'], ['CWW', 'B53', 'B53', 'B53']]
    test_2 = [[None], ['B53', 'CWW'], ['B53', 'CWW', 'B53']]
    print(compare_rings(test_1, test_2, K=3))
    res = compare_rings(test_1, test_2, K=3, r='R_IDF', IDF=idf)
    print(res)

    test_1 = [[None], ['CWW', 'B53'], ['CWW', 'B53', 'B53', 'B53']]
    test_2 = [[None], ['B53', 'CWW'], ['B53', 'B53', 'B53']]
    print(compare_rings(test_1, test_2, K=3))
    res = compare_rings(test_1, test_2, K=3, r='R_IDF', IDF=idf)
    print(res)

    l1, e1 = node_2_unordered_rings(g, list(g.nodes())[0])
    l2, e2 = node_2_unordered_rings(g, list(g.nodes())[0])
    print(compare_rings_graphlet(g, g, l1, l2))

    # print(compare_rings(test_1, test_2, K=3))
    # res = compare_rings(test_1, test_2, K=3, r='R_IDF', IDF=idf)
    # print(res)
    # # print(res)
    """

    """a = time.perf_counter()
        trees = build_dict_tree(graph)
        t = time.perf_counter() - a
        time_tree.append(t)

        a = time.perf_counter()
        rings = build_ring_tree_from_graph(graph)
        t = time.perf_counter() - a
        time_rings.append(t)

        print(i)
        annotated_subset.append((name, graph, trees, rings))
        # print(len(graph.edges))
        # print(trees[('A',1)])
        # print(rings)
        # if i > 1:
        #     break
    pickle.dump(annotated_subset, open('../data/subset_annotated.p', 'wb'))
    """
