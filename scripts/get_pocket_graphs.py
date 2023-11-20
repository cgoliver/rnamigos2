import pandas as pd
import os
import networkx as nx
from rnaglib.config import GRAPH_KEYS, TOOL
from rnaglib.utils import graph_from_pdbid, graph_utils, graph_io
from tqdm import tqdm

# This should work but sometimes has missing nodes
# csv_path = 'data/rnamigos2_dataset_consolidated.csv'
# systems = pd.read_csv(csv_path)[['PDB_ID_POCKET_NODES', 'PDB_ID_POCKET']]
# all_pockets_nodes = systems.groupby(["PDB_ID_POCKET"])['PDB_ID_POCKET_NODES'].first()

old_path = 'data/json_pockets'
annotated_path = 'data/json_pockets_annotated'
expanded_path = 'data/json_pockets_expanded'
os.makedirs(expanded_path, exist_ok=True)
os.makedirs(annotated_path, exist_ok=True)


def node_2_unordered_rings(G, node, depth=2):
    """
    Return rings centered at `node` up to depth `depth`.

    Return dict of dicts. One dict for each type of ring.
    Each inner dict is keyed by node id and its value is a list of lists.
    A ring is a list of lists with one list per depth ring.

    :param G: Networkx graph
    :param node: A node from G
    :param depth: The depth or number of hops starting from node to include in the ring annotation
    :param hasher: A hasher object to use for encoding the graphlets
    :param hash_table: A hash table to fill with the annotations

    :return: {'node_annots': list, 'edge_annots': list, 'graphlet_annots': list} each of the list is of length depth
    and contains lists of the nodes in the ring at each depth.

    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1,2, {'LW': 'A'}),\
                          (1, 3, {'LW': 'B'}),\
                          (2, 3, {'LW': 'C'}),\
                          (3, 4, {'LW': 'A'})])
    >>> rings = node_2_unordered_rings(G, 1, depth=2)
    >>> rings['edge']
    [[None], ['A', 'B'], ['C', 'A']]
    """
    node_rings = [[node]]
    edge_rings = [[None]]
    visited = set()
    visited.add(node)
    visited_edges = set()
    for k in range(depth):
        ring_k = []
        edge_ring_k = []
        for node in node_rings[k]:
            children = []
            e_labels = []
            for nei in G.neighbors(node):
                if nei not in visited:
                    visited.add(nei)
                    children.append(nei)
                # check if we've seen this edge.
                e_set = frozenset([node, nei])
                if e_set not in visited_edges:
                    e_labels.append(G[node][nei][GRAPH_KEYS['bp_type'][TOOL]])
                    visited_edges.add(e_set)
            ring_k.extend(children)
            edge_ring_k.extend(e_labels)
        node_rings.append(ring_k)
        edge_rings.append(edge_ring_k)
    return {'edge': edge_rings}


def build_ring_tree_from_graph(graph, depth=2):
    """
    This function mostly loops over nodes and calls the annotation function.
    It then puts the annotated data into the graph.

    :param graph: nx graph
    :param depth: The depth or number of hops starting from node to include in the ring annotation
    :return: dict (ring_level: node: ring)
    """
    edge_rings = dict()
    for node in sorted(graph.nodes()):
        rings = node_2_unordered_rings(graph,
                                       node,
                                       depth=depth)
        edge_rings[node] = rings['edge']
    return edge_rings


failed_set = set()
for pocket in tqdm(os.listdir(old_path)):
    old_pocket_path = os.path.join(old_path, pocket)
    expanded_pocket_path = os.path.join(expanded_path, pocket)

    pdb_id = pocket[:4].lower()
    rglib_graph = graph_from_pdbid(pdb_id, redundancy='all')
    if rglib_graph is None:
        failed_set.add(pocket)
    old_pocket_graph = graph_io.load_json(old_pocket_path)
    new_nodes = {s[:4].lower() + s[4:] for s in old_pocket_graph.nodes}
    # Some weird edge cases happen
    # 6YMJ_A_ADN_102.json 14 16
    # 6YMJ_F_ADN_103.json 12 14
    # 6YMJ_A_ADN_103.json 12 14
    # 6YMJ_F_ADN_102.json 14 16
    new_nodes_filtered = new_nodes.intersection(set(rglib_graph.nodes()))
    if len(new_nodes_filtered) < len(new_nodes):
        print(pocket, len(new_nodes_filtered), len(new_nodes))
    expanded_nodes = graph_utils.bfs(rglib_graph, new_nodes_filtered, depth=4, label='LW')
    new_pocket_graph = rglib_graph.subgraph(expanded_nodes)
    in_pocket = {node: node in new_nodes_filtered for node in expanded_nodes}
    nt_codes = nx.get_node_attributes(new_pocket_graph, 'nt_code')
    edge_types = nx.get_edge_attributes(new_pocket_graph, 'LW')

    # New graph creation enables removing old attributes. (more lightweight)
    expanded_graph = nx.DiGraph()  # or whatever type of graph `G` is
    expanded_graph.add_edges_from(new_pocket_graph.edges())
    nx.set_node_attributes(expanded_graph, name='in_pocket', values=in_pocket)
    nx.set_node_attributes(expanded_graph, name='nt_code', values=nt_codes)
    nx.set_edge_attributes(expanded_graph, name='LW', values=edge_types)
    graph_io.dump_json(expanded_pocket_path, expanded_graph)

    # Annotate nodes with rings for pretraining.
    annotated_pocket_path = os.path.join(annotated_path, pocket)
    edge_rings = build_ring_tree_from_graph(expanded_graph, depth=2)
    nx.set_node_attributes(expanded_graph, name='edge_annots', values=edge_rings)
    graph_io.dump_json(annotated_pocket_path, expanded_graph)

print(failed_set)
print(f"{len(failed_set)}/{len(os.listdir(old_path))} failed systems")
