import os
from rnaglib.utils import graph_from_pdbid, graph_utils, graph_io
from tqdm import tqdm
import networkx as nx

old_path = 'data/json_pockets'
new_path = 'data/json_pockets_new'
load_path = 'data/json_pockets_load'
expanded_path = 'data/json_pockets_expanded'
os.makedirs(new_path, exist_ok=True)
os.makedirs(load_path, exist_ok=True)
os.makedirs(expanded_path, exist_ok=True)

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
    expanded_graph = nx.DiGraph()  # or whatever type of graph `G` is
    expanded_graph.add_edges_from(new_pocket_graph.edges())
    nx.set_node_attributes(expanded_graph, name='in_pocket', values=in_pocket)
    nx.set_node_attributes(expanded_graph, name='nt_code', values=nt_codes)
    nx.set_edge_attributes(expanded_graph, name='LW', values=edge_types)
    graph_io.dump_json(expanded_pocket_path, expanded_graph)
#
# print(failed_set)
# print(f"{len(failed_set)}/{len(os.listdir(old_path))} failed systems")

#
# failed_set = set()
# for pocket in tqdm(os.listdir(old_path)):
#     old_pocket_path = os.path.join(old_path, pocket)
#     pdb_id = pocket[:4].lower()
#     rglib_graph = graph_from_pdbid(pdb_id, redundancy='all')
#     if rglib_graph is None:
#         failed_set.add(pocket)
#     old_pocket_graph = graph_io.load_json(old_pocket_path)
#     new_nodes = {s[:4].lower() + s[4:] for s in old_pocket_graph.nodes}
#     new_pocket_graph = rglib_graph.subgraph(new_nodes)
#     new_pocket_path = os.path.join(new_path, pocket)
#     graph_io.dump_json(new_pocket_path, new_pocket_graph)
# print(failed_set)
# print(f"{len(failed_set)}/{len(os.listdir(old_path))} failed systems")
#
# failed_set = set()
# for pocket in tqdm(os.listdir(new_path)):
#     new_pocket_path = os.path.join(new_path, pocket)
#     new_pocket_graph = graph_io.load_json(new_pocket_path)
#
#     nt_codes = nx.get_node_attributes(new_pocket_graph, 'nt_code')
#     edge_types = nx.get_edge_attributes(new_pocket_graph, 'LW')
#     load_graph = nx.DiGraph()  # or whatever type of graph `G` is
#     load_graph.add_edges_from(new_pocket_graph.edges())
#     nx.set_node_attributes(load_graph, name='nt_code', values=nt_codes)
#     nx.set_edge_attributes(load_graph, name='LW', values=edge_types)
#     load_pocket_path = os.path.join(load_path, pocket)
#     graph_io.dump_json(load_pocket_path, load_graph)
#
# print(failed_set)
# print(f"{len(failed_set)}/{len(os.listdir(old_path))} failed systems")
