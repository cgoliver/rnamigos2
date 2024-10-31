"""
In this file, we compute pockets corruptions:
- different perturbs
- get_expanded_subgraph_from_list : just a util to get a graph.json from a nodelist
- get_perturbed_pockets: different strategies to build perturbed node lists (and then graphs) from a given pocket
- compute_overlaps: a postprocessing function, for each perturbed pocket, it computes the overlap with the GT pocket
"""

import os

import networkx as nx
import numpy as np
import pandas as pd
from rnaglib.algorithms.graph_algos import bfs
from rnaglib.utils import graph_from_pdbid, graph_io
import torch
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)


def get_expanded_subgraph_from_list(rglib_graph, nodelist, bfs_depth=4):
    expanded_nodes = bfs(rglib_graph, nodelist, depth=bfs_depth, label="LW")
    new_pocket_graph = rglib_graph.subgraph(expanded_nodes)
    in_pocket = {node: node in nodelist for node in expanded_nodes}
    nt_codes = nx.get_node_attributes(new_pocket_graph, "nt_code")
    edge_types = nx.get_edge_attributes(new_pocket_graph, "LW")

    # New graph creation enables removing old attributes. (more lightweight)
    expanded_graph = nx.DiGraph()  # or whatever type of graph `G` is
    expanded_graph.add_edges_from(new_pocket_graph.edges())
    nx.set_node_attributes(expanded_graph, name="in_pocket", values=in_pocket)
    nx.set_node_attributes(expanded_graph, name="nt_code", values=nt_codes)
    nx.set_edge_attributes(expanded_graph, name="LW", values=edge_types)
    return expanded_graph


def random_perturb(around_pocket, n_nodes_to_sample):
    # just use random nodes from the list
    sorted_around_pocket = sorted(list(around_pocket))
    noisy_nodelist = list(np.random.choice(sorted_around_pocket, replace=False, size=n_nodes_to_sample))
    return noisy_nodelist


def soft_perturb(around_pocket, in_pocket_filtered, n_nodes_to_sample):
    # start from the pocket, and subsample/oversample, starting from the pocket
    sorted_neighbors_bfs = sorted(list(around_pocket.difference(in_pocket_filtered)))
    sorted_in_pocket = sorted(list(in_pocket_filtered))

    shuffled_in_pocket = list(np.random.choice(sorted_in_pocket, replace=False, size=len(in_pocket_filtered)))
    shuffled_neigh = list(np.random.choice(sorted_neighbors_bfs, replace=False, size=len(sorted_neighbors_bfs)))
    shuffled_in_pocket.extend(shuffled_neigh)
    noisy_nodelist = shuffled_in_pocket[:n_nodes_to_sample]
    return noisy_nodelist


def hard_perturb(around_pocket, rglib_graph, in_pocket_filtered, perturb_bfs_depth, n_nodes_to_sample):
    # Sample a pocket around a random node of the perimeter
    smaller_bfs = bfs(rglib_graph, in_pocket_filtered, depth=perturb_bfs_depth - 1, label="LW")
    perimeter = sorted(list(around_pocket.difference(smaller_bfs)))
    if len(perimeter) == 0:
        raise ValueError(f"Buggy pocket it spans the whole connected component and cannot be expanded")
    seed_pertubed_pocket = np.random.choice(perimeter, size=1).item()

    # Now expand this seed with increasing radius up to getting more than target node
    prev_perturbed_pocket = {}
    perturbed_pocket = {seed_pertubed_pocket}
    expander = 1
    while len(perturbed_pocket) < n_nodes_to_sample and expander <= 10:
        prev_perturbed_pocket = perturbed_pocket
        perturbed_pocket = bfs(rglib_graph, perturbed_pocket, depth=expander, label="LW")
        expander += 1
        # When querying with very large fractions, sometimes we cannot return as many nodes as queried
        # Note: nx.connected_component does not work for directed graphs...
        if expander > 10:
            print("Cannot craft a large enough pocket, maybe we seeded using a disconnected component")
            break

    # Finally, subsample the last parameter to get the final pocket.
    last_perimeter = sorted(list(perturbed_pocket.difference(prev_perturbed_pocket)))
    missing_nbr_nodes = n_nodes_to_sample - len(prev_perturbed_pocket)
    last_nodes = list(np.random.choice(list(last_perimeter), replace=False, size=missing_nbr_nodes))
    noisy_nodelist = list(prev_perturbed_pocket) + last_nodes
    return noisy_nodelist


def rognan_like(rglib_graph, n_nodes_to_sample):
    # Sample a pocket around a random node of the perimeter
    seed_pertubed_pocket = np.random.choice(rglib_graph.nodes(), size=1).item()

    # Now expand this seed with increasing radius up to getting more than target node
    prev_perturbed_pocket = {}
    perturbed_pocket = {seed_pertubed_pocket}
    expander = 1
    while len(perturbed_pocket) < n_nodes_to_sample and expander <= 10:
        prev_perturbed_pocket = perturbed_pocket
        perturbed_pocket = bfs(rglib_graph, perturbed_pocket, depth=expander, label="LW")
        expander += 1
    # When querying with very large fractions, sometimes we cannot return as many nodes as queried
    # Note: nx.connected_component does not work for directed graphs...
    if expander > 10:
        raise ValueError("Cannot craft a large enough pocket, maybe we seeded using a disconnected component")

    # Finally, subsample the last parameter to get the final pocket.
    last_perimeter = sorted(list(perturbed_pocket.difference(prev_perturbed_pocket)))
    missing_nbr_nodes = n_nodes_to_sample - len(prev_perturbed_pocket)
    last_nodes = list(np.random.choice(list(last_perimeter), replace=False, size=missing_nbr_nodes))
    noisy_nodelist = list(prev_perturbed_pocket) + last_nodes
    return noisy_nodelist


def get_perturbed_pockets(all_pockets,
                          unperturbed_path="data/json_pockets_expanded",
                          out_path="figs/perturbations/perturbed",
                          fractions=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
                          perturb_bfs_depth=1,
                          max_replicates=5,
                          recompute=True,
                          perturbation="random",
                          final_bfs=4,
                          ):
    existing_pockets = set([pocket.rstrip(".json") for pocket in os.listdir(unperturbed_path)])
    pockets_to_compute = sorted(list(existing_pockets.intersection(all_pockets)))
    failed_set = set()
    for pocket in tqdm(pockets_to_compute):
        # Get rglib grpah
        pdb_id = pocket[:4].lower()
        rglib_graph = graph_from_pdbid(pdb_id, redundancy="all")
        if rglib_graph is None:
            failed_set.add(pocket)

        # Get pocket graph and hence initial nodelist
        unperturbed_pocket_path = os.path.join(unperturbed_path, f"{pocket}.json")
        old_pocket_graph = graph_io.load_json(unperturbed_pocket_path)
        in_pocket_nodes = {
            node[:4].lower() + node[4:]
            for node, in_pocket in old_pocket_graph.nodes(data="in_pocket")
            if in_pocket
        }

        # Ensure all nodes are valid and expand with a small bfs
        in_pocket_filtered = in_pocket_nodes.intersection(set(rglib_graph.nodes()))
        around_pocket = bfs(rglib_graph, in_pocket_filtered, depth=perturb_bfs_depth, label="LW")

        # Now compute the perturbed pockets
        for fraction in fractions:
            n_nodes_to_sample = int(fraction * len(in_pocket_filtered))
            n_nodes_to_sample = min(max(n_nodes_to_sample, 1), len(around_pocket))
            for replicate in range(max_replicates):
                # Setup dirs
                out_dir = os.path.join(out_path, f"perturbed_{fraction}_{replicate}")
                os.makedirs(out_dir, exist_ok=True)
                if perturbation == "rognan_true":
                    actual_pocket_id = pockets_to_compute.index(pocket)
                    rognaned = (actual_pocket_id + replicate + 1) % len(pockets_to_compute)
                    out_name = os.path.join(out_dir, f"{pockets_to_compute[rognaned]}.json")
                else:
                    out_name = os.path.join(out_dir, f"{pocket}.json")

                if os.path.exists(out_name) and not recompute:
                    continue

                try:
                    # Sample a broken binding site
                    # To get reproducible results, we need to sort sets
                    if perturbation == "random":
                        noisy_nodelist = random_perturb(around_pocket, n_nodes_to_sample)
                    elif perturbation == "soft":
                        noisy_nodelist = soft_perturb(around_pocket, in_pocket_filtered, n_nodes_to_sample)
                    elif perturbation == "hard":
                        noisy_nodelist = hard_perturb(around_pocket, rglib_graph, in_pocket_filtered, perturb_bfs_depth,
                                                      n_nodes_to_sample)
                    elif perturbation == "rognan_like":
                        noisy_nodelist = rognan_like(rglib_graph, n_nodes_to_sample)
                    elif perturbation == "rognan_true":
                        noisy_nodelist = soft_perturb(around_pocket, in_pocket_filtered, n_nodes_to_sample)
                    else:
                        raise NotImplementedError
                    expanded_graph = get_expanded_subgraph_from_list(
                        rglib_graph=rglib_graph,
                        nodelist=noisy_nodelist,
                        bfs_depth=final_bfs,
                    )
                except Exception as e:
                    if isinstance(e, NotImplementedError):
                        raise e
                    else:
                        print(e)
                    expanded_graph = []
                if len(expanded_graph) == 0:
                    print("Tried to create empty graph, skipped system: ", pocket, fraction, replicate)
                    continue
                graph_io.dump_json(out_name, expanded_graph)


def compute_overlaps(original_pockets, modified_pockets_path, dump_path=None):
    """
    Given a directory of modified pockets and a dict of reference ones, return a dict
    with the overlap between modified and original pockets.
    """
    resdict = {}
    for modified_pocket_name in os.listdir(modified_pockets_path):
        modified_pocket_path = os.path.join(modified_pockets_path, modified_pocket_name)
        modified_pocket = graph_io.load_json(modified_pocket_path)
        pocket_id = modified_pocket_name[:-5]
        original_pocket = original_pockets[pocket_id]
        mod_nodes = set(modified_pocket.nodes())
        ori_nodes = set(original_pocket.nodes())
        extra_nodes = len(mod_nodes.difference(ori_nodes))
        missing_nodes = len(ori_nodes.difference(mod_nodes))
        pocket_size = len(ori_nodes)
        # if extra_nodes > 10:
        #     a = 1
        resdict[pocket_id] = (extra_nodes, missing_nodes, pocket_size)
    if dump_path is not None:
        rows = [{"pocket_id": pocket_id, "extra": extra, "missing": missing, "pocket_size": pocket_size}
                for pocket_id, (extra, missing, pocket_size) in resdict.items()]
        df = pd.DataFrame(rows)
        df.to_csv(dump_path, index=False)
    return resdict
