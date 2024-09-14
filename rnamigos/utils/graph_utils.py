import os
from pathlib import Path

import dgl
import torch
from loguru import logger
import networkx as nx
import numpy as np

import rnaglib
from rnaglib.prepare_data import fr3d_to_graph
from rnaglib.utils import graph_io
from rnaglib.config import NODE_FEATURE_MAP
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB
from rnaglib.transforms import RNAFMTransform


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
    undirected_edge_map = {
        old_label: new_map[remap[old_label]] for old_label in edge_map.keys()
    }
    return undirected_edge_map


def add_rnafm(pocket_nx, rna_path, cache_path="data/pocket_embeddings"):
    embs = np.load(Path("data/pocket_embeddings") / Path(Path(rna_path).stem + ".npz"))
    # Now if some nodes are still missing and complement those with the mean embedding
    nx.set_node_attributes(pocket_nx, embs, "rnafm")
    existing_nodes, existing_embs = list(
        zip(*nx.get_node_attributes(pocket_nx, "rnafm").items())
    )
    missing_nodes = set(pocket_nx.nodes()) - set(existing_nodes)
    if len(missing_nodes) > 0:
        mean_emb = torch.mean(torch.stack(existing_embs), dim=0)
    missing_embs = {node: mean_emb for node in missing_nodes}
    nx.set_node_attributes(pocket_nx, name="rnafm", values=missing_embs)


def load_rna_graph(
    rna_path,
    edge_map=EDGE_MAP_RGLIB,
    undirected=False,
    use_rings=False,
    use_rnafm=False,
):
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
    one_hot = {
        edge: torch.tensor(edge_map[label.upper()])
        for edge, label in (nx.get_edge_attributes(pocket_graph, "LW")).items()
    }
    nx.set_edge_attributes(pocket_graph, name="edge_type", values=one_hot)

    # Needed for graph creation with fred, the key changed from nt_code to nt
    _, ndata = list(pocket_graph.nodes(data=True))[0]
    if "nt" in ndata.keys():
        nx.set_node_attributes(
            pocket_graph,
            name="nt_code",
            values={node: d["nt"] for node, d in pocket_graph.nodes(data=True)},
        )

    # One-hot encode it
    one_hot_nucs = {
        node: NODE_FEATURE_MAP["nt_code"].encode(label)
        for node, label in (nx.get_node_attributes(pocket_graph, "nt_code")).items()
    }
    nx.set_node_attributes(pocket_graph, name="nt_features", values=one_hot_nucs)

    # Add 1/0 flag to indicate which nodes are actually in the pocket (as opposed to ones used for expansion)
    pocket_nodes = {
        node: label
        for node, label in (nx.get_node_attributes(pocket_graph, "in_pocket")).items()
    }
    # By default, or if some nodes were missing, set them to True
    pocket_nodes = {
        node: True if node not in pocket_nodes else pocket_nodes[node]
        for node in pocket_graph.nodes()
    }
    nx.set_node_attributes(pocket_graph, name="in_pocket", values=pocket_nodes)

    # Optionally add rna_fm embs
    node_attrs = ["nt_features", "in_pocket"]
    if use_rnafm:
        add_rnafm(pocket_graph, rna_path)
        node_attrs.append("rnafm")

    pocket_graph_dgl = dgl.from_networkx(
        nx_graph=pocket_graph,
        edge_attrs=["edge_type"],
        node_attrs=node_attrs,
    )
    rings = []
    if use_rings:
        for node, data in sorted(pocket_graph.nodes(data=True)):
            if data["in_pocket"]:
                rings.append(data["edge_annots"])
    return pocket_graph_dgl, rings


def get_dgl_graph(cif_path, residue_list, undirected=False, use_rnafm=False):
    """
    :param cif_path: toto/tata/1cqr.cif
    :param residue_list: list of strings "A.2","A.3",... ,"A.85" (missing pdb, useful for inference)
    :return:
    """
    ### DATA PREP
    # convert cif to graph and keep only relevant keys
    nx_graph = fr3d_to_graph(cif_path)

    buggy_nodes = []
    for node in nx_graph.nodes():
        try:
            nx_graph.nodes[node]["nt"]
        except KeyError:
            buggy_nodes.append(node)

    nx_graph.remove_nodes_from(buggy_nodes)

    logger.warning(
        f"Conversion of mmCIF to graph by fr3d-python created {len(buggy_nodes)} residues with missing residue IDs. Removing {buggy_nodes} from the graph."
    )

    # This is the pdbid used by fr3d
    pdbid = Path(cif_path).stem.lower()
    if residue_list is not None:
        # subset cif with given reslist
        reslist = [f"{pdbid}.{res}" for res in residue_list]
        expanded_reslist = rnaglib.utils.graph_utils.bfs(
            nx_graph, reslist, depth=4, label="LW"
        )
        in_pocket = {node: node in reslist for node in expanded_reslist}
        expanded_graph = nx_graph.subgraph(expanded_reslist)
        nx.set_node_attributes(expanded_graph, name="in_pocket", values=in_pocket)
    else:
        expanded_graph = nx_graph
        in_pocket = {node: True for node in nx_graph.nodes}
        nx.set_node_attributes(expanded_graph, name="in_pocket", values=in_pocket)
    dgl_graph, _ = load_rna_graph(
        expanded_graph, undirected=undirected, use_rnafm=use_rnafm
    )
    return dgl_graph


if __name__ == "__main__":
    g = load_rna_graph("data/json_pockets_expanded/1DDY_G_B12_701.json", use_rnafm=True)
    pass
