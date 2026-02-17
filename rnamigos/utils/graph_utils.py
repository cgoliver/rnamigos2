import os
from pathlib import Path

import dgl
import torch
from loguru import logger
import networkx as nx
import numpy as np

import rnaglib
from rnaglib.algorithms import multigraph_to_simple
from rnaglib.transforms import RNAFMTransform
from rnaglib.prepare_data import fr3d_to_graph
from rnaglib.utils import graph_io
from rnaglib.config import NODE_FEATURE_MAP
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB


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


def prepare_pocket(rna_path,
                   edge_map=EDGE_MAP_RGLIB,
                   undirected=False):
    """
    NetworkX Graph or path to a json => Networkx graph with the right fields
    """
    if isinstance(rna_path, (str, Path)):
        pocket_graph = graph_io.load_json(rna_path)
    if isinstance(rna_path, nx.Graph):
        pocket_graph = rna_path
        # Useful to keep a string for RNA-FM computations
        rna_path = pocket_graph.name
    # possibly undirected, just update the edge map to keep a DiGraph
    edge_map = to_undirected(edge_map) if undirected else edge_map
    edge_map = {key.upper(): value for key, value in edge_map.items()}
    one_hot = {
        edge: torch.tensor(edge_map[label.upper()])
        for edge, label in (nx.get_edge_attributes(pocket_graph, "LW")).items()
    }
    nx.set_edge_attributes(pocket_graph, name="edge_type", values=one_hot)

    # Needed for graph creation with fr3d, the key changed from nt_code to nt
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
    pocket_nodes = {node: label for node, label in (nx.get_node_attributes(pocket_graph, "in_pocket")).items()}
    # By default, or if some nodes were missing, set them to True
    pocket_nodes = {node: True if node not in pocket_nodes else pocket_nodes[node] for node in pocket_graph.nodes()}
    nx.set_node_attributes(pocket_graph, name="in_pocket", values=pocket_nodes)
    return pocket_graph


def add_rnafm(pocket_nx,
              rna_path,
              pocket_cache_path="data/pocket_embeddings",
              cache_path="data/pocket_chain_embeddings"):
    cached_pocket_embs_path = os.path.join(pocket_cache_path, f"{Path(rna_path).stem}.npz")
    embs = np.load(cached_pocket_embs_path)
    # Now if some nodes are still missing and complement those with the mean embedding
    nx.set_node_attributes(pocket_nx, embs, "rnafm")
    existing_rnafm_embs_dict = nx.get_node_attributes(pocket_nx, "rnafm")
    if len(existing_rnafm_embs_dict) > 0:
        existing_nodes, existing_embs = list(zip(*existing_rnafm_embs_dict.items()))
    else:
        existing_nodes, existing_embs = [], []
    missing_nodes = set(pocket_nx.nodes()) - set(existing_nodes)
    n = len(pocket_nx.nodes())
    # If only a fraction is missing, just skip and compute mean embedding
    if n * 0.15 > len(missing_nodes) > 0:
        if isinstance(existing_embs[0], np.ndarray):
            existing_embs = [torch.from_numpy(x) for x in existing_embs]
        mean_emb = torch.mean(torch.stack(existing_embs), dim=0)
        missing_embs = {node: mean_emb for node in missing_nodes}
        nx.set_node_attributes(pocket_nx, name="rnafm", values=missing_embs)

    # Otherwise load the whole graphs embeddings. This step is useful for pocket perturbation.
    elif len(missing_nodes) >= n * 0.15:
        pdb_id = list(missing_nodes)[0].split(".")[0].upper()
        large_embs = np.load(os.path.join(cache_path, f"{pdb_id}.npz"))
        nx.set_node_attributes(pocket_nx, large_embs, "rnafm")

        # Just triple check
        existing_nodes, existing_embs = list(zip(*nx.get_node_attributes(pocket_nx, "rnafm").items()))
        missing_nodes = set(pocket_nx.nodes()) - set(existing_nodes)
        if len(missing_nodes) > 0:
            raise Exception

    rnafm_embs = nx.get_node_attributes(pocket_nx, name="rnafm")
    pre_feats = nx.get_node_attributes(pocket_nx, name="nt_features")
    combined = {}
    for node, pre_feat in pre_feats.items():
        rna_emb = rnafm_embs[node]
        if isinstance(rna_emb, np.ndarray):
            rna_emb = torch.from_numpy(rna_emb)
        combined[node] = torch.cat((pre_feat, rna_emb)).float()
    nx.set_node_attributes(pocket_nx, name="nt_features", values=combined)


def nx_to_dgl(pocket_graph, use_rings=False):
    pocket_graph_dgl = dgl.from_networkx(
        nx_graph=pocket_graph,
        edge_attrs=["edge_type"],
        node_attrs=["nt_features", "in_pocket"],
    )
    rings = []
    if use_rings:
        for node, data in sorted(pocket_graph.nodes(data=True)):
            if data["in_pocket"]:
                rings.append(data["edge_annots"])
    return pocket_graph_dgl, rings


def load_rna_graph(rna_path,
                   edge_map=EDGE_MAP_RGLIB,
                   undirected=False,
                   use_rings=False,
                   use_rnafm=False,
                   ):
    """
    NetworkX Graph or path to a json => DGL graph
    """
    pocket_graph = prepare_pocket(rna_path=rna_path, edge_map=edge_map, undirected=undirected)
    # Optionally add rna_fm embs
    if use_rnafm:
        add_rnafm(pocket_graph, rna_path)
    # a = list(pocket_graph.nodes(data=True))
    # b = pocket_graph_dgl.ndata['rnafm']
    return nx_to_dgl(pocket_graph=pocket_graph, use_rings=use_rings)


def get_dgl_graph(cif_path, residue_list, undirected=False, use_rnafm=False):
    """
    :param cif_path: toto/tata/1cqr.cif
    :param residue_list: list of strings "A.2","A.3",... ,"A.85" (missing pdb, useful for inference)
    :return:
    """
    ### DATA PREP
    # convert cif to graph and keep only relevant keys
    multi_nx_graph = fr3d_to_graph(cif_path)
    nx_graph = multigraph_to_simple(multi_nx_graph)

    buggy_nodes = []
    for node in nx_graph.nodes():
        try:
            nx_graph.nodes[node]["nt"]
        except KeyError:
            buggy_nodes.append(node)
    nx_graph.remove_nodes_from(buggy_nodes)
    logger.warning(
        f"Conversion of mmCIF to graph by fr3d-python created {len(buggy_nodes)} residues with missing residue IDs. "
        f"Removing {buggy_nodes} from the graph."
    )

    if use_rnafm:
        nx_graph = RNAFMTransform()({"rna": nx_graph})['rna']

    # This is the pdbid used by fr3d
    pdbid = Path(cif_path).stem.lower()
    if residue_list is not None:
        # subset cif with given reslist
        reslist = [f"{pdbid}.{res}" for res in residue_list]
        expanded_reslist = rnaglib.algorithms.bfs(nx_graph, reslist, depth=4, label="LW")
        in_pocket = {node: node in reslist for node in expanded_reslist}
        expanded_graph = nx_graph.subgraph(expanded_reslist)
        nx.set_node_attributes(expanded_graph, name="in_pocket", values=in_pocket)
    else:
        expanded_graph = nx_graph
        in_pocket = {node: True for node in nx_graph.nodes}
        nx.set_node_attributes(expanded_graph, name="in_pocket", values=in_pocket)

    # Here, we don't use load_graph because default add_rna expects rnafm embs to be in cache,
    # while we just computed them
    pocket_graph = prepare_pocket(expanded_graph, undirected=undirected)
    if use_rnafm:
        rnafm_embs = nx.get_node_attributes(pocket_graph, name="rnafm")
        pre_feats = nx.get_node_attributes(pocket_graph, name="nt_features")
        combined = {}
        for node, pre_feat in pre_feats.items():
            rna_emb = torch.from_numpy(np.asarray(rnafm_embs[node]))
            combined[node] = torch.cat((pre_feat, rna_emb))
        nx.set_node_attributes(pocket_graph, name="nt_features", values=combined)

    dgl_graph, _ = nx_to_dgl(pocket_graph=pocket_graph)
    return dgl_graph


if __name__ == "__main__":
    g = load_rna_graph("data/json_pockets_expanded/1DDY_G_B12_701.json", use_rnafm=True)
    pass
