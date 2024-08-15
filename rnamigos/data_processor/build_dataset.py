"""
    Takes ligand residue IDs from lig_dict (built by `binding_pocket_filter.py`)
    and builds a networkx binding site graph for each ligand.
"""

import os, sys

if __name__ == "__main__":
    sys.path.append("..")
import pickle
import itertools
import subprocess

import numpy as np
import networkx as nx
from Bio.PDB import *
from tqdm import tqdm

from rnamigos.data_processor.graph_process import find_node, dangle_trim, kill_islands, to_orig, bfs_expand

faces = ['W', 'S', 'H']
orientations = ['C', 'T']
valid_edges = set(
    ['B53'] + [orient + "".join(sorted(e1 + e2)) for e1, e2 in itertools.product(faces, faces) for orient in
               orientations])


def lig_center(lig_atoms):
    return np.mean(np.array([a.coord for a in lig_atoms]), axis=0)


def find_residue(chain, pos):
    """
        Get BioPython residue object from chain and position.

        Arguments:
            chain (BioPython.Chain): chain object
            pos (int): position on chain
        Returns:
            Bio.PDB.Residue: residue object matching position.
    """
    for residue in chain:
        if residue.id[1] == pos:
            return residue
    return None


def graph_from_residues(full_graph, residues, expand_depth=0):
    """
        Build an NX graph from list of biopython residues.

        Arguments:
            full_graph (Networkx graph): networkx graph of full RNA structure.
            residues (list): list of BioPython Residues to get graph for.
            expand_depth (int): number of hops in graph to extend pocket by.

        Returns:
            Networkx graph: networkx graph with nodes matching `residuses`
    """
    pocket_nodes = []
    for r in residues:
        node = find_node(full_graph, r.get_parent().id, r.id[1])
        if node is not None:
            pocket_nodes.append(node)
    if expand_depth:
        pocket_nodes = bfs_expand(full_graph, pocket_nodes, depth=expand_depth)

    pocket_graph = full_graph.subgraph(pocket_nodes).copy()
    G = to_orig(pocket_graph)

    # remove bases with no connections
    kill_islands(G)
    # remove dangles (only backbone interactions not in loops)
    G = dangle_trim(G)

    return G


def get_pocket_graph(pdb_structure_path, ligand_id, graph,
                     ablate=None, dump_path="../data/pockets_nx", cutoff=10,
                     non_binding=False, max_non_bind_samples=5):
    """
        Main function for extracting a graph from a binding site.
        Dumps graph representing the pocket around given ligand in PDB.

        Arguments;
            pdb_structure_path (str): path to full PDB structure
            ligand_id (str): string describing ligand position in PDB
            graph (Networkx): full graph of RNA structure
            ablate (str): which ablation to perform on final graph (see `graph_process.py`)
            dump_path (str): path to write binding site graph
            cutoff (int): Maximum distance from ligand to include for graph residues. (default=10)
            non_binding (bool): If True, samples non-binding sites for each binding site (default=False).
            max_non_bind_sample (int): Number of non-binding sites to sample. (default=5)

        Returns:
            networkx graph: graph representing binding site around ligand.
    """
    # load PDB
    print(ligand_id)
    parser = MMCIFParser(QUIET=True)
    pdbid = os.path.basename(pdb_structure_path).split(".")[0]
    structure = parser.get_structure("", pdb_structure_path)[0]

    chain, resname, pos = ligand_id.split(":")[1:]

    # find ligand residue and get its center coordinates
    lig_residue = find_residue(structure[chain], int(pos))
    lig_res_atoms = lig_residue.get_atoms()
    lig_coord = lig_center(lig_res_atoms)

    # get atoms within radius
    kd = NeighborSearch(list(structure.get_atoms()))
    pocket = kd.search(lig_coord, cutoff, level='R')

    print('------------ información del pocket ---------------------')
    print(len(pocket))
    print(pocket)

    G = graph_from_residues(graph, pocket)

    # visualize on 3D structure
    # pdb_to_markers_(structure, G, "markers.cmm") # subprocess.call(['chimera', pdb_structure_path, 'markers.cmm'])

    # os.remove("markers.cmm")

    labels = {d['label'] for _, _, d in G.edges(data=True)}

    assert labels.issubset(valid_edges)

    # rna_draw(G, title="BINDING")
    if len(G.nodes()) < 4:
        return None

    if dump_path:
        nx.write_gpickle(G, os.path.join(dump_path, f"{pdbid}_{ligand_id}_BIND.nx"))

    # sample and build non-binding graph.
    if non_binding:
        sampled = 0
        site_sampler = sample_non_binding_sites(pdb_structure_path, lig_residue)
        for pocket in site_sampler:
            if sampled >= max_non_bind_samples:
                break
            if pocket:
                non_bind_graph = graph_from_residues(graph, pocket)
                if dump_path and (len(non_bind_graph.nodes()) > 4):
                    # nx.write_gpickle(G, os.path.join(dump_path, f"{pdbid}_{ligand_id}_NON_{sampled}.nx"))
                    sampled += 1
                # rna_draw(non_bind_graph, title="NON BINDING")
                # pdb_to_markers_(structure, non_bind_graph, "markers.cmm")
                # subprocess.call(['chimera', pdb_structure_path, 'markers.cmm'])
                # os.remove("markers.cmm")
        print(f">>> Sampled {sampled} non-binding sites for this pocket of {max_non_bind_samples}.")
    return G


def get_binding_site_graphs_all(lig_dict_path, dump_path, non_binding=False):
    """
        Get binding site graphs for each ligand in lig_dict.
        lig_dict is a dictionary with {'pdbid': [ligand_residues]} generated by `binding_sites.py`
        Saves a binding pocket graph for each ligand.

        Args:
            lig_dict_path (str): Path to pdb ligand annotations dictionary (see `binding_sites.py`)
            dump_path (str): Path to folder where annotated graphs will be dumped.
            non_binding (bool): If `True` will sample non-binding sites which can be used for binding
                                site finding.

    """

    lig_dict = pickle.load(open(lig_dict_path, 'rb'))

    print(f">>> building graphs for {len(lig_dict)} PDBs")
    print(f">>> dumping in {dump_path}")
    print(f">>> and {sum(map(len, lig_dict.values()))} binding sites.")

    failed = 0

    try:
        os.mkdir(dump_path)
    except:
        pass

    done_pdbs = {f.split('_')[0] for f in os.listdir(dump_path)}
    print(f">>> skipping {len(done_pdbs)}")

    failed = []
    empties = 0
    num_found = 0
    missing_graphs = []
    for pdbid, ligs in tqdm(lig_dict.items()):
        pdbid = pdbid.split(".")[0]
        # pdb_path = f"../data/all_rna_prot_lig_2019/{pdbid}.cif"
        # pdb_path = f"../../carlos_docking/data/all_rna_with_lig_2019/{pdbid}.cif"
        pdb_path = f"../data/cif_files/{pdbid}.cif"
        if pdbid in done_pdbs:
            continue
        # try:
        print(">>> ", pdbid)
        try:
            # pdb_graph = pickle.load(open(f'../data/RNA_Graphs/{pdbid}.pickle', 'rb'))
            pdb_graph = pickle.load(open(f'../data/networkx_graphs/{pdbid}.p', 'rb'))
        except FileNotFoundError:
            print(f"{pdbid} graph not found.")
            print(f"{pdbid} had {len(ligs)} binding sites")
            missing_graphs.append(pdbid)
            continue
        try:
            pdb_graph.nodes()
        except AttributeError:
            print("empty graph")
            continue
        # print(f"new guy: {pdbid}")
        # continue
        for lig in ligs:
            # dump binding site graphs
            try:
                g = get_pocket_graph(pdb_path, lig,
                                     pdb_graph, dump_path=dump_path,
                                     non_binding=non_binding)
                if g is None:
                    empties += 1
                else:
                    num_found += 1
                    print(f">>> pockets so far {num_found}")

            except FileNotFoundError:
                print(f"{pdbid} not found")
                failed.append(pdbid)
    print(f">>> missing graphs for {missing_graphs}")
    print(f">>> failed on {len(failed)} graphs")
    print(f">>> got {empties} empty graphs")


if __name__ == "__main__":
    # take all ligands with 8 angstrom sphere and 0.6 RNA concentration, build a graph for each.
    # get_binding_site_graphs_all('../data/lig_dict_c_8A_06rna.p','../data/pockets_nx_pfind',
    # non_binding=True)
    # get_binding_site_graphs_all('../data/lig_dict_ismb_rna06_rad10.p', '../data/pockets_nx_ismb', non_binding=False)
    get_binding_site_graphs_all('../data/lig_dict_filter.p', '../data/pockets_nx', non_binding=False)
    pass
