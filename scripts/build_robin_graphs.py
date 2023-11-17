from pathlib import Path
import networkx as nx
from collections import Counter

import numpy as np
import pandas as pd
from Bio.PDB import *
#from rnaglib.prepare_data import fr3d_to_graph
from rnaglib.drawing import rna_draw
from rnaglib.utils.graph_utils import bfs
from rnaglib.utils import dump_json
from rnaglib.utils import load_json
from rnaglib.utils import graph_from_pdbid

CANONICALS = ['B53', 'B35', 'CWW']

def ligand_center(residue):
    return np.mean(np.array([atom.coord for atom in residue.get_atoms()]), axis=0)


def get_reslist(pdb_path, ligand_id, cutoff=10):
    parser = MMCIFParser(QUIET=True)
    struc = parser.get_structure("", pdb_path)[0]

    for res in struc.get_residues():
        if res.get_resname() == ligand_id:
            ligand_res = res

    center = ligand_center(ligand_res)

    kd = NeighborSearch(list(struc.get_atoms()))
    pocket = kd.search(ligand_center(ligand_res), cutoff, level='R')

    return [f"{pdb_path.stem.lower()}.{p.get_parent().id}.{p.id[1]}" for p in pocket]

if __name__ == "__main__":
    df = pd.read_csv("../data/ROBIN.csv")
    pdbl = PDBList()
    dl_path = Path("ROBIN_PDBS")
    dl_path.mkdir(parents=True, exist_ok=True)

    backend = 'fr3d'
    backend = 'x3dna'

    pockets = []
    rows = []
    for pocket in df.itertuples():
        # get the pdb
        pdbl.retrieve_pdb_file(pocket.PDBID, pdir=dl_path)
        if backend == 'fr3d':
            big_G = fr3d_to_graph(Path(dl_path, pocket.PDBID + ".cif"))
        elif backend == 'x3dna':
            big_G = graph_from_pdbid(pocket.PDBID, redundancy='all')
            pass
        else:
            print("invalid backend")
            sys.exit()

        if big_G is None:
            print(f"{pocket.PDBID} no graph")
            continue

        print(pocket.LIGAND)
        print(pocket.PDBID)
        if not pd.isna(pocket.LIGAND): 
            print("Ligand found")
            pocket_res = get_reslist(Path(dl_path, pocket.PDBID + ".cif"), pocket.LIGAND)
            pocket_res = [r for r in pocket_res if r in big_G.nodes()]
            G = big_G.subgraph(pocket_res).copy()
        elif not pd.isna(pocket.RESLIST) :
            print("reslist")
            pocket_res = pocket.RESLIST.split(";")
            print(len(pocket_res))
            G = big_G.subgraph(pocket.RESLIST.split(";")).copy()
        else:
            pocket_res = list(G.nodes())
            G = big_G
            pass

        expanded_nodes = bfs(big_G, list(G.nodes()), depth=4, label='LW')
        G_expand = big_G.subgraph(expanded_nodes).copy()
        # rna_draw(G_expand, show=True)
        nx.set_node_attributes(G_expand, {n: True if n in pocket_res else False for n in G_expand.nodes()}, 'in_pocket')
        NC_before = Counter([d['LW'].upper() for _,_,d in G.edges(data=True)])
        NC_after = Counter([d['LW'].upper() for _,_,d in G_expand.edges(data=True)])

        pockets.append(G_expand)
        if sum([c for k,c in NC_after.items()]) > 1:
            dump_json(f"../data/robin_graphs_{backend}/{pocket.ROBIN_ID}.json", G_expand)

        if sum([c for k,c in NC_after.items() if k not in CANONICALS]) > 1:
            dump_json(f"../data/robin_graphs_{backend}/{pocket.ROBIN_ID}.json", G_expand)

        rows.append({"ROBIN ID": pocket.ROBIN_ID, 
                     "PDBID": pocket.PDBID,
                     "Ligand": pocket.LIGAND,
                     "Num Nodes": len(G.nodes()),
                     "Num Nodes 1-hop BFS": len(G_expand.nodes()),
                     "Num edges": len(G.edges()),
                     "Num edges 1-hop BFS": len(G_expand.edges()),
                     "Non-canonical": sum([c for k,c in NC_before.items() if k not in CANONICALS]),
                     "Non-canonical 1-hop BFS": sum([c for k,c in NC_after.items() if k not in CANONICALS]),
                     }
                    )

    df = pd.DataFrame(rows)
    print(df.to_markdown())
    print(df.to_latex())

