import os
import sys

from rnaglib.utils import graph_io
from collections import defaultdict

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.inference import inference
from rnaglib.drawing import rna_draw
from rnamigos.utils.graph_utils import load_rna_graph


# GET REFERENCE POCKETS
def group_reference_pockets(pockets_path='data/json_pockets_expanded'):
    all_pockets = os.listdir(pockets_path)
    # Group pockets by pdb_id
    grouped_pockets = defaultdict(list)
    for pocket in all_pockets:
        pdb_id = pocket.split('_')[0]
        grouped_pockets[pdb_id].append(pocket)
    return grouped_pockets


ROBIN_SYSTEMS = """2GDI	TPP TPP 
6QN3	GLN  Glutamine_RS
5BTP	AMZ  ZTP
2QWY	SAM  SAM_ll
3FU2	PRF  PreQ1
"""


def get_nodelists_and_ligands(robin_systems=ROBIN_SYSTEMS, pockets_path='data/json_pockets_expanded'):
    """
    This was useful to compare pockets appearing in a given ROBIN PDB.
    """
    grouped_pockets = group_reference_pockets(pockets_path=pockets_path)
    nodelists = dict()
    ligand_names = dict()
    for robin_sys in robin_systems.splitlines():
        robin_pdb_id = robin_sys.split()[0]
        ligand_name = robin_sys.split()[2]
        copies = grouped_pockets[robin_pdb_id]
        # print(copies)
        # ['2GDI_Y_TPP_100.json', '2GDI_X_TPP_100.json'] ~ copies
        # []
        # ['5BTP_A_AMZ_106.json', '5BTP_B_AMZ_108.json'] ~ copies
        # ['2QWY_A_SAM_100.json', '2QWY_B_SAM_300.json', '2QWY_C_SAM_500.json'] ~ copies
        # ['3FU2_C_PRF_101.json', '3FU2_A_PRF_101.json', '3FU2_B_PRF_101.json'] ~ copies
        # Since they are all ~ copies, we can remove them.
        if len(copies) == 0:
            continue
        else:
            representative_pocket = copies[0]
            pocket_path = os.path.join(pockets_path, representative_pocket)
            pocket_graph = graph_io.load_json(pocket_path)
            colors = ['blue' if in_pocket else 'white' for n, in_pocket in pocket_graph.nodes(data='in_pocket')]
            rna_draw(pocket_graph, node_colors=colors, layout='spring', show=True)

            node_list = [node[5:] for node, in_pocket in pocket_graph.nodes(data='in_pocket') if in_pocket]
            nodelists[representative_pocket] = node_list
            ligand_names[representative_pocket] = ligand_name
    return nodelists, ligand_names


def robin_inference(ligand_name, dgl_pocket_graph, out_path=None):
    actives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "actives.txt")
    actives_smiles_list = [s.lstrip().rstrip() for s in list(open(actives_ligands_path).readlines())]
    inactives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "decoys.txt")
    inactives_smiles_list = [s.lstrip().rstrip() for s in list(open(inactives_ligands_path).readlines())]
    smiles_list = actives_smiles_list + inactives_smiles_list
    is_active = [1 for _ in range(len(actives_smiles_list))] + [0 for _ in range(len(inactives_smiles_list))]
    final_df = inference(dgl_graph=dgl_pocket_graph,
                         smiles_list=smiles_list,
                         dump_all=True,
                         out_path=out_path)
    final_df['is_active'] = is_active
    return final_df


if __name__ == "__main__":
    pass
    expanded_path = 'data/json_pockets_expanded'
    nodelists, ligand_names = get_nodelists_and_ligands()
    out_dir = 'outputs/robin_docknative'
    os.makedirs(out_dir, exist_ok=True)
    for pocket, ligand_name in ligand_names.items():
        pocket_name = pocket.strip('.json')
        print('Doing pocket : ', pocket_name)

        # Get dgl pocket
        pocket_path = os.path.join(expanded_path, pocket)
        pocket_graph = graph_io.load_json(pocket_path)
        dgl_pocket_graph, _ = load_rna_graph(pocket_graph)

        # Do inference
        out_path = os.path.join(out_dir, f"{pocket_name}_results.txt")
        robin_inference(ligand_name, dgl_pocket_graph, out_path)
