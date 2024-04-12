import os
import sys

from rnaglib.utils import graph_io
from collections import defaultdict

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from experiments.inference import inference
from rnamigos_dock.tools.graph_utils import load_rna_graph


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
    Get one ligand pocket id and its corresponding ligands from decoyfinder.
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
            node_list = [node[5:] for node, in_pocket in pocket_graph.nodes(data='in_pocket') if in_pocket]
            nodelists[representative_pocket] = node_list
            ligand_names[representative_pocket] = ligand_name
    return nodelists, ligand_names


if __name__ == "__main__":
    pass
    expanded_path = 'data/json_pockets_expanded'
    nodelists, ligand_names = get_nodelists_and_ligands()

    models_path = {
        'dock': 'saved_models/dock',
        'is_native': 'saved_models/native',
        'native_fp': 'saved_models/fp'
    }
    out_dir = 'outputs/robin_docknative'
    os.makedirs(out_dir, exist_ok=True)
    decoys_ligands_dir = "data/robin_decoys_decoyfinder"
    new_mixing_coeffs = [0.3, 0., 0.3]
    # new_mixing_coeffs = [0.36841931, 0.26315665, 0.36841931]
    for pocket, ligand_name in ligand_names.items():
        pocket_name = pocket.strip('.json')
        print('Doing pocket : ', pocket_name)

        # Get dgl pocket
        pocket_path = os.path.join(expanded_path, pocket)
        pocket_graph = graph_io.load_json(pocket_path)
        dgl_pocket_graph, _ = load_rna_graph(pocket_graph, undirected=False)

        # Get smiles list for decoys
        # decoys_ligands_path = os.path.join(decoys_ligands_dir, f"{ligand_name}_decoys.txt")
        # out_path = os.path.join(out_dir, f"{pocket_name}_decoys.txt")
        # smiles_list = [s.lstrip().rstrip() for s in list(open(decoys_ligands_path).readlines())]
        # inference(dgl_graph=dgl_pocket_graph, smiles_list=smiles_list, out_path=out_path,
        #           dump_all=True, models_path=models_path, mixing_coeffs=new_mixing_coeffs)

        active_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "actives.txt")
        out_path = os.path.join(out_dir, f"{pocket_name}_actives.txt")
        smiles_list = [s.lstrip().rstrip() for s in list(open(active_ligands_path).readlines())]
        inference(dgl_graph=dgl_pocket_graph, smiles_list=smiles_list, out_path=out_path,
                  dump_all=True, models_path=models_path, mixing_coeffs=new_mixing_coeffs)

        inactives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "decoys.txt")
        out_path = os.path.join(out_dir, f"{pocket_name}_inactives.txt")
        smiles_list = [s.lstrip().rstrip() for s in list(open(inactives_ligands_path).readlines())]
        inference(dgl_graph=dgl_pocket_graph, smiles_list=smiles_list, out_path=out_path,
                  dump_all=True, models_path=models_path, mixing_coeffs=new_mixing_coeffs)
