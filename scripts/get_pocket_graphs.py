import os
from rnaglib.utils import graph_from_pdbid
from rnaglib.utils import graph_io
from tqdm import tqdm

old_path = 'data/json_pockets'
new_path = 'data/json_pockets_new'
failed_set = set()
for pocket in tqdm(os.listdir(old_path)):
    old_pocket_path = os.path.join(old_path, pocket)
    pdb_id = pocket[:4].lower()
    rglib_graph = graph_from_pdbid(pdb_id, redundancy='all')
    if rglib_graph is None:
        failed_set.add(pocket)
    old_pocket_graph = graph_io.load_json(old_pocket_path)
    new_nodes = {s[:4].lower() + s[4:] for s in old_pocket_graph.nodes}
    new_pocket_graph = rglib_graph.subgraph(new_nodes)
    new_pocket_path = os.path.join(new_path, pocket)
    graph_io.dump_json(new_pocket_path, new_pocket_graph)
print(failed_set)
print(f"{len(failed_set)}/{len(os.listdir(old_path))} failed systems")
