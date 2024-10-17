import os
import sys

from collections import defaultdict
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import AgglomerativeClustering

from rnaglib.utils import graph_io

np.random.seed(42)

# OPEN RMSCORE DATA
with open("data/rmscores/systems.txt", 'r') as f:
    columns = [s.strip() for s in f.readlines()]
df = pd.read_csv("data/rmscores/rmscore_normalized_by_average_length_complete_dataset.csv", header=None)
df.columns = columns
raw_val = df.values
indices = {name: idx for (idx, name) in enumerate(columns)}

# sam_cols = [name for name in columns if 'SAM' in name]
# sam_idx = [indices[name] for name in sam_cols]
# sam_pairwise = raw_val[sam_idx, :][:, sam_idx]
# copies = ['4OQU_A_SAM_101', '5FK5_A_SAM_1095', '7DWH_X_SAM_102', '6UET_A_SAM_301', '6FZ0_A_SAM_104', '2QWY_A_SAM_100',
#           '2QWY_B_SAM_300', '2QWY_C_SAM_500', '6YMM_B_SAM_201']
# copies_idx = [indices[name] for name in copies]
# copies_pairwise = raw_val[copies_idx, :][:, copies_idx]

# CLUSTER AND KEEP TRACK OF CLUSTER : NAMES mappings
clustering = AgglomerativeClustering(metric='precomputed',
                                     n_clusters=None,
                                     compute_full_tree=True,
                                     linkage='single',
                                     distance_threshold=0.25).fit(1 - raw_val)
labels = clustering.labels_
unique_values = np.unique(labels)
groups = {}
# This reverse map (cluster id : systems) is useful to remove robin systems
label_value_to_rep = {}
for value in unique_values:
    cluster = np.where(labels == value)[0]
    cluster_names = [columns[idx] for idx in cluster]
    group_rep = cluster_names[0]
    groups[group_rep] = cluster_names
    label_value_to_rep[value] = group_rep

# Analysis of the number of different ligands in each group
# different_ligs = 0
# for name, group in groups.items():
#     ligands = [s.split('_')[2] for s in group]
#     if len(np.unique(ligands)) > 1:
#         print(name, len(np.unique(ligands)), len(ligands), ligands)
#     different_ligs += len(np.unique(ligands))
#     # print(name, len(group))
#     # if 'SAM' in name:
#     #     print(group)
# # plt.hist([len(group) for group in groups.values()])
# # plt.show()
# # print(groups)
# print("Different", different_ligs, len(groups))

# HANDLE ROBIN SYSTEMS: remove from train and later add in test
robin_pdb_names = ["2GDI", "5BTP", "2QWY", "3FU2"]
robin_groups = {}
for name in robin_pdb_names:
    # Find the corresponding name and group id (should be unique)
    robin_ids = [indices[column] for column in columns if column.startswith(name)]
    robin_clusters = labels[robin_ids]
    assert len(np.unique(robin_clusters)) == 1
    robin_cluster = robin_clusters[0]
    robin_rep = label_value_to_rep[robin_cluster]
    # Remove this group from groups (and later add it in the test set)
    robin_groups[robin_rep] = groups.pop(robin_rep)

# Split groups based on keys + add ROBIN
train_cut = int(0.85 * len(groups))
train_groups_keys = list(groups.keys())[:train_cut]
test_groups_keys = list(groups.keys())[train_cut:]
train_groups = {key: groups[key] for key in train_groups_keys}
test_groups = {key: groups[key] for key in test_groups_keys}
test_groups.update(robin_groups)
print("Number of groups", len(train_groups), len(test_groups))

# Also dump the actual pdb ids as flat lists, including redundancy
train_names = set(chain.from_iterable([[name for name in group] for group in train_groups.values()]))
test_names = set(chain.from_iterable([[name for name in group] for group in test_groups.values()]))
print("Number of examples", len(train_names), len(test_names))
pickle.dump((train_names, test_names, train_groups, test_groups), open("data/train_test_75.p", 'wb'))

# Also propose a train/val split
print('trainval')
trainval_cut = int(0.8 * len(groups))
train_groups_keys = list(groups.keys())[:trainval_cut]
val_groups_keys = list(groups.keys())[trainval_cut:train_cut]
train_groups = {key: groups[key] for key in train_groups_keys}
val_groups = {key: groups[key] for key in val_groups_keys}
print("Number of groups", len(train_groups), len(val_groups))
# Also dump the actual pdb ids as flat lists, including redundancy
train_names = set(chain.from_iterable([[name for name in group] for group in train_groups.values()]))
val_names = set(chain.from_iterable([[name for name in group] for group in val_groups.values()]))
print("Number of examples", len(train_names), len(val_names))
pickle.dump((train_names, val_names, train_groups, val_groups), open("data/train_val_75.p", 'wb'))


def assess_copies():
    """
    Just plotting copy data => We need splits !
    """
    ROBIN_SYSTEMS = """2GDI	TPP TPP 
    6QN3	GLN  Glutamine_RS
    5BTP	AMZ  ZTP
    2QWY	SAM  SAM_ll
    3FU2	PRF  PreQ1
    """

    # GET REFERENCE POCKETS
    def group_reference_pockets(pockets_path="data/json_pockets_expanded"):
        all_pockets = os.listdir(pockets_path)
        # Group pockets by pdb_id
        grouped_pockets = defaultdict(list)
        for pocket in all_pockets:
            pdb_id = pocket.split("_")[0]
            grouped_pockets[pdb_id].append(pocket)
        return grouped_pockets

    def get_nodelists_and_ligands(
            robin_systems=ROBIN_SYSTEMS, pockets_path="data/json_pockets_expanded"
    ):
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
                colors = [
                    "blue" if in_pocket else "white"
                    for n, in_pocket in pocket_graph.nodes(data="in_pocket")
                ]
                # rna_draw(pocket_graph, node_colors=colors, layout="spring", show=True)

                node_list = [
                    node[5:]
                    for node, in_pocket in pocket_graph.nodes(data="in_pocket")
                    if in_pocket
                ]
                nodelists[representative_pocket] = node_list
                ligand_names[representative_pocket] = ligand_name
        return nodelists, ligand_names

    nodelists, ligand_names = get_nodelists_and_ligands()
    return nodelists, ligand_names


def assess_gain():
    def compute_max_train_test(train_names, test_names):
        train_ids = [indices[name] for name in train_names]
        test_ids = [indices[name] for name in test_names]
        pairwise = raw_val[train_ids, :][:, test_ids]
        max_test = np.max(pairwise, axis=1)
        print(max_test.mean(), max_test.std())
        return max_test

    previous_data = pd.read_csv("data/rnamigos2_dataset_consolidated.csv")[["PDB_ID_POCKET", "TYPE"]]
    previous_data = previous_data.drop_duplicates()
    train, test = set(), set()
    for pocket_name, split in previous_data.values:
        if split == 'TEST':
            test.add(pocket_name)
        else:
            train.add(pocket_name)
    pickle.dump((train, test), open("temp_train_test.p", 'wb'))
    train, test = pickle.load(open("temp_train_test.p", 'rb'))
    max_test_previous = compute_max_train_test(train, test)
    plt.hist(max_test_previous, alpha=0.5, bins=20)

    max_test_new = compute_max_train_test(train_names, test_names)
    plt.hist(max_test_new, alpha=0.5, bins=20)
    max_test_new = compute_max_train_test(train_groups, test_groups)
    plt.hist(max_test_new, alpha=0.5, bins=20)
    plt.show()
