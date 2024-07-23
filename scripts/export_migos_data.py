"""
In this script we try to get a minimal reproducing script
"""

import numpy as np
import os
from pathlib import Path
import pickle
from rnaglib.utils import graph_io


class LigandComputer:
    def __init__(self, ligands_path='../data/ligand_db/'):
        self.ligands_path = ligands_path
        # Get the full groups to group actives/inactives together
        splits_file = os.path.join(SCRIPT_DIR, '../data/train_test_75.p')
        train_names, test_names, train_names_grouped, test_names_grouped = pickle.load(open(splits_file, 'rb'))
        # Use all actives
        self.groups = {**train_names_grouped, **test_names_grouped}
        self.reverse_groups = {group_member: group_rep for group_rep, group_members in self.groups.items()
                               for group_member in group_members}

    @staticmethod
    def parse_smiles(smiles_path):
        sm_list = list(open(smiles_path).readlines())
        sm_list = [sm.strip() for sm in sm_list]
        return sm_list

    def get_actives(self, group_pockets):
        group_list = []
        for pocket in group_pockets:
            try:
                active = self.parse_smiles(Path(self.ligands_path, pocket, 'pdb', 'actives.txt'))[0]
                group_list.append(active)
            except Exception as e:
                # print(e)
                pass
        group_actives = set(group_list)
        return group_actives

    def get_group(self, pocket_name):
        return self.groups[self.reverse_groups[pocket_name]]

    def get_ligands(self, pocket_name, decoy_mode='pdb', ):
        # We need to return all actives and ensure they are not in the inactives of a pocket
        group_pockets = self.get_group(pocket_name)
        group_actives = self.get_actives(group_pockets)
        decoys_smiles = self.parse_smiles(Path(self.ligands_path, pocket_name, decoy_mode, 'decoys.txt'))

        # Filter actives flagged as inactives
        decoys_smiles = [smile for smile in decoys_smiles if smile not in group_actives]
        actives_smiles = list(group_actives)

        # Filter None
        actives_smiles = [x for x in actives_smiles if x is not None]
        decoys_smiles = [x for x in decoys_smiles if x is not None]
        return actives_smiles, decoys_smiles


SCRIPT_DIR = os.path.dirname(__file__)

# Use reps (members of the groups)
reps_file = os.path.join(SCRIPT_DIR, '../data/group_reps_75.p')
train_group_reps, test_group_reps = pickle.load(open(reps_file, 'rb'))
reps = set(train_group_reps + test_group_reps)
lc = LigandComputer()
all_groups = {}
pockets_path = "../data/json_pockets_expanded"
failures = 0
for i, group in enumerate(reps):
    if i % 20 == 0:
        print(i, failures, len(reps))
    try:
        group_positive, pdb_decoys = lc.get_ligands(group)
        _, chembl_decoys = lc.get_ligands(group, decoy_mode='chembl')
        rna_path = os.path.join(pockets_path, f"{group}.json")
        pocket_graph = graph_io.load_json(rna_path)
        nodes = dict(pocket_graph.nodes(data=True))
        for key, val in nodes.items():
            del val['nt_code']
        group_object = {'group': lc.get_group(group),
                        'nodes': nodes,
                        'actives': group_positive,
                        'pdb_decoys': pdb_decoys,
                        'chembl_decoys': chembl_decoys}
        all_groups[group] = group_object
    except FileNotFoundError as e:
        failures += 1
        print(e)
    except KeyError as e:
        failures += 1
        print(e)

train_groups = {group: all_groups[group] for group in train_group_reps if group in all_groups}
test_groups = {group: all_groups[group] for group in test_group_reps if group in all_groups}

json_dump = "../data/dataset_as_json.json"
pickle.dump((train_groups, test_groups), open(json_dump, 'wb'))
