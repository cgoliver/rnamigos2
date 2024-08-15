"""
Convert consolidated csv with RDOCK to output/ format

JUST ADAPT THE EVALUATE SCRIPT
"""
import os
import sys

from dgl.dataloading import GraphDataLoader
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.utils.virtual_screen import mean_active_rank
from rnamigos.learning.dataset import get_systems



class VirtualScreenDatasetDocking:
    def __init__(self,
                 systems,
                 ligands_path,
                 decoy_mode='pdb',
                 group_ligands=True,
                 reps_only=False
                 ):
        self.ligands_path = ligands_path
        self.systems = systems
        self.decoy_mode = decoy_mode
        self.all_pockets_names = list(self.systems['PDB_ID_POCKET'].unique())

        self.group_ligands = group_ligands
        self.reps_only = reps_only
        script_dir = os.path.dirname(__file__)
        if self.reps_only:
            # This amounts to choosing only reps.
            # Previously, the retained ones were the centroids.
            reps_file = os.path.join(script_dir, '../data/group_reps_75.p')
            train_group_reps, test_group_reps = pickle.load(open(reps_file, 'rb'))
            reps = set(train_group_reps + test_group_reps)
            self.all_pockets_names = [pocket for pocket in self.all_pockets_names if pocket in reps]

        if self.group_ligands:
            splits_file = os.path.join(script_dir, '../data/train_test_75.p')
            _, _, train_names_grouped, test_names_grouped = pickle.load(open(splits_file, 'rb'))
            self.groups = {**train_names_grouped, **test_names_grouped}
            self.reverse_groups = {group_member: group_rep for group_rep, group_members in self.groups.items()
                                   for group_member in group_members}

    def __len__(self):
        return len(self.all_pockets_names)

    def parse_smiles(self, smiles_path):
        sm_list = list(open(smiles_path).readlines())
        sm_list = [sm.strip() for sm in sm_list]
        return sm_list

    def get_ligands(self, pocket_name):
        actives_smiles = self.parse_smiles(Path(self.ligands_path, pocket_name, self.decoy_mode, 'actives.txt'))
        decoys_smiles = self.parse_smiles(Path(self.ligands_path, pocket_name, self.decoy_mode, 'decoys.txt'))
        # We need to return all actives and ensure they are not in the inactives of a pocket
        if self.group_ligands:
            group_pockets = self.groups[self.reverse_groups[pocket_name]]
            group_list = []
            for pocket in group_pockets:
                try:
                    active = self.parse_smiles(Path(self.ligands_path, pocket, self.decoy_mode, 'actives.txt'))[0]
                    group_list.append(active)
                except Exception as e:
                    pass
                    # print(e)
            group_actives = set(group_list)
            decoys_smiles = [smile for smile in decoys_smiles if smile not in group_actives]
            actives_smiles = list(group_actives)
        actives_smiles = [x for x in actives_smiles if x is not None]
        decoys_smiles = [x for x in decoys_smiles if x is not None]
        return actives_smiles, decoys_smiles

    def __getitem__(self, idx):
        try:
            pocket_name = self.all_pockets_names[idx]
            actives_smiles, decoys_smiles = self.get_ligands(pocket_name)
            all_smiles = actives_smiles + decoys_smiles
            is_active = np.zeros(len(all_smiles))
            is_active[:len(actives_smiles)] = 1.
            return pocket_name, all_smiles, torch.tensor(is_active)
        except FileNotFoundError as e:
            # print(e)
            return None, None, None


def run_virtual_screen_docking(systems, dataloader, score_to_use='INTER', **kwargs):
    """run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataloader: Loader of VirtualScreenDataset object
    :param metric: function that takes a list of prediction and an is_active indicator and returns a score
    :param return_model_outputs: whether to return the scores given by the model.

    :returns scores: list of scores, one for each graph in the dataset
    :returns inds: list of indices in the dataloader for which the score computation was successful
    """
    efs, all_scores, status, all_smiles, pocket_names = [], [], [], [], []
    print(f"Doing VS on {len(dataloader)} pockets.")
    failed = 0
    for i, (pocket_name, smiles, is_active) in enumerate(dataloader):
        # Some ligfiles are missing
        if pocket_name is None:
            failed += 1
            continue
        if not i % 20:
            print(f"Done {i}/{len(dataloader)}")
        if len(smiles) < 10:
            print(f"Skipping pocket{i}, not enough decoys")
            failed += 1
            continue
        pocket_scores = systems.loc[systems['PDB_ID_POCKET'] == pocket_name]
        selected_actives, selected_smiles, scores = [], [], []
        # We need to loop to reorder the smiles and handle potential missing systems
        for i, sm in enumerate(smiles):
            relevant_row = pocket_scores[pocket_scores['LIGAND_SMILES'] == sm]
            if len(relevant_row) == 0:
                pass
            else:
                score = float(relevant_row[score_to_use].values)
                scores.append(score)
                selected_actives.append(is_active[i])
                selected_smiles.append(sm)
        selected_actives = np.array(selected_actives)
        scores = np.array(scores)
        efs.append(mean_active_rank(scores, selected_actives, **kwargs))
        all_scores.append(list(scores))
        status.append(list(selected_actives))
        pocket_names.append(pocket_name)
        all_smiles.append(selected_smiles)
    print(f"VS failed on {failed} systems")
    print(efs)
    return efs, all_scores, status, pocket_names, all_smiles


rows, raw_rows = [], []
test_systems = get_systems(target="native_fp",
                           rnamigos1_split=-2,
                           use_rnamigos1_train=False,
                           use_rnamigos1_ligands=False,
                           return_test=True)

df = pd.read_csv("data/rnamigos2_dataset_consolidated.csv")
df = df[['PDB_ID_POCKET', 'LIGAND_SMILES', 'LIGAND_SOURCE', 'TOTAL', 'INTER']]
df = df[df['PDB_ID_POCKET'].isin(test_systems['PDB_ID_POCKET'].unique())]
script_dir = os.path.dirname(__file__)

decoys = ['chembl', 'pdb', 'pdb_chembl']
for decoy_mode in decoys:
    dataset = VirtualScreenDatasetDocking(ligands_path=os.path.join(script_dir, '../data/ligand_db'),
                                          systems=test_systems,
                                          decoy_mode=decoy_mode,
                                          group_ligands=True,
                                          reps_only=False)

    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }
    dataloader = GraphDataLoader(dataset=dataset, **loader_args)
    # df_decoy = df[df['LIGAND_SOURCE'].isin([mode.upper() for mode in decoy_mode.split('_')])]
    efs, scores, status, pocket_names, all_smiles = run_virtual_screen_docking(systems=df,
                                                                               dataloader=dataloader,
                                                                               lower_is_better=True,
                                                                               )
    for pocket_id, score_list, status_list, smiles_list in zip(pocket_names, scores, status, all_smiles):
        for score, status, smiles in zip(score_list, status_list, smiles_list):
            raw_rows.append({'raw_score': score, 'is_active': status, 'pocket_id': pocket_id, 'smiles': smiles,
                             'decoys': decoy_mode})

    for ef, score, pocket_id in zip(efs, scores, pocket_names):
        rows.append({
            'score': ef,
            'metric': 'EF' if decoy_mode == 'robin' else 'MAR',
            'decoys': decoy_mode,
            'pocket_id': pocket_id})
    print('Mean EF :', np.mean(efs))

df = pd.DataFrame(rows)
df_raw = pd.DataFrame(raw_rows)
df.to_csv("outputs/rdock.csv")
df_raw.to_csv("outputs/rdock_raw.csv")
