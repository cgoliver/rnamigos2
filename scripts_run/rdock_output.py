"""
Convert consolidated csv with RDOCK to output/ format

JUST ADAPT THE EVALUATE SCRIPT
"""
import os
import pathlib
import sys

from dgl.dataloading import GraphDataLoader
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.utils.virtual_screen import mean_active_rank, run_results_to_raw_df, run_results_to_ef_df
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


def run_virtual_screen_docking(systems, dataloader, score_to_use='INTER'):
    """run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataloader: Loader of VirtualScreenDataset object
    :param metric: function that takes a list of prediction and an is_active indicator and returns a score
    :param return_model_outputs: whether to return the scores given by the model.

    :returns scores: list of scores, one for each graph in the dataset
    :returns inds: list of indices in the dataloader for which the score computation was successful
    """
    efs, all_scores, status, all_smiles, pocket_names = [], [], [], [], []
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
                score = float(relevant_row[score_to_use].item())
                scores.append(score)
                selected_actives.append(is_active[i])
                selected_smiles.append(sm)
        selected_actives = np.array(selected_actives)
        scores = -np.array(scores)
        efs.append(mean_active_rank(scores, selected_actives))
        all_scores.append(list(scores))
        status.append(list(selected_actives))
        pocket_names.append(pocket_name)
        all_smiles.append(selected_smiles)
    return efs, all_scores, status, pocket_names, all_smiles


def get_dfs_rdock(test_systems, data_df):
    script_dir = os.path.dirname(__file__)
    rows, raw_rows = [], []
    decoys = ['chembl', 'pdb', 'pdb_chembl']
    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }
    for decoy_mode in decoys:
        print(f"Doing rDock inference and VS on {decoy_mode} decoys.")
        dataset = VirtualScreenDatasetDocking(ligands_path=os.path.join(script_dir, '../data/ligand_db'),
                                              systems=test_systems,
                                              decoy_mode=decoy_mode,
                                              group_ligands=True,
                                              reps_only=False)
        dataloader = GraphDataLoader(dataset=dataset, **loader_args)
        efs, scores, status, pocket_names, all_smiles = run_virtual_screen_docking(systems=data_df,
                                                                                   dataloader=dataloader)
        print('Mean EF :', np.mean(efs))
        raw_df = run_results_to_raw_df(scores, status, pocket_names, all_smiles, decoy_mode)
        ef_df = run_results_to_ef_df(efs, scores, pocket_names, decoy_mode)
        rows.append(ef_df)
        raw_rows.append(raw_df)
    df_ef = pd.concat(rows)
    df_raw = pd.concat(raw_rows)
    return df_ef, df_raw


if __name__ == '__main__':
    # Get the systems name and the docking values
    test_systems = get_systems(target="native_fp",
                               rnamigos1_split=-2,
                               use_rnamigos1_train=False,
                               use_rnamigos1_ligands=False,
                               return_test=True)
    df_data = pd.read_csv("data/rnamigos2_dataset_consolidated.csv")
    df_data = df_data[['PDB_ID_POCKET', 'LIGAND_SMILES', 'LIGAND_SOURCE', 'TOTAL', 'INTER']]
    df_data = df_data[df_data['PDB_ID_POCKET'].isin(test_systems['PDB_ID_POCKET'].unique())]

    # For each decoy set, do an rDock "prediction" and compute EFs
    df_ef, df_raw = get_dfs_rdock(test_systems, df_data)

    # Finally, dump the results as CSVs
    dump_path = pathlib.Path("outputs/pockets/rdock.csv")
    dump_path_raw = pathlib.Path("outputs/pockets/rdock_raw.csv")
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    df_ef.to_csv(dump_path, index=False)
    df_raw.to_csv(dump_path_raw, index=False)
