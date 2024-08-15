import itertools
import os
import pickle

import numpy as np
import torch
from dgl.dataloading import GraphCollator
from numpy import random
from torch.utils.data import Sampler

from dgl.dataloading import GraphDataLoader
from rnamigos_dock.learning.dataset import VirtualScreenDataset


class IsNativeSampler(Sampler):
    def __init__(self, systems_dataframe, group_sampling=True):
        super().__init__(data_source=None)
        self.group_sampling = group_sampling
        positives = (systems_dataframe['IS_NATIVE'] == 1).values
        negatives = 1 - positives
        if not group_sampling:
            self.positive_rows = np.where(positives)[0]
            self.negative_rows = np.where(negatives)[0]
            self.num_pos_examples = len(self.positive_rows)
        else:
            script_dir = os.path.dirname(__file__)
            splits_file = os.path.join(script_dir, '../../data/train_test_75.p')
            _, _, train_names_grouped, _ = pickle.load(open(splits_file, 'rb'))
            #  Build positive and negative rows for each group as the list of positive and negative indices
            # Useful for sampling, also keep track of the amount of positive and negative for each group
            self.all_positives = list()
            self.all_negatives = list()
            num_pos, num_neg = [], []
            for group_rep, group in train_names_grouped.items():
                in_group = (systems_dataframe['PDB_ID_POCKET'].isin(group)).values
                group_positive = np.logical_and(in_group, positives)
                group_negative = np.logical_and(in_group, negatives)
                positive_rows = np.where(group_positive)[0]
                # This can happen (rarely) if all positives are in validation.
                if len(positive_rows) == 0:
                    continue
                negative_rows = np.where(group_negative)[0]
                self.all_positives.append(positive_rows)
                self.all_negatives.append(negative_rows)
                num_pos.append(len(positive_rows))
                num_neg.append(len(negative_rows))
            self.num_pos = np.array(num_pos)
            self.num_neg = np.array(num_neg)
            # Computing length now avoids problem with empty examples
            self.num_pos_examples = len(self.num_pos)

    def __iter__(self):
        if not self.group_sampling:
            selected_neg_rows = np.random.choice(self.negative_rows,
                                                 self.num_pos_examples,
                                                 replace=False)
            selected_positive_rows = self.positive_rows
        else:
            selected_pos = np.random.randint(0, self.num_pos)
            selected_neg = np.random.randint(0, self.num_neg)
            selected_positive_rows = []
            selected_neg_rows = []
            for i, (group_pos, group_neg) in enumerate(zip(self.all_positives, self.all_negatives)):
                selected_positive_rows.append(group_pos[selected_pos[i]])
                selected_neg_rows.append(group_neg[selected_neg[i]])
            selected_positive_rows = np.array(selected_positive_rows)
            selected_neg_rows = np.array(selected_neg_rows)
        systems = np.concatenate((selected_neg_rows, selected_positive_rows))
        np.random.shuffle(systems)
        yield from systems

    def __len__(self) -> int:
        return self.num_pos_examples * 2


class NativeFPSampler(Sampler):
    def __init__(self, systems_dataframe, group_sampling=True):
        super().__init__(data_source=None)
        self.group_sampling = group_sampling
        if not group_sampling:
            self.num_examples = len(systems_dataframe)
        else:
            script_dir = os.path.dirname(__file__)
            splits_file = os.path.join(script_dir, '../../data/train_test_75.p')
            _, _, train_names_grouped, _ = pickle.load(open(splits_file, 'rb'))
            #  Build positive and negative rows for each group as the list of positive and negative indices
            # Useful for sampling, also keep track of the amount of positive and negative for each group
            self.grouped_rows = list()
            num_group = []
            for group_rep, group in train_names_grouped.items():
                in_group = (systems_dataframe['PDB_ID_POCKET'].isin(group)).values
                group_rows = np.where(in_group)[0]
                # This can happen (rarely) if all positives are in validation.
                if len(group_rows) == 0:
                    continue
                self.grouped_rows.append(group_rows)
                num_group.append(len(group_rows))
            self.num_group = np.array(num_group)
            self.num_examples = len(self.num_group)

    def __iter__(self):
        if not self.group_sampling:
            systems = np.arange(self.num_examples)
        else:
            selected = np.random.randint(0, self.num_group)
            selected_rows = []
            for i, group_selected in enumerate(self.grouped_rows):
                selected_rows.append(group_selected[selected[i]])
            systems = np.array(selected_rows)
        np.random.shuffle(systems)
        yield from systems

    def __len__(self) -> int:
        return self.num_examples


class RingCollater():
    def __init__(self, node_simfunc=None, max_size_kernel=None):
        self.node_simfunc = node_simfunc
        self.max_size_kernel = max_size_kernel
        self.graph_collator = GraphCollator()

    def k_block(self, node_rings):
        # We need to reimplement because current one expects dicts
        block = np.zeros((len(node_rings), len(node_rings)))
        assert self.node_simfunc.compare(node_rings[0],
                                         node_rings[0]) > 0.99, "Identical rings giving non 1 similarity."
        sims = [self.node_simfunc.compare(n1, n2)
                for i, (n1, n2) in enumerate(itertools.combinations(node_rings, 2))]
        block[np.triu_indices(len(node_rings), 1)] = sims
        block += block.T
        block += np.eye(len(node_rings))
        return block

    def collate(self, items):
        batch = {}
        for key, value in items[0].items():
            values = [d[key] for d in items]
            if key == 'rings':
                if self.node_simfunc is None:
                    batch[key] = None, None
                    continue
                # Taken from rglib
                flat_rings = list()
                for ring in values:
                    flat_rings.extend(ring)
                if self.max_size_kernel is None or len(flat_rings) < self.max_size_kernel:
                    # Just take them all
                    node_ids = [1 for _ in flat_rings]
                else:
                    # Take only 'max_size_kernel' elements
                    node_ids = [1 for _ in range(self.max_size_kernel)] + \
                               [0 for _ in range(len(flat_rings) - self.max_size_kernel)]
                    random.shuffle(node_ids)
                    flat_rings = [node for i, node in enumerate(flat_rings) if node_ids[i] == 1]
                k_block = self.k_block(flat_rings)
                batch[key] = torch.from_numpy(k_block).detach().float(), node_ids
            else:
                batch[key] = self.graph_collator.collate(items=values)
        return batch


VS_LOADER_ARGS = {'shuffle': False,
                  'batch_size': 1,
                  'num_workers': 4,
                  'collate_fn': lambda x: x[0]
                  }


def get_vs_loader(systems, decoy_mode, cfg, reps_only=False, rognan=False):
    global VS_LOADER_ARGS
    vs_dataset = VirtualScreenDataset(pockets_path=cfg.data.pocket_graphs,
                                      ligands_path=cfg.data.ligand_db,
                                      systems=systems,
                                      decoy_mode=decoy_mode,
                                      use_graphligs=cfg.model.use_graphligs,
                                      group_ligands=True,
                                      reps_only=reps_only,
                                      rognan=rognan)
    dataloader = GraphDataLoader(dataset=vs_dataset, **VS_LOADER_ARGS)
    return dataloader
