import os
import sys

from dgl.dataloading import GraphDataLoader
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from rnaglib.utils import graph_from_pdbid, graph_utils, graph_io
from sklearn import metrics
import torch
import time
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos_dock.learning.models import get_model_from_dirpath
from rnamigos_dock.learning.loader import VirtualScreenDataset, get_systems
from rnamigos_dock.post.virtual_screen import mean_active_rank, run_virtual_screen

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


def get_expanded_subgraph_from_list(rglib_graph, nodelist, bfs_depth=4):
    expanded_nodes = graph_utils.bfs(rglib_graph, nodelist, depth=bfs_depth, label='LW')
    new_pocket_graph = rglib_graph.subgraph(expanded_nodes)
    in_pocket = {node: node in nodelist for node in expanded_nodes}
    nt_codes = nx.get_node_attributes(new_pocket_graph, 'nt_code')
    edge_types = nx.get_edge_attributes(new_pocket_graph, 'LW')

    # New graph creation enables removing old attributes. (more lightweight)
    expanded_graph = nx.DiGraph()  # or whatever type of graph `G` is
    expanded_graph.add_edges_from(new_pocket_graph.edges())
    nx.set_node_attributes(expanded_graph, name='in_pocket', values=in_pocket)
    nx.set_node_attributes(expanded_graph, name='nt_code', values=nt_codes)
    nx.set_edge_attributes(expanded_graph, name='LW', values=edge_types)
    return expanded_graph


def get_perturbed_pockets(unperturbed_path='data/json_pockets_expanded', out_path='figs/perturbed'):
    test_systems = get_systems(target="is_native",
                               rnamigos1_split=-2,
                               use_rnamigos1_train=False,
                               use_rnamigos1_ligands=False,
                               return_test=True)
    test_pockets_redundant = test_systems[['PDB_ID_POCKET']].values.squeeze()
    test_pockets = set(list(test_pockets_redundant))
    # print(test_pockets, len(test_pockets))

    computed_pockets = set([pocket.rstrip('.json') for pocket in os.listdir(unperturbed_path)])
    to_compute = computed_pockets.intersection(test_pockets)

    failed_set = set()
    for pocket in tqdm(to_compute):
        # Get rglib grpah
        pdb_id = pocket[:4].lower()
        rglib_graph = graph_from_pdbid(pdb_id, redundancy='all')
        if rglib_graph is None:
            failed_set.add(pocket)

        # Get pocket graph and hence initial nodelist
        unperturbed_pocket_path = os.path.join(unperturbed_path, f'{pocket}.json')
        old_pocket_graph = graph_io.load_json(unperturbed_pocket_path)
        in_pocket_nodes = {node[:4].lower() + node[4:]
                           for node, in_pocket in old_pocket_graph.nodes(data='in_pocket')
                           if in_pocket}

        # Ensure all nodes are valid and expand with a small bfs
        in_pocket_filtered = in_pocket_nodes.intersection(set(rglib_graph.nodes()))
        around_pocket = graph_utils.bfs(rglib_graph, in_pocket_filtered, depth=1, label='LW')

        # Now compute the perturbed pockets
        for fraction in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            n_nodes_to_sample = int(fraction * len(in_pocket_filtered))
            for replicate in range(5):
                # Setup dirs
                out_dir = os.path.join(out_path, f'perturbed_{fraction}_{replicate}')
                os.makedirs(out_dir, exist_ok=True)
                out_name = os.path.join(out_dir, f'{pocket}.json')

                # Sample a broken binding site
                noisy_nodelist = list(np.random.choice(list(around_pocket), replace=False, size=n_nodes_to_sample))
                expanded_graph = get_expanded_subgraph_from_list(rglib_graph=rglib_graph, nodelist=noisy_nodelist)
                graph_io.dump_json(out_name, expanded_graph)


def compute_efs_model(model, dataloader, lower_is_better):
    rows, raw_rows = [], []
    efs, scores, status, pocket_names, all_smiles = run_virtual_screen(model,
                                                                       dataloader,
                                                                       metric=mean_active_rank,
                                                                       lower_is_better=lower_is_better)
    for pocket_id, score_list, status_list, smiles_list in zip(pocket_names, scores, status, all_smiles):
        for score, status, smiles in zip(score_list, status_list, smiles_list):
            raw_rows.append({'raw_score': score,
                             'is_active': status,
                             'pocket_id': pocket_id,
                             'smiles': smiles})

    for ef, score, pocket_id in zip(efs, scores, pocket_names):
        rows.append({
            'score': ef,
            'pocket_id': pocket_id})
    print('Mean EF :', np.mean(efs))
    df = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows)
    return df, df_raw


def mix_two_scores(df, score1, score2):
    """
    adapted from mixing to return a raw df
    """

    def normalize(scores):
        out_scores = (scores - scores.min()) / (scores.max() - scores.min())
        return out_scores

    pockets = df['pocket_id'].unique()
    all_efs = []
    all_pocket_raw = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        pocket_df = pocket_df.reset_index(drop=True)
        docking_scores = pocket_df[score1]
        new_scores = pocket_df[score2]
        normalized_docking = normalize(docking_scores)
        normalized_new = normalize(new_scores)
        pocket_df['mixed'] = (0.5 * normalized_docking + 0.5 * normalized_new).values
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'], pocket_df['mixed'],
                                                 drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)
        all_efs.append({'score': enrich, 'pocket_id': p})
        all_pocket_raw.append(pocket_df)
    mixed_df = pd.DataFrame(all_efs)
    print('Mean EF mixed:', np.mean(mixed_df['score'].values))
    mixed_df_raw = pd.concat(all_pocket_raw)
    return mixed_df, mixed_df_raw


# Copied from evaluate except reps_only=True to save time
#   cache_graphs=True to save time over two model runs
#   target is set to "is_native" which has no impact since it's just used to get pdb lists
# The goal here is just to have easy access to the loader and modify its pockets_path
def get_perf(pocket_path, basename=None):
    # Setup loader
    test_systems = get_systems(target="is_native",
                               rnamigos1_split=-2,
                               use_rnamigos1_train=False,
                               use_rnamigos1_ligands=False,
                               return_test=True)
    loader_args = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }
    dataset = VirtualScreenDataset(pocket_path,
                                   cache_graphs=True,
                                   ligands_path="data/ligand_db",
                                   systems=test_systems,
                                   decoy_mode='chembl',
                                   use_graphligs=True,
                                   group_ligands=True,
                                   reps_only=True)
    dataloader = GraphDataLoader(dataset=dataset, **loader_args)

    # Setup path and models
    d = Path("figs/perturbed", parents=True, exist_ok=True)
    if basename is None:
        base_name = Path(pocket_path).stem
    dock_model_path = 'results/trained_models/dock/dock_42'
    dock_model = get_model_from_dirpath(dock_model_path)
    native_model_path = 'results/trained_models/is_native/native_42'
    native_model = get_model_from_dirpath(native_model_path)

    # Get dock performance
    df_dock, df_dock_raw = compute_efs_model(dock_model, dataloader=dataloader, lower_is_better=True)
    df_dock.to_csv(d / (base_name + '_dock.csv'))
    df_dock_raw.to_csv(d / (base_name + "_dock_raw.csv"))

    # Get native performance
    df_native, df_native_raw = compute_efs_model(native_model, dataloader=dataloader, lower_is_better=False)
    df_native.to_csv(d / (base_name + '_native.csv'))
    df_native_raw.to_csv(d / (base_name + "_native_raw.csv"))

    # Now merge those two results to get a final mixed performance
    # Inspired from mixing.py, function find_best_mix and mix_two_scores
    df_dock_raw['dock'] = -df_dock_raw['raw_score'].values
    df_native_raw['native'] = df_native_raw['raw_score'].values
    big_df_raw = df_dock_raw.merge(df_native_raw, on=['pocket_id', 'smiles', 'is_active'], how='outer')

    mixed_df, mixed_df_raw = mix_two_scores(big_df_raw, score1='dock', score2='native')
    mixed_df.to_csv(d / (base_name + '_mixed.csv'))
    mixed_df_raw.to_csv(d / (base_name + "_mixed_raw.csv"))
    return np.mean(mixed_df['score'].values)


if __name__ == '__main__':
    get_perturbed_pockets()

    # Check that inference works
    os.makedirs("figs/perturbed", exist_ok=True)
    # get_perf(pocket_path="data/json_pockets_expanded")

    list_of_results = []
    for fraction in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        for replicate in range(5):
            print(fraction, replicate)
            # Setup dirs
            perturbed_pocket_path = os.path.join('figs/perturbed', f'perturbed_{fraction}_{replicate}')
            mean_score = get_perf(pocket_path=perturbed_pocket_path, basename=perturbed_pocket_path)
            list_of_results.append({"thresh": fraction, "replicate": replicate, "score": mean_score})
    df = pd.DataFrame(list_of_results)
    df.to_csv('figs/perturbed/aggregated.csv')

    means = df.groupby('thresh')['score'].mean()
    labels = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    import matplotlib.pyplot as plt

    plt.plot(labels, means.values)
    plt.hlines(y=0.985, xmin=min(labels), xmax=max(labels))
    plt.show()
