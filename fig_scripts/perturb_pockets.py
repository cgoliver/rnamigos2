import os
import random
import sys

from dgl.dataloading import GraphDataLoader
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from rnaglib.utils import graph_from_pdbid, graph_utils, graph_io
import seaborn as sns
from sklearn import metrics
import torch
import time
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos_dock.learning.models import get_model_from_dirpath
from rnamigos_dock.learning.loader import VirtualScreenDataset, get_systems
from rnamigos_dock.post.virtual_screen import mean_active_rank, run_virtual_screen
from fig_scripts.plot_utils import PALETTE_DICT

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


def get_perturbed_pockets(unperturbed_path='data/json_pockets_expanded',
                          out_path='figs/perturbed',
                          fractions=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
                          perturb_bfs_depth=1,
                          max_replicates=5,
                          recompute=False,
                          perturbation='random'):
    test_systems = get_systems(target="is_native",
                               rnamigos1_split=-2,
                               use_rnamigos1_train=False,
                               use_rnamigos1_ligands=False,
                               return_test=True)
    test_pockets_redundant = test_systems[['PDB_ID_POCKET']].values.squeeze()
    test_pockets = set(list(test_pockets_redundant))
    # print(test_pockets, len(test_pockets))

    existing_pockets = set([pocket.rstrip('.json') for pocket in os.listdir(unperturbed_path)])
    pockets_to_compute = sorted(list(existing_pockets.intersection(test_pockets)))

    failed_set = set()
    for pocket in tqdm(pockets_to_compute):
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
        around_pocket = graph_utils.bfs(rglib_graph, in_pocket_filtered, depth=perturb_bfs_depth, label='LW')

        # Now compute the perturbed pockets
        for fraction in fractions:
            n_nodes_to_sample = int(fraction * len(in_pocket_filtered))
            n_nodes_to_sample = min(max(n_nodes_to_sample, 1), len(around_pocket))
            for replicate in range(max_replicates):
                # Setup dirs
                out_dir = os.path.join(out_path, f'perturbed_{fraction}_{replicate}')
                os.makedirs(out_dir, exist_ok=True)
                out_name = os.path.join(out_dir, f'{pocket}.json')

                if os.path.exists(out_name) and not recompute:
                    continue

                # Sample a broken binding site
                # To get reproducible results, we need to sort sets
                if perturbation == 'random':
                    # just use random nodes from the list
                    sorted_around_pocket = sorted(list(around_pocket))
                    noisy_nodelist = list(np.random.choice(sorted_around_pocket, replace=False, size=n_nodes_to_sample))
                elif perturbation == 'soft':
                    # start from the pocket, and subsample/oversample, starting from the pocket
                    sorted_neighbors_bfs = sorted(list(around_pocket.difference(in_pocket_filtered)))
                    sorted_in_pocket = sorted(list(in_pocket_filtered))

                    shuffled_in_pocket = list(np.random.choice(sorted_in_pocket,
                                                               replace=False,
                                                               size=len(in_pocket_filtered)))
                    shuffled_neigh = list(np.random.choice(sorted_neighbors_bfs,
                                                           replace=False,
                                                           size=len(sorted_neighbors_bfs)))
                    shuffled_in_pocket.extend(shuffled_neigh)
                    noisy_nodelist = shuffled_in_pocket[:n_nodes_to_sample]
                elif perturbation == 'hard':
                    # Sample a pocket around a random node of the perimeter
                    smaller_bfs = graph_utils.bfs(rglib_graph,
                                                  in_pocket_filtered,
                                                  depth=perturb_bfs_depth - 1,
                                                  label='LW')
                    perimeter = sorted(list(around_pocket.difference(smaller_bfs)))
                    if len(perimeter) == 0:
                        print(f"Buggy pocket: {pocket}, it spans the whole connected component and cannot be expanded")
                        continue
                    seed_pertubed_pocket = np.random.choice(perimeter, size=1).item()

                    # Now expand this seed with increasing radius up to getting more than target node
                    prev_perturbed_pocket = {}
                    perturbed_pocket = {seed_pertubed_pocket}
                    expander = 1
                    while len(perturbed_pocket) < n_nodes_to_sample and expander <= 10:
                        prev_perturbed_pocket = perturbed_pocket
                        perturbed_pocket = graph_utils.bfs(rglib_graph,
                                                           perturbed_pocket,
                                                           depth=expander,
                                                           label='LW')
                        expander += 1
                    # When querying with very large fractions, sometimes we cannot return as many nodes as queried
                    # Note: nx.connected_component does not work for directed graphs...
                    if expander > 10:
                        print('Cannot craft a large enough pocket, maybe we seeded using a disconnected component')
                        break

                    # Finally, subsample the last parameter to get the final pocket.
                    last_perimeter = sorted(list(perturbed_pocket.difference(prev_perturbed_pocket)))
                    missing_nbr_nodes = n_nodes_to_sample - len(prev_perturbed_pocket)
                    last_nodes = list(np.random.choice(list(last_perimeter), replace=False, size=missing_nbr_nodes))
                    noisy_nodelist = list(prev_perturbed_pocket) + last_nodes

                else:
                    raise NotImplementedError
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
def get_perf(pocket_path, base_name=None, out_dir=None):
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
    all_pockets = set(test_systems['PDB_ID_POCKET'].unique())
    all_pockets_available = set([x[:-5] for x in os.listdir(pocket_path)])
    missing_pockets = all_pockets - all_pockets_available
    # When using hard_3, systems that fail are : '6E8S_B_SPM_107' (only for r=5) and '5V3F_B_74G_104', '7REX_C_PRF_102
    # 5V3F also fails from the ligand perspective,
    # Others don't and have a ~ bad perf, giving an edge to hard_3.
    if len(missing_pockets) > 0:
        print("missing_pockets : ", missing_pockets)
        test_systems = test_systems[~test_systems["PDB_ID_POCKET"].isin(missing_pockets)]
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
    out_dir = Path(pocket_path).parent if out_dir is None else Path(out_dir)
    if base_name is None:
        base_name = Path(pocket_path).name

    dock_model_path = 'results/trained_models/dock/dock_42'
    dock_model = get_model_from_dirpath(dock_model_path)
    native_model_path = 'results/trained_models/is_native/native_42'
    native_model = get_model_from_dirpath(native_model_path)

    # Get dock performance
    df_dock, df_dock_raw = compute_efs_model(dock_model, dataloader=dataloader, lower_is_better=True)
    df_dock.to_csv(out_dir / (base_name + '_dock.csv'))
    df_dock_raw.to_csv(out_dir / (base_name + "_dock_raw.csv"))

    # Get native performance
    df_native, df_native_raw = compute_efs_model(native_model, dataloader=dataloader, lower_is_better=False)
    df_native.to_csv(out_dir / (base_name + '_native.csv'))
    df_native_raw.to_csv(out_dir / (base_name + "_native_raw.csv"))

    # Now merge those two results to get a final mixed performance
    # Inspired from mixing.py, function find_best_mix and mix_two_scores
    df_dock_raw['dock'] = -df_dock_raw['raw_score'].values
    df_native_raw['native'] = df_native_raw['raw_score'].values
    big_df_raw = df_dock_raw.merge(df_native_raw, on=['pocket_id', 'smiles', 'is_active'], how='outer')

    mixed_df, mixed_df_raw = mix_two_scores(big_df_raw, score1='dock', score2='native')
    mixed_df.to_csv(out_dir / (base_name + '_mixed.csv'))
    mixed_df_raw.to_csv(out_dir / (base_name + "_mixed_raw.csv"))
    return np.mean(mixed_df['score'].values)


def get_efs(all_perturbed_pockets_path='figs/perturbed',
            out_df='figs/perturbed/aggregated.csv',
            recompute=False,
            fractions=None):
    list_of_results = []
    todo = list(sorted([x for x in os.listdir(all_perturbed_pockets_path) if not x.endswith('.csv')]))

    if fractions is not None:
        fractions = set(fractions)
        todo = [x for x in todo if float(x.split('_')[1]) in fractions]
    for i, perturbed_pocket in enumerate(todo):
        print(all_perturbed_pockets_path, i, len(todo))
        perturbed_pocket_path = os.path.join(all_perturbed_pockets_path, perturbed_pocket)

        # Only recompute if the csv ain't here or can't be parsed correclty
        mean_score = None
        if not recompute:
            out_dir = Path(perturbed_pocket_path).parent
            base_name = Path(perturbed_pocket_path).name
            out_csv_path = out_dir / (base_name + "_mixed.csv")
            if os.path.exists(out_csv_path):
                df = pd.read_csv(out_csv_path)
                mean_score = np.mean(df['score'].values)
        if mean_score is None:
            mean_score = get_perf(pocket_path=perturbed_pocket_path)

        _, fraction, replicate = perturbed_pocket.split('_')
        list_of_results.append({"thresh": fraction, "replicate": replicate, "score": mean_score})
    df = pd.DataFrame(list_of_results)
    df.to_csv(out_df)
    return df

def get_all_perturbed_bfs(fractions=(0.7, 0.85, 1.0, 1.15, 1.3), max_replicates=10, hard=False,
                          recompute=False, use_cached_pockets=True):
    dfs = []
    for i in range(1, 4):
        out_path = f'figs/perturbed{"_hard" if hard else ""}_{i}'
        out_df = f'figs/aggregated{"_hard" if hard else ""}_{i}.csv'
        if not use_cached_pockets:
            get_perturbed_pockets(out_path=out_path,
                                  perturb_bfs_depth=i,
                                  perturbation="hard" if hard else "random",
                                  fractions=fractions,
                                  max_replicates=max_replicates,
                                  recompute=recompute)
        df = get_efs(all_perturbed_pockets_path=out_path, out_df=out_df, fractions=fractions, recompute=recompute)
        dfs.append(df)
    return dfs


def get_all_perturbed_soft(fractions=(0.7, 0.85, 1.0, 1.15, 1.3), max_replicates=10,
                           recompute=False, use_cached_pockets=True):
    out_path = f'figs/perturbed_soft'
    out_df = f'figs/aggregated_soft.csv'
    if not use_cached_pockets:
        get_perturbed_pockets(out_path=out_path,
                              perturb_bfs_depth=2,
                              fractions=fractions,
                              max_replicates=max_replicates,
                              perturbation='soft',
                              recompute=recompute)
    df = get_efs(all_perturbed_pockets_path=out_path, out_df=out_df, fractions=fractions, recompute=recompute)
    return df


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # Check that inference works, we should get 0.9848
    # os.makedirs("figs/perturbed", exist_ok=True)
    # get_perf(pocket_path="data/json_pockets_expanded", out_dir="figs/perturbed")

    # Check pocket computation works
    # get_perturbed_pockets(unperturbed_path='data/json_pockets_expanded',
    #                       out_path='figs/perturbed_1',
    #                       fractions=(0.7, 0.85, 1.0, 1.15, 1.3),
    #                       perturb_bfs_depth=1,
    #                       max_replicates=2)

    # Get a first result
    # df = get_efs(all_perturbed_pockets_path='figs/perturbed', out_df='figs/perturbed/aggregated.csv')
    # df = pd.read_csv('figs/perturbed/aggregated.csv')

    # fractions = (0.1, 0.7, 0.85, 1.0, 1.15, 1.3, 5)
    fractions = (0.7, 0.85, 1.0, 1.15, 1.3)
    # fractions = (0.1, 5)
    # Now compute perturbed scores using the random BFS approach
    # dfs = get_all_perturbed_bfs(fractions=fractions, recompute=False, use_cached_pockets=True)
    dfs_hard = get_all_perturbed_bfs(fractions=fractions, recompute=False, use_cached_pockets=True, hard=True)
    # dfs = dfs[:-1]

    # Now compute perturbed scores using the soft approach
    df_soft = get_all_perturbed_soft(fractions=fractions, recompute=False, use_cached_pockets=True)

    colors = sns.light_palette('royalblue', n_colors=4, reverse=True)


    def get_low_high(df, fractions):
        if not isinstance(fractions, (list, tuple)):
            fractions = [fractions]
        df = df[df['thresh'].isin([str(x) for x in fractions])]
        means = df.groupby('thresh')['score'].mean().values
        stds = df.groupby('thresh')['score'].std().values
        means_low = means - stds
        means_high = means + stds
        return means, means_low, means_high


    # # Plot BFS perturbed
    for i, df in enumerate(dfs_hard):
        means, means_low, means_high = get_low_high(df, fractions)
        color = colors[i]
        plt.plot(fractions, means, linewidth=2, color=color, label=rf'Perturbed pockets with BFS:{i + 1}')
        plt.fill_between(fractions, means_low, means_high, alpha=0.2, color=color)

    # Plot soft perturbed
    means, means_low, means_high = get_low_high(df_soft, fractions)
    color = 'purple'
    plt.plot(fractions, means, linewidth=2, color=color, label=rf'Perturbed pockets with soft stategy')
    plt.fill_between(fractions, means_low, means_high, alpha=0.2, color=color)

    # End of the plot + pretty plot
    plt.hlines(y=0.984845, xmin=min(fractions), xmax=max(fractions),
               label=r'Original pockets', color=PALETTE_DICT['mixed'], linestyle='--')
    # plt.hlines(y=0.9593, xmin=min(fractions), xmax=max(fractions),
    #            label=r'rDock', color=PALETTE_DICT['rdock'], linestyle='--')
    plt.legend(loc='lower right')
    plt.ylabel(r"mean AuROC over pockets")
    plt.xlabel(r"Fraction of nodes sampled")
    plt.show()
