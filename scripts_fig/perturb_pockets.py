"""
In this file, a first set of functions computes pockets corruptions:
- get_expanded_subgraph_from_list : just a util to get a graph.json from a nodelist
- get_perturbed_pockets: different strategies to build perturbed node lists (and then graphs) from a given pocket
- compute_overlaps: a postprocessing function, for each perturbed pocket, it computes the overlap with the GT pocket

Then a second set of function computes AuROCs and EFs from a directory containing pockets.
The computation is different for pdb/chembl pockets/decoys and ROBIN systems.
Indeed, in the first case, we have 60*.7k pockets ligands pairs and in the second we have 4*20k.
Hence, we have:
- get_perf <- compute_efs_model <- enrichment_factor : returns a df for the classical scenario
- get_perf_robin <- do_robin <- enrichment_factor : returns a df for the ROBIN scenario
- get_efs uses one of these functions on a directory containing directories of perturbed pockets with different
conditions (fractions, replicates and so on..)

A third set of functions launches both computations automatically:
1. compute one kind of pocket perturbation
2. compute EFs over it
- get_all_perturbed_bfs
- get_all_perturbed_soft
- get_all_perturbed_rognan
 TODO: those could probably be factored more compactly

A fourth set of functions is used to produce plots.

Finally, two main() are defined, one for ROBIN and one for normal scenario. These redefine global variables and make
the right calls to get the relevant final plots.
"""

import os
import sys

from dgl.dataloading import GraphDataLoader
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd
import random
from rnaglib.utils import graph_from_pdbid, graph_utils, graph_io
import seaborn as sns
from sklearn import metrics
import torch
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.learning.dataset import VirtualScreenDataset, get_systems
from rnamigos.utils.virtual_screen import mean_active_rank, run_virtual_screen
from rnamigos.utils.graph_utils import load_rna_graph
from scripts_run.robin_inference import robin_inference

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
                          out_path='figs/perturbations/perturbed',
                          fractions=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
                          perturb_bfs_depth=1,
                          max_replicates=5,
                          recompute=True,
                          perturbation='random',
                          final_bfs=4):
    existing_pockets = set([pocket.rstrip('.json') for pocket in os.listdir(unperturbed_path)])
    pockets_to_compute = sorted(list(existing_pockets.intersection(ALL_POCKETS)))
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
                elif perturbation == 'rognan like':
                    # Sample a pocket around a random node of the perimeter
                    seed_pertubed_pocket = np.random.choice(rglib_graph.nodes(), size=1).item()

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
                expanded_graph = get_expanded_subgraph_from_list(rglib_graph=rglib_graph,
                                                                 nodelist=noisy_nodelist,
                                                                 bfs_depth=final_bfs)
                if len(expanded_graph) == 0:
                    print('Tried to create empty graph, skipped system: ', pocket, fraction, replicate)
                    continue
                graph_io.dump_json(out_name, expanded_graph)


def compute_overlaps(original_pockets, modified_pockets_path, dump_path=None):
    """
    Given a directory of modified pockets and a dict of reference ones, return a dict
    with the overlap between modified and original pockets.
    """
    resdict = {}
    for modified_pocket_name in os.listdir(modified_pockets_path):
        modified_pocket_path = os.path.join(modified_pockets_path, modified_pocket_name)
        modified_pocket = graph_io.load_json(modified_pocket_path)
        pocket_id = modified_pocket_name[:-5]
        original_pocket = original_pockets[pocket_id]
        mod_nodes = set(modified_pocket.nodes())
        ori_nodes = set(original_pocket.nodes())
        extra_nodes = len(mod_nodes.difference(ori_nodes))
        missing_nodes = len(ori_nodes.difference(mod_nodes))
        pocket_size = len(ori_nodes)
        # if extra_nodes > 10:
        #     a = 1
        resdict[pocket_id] = (extra_nodes, missing_nodes, pocket_size)
    if dump_path is not None:
        rows = [{"pocket_id": pocket_id,
                 "extra": extra,
                 "missing": missing,
                 "pocket_size": pocket_size} for pocket_id, (extra, missing, pocket_size) in
                resdict.items()]
        df = pd.DataFrame(rows)
        df.to_csv(dump_path, index=False)
    return resdict


def enrichment_factor(scores, is_active, lower_is_better=True, frac=0.01):
    # ddf = pd.DataFrame({'score': scores, 'is_active': is_active})
    # sns.kdeplot(ddf, x='score', hue='is_active', common_norm=False)
    # plt.show()

    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=not lower_is_better)]
    scores_sorted = [s for s, _ in sorted(zip(scores, is_active), reverse=not lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    ef = (n_actives_screened / n_screened) / (n_actives / len(scores))
    return ef, scores_sorted[n_screened]


def compute_efs_model(model, dataloader, lower_is_better):
    """
    Given a model and a dataloader, make the inference on all pocket-ligand pairs and dump raw and aggregated csvs
    """
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

    # compute real EFs
    raw_df = pd.DataFrame(raw_rows)

    ef_rows = []
    for frac in (0.01, 0.02, 0.05):
        for pocket, group in raw_df.groupby('pocket_id'):
            ef_frac, _ = enrichment_factor(group['raw_score'], group['is_active'], frac=frac,
                                           lower_is_better=lower_is_better)
            ef_rows.append({'score': ef_frac,
                            'pocket_id': pocket,
                            'frac': frac
                            })

    df_ef = pd.DataFrame(ef_rows)
    print(df_ef)

    for ef, score, pocket_id in zip(efs, scores, pocket_names):
        grouped = {
            'score': ef,
            'pocket_id': pocket_id}
        rows.append(grouped)
    print('Mean EF :', np.mean(efs))
    df = pd.DataFrame(rows)
    df_raw = pd.DataFrame(raw_rows)
    return df, df_raw, df_ef


# Copied from evaluate except reps_only=True to save time
#   cache_graphs=True to save time over two model runs
#   target is set to "is_native" which has no impact since it's just used to get pdb lists
# The goal here is just to have easy access to the loader and modify its pockets_path
def get_perf(pocket_path, base_name=None, out_dir=None):
    """
    Starting from a pocket path containing pockets, and using global variables to set things like pockets to use or
    paths, dump the native/dock/mixed results of a virtual screening
    """
    # Setup loader
    print(f"get_perf {pocket_path}")
    all_pockets_available = set([x[:-5] for x in os.listdir(pocket_path)])
    missing_pockets = ALL_POCKETS - all_pockets_available

    # When using hard_3, systems that fail are : '6E8S_B_SPM_107' (only for r=5) and '5V3F_B_74G_104', '7REX_C_PRF_102
    # 5V3F also fails from the ligand perspective,
    # Others don't and have a ~ bad perf, giving an edge to hard_3.
    if len(missing_pockets) > 0:
        print("missing_pockets : ", missing_pockets)
        test_systems = TEST_SYSTEMS[~TEST_SYSTEMS["PDB_ID_POCKET"].isin(missing_pockets)]
    else:
        test_systems = TEST_SYSTEMS
    decoy_mode = 'robin' if ROBIN else 'chembl'
    ligand_cache = f'data/ligands/{"robin_" if ROBIN else ""}lig_graphs.p'
    dataset = VirtualScreenDataset(pocket_path,
                                   cache_graphs=False,
                                   ligands_path="data/ligand_db",
                                   systems=test_systems,
                                   decoy_mode=decoy_mode,
                                   use_graphligs=True,
                                   group_ligands=False,
                                   reps_only=not ROBIN,
                                   ligand_cache=ligand_cache,
                                   use_ligand_cache=True,
                                   )
    dataloader = GraphDataLoader(dataset=dataset, **LOADER_ARGS)

    # Setup path and models
    out_dir = Path(pocket_path).parent if out_dir is None else Path(out_dir)
    if base_name is None:
        base_name = Path(pocket_path).name

    dock_model_path = 'results/trained_models/dock/dock_42'
    dock_model = get_model_from_dirpath(dock_model_path)
    native_model_path = 'results/trained_models/is_native/native_42'
    native_model = get_model_from_dirpath(native_model_path)

    # Get dock performance
    df_dock, df_dock_raw, df_dock_ef = compute_efs_model(dock_model, dataloader=dataloader, lower_is_better=True)
    df_dock.to_csv(out_dir / (base_name + '_dock.csv'))
    df_dock_raw.to_csv(out_dir / (base_name + "_dock_raw.csv"))
    df_dock_ef.to_csv(out_dir / (base_name + "_dock_ef.csv"))

    # Get native performance
    df_native, df_native_raw, df_native_ef = compute_efs_model(native_model,
                                                               dataloader=dataloader,
                                                               lower_is_better=False)
    df_native.to_csv(out_dir / (base_name + '_native.csv'))
    df_native_raw.to_csv(out_dir / (base_name + "_native_raw.csv"))
    df_native_ef.to_csv(out_dir / (base_name + "_native_ef.csv"))

    # Now merge those two results to get a final mixed performance
    # Inspired from mixing.py, function find_best_mix and mix_two_scores
    df_dock_raw['dock'] = -df_dock_raw['raw_score'].values
    df_native_raw['native'] = df_native_raw['raw_score'].values
    big_df_raw = df_dock_raw.merge(df_native_raw, on=['pocket_id', 'smiles', 'is_active'], how='outer')

    def mix_two_scores(df, score1, score2):
        """
        Adapted from mixing to return a raw df
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

        ef_rows = []
        for frac in (0.01, 0.02, 0.05):
            for pocket, group in mixed_df_raw.groupby('pocket_id'):
                ef_frac, _ = enrichment_factor(group['mixed'], group['is_active'], frac=frac, lower_is_better=False)
                ef_rows.append({'score': ef_frac,
                                'pocket_id': pocket,
                                'frac': frac
                                })

        mixed_df_ef = pd.DataFrame(ef_rows)
        return mixed_df, mixed_df_raw, mixed_df_ef

    mixed_df, mixed_df_raw, mixed_df_ef = mix_two_scores(big_df_raw, score1='dock', score2='native')
    mixed_df.to_csv(out_dir / (base_name + '_mixed.csv'))
    mixed_df_raw.to_csv(out_dir / (base_name + "_mixed_raw.csv"))
    mixed_df_ef.to_csv(out_dir / (base_name + "_mixed_ef.csv"))
    return np.mean(mixed_df['score'].values)


def do_robin(ligand_name, pocket_path):
    print('Doing pocket : ', pocket_path)

    # Get dgl pocket
    dgl_pocket_graph, _ = load_rna_graph(pocket_path + '.json')

    # Compute scores and EFs
    final_df = robin_inference(ligand_name, dgl_pocket_graph)
    pocket_id = Path(pocket_path).stem
    final_df['pocket_id'] = pocket_id
    ef_rows = []
    for frac in (0.01, 0.02, 0.05):
        ef, _ = enrichment_factor(final_df['mixed_score'],
                                  final_df['is_active'],
                                  lower_is_better=False,
                                  frac=frac)
        ef_rows.append({'pocket_id': pocket_id, 'score': ef, 'frac': frac})
    ef_df = pd.DataFrame(ef_rows)
    return ef_df, final_df


def get_perf_robin(pocket_path, base_name=None, out_dir=None):
    # Setup loader
    # Setup path and models
    out_dir = Path(pocket_path).parent if out_dir is None else Path(out_dir)
    if base_name is None:
        base_name = Path(pocket_path).name
    ef_dfs = []
    raw_dfs = []
    for ef_df, raw in Parallel(n_jobs=4)(
            delayed(do_robin)(ligand_name, os.path.join(pocket_path, pocket)) for ligand_name, pocket in
            ROBIN_POCKETS.items()):
        ef_dfs.append(ef_df)
        raw_dfs.append(raw)
    df_raw = pd.concat(raw_dfs)
    df_score = pd.concat(ef_dfs)
    df_raw.to_csv(out_dir / (base_name + "_raw.csv"))
    df_score.to_csv(out_dir / (base_name + "_ef.csv"))
    return np.mean(df_score['score'].values)


def get_efs(all_perturbed_pockets_path='figs/perturbations/perturbed',
            out_df='figs/perturbations/perturbed/aggregated.csv',
            recompute=True,
            fractions=None,
            compute_overlap=False,
            metric='ef',
            ef_frac=0.02):
    list_of_results = []
    todo = list(sorted([x for x in os.listdir(all_perturbed_pockets_path) if not x.endswith('.csv')]))

    if fractions is not None:
        fractions = set(fractions)
        todo = [x for x in todo if float(x.split('_')[1]) in fractions]
    for i, perturbed_pocket_dir in enumerate(todo):
        _, fraction, replicate = perturbed_pocket_dir.split('_')

        perturbed_pocket_path = os.path.join(all_perturbed_pockets_path, perturbed_pocket_dir)

        # Only recompute if the csv ain't here or can't be parsed correclty
        out_dir = Path(perturbed_pocket_path).parent
        base_name = Path(perturbed_pocket_path).name
        if ROBIN:
            out_csv_path = out_dir / (base_name + f"{'_ef' if metric == 'ef' else ''}.csv")
        else:
            # out_csv_path = out_dir / (base_name + "_dock.csv")
            # out_csv_path = out_dir / (base_name + "_native.csv")
            # out_csv_path = out_dir / (base_name + "_mixed.csv")
            out_csv_path = out_dir / (base_name + f"_mixed{'_ef' if metric == 'ef' else ''}.csv")
        if recompute or not os.path.exists(out_csv_path):
            if ROBIN:
                _ = get_perf_robin(pocket_path=perturbed_pocket_path)
            else:
                _ = get_perf(pocket_path=perturbed_pocket_path)
        if not metric == 'ef':
            df = pd.read_csv(out_csv_path)[['pocket_id', 'score']]
        else:
            df = pd.read_csv(out_csv_path)[['pocket_id', 'score', 'frac']]
            df = df.loc[df['frac'] == ef_frac]

        mean_score = np.mean(df['score'].values)
        if compute_overlap:
            overlap_csv_path = out_dir / (base_name + "_overlap.csv")
            if not os.path.exists(overlap_csv_path):
                compute_overlaps(original_pockets=ALL_POCKETS_GRAPHS,
                                 modified_pockets_path=perturbed_pocket_path,
                                 dump_path=overlap_csv_path)
            overlap_df = pd.read_csv(overlap_csv_path)
            perturb_df = df.merge(overlap_df, on=['pocket_id'], how='left')
        else:
            # Aggregated version
            # perturb_df = pd.DataFrame({"thresh": fraction, "replicate": replicate, "score": mean_score})
            df["thresh"] = fraction
            df["replicate"] = replicate
            perturb_df = df

        list_of_results.append(perturb_df)
    df = pd.concat(list_of_results)
    df.to_csv(out_df)
    return df


def get_all_perturbed_bfs(fractions=(0.7, 0.85, 1.0, 1.15, 1.3), max_replicates=10, hard=False,
                          recompute=True, use_cached_pockets=True, compute_overlap=False,
                          metric='ef', ef_frac=0.02):
    dfs = []
    for i in range(1, 4):
        out_path = f'figs/perturbations/perturbed{"_hard" if hard else ""}{"robin_" if ROBIN else ""}_{i}'
        out_df = f'figs/perturbations/aggregated{"_hard" if hard else ""}{"robin_" if ROBIN else ""}_{i}.csv'
        if not use_cached_pockets:
            get_perturbed_pockets(out_path=out_path,
                                  perturb_bfs_depth=i,
                                  perturbation="hard" if hard else "random",
                                  fractions=fractions,
                                  max_replicates=max_replicates,
                                  recompute=recompute)
        df = get_efs(all_perturbed_pockets_path=out_path,
                     out_df=out_df,
                     fractions=fractions,
                     recompute=recompute,
                     compute_overlap=compute_overlap,
                     metric=metric,
                     ef_frac=ef_frac)
        dfs.append(df)
    return dfs


def get_all_perturbed_soft(fractions=(0.7, 0.85, 1.0, 1.15, 1.3),
                           max_replicates=10,
                           recompute=True,
                           use_cached_pockets=True,
                           final_bfs=4,
                           compute_overlap=False,
                           metric='ef',
                           ef_frac=0.02):
    out_path = f'figs/perturbations/perturbed_soft_robin_{final_bfs}'
    out_df = f'figs/perturbations/aggregated_soft_robin_{final_bfs}.csv'
    if not use_cached_pockets:
        get_perturbed_pockets(out_path=out_path,
                              perturb_bfs_depth=2,
                              fractions=fractions,
                              max_replicates=max_replicates,
                              perturbation='soft',
                              recompute=recompute,
                              final_bfs=final_bfs)
    df = get_efs(all_perturbed_pockets_path=out_path,
                 out_df=out_df,
                 fractions=fractions,
                 recompute=recompute,
                 compute_overlap=compute_overlap,
                 metric=metric,
                 ef_frac=ef_frac)
    return df


def get_all_perturbed_rognan(fractions=(0.7, 0.85, 1.0, 1.15, 1.3), max_replicates=10,
                             recompute=True, use_cached_pockets=False, final_bfs=4,
                             metric='ef', ef_frac=0.02):
    out_path = f'figs/perturbations/perturbed_rognan_robin'
    out_df = f'figs/perturbations/aggregated_rognan_robin.csv'
    if not use_cached_pockets:
        get_perturbed_pockets(out_path=out_path,
                              perturb_bfs_depth=2,
                              fractions=fractions,
                              max_replicates=max_replicates,
                              perturbation='rognan like',
                              recompute=recompute,
                              final_bfs=final_bfs)
    df = get_efs(all_perturbed_pockets_path=out_path,
                 out_df=out_df,
                 fractions=fractions,
                 recompute=recompute,
                 metric=metric,
                 ef_frac=ef_frac)
    return df


def add_delta(df):
    df = df.merge(DF_UNPERTURBED, how="left", on='pocket_id')
    df['delta'] = df['score'] - df['unpert_score']
    return df


def add_pert_magnitude(df):
    # pert_magn = (df['extra'].values) / df['pocket_size'].values
    # pert_magn = (df['missing'].values) / df['pocket_size'].values
    # pert_magn = df['missing'].values
    pert_magn = (df['extra'].values + df['missing'].values) / df['pocket_size'].values
    # jaccard
    pert_magn = (df['pocket_size'].values - df['missing'].values) / (df['pocket_size'].values + df['extra'].values)
    df['magnitude'] = pert_magn
    return df


def plot_overlap(df, filter_good=True, **kwargs):
    df = add_pert_magnitude(df)
    df = add_delta(df)
    if filter_good:
        df = filter_on_good_pockets(df)
    plt.scatter(df['magnitude'], df['delta'], **kwargs)


def get_low_high(df, fractions, to_plot='score', filter_good=True, error_bar=True, metric='ef'):
    if not isinstance(fractions, (list, tuple)):
        fractions = [fractions]
    # df = df[df['replicate'].isin([str(x) for x in (0, 1)])]
    if filter_good:
        df = filter_on_good_pockets(df)
    df = df[df['thresh'].isin([str(x) for x in fractions])]
    if metric != 'ef':
        df[to_plot] = 100 * df[to_plot]
    means = df.groupby('thresh')[to_plot].mean().values
    if error_bar:
        stds = df.groupby('thresh').agg({'score': lambda x: x.std() / np.sqrt(len(x))}).values.flatten()
    else:
        stds = df.groupby('thresh')[to_plot].std().values
    means_low = means - stds
    means_high = means + stds
    return means, means_low, means_high


def plot_one(df, fractions, filter_good=True, plot_delta=True, color='blue', label='default_label', metric='ef'):
    if plot_delta:
        df = add_delta(df)
        to_plot = 'delta'
    else:
        to_plot = 'score'
    means, means_low, means_high = get_low_high(df, fractions, to_plot=to_plot, filter_good=filter_good, metric=metric)
    plt.plot(fractions, means, linewidth=2, color=color, label=label)
    plt.fill_between(fractions, means_low, means_high, alpha=0.2, color=color)


def plot_list(dfs, fractions, colors='blue', label='default_label', **kwargs):
    for i, df in enumerate(dfs):
        plot_one(df, fractions, color=colors[i], label=f"{label}: {i}", **kwargs)


def end_plot():
    # End of the plot + pretty plot
    # plt.hlines(y=0.934, xmin=min(fractions), xmax=max(fractions),  # dock
    # plt.hlines(y=0.951, xmin=min(fractions), xmax=max(fractions),  # native
    # plt.hlines(y=0.984845, xmin=min(fractions), xmax=max(fractions),
    #            label=r'Original pockets', color=PALETTE_DICT['mixed'], linestyle='--')
    # plt.hlines(y=0.9593, xmin=min(fractions), xmax=max(fractions),
    #            label=r'rDock', color=PALETTE_DICT['rdock'], linestyle='--')
    plt.legend(loc='lower right')
    plt.ylabel(r"mean AuROC over pockets")
    plt.xlabel(r"Fraction of nodes sampled")
    plt.show()


def filter_on_good_pockets(df):
    return df[df['pocket_id'].isin(GOOD_POCKETS)]


def main_chembl():
    global TEST_SYSTEMS
    global ALL_POCKETS
    global ALL_POCKETS_GRAPHS
    global DF_UNPERTURBED
    global ROBIN
    ROBIN = False
    TEST_SYSTEMS = get_systems(target="is_native",
                               rnamigos1_split=-2,
                               use_rnamigos1_train=False,
                               use_rnamigos1_ligands=False,
                               return_test=True)
    ALL_POCKETS = set(TEST_SYSTEMS['PDB_ID_POCKET'].unique())
    ALL_POCKETS_GRAPHS = {pocket_id: graph_io.load_json(os.path.join("data/json_pockets_expanded", f"{pocket_id}.json"))
                          for pocket_id in ALL_POCKETS}
    # # Check that inference works, we should get 0.9848
    os.makedirs("figs/perturbations/unperturbed", exist_ok=True)
    get_perf(pocket_path="data/json_pockets_expanded",
             out_dir="figs/perturbations/unperturbed")
    DF_UNPERTURBED = pd.read_csv("figs/perturbations/unperturbed/json_pockets_expanded_mixed.csv", index_col=False)
    DF_UNPERTURBED.rename(columns={'score': 'unpert_score'}, inplace=True)
    global GOOD_POCKETS
    GOOD_POCKETS = DF_UNPERTURBED[DF_UNPERTURBED['unpert_score'] >= 0.98]['pocket_id'].unique()

    # fractions = (0.1, 0.7, 0.85, 1.0, 1.15, 1.3, 5)
    fractions = (0.7, 0.85, 1.0, 1.15, 1.3)
    # fractions = (1, 1)
    colors = sns.light_palette('royalblue', n_colors=4, reverse=True)

    # Check pocket computation works
    # get_perturbed_pockets(unperturbed_path='data/json_pockets_expanded',
    #                      out_path='figs/perturbations/perturbed_robin',
    #                      fractions=(0.9, 1.0),
    #                      perturb_bfs_depth=1,
    #                      max_replicates=2)
    # # Get a first result
    # df = get_efs(all_perturbed_pockets_path='figs/perturbations/perturbed_robin',
    #            out_df='figs/perturbations/perturbed_robin/aggregated_test.csv',
    #            compute_overlap=True)

    # Now compute perturbed scores using the random BFS approach
    # dfs_random = get_all_perturbed_bfs(fractions=fractions, recompute=False, use_cached_pockets=True)
    # plot_list(dfs=dfs_random, fractions=fractions, colors=colors, label="Random strategy")

    # Hard: sample on the border
    # dfs_hard = get_all_perturbed_bfs(fractions=fractions, recompute=False, use_cached_pockets=True, hard=True)
    # plot_list(dfs=dfs_hard, fractions=fractions, colors=colors, label="Hard strategy")

    use_cached_pockets = False
    recompute = False
    metric = 'ef'
    # Rognan like
    df_rognan = get_all_perturbed_rognan(fractions=fractions, recompute=recompute,
                                         use_cached_pockets=use_cached_pockets)
    # plot_one(df_rognan, fractions=fractions, color='black', label='Rognan strategy')    # Plot rognan

    # Now compute perturbed scores using the soft approach.
    # Vary unexpanding. You can't do BFS0, since this makes small graphs with no edges,
    # resulting in empty graph when subgraphing
    df_soft_4 = get_all_perturbed_soft(fractions=fractions, use_cached_pockets=use_cached_pockets, final_bfs=4,
                                       recompute=recompute, metric=metric)
    df_soft_1 = get_all_perturbed_soft(fractions=fractions, use_cached_pockets=use_cached_pockets, final_bfs=1,
                                       recompute=recompute, metric=metric)
    """
    df_soft_2 = get_all_perturbed_soft(fractions=fractions, use_cached_pockets=use_cached_pockets, final_bfs=2, recompute=recompute, metric=metric, robin=robin)
    df_soft_3 = get_all_perturbed_soft(fractions=fractions, use_cached_pockets=use_cached_pockets, final_bfs=3, recompute=recompute, metric=metric, robin=robin)
    """
    # dfs_soft = [
    #     df_soft_1,
    #     df_soft_2,
    #     df_soft_3,
    #     df_soft_4
    # ]
    plot_one(df_soft_1, plot_delta=False, filter_good=False, fractions=fractions, color='purple',
             label='soft 1')  # Plot soft perturbed
    # plot_list(dfs=dfs_soft, colors=colors, label="Soft strategy")
    end_plot()

    # Compute plots with overlap
    # With soft strategy
    # df_soft_4_overlap = get_all_perturbed_soft(fractions=fractions, use_cached_pockets=use_cached_pockets,
    #                                            compute_overlap=True)
    # plot_overlap(df_soft_4_overlap)

    # With hard strategy
    # dfs_hard_overlap = get_all_perturbed_bfs(fractions=fractions,
    #                                          hard=True,
    #                                          use_cached_pockets=use_cached_pockets,
    #                                          compute_overlap=True)
    # for i, df_hard in enumerate(dfs_hard_overlap):
    #     plot_overlap(df_hard, color=colors[i])
    # plt.show()


def main_robin():
    global TEST_SYSTEMS
    global LOADER_ARGS
    global ALL_POCKETS
    global ALL_POCKETS_GRAPHS
    global DF_UNPERTURBED
    global ROBIN
    global ROBIN_POCKETS

    ROBIN_POCKETS = {'TPP': '2GDI_Y_TPP_100',
                     'ZTP': '5BTP_A_AMZ_106',
                     'SAM_ll': '2QWY_B_SAM_300',
                     'PreQ1': '3FU2_A_PRF_101'
                     }
    TEST_SYSTEMS = pd.DataFrame({'PDB_ID_POCKET': list(ROBIN_POCKETS.values())})
    ALL_POCKETS = set(ROBIN_POCKETS.values())
    ROBIN = True
    ALL_POCKETS_GRAPHS = {pocket_id: graph_io.load_json(os.path.join("data/json_pockets_expanded", f"{pocket_id}.json"))
                          for pocket_id in ALL_POCKETS}
    os.makedirs("figs/perturbations/unperturbed_robin", exist_ok=True)
    get_perf_robin(pocket_path="data/json_pockets_expanded",
                   out_dir="figs/perturbations/unperturbed_robin")
    DF_UNPERTURBED = pd.read_csv("figs/perturbations/unperturbed_robin/json_pockets_expanded_mixed.csv",
                                 index_col=False)

    # fractions = (0.1, 0.7, 0.85, 1.0, 1.15, 1.3, 5)
    fractions = (0.7, 0.85, 1.0, 1.15, 1.3)
    # fractions = (0.1, 5)
    colors = sns.light_palette('royalblue', n_colors=4, reverse=True)
    ef_frac = 0.02
    use_cached_pockets = False
    recompute = False
    metric = 'ef'
    df_soft_1 = get_all_perturbed_soft(fractions=fractions, use_cached_pockets=use_cached_pockets, final_bfs=1,
                                       recompute=recompute, metric=metric, ef_frac=ef_frac)
    plot_one(df_soft_1, plot_delta=False, filter_good=False, fractions=fractions, color='purple',
             label='bfs 1')  # Plot soft perturbed
    end_plot()


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    LOADER_ARGS = {'shuffle': False,
                   'batch_size': 1,
                   'num_workers': 4,
                   'collate_fn': lambda x: x[0]
                   }
    TEST_SYSTEMS, ALL_POCKETS, ALL_POCKETS_GRAPHS, DF_UNPERTURBED, ROBIN = [None, ] * 5

    # main_chembl()
    main_robin()
