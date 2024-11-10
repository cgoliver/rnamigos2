"""
In this file, a first set of functions computes AuROCs and EFs from a directory containing pockets.
The computation is different for pdb/chembl pockets/decoys and ROBIN systems.
Indeed, in the first case, we have 60*.7k pockets ligands pairs and in the second we have 4*20k.
Hence, we have:
- get_perf <- compute_efs_model <- enrichment_factor : returns a df for the classical scenario
- get_perf_robin <- do_robin <- enrichment_factor : returns a df for the ROBIN scenario
- get_efs uses one of these functions on a directory containing directories of perturbed pockets with different
conditions (fractions, replicates and so on...)

A fourth set of functions is used to produce plots.

Finally, two main() are defined, one for ROBIN and one for normal scenario. These redefine global variables and make
the right calls to get the relevant final plots.
"""

import os
import sys

from dgl.dataloading import GraphDataLoader
from joblib import Parallel, delayed
from functools import partial
import numpy as np
from pathlib import Path
import pandas as pd
import random
from rnaglib.utils import graph_io
import seaborn as sns
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from rnamigos.learning.dataset import VirtualScreenDataset, get_systems
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.utils.mixing_utils import mix_two_dfs
from rnamigos.utils.virtual_screen import get_auroc, run_virtual_screen, enrichment_factor
from rnamigos.utils.virtual_screen import run_results_to_raw_df, run_results_to_auroc_df, raw_df_to_efs
from scripts_run.robin_inference import robin_inference_raw
from scripts_fig.perturbations.get_perturbed_pockets import get_perturbed_pockets, compute_overlaps
from scripts_fig.perturbations.pertub_plot_utils import end_plot, plot_one, plot_list

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)


def compute_efs_model(model, dataloader, lower_is_better):
    """
    Given a model and a dataloader, make the inference on all pocket-ligand pairs and dump raw and aggregated csvs
    """
    efs, scores, status, pocket_names, all_smiles = run_virtual_screen(
        model, dataloader, metric=get_auroc, lower_is_better=lower_is_better
    )
    df_aurocs = run_results_to_auroc_df(efs, scores, pocket_names, decoy_mode=DECOYS)
    df_raw = run_results_to_raw_df(scores, status, pocket_names, all_smiles, decoy_mode=DECOYS)
    df_ef = raw_df_to_efs(df_raw)
    return df_aurocs, df_raw, df_ef


# Copied from evaluate except reps_only=True to save time
#   cache_graphs=True to save time over two model runs
#   target is set to "is_native" which has no impact since it's just used to get pdb lists
# The goal here is just to have easy access to the loader and modify its pockets_path
def get_perf(pocket_path, base_name=None, out_dir=None):
    """
    Starting from a pocket path containing pockets, and using global variables to set things like pockets to use or
    paths, dump the native/dock/mixed results (raw, mar and efs) of a virtual screening
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
    ligand_cache = f'data/ligands/{"robin_" if ROBIN else ""}lig_graphs.p'
    dataset = VirtualScreenDataset(
        pocket_path,
        cache_graphs=False,
        ligands_path="data/ligand_db",
        systems=test_systems,
        decoy_mode=DECOYS,
        use_graphligs=True,
        group_ligands=False,
        reps_only=not ROBIN,
        ligand_cache=ligand_cache,
        use_rnafm=True,
        use_ligand_cache=True,
    )
    dataloader = GraphDataLoader(dataset=dataset, **LOADER_ARGS)

    # Setup path and models
    out_dir = Path(pocket_path).parent if out_dir is None else Path(out_dir)
    if base_name is None:
        base_name = Path(pocket_path).name

    # Get dock performance
    dock_model_path = "results/trained_models/dock/dock_42"
    dock_model = get_model_from_dirpath(dock_model_path)
    df_dock_aurocs, df_dock_raw, df_dock_ef = compute_efs_model(dock_model, dataloader=dataloader, lower_is_better=True)
    df_dock_aurocs.to_csv(out_dir / (base_name + f"_dock{'_chembl' if DECOYS == 'chembl' else ''}.csv"))
    df_dock_raw.to_csv(out_dir / (base_name + f"_dock{'_chembl' if DECOYS == 'chembl' else ''}_raw.csv"))
    df_dock_ef.to_csv(out_dir / (base_name + f"_dock{'_chembl' if DECOYS == 'chembl' else ''}_ef.csv"))
    # df_dock_raw = pd.read_csv(out_dir / (base_name + "_dock_raw.csv"))

    # Get native performance
    # native_model_path = "results/trained_models/is_native/native_w0.01_gap_nobn"
    # native_model_path = "results/trained_models/is_native/native_bce0.02"
    native_model_path = "results/trained_models/is_native/native_bce0.02_groupsample"
    native_model = get_model_from_dirpath(native_model_path)
    df_native_aurocs, df_native_raw, df_native_ef = compute_efs_model(
        native_model, dataloader=dataloader, lower_is_better=False
    )
    df_native_aurocs.to_csv(out_dir / (base_name + f"_native{'_chembl' if DECOYS == 'chembl' else ''}_sample.csv"))
    df_native_raw.to_csv(out_dir / (base_name + f"_native{'_chembl' if DECOYS == 'chembl' else ''}_sample_raw.csv"))
    df_native_ef.to_csv(out_dir / (base_name + f"_native{'_chembl' if DECOYS == 'chembl' else ''}_sample_ef.csv"))

    # Now merge those two results to get a final mixed performance
    all_aurocs, mixed_df_aurocs, mixed_df_raw = mix_two_dfs(
        df_native_raw, df_dock_raw, score_1="raw_score", outname_col="mixed", use_max=True
    )
    mixed_df_ef = raw_df_to_efs(mixed_df_raw, score="mixed")
    mixed_df_aurocs.to_csv(out_dir / (base_name + f"_mixed{'_chembl' if DECOYS == 'chembl' else ''}.csv"))
    mixed_df_raw.to_csv(out_dir / (base_name + f"_mixed{'_chembl' if DECOYS == 'chembl' else ''}_raw.csv"))
    mixed_df_ef.to_csv(out_dir / (base_name + f"_mixed{'_chembl' if DECOYS == 'chembl' else ''}_ef.csv"))
    return np.mean(mixed_df_aurocs["score"].values)


def do_robin(ligand_name, pocket_path, use_rnafm=False):
    print("Doing pocket : ", pocket_path)

    # Get dgl pocket
    dgl_pocket_graph, _ = load_rna_graph(pocket_path + ".json", use_rnafm=use_rnafm)

    # Compute scores and EFs
    final_df = robin_inference_raw(ligand_name, dgl_pocket_graph)
    pocket_id = Path(pocket_path).stem
    final_df["pocket_id"] = pocket_id
    ef_df = raw_df_to_efs(final_df, score="mixed_score")
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
        delayed(do_robin)(ligand_name, os.path.join(pocket_path, pocket))
        for ligand_name, pocket in ROBIN_POCKETS.items()
    ):
        ef_dfs.append(ef_df)
        raw_dfs.append(raw)
    df_raw = pd.concat(raw_dfs)
    df_score = pd.concat(ef_dfs)
    df_raw.to_csv(out_dir / (base_name + "_raw.csv"))
    df_score.to_csv(out_dir / (base_name + "_ef.csv"))
    return np.mean(df_score["score"].values)


def get_results_dfs(
    all_perturbed_pockets_path="figs/perturbations/perturbed",
    out_df="figs/perturbations/perturbed/aggregated.csv",
    recompute=True,
    fractions=None,
    compute_overlap=False,
    metric="ef",
    ef_frac=0.02,
):
    list_of_results = []
    todo = list(sorted([x for x in os.listdir(all_perturbed_pockets_path) if not x.endswith(".csv")]))

    if fractions is not None:
        fractions = set(fractions)
        todo = [x for x in todo if float(x.split("_")[1]) in fractions]
    for i, perturbed_pocket_dir in enumerate(todo):
        _, fraction, replicate = perturbed_pocket_dir.split("_")

        perturbed_pocket_path = os.path.join(all_perturbed_pockets_path, perturbed_pocket_dir)

        # Only recompute if the csv ain't here or can't be parsed correclty
        out_dir = Path(perturbed_pocket_path).parent
        base_name = Path(perturbed_pocket_path).name
        if ROBIN:
            out_csv_path = out_dir / (base_name + f"{'_ef' if metric == 'ef' else ''}.csv")
        else:
            # out_csv_path = out_dir / (base_name + "_dock.csv")
            # out_csv_path = out_dir / (base_name + "_native_sample.csv")
            # out_csv_path = out_dir / (base_name + "_mixed.csv")
            out_csv_path = out_dir / (
                base_name + f"_mixed{'_chembl' if DECOYS == 'chembl' else ''}{'_ef' if metric == 'ef' else ''}.csv"
            )
        if recompute or not os.path.exists(out_csv_path):
            if ROBIN:
                _ = get_perf_robin(pocket_path=perturbed_pocket_path)
            else:
                _ = get_perf(pocket_path=perturbed_pocket_path)
        if not metric == "ef":
            df = pd.read_csv(out_csv_path)[["pocket_id", "score"]]
        else:
            df = pd.read_csv(out_csv_path)[["pocket_id", "score", "frac"]]
            df = df.loc[df["frac"] == ef_frac]

        if compute_overlap:
            overlap_csv_path = out_dir / (base_name + "_overlap.csv")
            if not os.path.exists(overlap_csv_path):
                compute_overlaps(
                    original_pockets=ALL_POCKETS_GRAPHS,
                    modified_pockets_path=perturbed_pocket_path,
                    dump_path=overlap_csv_path,
                )
            overlap_df = pd.read_csv(overlap_csv_path)
            perturb_df = df.merge(overlap_df, on=["pocket_id"], how="left")
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


def get_all_perturbations_factory(
    fractions=(0.7, 0.85, 1.0, 1.15, 1.3),
    max_replicates=10,
    mode="hard",
    final_bfs=(4,),
    perturb_bfs=(1,),
    recompute=True,
    use_cached_pockets=True,
    compute_overlap=False,
    metric="mar",
    ef_frac=0.02,
):
    dfs = []
    for final_bf in final_bfs:
        for perturb_bf in perturb_bfs:
            setup_name = f'{mode}{"_robin" if ROBIN else ""}_p{perturb_bf}_f{final_bf}'
            out_path = f"figs/perturbations/perturbed_{setup_name}"
            os.makedirs(out_path, exist_ok=True)
            out_path = f"figs/perturbations/perturbed_{setup_name}"
            out_df = f"figs/perturbations/aggregated_{setup_name}.csv"
            if not use_cached_pockets:
                get_perturbed_pockets(
                    out_path=out_path,
                    perturb_bfs_depth=perturb_bf,
                    final_bfs=final_bf,
                    perturbation=mode,
                    fractions=fractions,
                    max_replicates=max_replicates,
                    recompute=recompute,
                    all_pockets=ALL_POCKETS,
                )
            df = get_results_dfs(
                all_perturbed_pockets_path=out_path,
                out_df=out_df,
                fractions=fractions,
                recompute=recompute,
                compute_overlap=compute_overlap,
                metric=metric,
                ef_frac=ef_frac,
            )
            dfs.append(df)
    return dfs


def instantiate_functions(fractions, metric):
    smaller_factory = partial(get_all_perturbations_factory, fractions=fractions, metric=metric)
    get_random = partial(smaller_factory, final_bfs=(4,), perturb_bfs=(1, 2, 3, 4), mode="random")
    get_hard = partial(smaller_factory, final_bfs=(4,), perturb_bfs=(1, 2, 3, 4), mode="hard")
    get_soft = partial(smaller_factory, final_bfs=(1, 2, 3, 4), perturb_bfs=(2,), mode="soft")
    get_rognan_like = partial(smaller_factory, perturb_bfs=(2,), mode="rognan_like")
    get_rognan_true = partial(smaller_factory, mode="rognan_true")
    return get_random, get_hard, get_soft, get_rognan_like, get_rognan_true


def main_chembl():
    global TEST_SYSTEMS
    global ALL_POCKETS
    global ALL_POCKETS_GRAPHS
    global DF_UNPERTURBED
    global ROBIN
    global DECOYS
    ROBIN = False
    metric = "auroc"
    # metric = 'ef'
    # DECOYS = 'pdb'
    DECOYS = "chembl"
    # DECOYS = "pdb_chembl"
    TEST_SYSTEMS = get_systems(
        target="is_native",
        rnamigos1_split=-2,
        use_rnamigos1_train=False,
        use_rnamigos1_ligands=False,
        return_test=True,
    )
    ALL_POCKETS = set(TEST_SYSTEMS["PDB_ID_POCKET"].unique())
    ALL_POCKETS_GRAPHS = {
        pocket_id: graph_io.load_json(os.path.join("data/json_pockets_expanded", f"{pocket_id}.json"))
        for pocket_id in ALL_POCKETS
    }
    # # Check that inference works, we should get 0.9848
    os.makedirs("figs/perturbations/unperturbed", exist_ok=True)
    # unpert_df = f"figs/perturbations/unperturbed/json_pockets_expanded_mixed{'_ef' if metric == 'ef' else ''}.csv"
    unpert_df = f"figs/perturbations/unperturbed/json_pockets_expanded_mixed{'_chembl' if DECOYS == 'chembl' else ''}{'_ef' if metric == 'ef' else ''}.csv"
    if not os.path.exists(unpert_df):
        get_perf(pocket_path="data/json_pockets_expanded", out_dir="figs/perturbations/unperturbed")
    DF_UNPERTURBED = pd.read_csv(unpert_df, index_col=False)
    DF_UNPERTURBED.rename(columns={"score": "unpert_score"}, inplace=True)

    global GOOD_POCKETS
    good_cutoff = 0.98 if DECOYS == "chembl" else 0.75
    GOOD_POCKETS = DF_UNPERTURBED[DF_UNPERTURBED["unpert_score"] >= good_cutoff]["pocket_id"].unique()

    # fractions = (0.1, 0.7, 0.85, 1.0, 1.15, 1.3, 5)
    fractions = (0.5, 0.6, 0.7, 0.85, 1.0, 1.15, 1.3)

    # Check pocket computation works
    # get_perturbed_pockets(unperturbed_path='data/json_pockets_expanded',
    #                      out_path='figs/perturbations/perturbed_robin',
    #                      fractions=(0.9, 1.0),
    #                      perturb_bfs_depth=1,
    #                      max_replicates=2,
    #                      all_pockets=ALL_POCKETS)
    # # Get a first result
    # df = get_efs(all_perturbed_pockets_path='figs/perturbations/perturbed_robin',
    #            out_df='figs/perturbations/perturbed_robin/aggregated_test.csv',
    #            compute_overlap=True,
    #            metric=metric)

    use_cached_pockets = True
    recompute = False

    get_random, get_hard, get_soft, get_rognan_like, get_rognan_true = instantiate_functions(
        fractions=fractions, metric=metric
    )
    # Now compute perturbed scores using the random BFS approach
    dfs_random = get_random(recompute=recompute, use_cached_pockets=use_cached_pockets)

    # Hard: sample on the border
    dfs_hard = get_hard(recompute=recompute, use_cached_pockets=use_cached_pockets)

    # Get dfs soft
    # dfs_soft = get_soft(recompute=recompute, use_cached_pockets=use_cached_pockets)

    # Rognan like and true
    # dfs_rognan_like = get_rognan_like(recompute=recompute, use_cached_pockets=use_cached_pockets)
    # dfs_rognan_true = get_rognan_true(recompute=recompute, use_cached_pockets=use_cached_pockets)

    # PLOT
    # plot_one(df_soft_1, plot_delta=False, filter_good=False, fractions=fractions, color='purple',metric=metric,
    #          label='soft 1')  # Plot soft perturbed
    # colors = sns.dark_palette("#69d", n_colors=5, reverse=True)
    # colors_2 = sns.light_palette("firebrick", n_colors=5, reverse=True)

    # colors = sns.light_palette("royalblue", n_colors=5, reverse=True)
    colors_2 = sns.light_palette("seagreen", n_colors=5, reverse=True)
    plot_list_partial = partial(
        plot_list,
        metric=metric,
        fractions=fractions,
        df_ref=DF_UNPERTURBED,
        plot_delta=False,
        filter_good=False,
        good_pockets=GOOD_POCKETS,
    )
    # plot_list_partial_color = partial(plot_list_partial, colors=colors)
    end_plot_partial = partial(end_plot, fractions=fractions)

    # Actually plot
    plot_list_partial(dfs=dfs_random, title="Noised pockets", colors=colors_2)
    fig_name = f"figs/perturbs_random{'_chembl' if DECOYS == 'chembl' else ''}.pdf"
    end_plot_partial(colors=colors_2, fig_name=fig_name)
    plot_list_partial(dfs=dfs_hard, title="Shifted pockets", colors=colors_2)
    fig_name = f"figs/perturbs_hard{'_chembl' if DECOYS == 'chembl' else ''}.pdf"
    end_plot_partial(colors=colors_2, fig_name=fig_name)
    # plot_list_partial(dfs_rognan_like, colors=["grey"], label="Rognan like")
    # plot_list_partial(dfs_rognan_true, colors=["black"], label="Rognan true")
    # plot_list_partial_color(dfs=dfs_soft, label="Soft strategy")
    # end_plot_partial()

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
    global DECOYS
    DECOYS = "robin"

    ROBIN_POCKETS = {
        "TPP": "2GDI_Y_TPP_100",
        "ZTP": "5BTP_A_AMZ_106",
        "SAM_ll": "2QWY_B_SAM_300",
        "PreQ1": "3FU2_A_PRF_101",
    }
    TEST_SYSTEMS = pd.DataFrame({"PDB_ID_POCKET": list(ROBIN_POCKETS.values())})
    ALL_POCKETS = set(ROBIN_POCKETS.values())
    ROBIN = True
    ALL_POCKETS_GRAPHS = {
        pocket_id: graph_io.load_json(os.path.join("data/json_pockets_expanded", f"{pocket_id}.json"))
        for pocket_id in ALL_POCKETS
    }
    os.makedirs("figs/perturbations/unperturbed_robin", exist_ok=True)
    get_perf_robin(pocket_path="data/json_pockets_expanded", out_dir="figs/perturbations/unperturbed_robin")
    DF_UNPERTURBED = pd.read_csv(
        "figs/perturbations/unperturbed_robin/json_pockets_expanded_mixed.csv", index_col=False
    )

    # fractions = (0.1, 0.7, 0.85, 1.0, 1.15, 1.3, 5)
    fractions = (0.7, 0.85, 1.0, 1.15, 1.3)
    colors = sns.light_palette("royalblue", n_colors=4, reverse=True)
    ef_frac = 0.02
    use_cached_pockets = False
    recompute = False
    metric = "ef"
    df_soft_1 = get_all_perturbations_factory(
        fractions=fractions, metric=metric, final_bfs=(1,), perturb_bfs=(2,), mode="soft"
    )
    plot_one(
        df_soft_1,
        plot_delta=False,
        filter_good=False,
        fractions=fractions,
        color="purple",
        label="bfs 1",
    )  # Plot soft perturbed
    end_plot()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    LOADER_ARGS = {
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 4,
        "collate_fn": lambda x: x[0],
    }
    TEST_SYSTEMS, ALL_POCKETS, ALL_POCKETS_GRAPHS, DF_UNPERTURBED, ROBIN, DECOYS = [
        None,
    ] * 6

    ALL_POCKETS = [Path(f).stem for f in os.listdir("data/json_pockets_expanded")]
    ALL_POCKETS_GRAPHS = {
        pocket_id: graph_io.load_json(os.path.join("data/json_pockets_expanded", f"{pocket_id}.json"))
        for pocket_id in ALL_POCKETS
    }

    # get_perturbed_pockets(perturbation="rognan true", all_pockets=ALL_POCKETS)

    main_chembl()
    # main_robin()
