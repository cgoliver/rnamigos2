import os
import sys

from pathlib import Path

from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
import pandas as pd
import pathlib
import torch

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.utils.virtual_screen import raw_df_to_efs, raw_df_to_aurocs
from rnamigos.utils.graph_utils import load_rna_graph
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.inference import inference_raw, get_models
from rnamigos.utils.mixing_utils import mix_all, mix_two_scores

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)

ROBIN_POCKETS = {
    "TPP": "2GDI_Y_TPP_100",
    "ZTP": "5BTP_A_AMZ_106",
    "SAM_ll": "2QWY_B_SAM_300",
    "PreQ1": "3FU2_A_PRF_101",
}

POCKET_PATH = "data/json_pockets_expanded"


def robin_inference_raw(
    ligand_name,
    dgl_pocket_graph,
    models=None,
    out_path=None,
    ligand_cache=None,
    use_ligand_cache=False,
    debug=False,
):
    """
    Given the graph pocket and ligand name, as well as models as expected by the inference script,
    output the raw df
    """
    actives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "actives.txt")
    actives_smiles_set = set([s.lstrip().rstrip() for s in list(open(actives_ligands_path).readlines())])

    inactives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "decoys.txt")
    inactives_smiles_set = set([s.lstrip().rstrip() for s in list(open(inactives_ligands_path).readlines())])
    inactives_smiles_set = inactives_smiles_set.difference(actives_smiles_set)

    actives_smiles_list = list(actives_smiles_set)
    inactives_smiles_list = list(inactives_smiles_set)
    smiles_list = actives_smiles_list + inactives_smiles_list
    is_active = [1 for _ in range(len(actives_smiles_list))] + [0 for _ in range(len(inactives_smiles_list))]
    if debug:
        smiles_list = smiles_list[:100]
        is_active = is_active[:100]

    models = get_models(model=models)
    final_df = inference_raw(
        models=models,
        dgl_graph=dgl_pocket_graph,
        smiles_list=smiles_list,
        ligand_cache=ligand_cache,
        use_ligand_cache=use_ligand_cache,
    )
    final_df["is_active"] = is_active
    final_df.to_csv(out_path)
    return final_df


def one_robin(ligand_name, pocket_id, models=None, use_rna_fm=False):
    dgl_pocket_graph, _ = load_rna_graph(POCKET_PATH / Path(pocket_id).with_suffix(".json"), use_rnafm=use_rna_fm)
    raw_df = robin_inference_raw(
        ligand_name=ligand_name,
        dgl_pocket_graph=dgl_pocket_graph,
        models=models,
        use_ligand_cache=True,
        ligand_cache="data/ligands/robin_lig_graphs.p",
        debug=False,
    )
    raw_df["pocket_id"] = pocket_id
    return pd.DataFrame(raw_df)


def get_swapped_pocketlig(swap=0):
    robin_ligs = list(ROBIN_POCKETS.keys())
    robin_pockets = list(ROBIN_POCKETS.values())

    # Associate the ligands with the wrong pockets.
    robin_lig_pocket_dict = {lig: robin_pockets[(i + swap) % len(robin_ligs)] for i, lig in enumerate(robin_ligs)}
    old_to_new = {robin_pockets[i]: robin_pockets[(i + swap) % len(robin_ligs)] for i in range(len(robin_ligs))}
    new_to_old = {v: k for k, v in old_to_new.items()}
    return robin_lig_pocket_dict, new_to_old


def get_all_preds(model, use_rna_fm, swap=0):
    robin_lig_pocket_dict, new_to_old = get_swapped_pocketlig(swap=swap)
    robin_raw_dfs = [
        df
        for df in Parallel(n_jobs=4)(
            delayed(one_robin)(ligand_name, pocket_id, model, use_rna_fm)
            for ligand_name, pocket_id in robin_lig_pocket_dict.items()
        )
    ]
    robin_raw_df = pd.concat(robin_raw_dfs)

    # The naming is based on the pocket. So to keep ligands swap consistent, we need to change that
    robin_raw_df["pocket_id"] = robin_raw_df["pocket_id"].map(new_to_old)

    # Finally, compute EFs
    robin_ef_df = raw_df_to_efs(robin_raw_df, fracs=(0.01, 0.02, 0.05, 0.1, 0.2))
    return robin_ef_df, robin_raw_df


def timing():
    """
    Compute timing when cached ligands are available
    :return:
    """
    robin_lig_pocket_dict, new_to_old = get_swapped_pocketlig()
    model_dir = "results/trained_models/"
    models = {"dock_42": "dock/dock_42", "native_42": "is_native/native_42"}
    models = {name: os.path.join(model_dir, model_path) for name, model_path in models.items()}
    models = {name: get_model_from_dirpath(model_path) for name, model_path in models.items()}
    import time

    t0 = time.time()
    for ligand_name, pocket_id in robin_lig_pocket_dict.items():
        raw_df = one_robin(ligand_name, pocket_id, models, use_rna_fm=True)
        mixed_df = mix_two_scores(raw_df, score1="dock_42", score2="native_42")
    tf = time.time()
    print("Total_time to score:", tf - t0)


def robin_eval(cfg, model):
    robin_ef_df, robin_raw_df = get_all_preds(model, use_rna_fm=cfg.model.use_rnafm)
    d = pathlib.Path(cfg.result_dir, parents=True, exist_ok=True)
    base_name = pathlib.Path(cfg.name).stem
    robin_ef_df.to_csv(d / (base_name + "_robin.csv"))
    robin_raw_df.to_csv(d / (base_name + "_robin_raw.csv"))
    ef_frac05 = robin_ef_df.loc[robin_ef_df["frac"] == 0.05]
    ef05 = np.mean(ef_frac05["score"].values)
    print(f"ROBIN:", ef05)
    return ef05


def get_all_csvs(recompute=False, swap=0):
    model_dir = "results/trained_models/"
    res_dir = "outputs/robin/" if swap == 0 else f"outputs/robin_swap_{swap}"
    os.makedirs(res_dir, exist_ok=True)
    for model, model_path in MODELS.items():
        out_csv = os.path.join(res_dir, f"{model}.csv")
        out_csv_raw = os.path.join(res_dir, f"{model}_raw.csv")
        if os.path.exists(out_csv) and not recompute:
            continue
        full_model_path = os.path.join(model_dir, model_path)
        model, cfg = get_model_from_dirpath(full_model_path, return_cfg=True)
        use_rnafm = cfg.model.use_rnafm if "use_rnafm" in cfg.model else False
        df_ef, df_raw = get_all_preds(model, use_rna_fm=use_rnafm, swap=swap)
        print(f"dumping {out_csv}")
        df_ef.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)


def assess_overlap_robin():
    res_dir = "outputs/robin"
    docking_df = pd.read_csv(os.path.join(res_dir, "dock_42_raw.csv"))
    smiles_dict = {}
    pocket_ids = list(ROBIN_POCKETS.values())
    for pocket_id in pocket_ids:
        docking_df_lig = docking_df[docking_df["pocket_id"] == pocket_id]
        smiles = docking_df_lig[["smiles"]].values.flatten().tolist()
        smiles_dict[pocket_id] = list(sorted(smiles))
    smiles_0 = smiles_dict.pop(pocket_ids[0])
    smiles_1 = smiles_dict.pop(pocket_ids[1])
    overlap = set(smiles_0).intersection(set(smiles_1))
    return overlap


def assess_overlap_docking():
    docking_df = pd.read_csv("data/robin_docking_consolidated_v2.csv")
    smiles_dict = {}
    for ligand_name, pocket_id in ROBIN_POCKETS.items():
        docking_df_lig = docking_df[docking_df["TARGET"] == ligand_name]
        smiles = docking_df_lig[["SMILE"]].values.flatten().tolist()
        smiles_dict[ligand_name] = list(sorted(smiles))
    robin_sys = list(ROBIN_POCKETS.keys())
    smiles_0 = smiles_dict.pop(robin_sys[0])
    smiles_1 = smiles_dict.pop(robin_sys[1])
    overlap = set(smiles_0).intersection(set(smiles_1))
    return overlap


def get_dfs_docking(swap=0):
    """
    Go from columns:
    TARGET,_TITLE1,SMILE,TOTAL,INTER,INTRA,RESTR,VDW,TYPE

    To columns:
    raw: score,smiles,is_active,pocket_id
    efs: pocket_id,score,frac
    """

    res_dir = "outputs/robin" if swap == 0 else f"outputs/robin_swap_{swap}"
    ref_raw_df = pd.read_csv(os.path.join(res_dir, "dock_42_raw.csv"))
    docking_df = pd.read_csv("data/robin_docking_consolidated_v2.csv")
    # For each pocket, get relevant mapping smiles : normalized score,
    # then use it to create the appropriate raw, and clean csvs
    all_raws, all_dfs = [], []
    robin_lig_pocket_dict, new_to_old = get_swapped_pocketlig(swap=swap)
    for ligand_name, pocket_id in robin_lig_pocket_dict.items():
        # Get relevant subpart of the original docking data and normalize it
        docking_df_lig = docking_df[docking_df["TARGET"] == ligand_name]
        scores = -pd.to_numeric(docking_df_lig["INTER"], errors="coerce").values.squeeze()
        scores[scores < 0] = 0
        scores = np.nan_to_num(scores, nan=0)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Now get the docking specific mapping {smiles: score}
        mapping = {}
        for smiles, score in zip(docking_df_lig[["SMILE"]].values, normalized_scores):
            mapping[smiles[0]] = score
        mapping = defaultdict(int, mapping)
        ref_raw_df_lig = ref_raw_df[ref_raw_df["pocket_id"] == pocket_id].copy()
        docking_scores = [mapping[smile] for smile in ref_raw_df_lig["smiles"].values]
        ref_raw_df_lig["raw_score"] = docking_scores
        all_raws.append(ref_raw_df_lig)
    all_raws = pd.concat(all_raws)
    all_raws.to_csv(f"{res_dir}/rdock_raw.csv")
    all_efs = raw_df_to_efs(all_raws, fracs=(0.01, 0.02, 0.05, 0.1, 0.2))
    all_efs.to_csv(f"{res_dir}/rdock.csv")


def get_merged_df(swap=0, recompute=False):
    """
    Aggregate several scores in one big_df (like for pockets)
    This is useful for plotting scripts, such as time_ef.py
    """
    res_dir = "outputs/robin" if swap == 0 else f"outputs/robin_swap_{swap}"
    out_csv = os.path.join(res_dir, "big_df_raw.csv")
    if not recompute and os.path.exists(out_csv):
        return
    to_mix = [
        "rdock",
        "dock_42",
        "native_42",
        "rnamigos_42",
        "rdocknat_42",
        "combined_42",
    ]
    big_df = None
    for name in to_mix:
        df = pd.read_csv(os.path.join(res_dir, f"{name}_raw.csv"))
        df = df[["pocket_id", "smiles", "is_active", "raw_score"]]
        df = df.rename(columns={"raw_score": name})
        if big_df is None:
            big_df = df
        else:
            big_df = big_df.merge(df, on=["pocket_id", "smiles", "is_active"], how="inner")
    big_df.to_csv(out_csv)


def print_results(swap=0):
    res_dir = "outputs/robin" if swap == 0 else f"outputs/robin_swap_{swap}"
    to_print = ["rdock"] + list(MODELS.keys()) + list(PAIRS.values())
    for method in to_print:
        in_csv = os.path.join(res_dir, f"{method}_raw.csv")
        raw_df = pd.read_csv(in_csv)
        frac = 0.2
        # efs = raw_df_to_efs(raw_df, fracs=(frac,))
        # ef = np.mean(efs['score'].values)
        # print(f"EF@:{frac} {method:<30} \t {ef:>8.4f}")
        df_aurocs = raw_df_to_aurocs(raw_df)
        # print(df_aurocs['score'].values)
        auroc = np.mean(df_aurocs["score"].values)
        print(f"AUROC:\t{method:<16}\t{auroc:>10.4f}")


if __name__ == "__main__":
    MODELS = {
        "dock_42": "dock/dock_42",
        "native_42": "is_native/native_42",
    }
    PAIRS = {
        # ("rdock", "dock_42"): "dock_rdock",
        ("native_42", "dock_42"): "rnamigos_42",
        # ("native_nornafm", "dock_42"): "rnamigos_42_nornafm",
        ("rnamigos_42", "rdock"): "combined_42",
        # ("native_42", "rdock"): "rdocknat_42",
    }
    SWAP = 0
    # TEST ONE INFERENCE
    # pocket_id = "TPP"
    # lig_name = "2GDI_Y_TPP_100"
    # model_dir = "results/trained_models/"
    # model_path = "is_native/native_42"
    # full_model_path = os.path.join(model_dir, model_path)
    # model = get_model_from_dirpath(full_model_path)
    # one_robin(pocket_id, lig_name, model, use_rna_fm=False)
    # assess_overlap_robin()

    # timing()

    res_dir = "outputs/robin" if SWAP == 0 else f"outputs/robin_swap_{SWAP}"
    # GET ALL CSVs for the models and plot them
    get_all_csvs(recompute=False, swap=SWAP)
    get_dfs_docking(swap=SWAP)
    mix_all(res_dir=res_dir, pairs=PAIRS, recompute=True, use_max=True)
    get_merged_df(recompute=True, swap=SWAP)
    # print_results(swap=SWAP)

    # COMPUTE PERTURBED VERSIONS
    for swap in range(1, 4):
        pass
        # get_all_csvs(recompute=True, swap=swap)
        # mix_all(swap=swap)
