import os
import sys

from loguru import logger
import numpy as np
import pandas as pd
import pathlib
from sklearn import metrics

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rnamigos.learning.dataset import get_systems_from_cfg
from rnamigos.learning.dataloader import get_vs_loader
from rnamigos.learning.models import get_model_from_dirpath
from rnamigos.utils.mixing_utils import mix_two_scores, mix_two_dfs, get_mix_score, unmix, mix_all
from rnamigos.utils.virtual_screen import get_results_dfs, raw_df_to_mean_auroc
from scripts_fig.plot_utils import group_df


def pdb_eval(cfg, model, dump=True, verbose=True, decoys=None, rognan=False, reps_only=False):
    # Final VS validation on each decoy set
    if verbose:
        logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
        logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")

    test_systems = get_systems_from_cfg(cfg, return_test=True)
    model = model.to("cpu")
    rows_aurocs, rows_raws = [], []
    if decoys is None:
        decoys = ["chembl", "pdb", "pdb_chembl", "decoy_finder"]
    elif isinstance(decoys, str):
        decoys = [decoys]
    for decoy_mode in decoys:
        dataloader = get_vs_loader(
            systems=test_systems,
            decoy_mode=decoy_mode,
            cfg=cfg,
            cache_graphs=False,
            reps_only=reps_only,
            verbose=verbose,
            rognan=rognan,
        )
        decoy_df_aurocs, decoys_dfs_raws = get_results_dfs(
            model=model, dataloader=dataloader, decoy_mode=decoy_mode, cfg=cfg, verbose=verbose
        )
        rows_aurocs.append(decoy_df_aurocs)
        rows_raws.append(decoys_dfs_raws)

    # Make it a df
    df_aurocs = pd.concat(rows_aurocs)
    df_raw = pd.concat(rows_raws)
    if dump:
        d = pathlib.Path(cfg.result_dir, parents=True, exist_ok=True)
        base_name = pathlib.Path(cfg.name).stem
        out_csv = d / (base_name + ".csv")
        out_csv_raw = d / (base_name + "_raw.csv")
        df_aurocs.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)

        # Just printing the results
        df_chembl = df_aurocs.loc[df_aurocs["decoys"] == "chembl"]
        df_pdbchembl = df_aurocs.loc[df_aurocs["decoys"] == "pdb_chembl"]
        df_chembl_grouped = group_df(df_chembl)
        df_pdbchembl_grouped = group_df(df_pdbchembl)
        logger.info(f"{cfg.name} Mean AuROC on chembl: {np.mean(df_chembl['score'].values)}")
        logger.info(f"{cfg.name} Mean grouped AuROC on chembl: {np.mean(df_chembl_grouped['score'].values)}")
        logger.info(f"{cfg.name} Mean AuROC on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")
        logger.info(f"{cfg.name} Mean grouped AuROC on pdbchembl: {np.mean(df_pdbchembl_grouped['score'].values)}")
    return df_aurocs, df_raw


def get_perf_model(models, res_dir, decoy_modes=("pdb", "chembl", "pdb_chembl"), reps_only=True, recompute=False):
    """
    This is quite similar to below, but additionally computes rognan.
     Also, only does it on just one decoy, and only on representatives.
    Could be merged
    """
    model_dir = "results/trained_models/"
    os.makedirs(res_dir, exist_ok=True)
    for model_name, model_path in models.items():
        decoys_df_aurocs, decoys_df_raws = list(), list()
        out_csv = os.path.join(res_dir, f"{model_name}.csv")
        out_csv_raw = os.path.join(res_dir, f"{model_name}_raw.csv")

        decoys_df_aurocs_rognan, decoys_df_raws_rognan = list(), list()
        out_csv_rognan = os.path.join(res_dir, f"{model_name}_rognan.csv")
        out_csv_raw_rognan = os.path.join(res_dir, f"{model_name}_rognan_raw.csv")
        if recompute or not os.path.exists(out_csv):
            for decoy_mode in decoy_modes:
                # get model
                full_model_path = os.path.join(model_dir, model_path)
                model, cfg = get_model_from_dirpath(full_model_path, return_cfg=True)
                # get normal results
                df_aurocs, df_raw = pdb_eval(
                    cfg, model, verbose=False, dump=False, decoys=decoy_mode, reps_only=reps_only
                )
                decoys_df_aurocs.append(df_aurocs)
                decoys_df_raws.append(df_raw)

                # get rognan results
                df_aurocs_rognan, df_raw_rognan = pdb_eval(
                    cfg, model, verbose=False, dump=False, decoys=decoy_mode, rognan=True, reps_only=reps_only
                )
                decoys_df_aurocs_rognan.append(df_aurocs_rognan)
                decoys_df_raws_rognan.append(df_raw_rognan)

            all_df_aurocs = pd.concat(decoys_df_aurocs)
            all_df_raws = pd.concat(decoys_df_raws)
            all_df_aurocs.to_csv(out_csv, index=False)
            all_df_raws.to_csv(out_csv_raw, index=False)

            all_df_aurocs_rognan = pd.concat(decoys_df_aurocs_rognan)
            all_df_raws_rognan = pd.concat(decoys_df_raws_rognan)
            all_df_aurocs_rognan.to_csv(out_csv_rognan, index=False)
            all_df_raws_rognan.to_csv(out_csv_raw_rognan, index=False)
        else:
            df_aurocs = pd.read_csv(out_csv)
            df_aurocs_rognan = pd.read_csv(out_csv_rognan)

        # Just printing the results
        # We need this special case for rdock
        decoy = None
        if "decoys" in df_aurocs.columns:
            decoy = decoy_modes[-1]
            df_aurocs = df_aurocs.loc[df_aurocs["decoys"] == decoy]
            df_aurocs_rognan = df_aurocs_rognan.loc[df_aurocs_rognan["decoys"] == decoy]
        test_auroc = np.mean(df_aurocs["score"].values)
        test_auroc_rognan = np.mean(df_aurocs_rognan["score"].values)
        gap_score = 2 * test_auroc - test_auroc_rognan
        print(f"{model_name}, {decoy}: AuROC {test_auroc:.3f} Rognan {test_auroc_rognan:.3f} GapScore {gap_score:.3f}")


def mix_all_chembl(pairs, res_dir, recompute=False, use_max=True):
    new_pairs = {}
    for (model1, model2), outname in pairs.items():
        new_pairs[(f"{model1}_rognan", f"{model2}_rognan")] = f"{outname}_rognan"
    new_pairs.update(pairs)
    mix_all(pairs=new_pairs, res_dir=res_dir, recompute=recompute, use_max=use_max)
    for outname in pairs.values():
        raw_result = pd.read_csv(os.path.join(res_dir, f"{outname}_raw.csv"))
        raw_result_rognan = pd.read_csv(os.path.join(res_dir, f"{outname}_rognan_raw.csv"))
        test_auroc = raw_df_to_mean_auroc(raw_result)
        test_auroc_rognan = raw_df_to_mean_auroc(raw_result_rognan)
        gap_score = 2 * test_auroc - test_auroc_rognan
        print(f"{outname}: AuROC {test_auroc:.3f} Rognan {test_auroc_rognan:.3f} GapScore {gap_score:.3f}")


def compute_mix_csvs(recompute=False):
    def merge_csvs(to_mix):
        """
        Aggregate rdock, native and dock results add mixing strategies
        """
        decoy_modes = ("pdb", "pdb_chembl", "chembl")
        all_big_raws = []
        for decoy in decoy_modes:
            raw_dfs = [pd.read_csv(f"outputs/pockets/{r}_raw.csv") for r in to_mix]
            raw_dfs = [df.loc[df["decoys"] == decoy] for df in raw_dfs]
            raw_dfs = [df[["pocket_id", "smiles", "is_active", "raw_score"]] for df in raw_dfs]
            raw_dfs = [group_df(df) for df in raw_dfs]

            for df in raw_dfs:
                df["smiles"] = df["smiles"].str.strip()

            raw_dfs[0]["rdock"] = raw_dfs[0]["raw_score"].values
            raw_dfs[1]["dock"] = raw_dfs[1]["raw_score"].values
            raw_dfs[2]["native"] = raw_dfs[2]["raw_score"].values
            raw_dfs = [df.drop("raw_score", axis=1) for df in raw_dfs]

            big_df_raw = raw_dfs[1]
            big_df_raw = big_df_raw.merge(raw_dfs[2], on=["pocket_id", "smiles", "is_active"], how="inner")
            big_df_raw = big_df_raw.merge(raw_dfs[0], on=["pocket_id", "smiles", "is_active"], how="inner")
            big_df_raw = big_df_raw[["pocket_id", "smiles", "is_active", "rdock", "dock", "native"]]

            def smaller_merge(df, score1, score2, outname):
                return mix_two_scores(df,
                                      score1=score1,
                                      score2=score2,
                                      outname_col=outname,
                                      use_max=True,
                                      add_decoy=False)[2]

            raw_df_docknat = smaller_merge(big_df_raw, "dock", "native", "docknat")
            big_df_raw = big_df_raw.merge(raw_df_docknat, on=["pocket_id", "smiles", "is_active"], how="outer")

            raw_df_rdocknat = smaller_merge(big_df_raw, "rdock", "native", "rdocknat")
            big_df_raw = big_df_raw.merge(raw_df_rdocknat, on=["pocket_id", "smiles", "is_active"], how="outer")

            raw_df_combined = smaller_merge(big_df_raw, "docknat", "rdock", "combined")
            big_df_raw = big_df_raw.merge(raw_df_combined, on=["pocket_id", "smiles", "is_active"], how="outer")

            raw_df_rdockdock = smaller_merge(big_df_raw,"dock","rdock","rdockdock")
            big_df_raw = big_df_raw.merge(raw_df_rdockdock, on=["pocket_id", "smiles", "is_active"], how="outer")

            dumb_decoy = [decoy for _ in range(len(big_df_raw))]
            big_df_raw.insert(len(big_df_raw.columns), "decoys", dumb_decoy)
            all_big_raws.append(big_df_raw)
        big_df_raw = pd.concat(all_big_raws)
        return big_df_raw

    for seed in SEEDS:
        for rognan in [True, False]:
            out_path_raw = f'outputs/pockets/big_df_{seed}{"_rognan" if rognan else ""}_raw.csv'
            if not os.path.exists(out_path_raw) or recompute:
                # Combine the learnt methods and dump results
                TO_MIX = [f'rdock{"_rognan" if rognan else ""}',
                          f'dock_{seed}{"_rognan" if rognan else ""}',
                          f'native_{seed}{"_rognan" if rognan else ""}']
                big_df_raw = merge_csvs(to_mix=TO_MIX)
                big_df_raw.to_csv(out_path_raw)

                # Dump aurocs dataframes for newly combined methods
                for method in ["docknat", "rdocknat", "combined"]:
                    outpath = f"outputs/pockets/{method}_{seed}{'_rognan' if rognan else ''}.csv"
                    unmix(big_df_raw, score=method, outpath=outpath)


def compute_all_self_mix():
    for i in range(len(SEEDS)):
        to_compare = i, (i + 1) % len(SEEDS)
        out_path_raw_1 = f'outputs/pockets/big_df_{SEEDS[to_compare[0]]}_raw.csv'
        big_df_raw_1 = pd.read_csv(out_path_raw_1)
        big_df_raw_1 = big_df_raw_1[big_df_raw_1["decoys"] == DECOY]

        out_path_raw_2 = f'outputs/pockets/big_df_{SEEDS[to_compare[1]]}_raw.csv'
        big_df_raw_2 = pd.read_csv(out_path_raw_2)
        big_df_raw_2 = big_df_raw_2[big_df_raw_2["decoys"] == DECOY]
        for score in ["native", "dock"]:
            all_aurocs, _, _ = mix_two_dfs(big_df_raw_1, big_df_raw_2, score)
            print(f"Self-mixing with {score}:", np.mean(all_aurocs))


def get_one_mixing_table(raw_df):
    all_methods = ["native", "dock", "rdock"]
    all_res = {}
    # Do singletons
    for method in all_methods:
        result = raw_df_to_mean_auroc(raw_df, score=method)
        all_res[method] = result

    mean_aurocs = get_mix_score(raw_df, score1="dock", score2="rdock")
    all_res["dock/rdock"] = mean_aurocs

    all_methods_2 = ["docknat", "rdocknat", "combined"]
    # Do singletons but dump them as a single csv since they did not exist
    for method in all_methods_2:
        result = raw_df_to_mean_auroc(raw_df, score=method)
        all_res[method] = result

    for k, v in all_res.items():
        print(f"{k:<20}:{v:.4f}")


def get_table_mixing():
    for seed in SEEDS:
        out_path_raw = f'outputs/pockets/big_df_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)
        big_df_raw = big_df_raw[big_df_raw["decoys"] == DECOY]
        get_one_mixing_table(big_df_raw)


if __name__ == "__main__":
    # DECOY = "pdb_chembl"
    DECOY = 'chembl'
    GROUPED = True
    SEEDS = [42]
    # SEEDS = [0, 1, 42]

    MODELS = {
        "dock_42": "dock/dock_42",
        "native_42": "is_native/native_42",
        # "native_nornafm": "is_native/native_bce0.02_groupsample_nornafm",
        # "dock_0": "dock/dock_0",
        # "native_0": "is_native/native_0",
        # "dock_1": "dock/dock_1",
        # "native_1": "is_native/native_1",
    }
    RUNS = list(MODELS.keys())

    PAIRS = {
        # ("rdock", "dock_42"): "dock_rdock",
        ("native_42", "dock_42"): "docknat_42",
        # ("native_nornafm", "dock_42"): "docknat_nornafm_42",
        ("rnamigos_42", "rdock"): "combined_42",
        # ("native_42", "dock_42"): "docknat_42",
    }

    # Just print perfs compared to Rognan, make inference on just one decoy
    # res_dir = "outputs/pockets_quick_chembl" if DECOY == 'chembl' else "outputs/pockets_quick"
    # get_perf_model(models={'rdock': 'rdock'}, res_dir=res_dir, decoy_modes=(DECOY,), reps_only=GROUPED, recompute=False)
    # get_perf_model(models=MODELS, res_dir=res_dir, decoy_modes=(DECOY,), reps_only=GROUPED, recompute=True)
    # mix_all_chembl(pairs=PAIRS, res_dir=res_dir, recompute=False)

    # GET INFERENCE CSVS
    decoys = ("pdb", "chembl", "pdb_chembl") if DECOY != 'chembl' else ("pdb", "pdb_chembl", "chembl")
    get_perf_model(models=MODELS, res_dir="outputs/pockets",
                   decoy_modes=decoys,
                   reps_only=GROUPED,
                   recompute=False)

    # PARSE INFERENCE CSVS AND MIX THEM
    compute_mix_csvs(recompute=True)

    # To compare to ensembling the same method with different seeds
    # compute_all_self_mix()

    # Get table with all mixing
    # get_table_mixing()
