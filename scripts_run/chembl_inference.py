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
from rnamigos.utils.mixing_utils import mix_two_scores, mix_two_dfs, get_mix_score
from rnamigos.utils.virtual_screen import get_efs
from scripts_fig.plot_utils import group_df


def pdb_eval(cfg, model, dump=True, verbose=True, decoys=None):
    # Final VS validation on each decoy set
    if verbose:
        logger.info(f"Loading VS graphs from {cfg.data.pocket_graphs}")
        logger.info(f"Loading VS ligands from {cfg.data.ligand_db}")

    test_systems = get_systems_from_cfg(cfg, return_test=True)
    model = model.to("cpu")
    ef_rows, raw_rows = [], []
    if decoys is None:
        decoys = ["chembl", "pdb", "pdb_chembl", "decoy_finder"]
    elif isinstance(decoys, str):
        decoys = [decoys]
    for decoy_mode in decoys:
        dataloader = get_vs_loader(systems=test_systems, decoy_mode=decoy_mode, cfg=cfg, cache_graphs=False)
        decoy_ef_rows, decoys_raw_rows = get_efs(
            model=model,
            dataloader=dataloader,
            decoy_mode=decoy_mode,
            cfg=cfg,
            verbose=verbose,
        )
        ef_rows += decoy_ef_rows
        raw_rows += decoys_raw_rows

    # Make it a df
    df_ef = pd.DataFrame(ef_rows)
    df_raw = pd.DataFrame(raw_rows)
    if dump:
        d = pathlib.Path(cfg.result_dir, parents=True, exist_ok=True)
        base_name = pathlib.Path(cfg.name).stem
        out_csv = d / (base_name + ".csv")
        out_csv_raw = d / (base_name + "_raw.csv")
        df_ef.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)

        # Just printing the results
        df_chembl = df_ef.loc[df_ef["decoys"] == "chembl"]
        df_pdbchembl = df_ef.loc[df_ef["decoys"] == "pdb_chembl"]
        df_chembl_grouped = group_df(df_chembl)
        df_pdbchembl_grouped = group_df(df_pdbchembl)
        logger.info(f"{cfg.name} Mean EF on chembl: {np.mean(df_chembl['score'].values)}")
        logger.info(f"{cfg.name} Mean grouped EF on chembl: {np.mean(df_chembl_grouped['score'].values)}")
        logger.info(f"{cfg.name} Mean EF on pdbchembl: {np.mean(df_pdbchembl['score'].values)}")
        logger.info(f"{cfg.name} Mean grouped EF on pdbchembl: {np.mean(df_pdbchembl_grouped['score'].values)}")
    return df_ef, df_raw


def get_all_csvs(recompute=False, decoys=None):
    model_dir = "results/trained_models/"
    res_dir = "outputs/pockets"
    os.makedirs(res_dir, exist_ok=True)
    for model, model_path in MODELS.items():
        out_csv = os.path.join(res_dir, f"{model}.csv")
        out_csv_raw = os.path.join(res_dir, f"{model}_raw.csv")
        if os.path.exists(out_csv) and not recompute:
            continue
        full_model_path = os.path.join(model_dir, model_path)
        model, cfg = get_model_from_dirpath(full_model_path, return_cfg=True)
        df_ef, df_raw = pdb_eval(cfg, model, dump=False, verbose=True, decoys=decoys)
        df_ef.to_csv(out_csv, index=False)
        df_raw.to_csv(out_csv_raw, index=False)


def compute_mix_csvs():
    def merge_csvs(to_mix, grouped=True, decoy=DECOY):
        """
        Aggregate rdock, native and dock results for a given decoy + add mixing strategies
        """
        raw_dfs = [pd.read_csv(f"outputs/{r}_raw.csv") for r in to_mix]
        raw_dfs = [df.loc[df['decoys'] == decoy] for df in raw_dfs]
        raw_dfs = [df[['pocket_id', 'smiles', 'is_active', 'raw_score']] for df in raw_dfs]
        if grouped:
            raw_dfs = [group_df(df) for df in raw_dfs]

        for df in raw_dfs:
            df['smiles'] = df['smiles'].str.strip()

        raw_dfs[0]['rdock'] = -raw_dfs[0]['raw_score'].values
        raw_dfs[1]['dock'] = -raw_dfs[1]['raw_score'].values
        raw_dfs[2]['native'] = raw_dfs[2]['raw_score'].values
        raw_dfs = [df.drop('raw_score', axis=1) for df in raw_dfs]

        big_df_raw = raw_dfs[1]
        big_df_raw = big_df_raw.merge(raw_dfs[2], on=['pocket_id', 'smiles', 'is_active'], how='outer')
        big_df_raw = big_df_raw.merge(raw_dfs[0], on=['pocket_id', 'smiles', 'is_active'], how='inner')
        big_df_raw = big_df_raw[['pocket_id', 'smiles', 'is_active', 'rdock', 'dock', 'native']]

        _, _, raw_df_docknat = mix_two_scores(big_df_raw, score1='dock', score2='native', outname_col='docknat',
                                              add_decoy=True)
        big_df_raw = big_df_raw.merge(raw_df_docknat, on=['pocket_id', 'smiles', 'is_active'], how='outer')

        _, _, raw_df_rdocknat = mix_two_scores(big_df_raw, score1='rdock', score2='native', outname_col='rdocknat',
                                               add_decoy=False)
        big_df_raw = big_df_raw.merge(raw_df_rdocknat, on=['pocket_id', 'smiles', 'is_active'], how='outer')

        _, _, raw_df_combined = mix_two_scores(big_df_raw, score1='docknat', score2='rdock', outname_col='combined',
                                               add_decoy=False)
        big_df_raw = big_df_raw.merge(raw_df_combined, on=['pocket_id', 'smiles', 'is_active'], how='outer')

        _, _, raw_df_rdockdock = mix_two_scores(big_df_raw, score1='dock', score2='rdock', outname_col='rdockdock',
                                                add_decoy=False)
        big_df_raw = big_df_raw.merge(raw_df_rdockdock, on=['pocket_id', 'smiles', 'is_active'], how='outer')
        return big_df_raw

    for seed in SEEDS:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = merge_csvs(to_mix=TO_MIX, grouped=GROUPED, decoy=DECOY)
        big_df_raw.to_csv(out_path_raw)


def compute_all_self_mix():
    for i in range(len(SEEDS)):
        to_compare = i, (i + 1) % len(SEEDS)
        out_path_raw_1 = f'outputs/big_df{"_grouped" if GROUPED else ""}_{SEEDS[to_compare[0]]}_raw.csv'
        big_df_raw_1 = pd.read_csv(out_path_raw_1)
        out_path_raw_2 = f'outputs/big_df{"_grouped" if GROUPED else ""}_{SEEDS[to_compare[1]]}_raw.csv'
        big_df_raw_2 = pd.read_csv(out_path_raw_2)
        for score in ['native', 'dock']:
            all_efs, _, _ = mix_two_dfs(big_df_raw_1, big_df_raw_2, score)
            print(score, np.mean(all_efs))


def get_mar_one(df, score, outname=None):
    pockets = df['pocket_id'].unique()
    all_efs = []
    rows = []
    for pi, p in enumerate(pockets):
        pocket_df = df.loc[df['pocket_id'] == p]
        fpr, tpr, thresholds = metrics.roc_curve(pocket_df['is_active'], pocket_df[score], drop_intermediate=True)
        enrich = metrics.auc(fpr, tpr)
        all_efs.append(enrich)
        rows.append({'pocket_id': p, "score": enrich, "decoys": DECOY, "metric": "MAR"})
    if outname is not None:
        df = pd.DataFrame(rows)
        df.to_csv(outname)
    pocket_ef = np.mean(all_efs)
    return pocket_ef


def get_one_mixing_table(df, seed=42):
    all_methods = ['native', 'dock', 'rdock']
    all_res = {}
    # Do singletons
    for method in all_methods:
        outname = f'outputs/{method}_{seed}.csv'
        result = get_mar_one(df, score=method, outname=outname)
        all_res[method] = result
    # Do pairs
    # for pair in itertools.combinations(all_methods, 2):
    #     mean_ef = get_mix_score(df, score1=pair[0], score2=pair[1])
    #     all_res[pair] = mean_ef
    mean_ef = get_mix_score(df, score1="dock", score2="rdock")
    all_res['dock/rdock'] = mean_ef

    result_mixed = get_mar_one(df, score='docknat', outname=f'outputs/docknat_{seed}.csv')
    all_res['docknat'] = result_mixed
    result_mixed = get_mar_one(df, score='rdocknat', outname=f'outputs/rdocknat_{seed}.csv')
    all_res['rdocknat'] = result_mixed
    result_mixed = get_mar_one(df, score='combined', outname=f'outputs/combined_{seed}.csv')
    all_res['combined'] = result_mixed

    for k, v in all_res.items():
        print(f"{k} \t: {v:.4f}")


def get_table_mixing():
    for seed in SEEDS:
        out_path_raw = f'outputs/big_df{"_grouped" if GROUPED else ""}_{seed}_raw.csv'
        big_df_raw = pd.read_csv(out_path_raw)
        get_one_mixing_table(big_df_raw)


if __name__ == "__main__":
    # Fix groups
    # get_groups()

    DECOY = 'pdb_chembl'
    # DECOY = 'chembl'
    GROUPED = True
    SEEDS = [42]
    # SEEDS = [0, 1, 42]

    MODELS = {
        "dock_rnafm": "dock/dock_new_pdbchembl_rnafm",
        "native_pre_rnafm": "is_native/native_pretrain_new_pdbchembl_rnafm",
    }
    RUNS = list(MODELS.keys())
    # GET INFERENCE CSVS FOR SEVERAL MODELS
    recompute = False
    get_all_csvs(recompute=recompute, decoys=DECOY)

    # PARSE INFERENCE CSVS AND MIX THEM
    TO_MIX = ['rdock'] + RUNS
    compute_mix_csvs()

    # To compare to ensembling the same method with different seeds
    # compute_all_self_mix()

    # Get table with all mixing
    # get_table_mixing()
