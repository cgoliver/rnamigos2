import os
import pandas as pd

if __name__ == "__main__":

    robin_runs = [
                  'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin',
                  'definitive_chembl_fp_dim64_simhungarian_prew0_robin',
                  'final_chembl_native_graphligs_dim64_optimol1_robin'
                  ]

    runs = ['rdock',
            'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0',
            'definitive_chembl_fp_dim64_simhungarian_prew0',
            'final_chembl_native_graphligs_dim64_optimol1'
            ]
    raw_dfs = [pd.read_csv(f"../outputs/{r}_raw.csv").sort_values(by=['pocket_id', 'smiles', 'decoys', 'is_active']) for r in runs]
    ef_dfs = [pd.read_csv(f"../outputs/{r}.csv") for r in runs]

    big_df = pd.concat(ef_dfs)
    big_df_raw = raw_dfs[0][['pocket_id', 'smiles', 'decoys', 'is_active']]

    for i in range(len(runs)):
        # raw_dfs[i]['model'] = runs[i]
        big_df_raw[runs[i]] = raw_dfs[i]['raw_score']
        ef_dfs[i]['model'] = runs[i]

    ef_df = pd.concat(ef_dfs)

    big_df_raw.to_csv("big_df_raw.csv")
    ef_df.to_csv("big_ef_df.csv")
