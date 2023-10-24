import os
import pandas as pd

if __name__ == "__main__":

    robin_runs = [
                  'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin',
                  'final_chembl_fp_dim64_simhungarian_prew0_robin',
                  'final_chembl_native_graphligs_dim64_optimol1_robin'
                  ]

    runs = ['rdock',
            'final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0',
            'final_chembl_fp_dim64_simhungarian_prew0',
            'final_chembl_native_graphligs_dim64_optimol1'
            ]
    raw_dfs = [pd.read_csv(f"../outputs/{r}_raw.csv") for r in robin_runs]
    ef_dfs = [pd.read_csv(f"../outputs/{r}.csv") for r in robin_runs]

    for i in range(len(robin_runs)):
        raw_dfs[i]['model'] = robin_runs[i]
        ef_dfs[i]['model'] = robin_runs[i]

    pd.concat(raw_dfs).to_csv("final_outputs_robin_raw.csv")
    pd.concat(ef_dfs).to_csv("final_outputs_robin.csv")


    
    pass
