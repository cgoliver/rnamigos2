import os
import glob
from pathlib import Path 
import numpy as np
import pandas as pd

def get_runs(run_dir):
    return set(["_".join(r.split("_")[:-1]) for r in os.listdir(run_dir)])

if __name__ == "__main__":
    RESULTS_DIR = Path('../..', 'results', 'trained_models')

    targets = ['dock', 'is_native', 'native_fp']

    results = []
    for target in targets:
        runs = get_runs(RESULTS_DIR / target)
        for run in runs:
            for split in range(10): 
                try:
                    p = Path(RESULTS_DIR, target, f"{run}_{split}", "ef.csv")
                    ef = pd.read_csv(p)
                    results.append({'run': run,
                                    'MAR_mean': np.mean(ef['ef']),
                                    'MAR_std': np.std(ef['ef']),
                                    'split': split 
                                    }
                                   )
                except FileNotFoundError:
                    pass
                    # print("missing ", target, run, split)

    df = pd.DataFrame(results)
    # print(df)
    # split_meaned = df.groupby('run').mean().reset_index().drop_duplicates(subset='run').drop(['split'], axis=1).sort_values(by='MAR_mean')
    split_meaned = df.groupby('run').mean().reset_index().drop_duplicates(subset='run').sort_values(by='MAR_mean')
    print(split_meaned.to_markdown())
