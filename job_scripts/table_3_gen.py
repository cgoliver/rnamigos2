import os
import glob
from pathlib import Path 
import numpy as np
import pandas as pd

if __name__ == "__main__":
    RESULTS_DIR = Path('..', 'results', 'trained_models', 'native_fp')

    runs = [
            'retrain',
            'bce_loss',
            'directed',
            'pre_r1',
            'pre_riso',
            'pre_riso_big',
            'retrain',
            'whole_data',
            'train_native',
            'train_dock']

    results = []
    for run in runs:
        for i in range(10):
            try:
                ef = pd.read_csv(RESULTS_DIR / f"{run}_{i}" / "ef.csv") 
                results.append({'run': run,
                                'MAR_mean': np.mean(ef['ef']),
                                'MAR_std': np.std(ef['ef']),
                                'split': i
                                }
                               )
            except FileNotFoundError:
                print("missing ", RESULTS_DIR / f"{run}_{i}" / "ef.csv") 

    df = pd.DataFrame(results)
    split_meaned = df.groupby('run').mean().reset_index().drop_duplicates(subset='run').drop(['split'], axis=1).sort_values(by='MAR_mean')
    print(split_meaned.to_markdown())


    
