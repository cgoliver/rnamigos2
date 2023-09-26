
import os
import glob
from pathlib import Path 
import numpy as np
import pandas as pd
from yaml import safe_load 

# fp_native_grid-true-0.0--1-bce-true-1e-4

if __name__ == "__main__":
    RESULTS_DIR = Path('..', 'results', 'trained_models', 'native_fp')

    runs = glob.glob(str(RESULTS_DIR) + '/fp_native_grid2*')
    grid_params = ['model.batch_norm', 
                   'model.dropout', 
                   'model.encoder.num_bases',
                   'train.loss',
                   'model.use_pretrained',
                   ]

    results = []
    for run in runs:
        try:
            with open(Path(run, 'config.yaml'), 'r') as f:
                df = pd.json_normalize(safe_load(f))
            p = Path(run, "ef.csv")
            ef = pd.read_csv(p)
                
            results.append({
                            'MAR_mean': np.mean(ef['ef']),
                            'MAR_std': np.std(ef['ef']),
                            **{p.split(".")[-1]: df[p][0] for p in grid_params}
                            }
                           )
        except FileNotFoundError:
            print("missing ", run) 

    df = pd.DataFrame(results)
    df = df.sort_values(by=['MAR_mean'])
    print(df.to_markdown(index=False))


    
