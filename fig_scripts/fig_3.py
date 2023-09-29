import glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    decoy_mode = 'pdb_chembl'
    csvs = glob.glob("outputs/*.csv")[:5]

    dfs = (pd.read_csv(c) for c in csvs)
    dfs = (df.loc[df['decoys'] == decoy_mode] for df in dfs)
    dfs = (df.assign(name=csvs[i][:-8]) for i, df in enumerate(dfs))
    big_df = pd.concat(dfs)

    bp = big_df.boxplot(column='score', by='name', grid=False)
    for i, n in enumerate(big_df['name'].unique()):
        y = big_df.score[big_df.name==n].dropna()
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.2)
    plt.show()

