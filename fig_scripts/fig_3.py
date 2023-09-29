import glob
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    decoy_mode = 'pdb_chembl'
    csvs = glob.glob("outputs/*.csv")

    dfs = (pd.read_csv(c) for c in csvs)
    dfs = (df.loc[df['decoys'] == decoy_mode] for df in dfs)
    dfs = (df.assign(name=csvs[i]) for i, df in enumerate(dfs))
    big_df = pd.concat(dfs)

    boxplot = big_df.boxplot(by='name')
    plt.show()

