import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
from matplotlib.lines import Line2D

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.manifold import TSNE

""" BARCODES """

def barcodes():
    methods = ['dock_split_grouped1',
               'fp_split_grouped1',
               'native_split_grouped1',
               'mixed'
               ]
    rows = []
    for m in methods:
        print(m)
        df = pd.read_csv(f"outputs/{m}.csv")
        row = df[df['decoys'] == 'chembl'].sort_values(by='pocket_id')
        print(row['pocket_id'].iloc[:10])
        rows.append(row['score'])
    sns.heatmap(rows, cmap='binary_r')
    plt.xlabel("Pocket")
    plt.ylabel("Method")
    plt.yticks(np.arange(len(methods)) + 0.5, [m.split("_")[0] for m in methods], rotation=0, va="center")
    plt.savefig("figs/fig_1c.pdf", format="pdf")
    plt.show()
    pass


""" VIOLINS """

def violins():
    pass

if __name__ == "__main__":
    barcodes()
    pass
