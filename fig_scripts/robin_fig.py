import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rnaglib.drawing import rna_draw
from rnaglib.utils import load_json

def enrichment_factor(scores, is_active, lower_is_better=True, frac=0.01):
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=not lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    return (n_actives_screened / n_screened) / (n_actives / len(scores))


def mean_active_rank(scores, is_active, lower_is_better=True, **kwargs):
    """ Compute the average rank of actives in the scored ligand set

    Arguments
    ----------
    scores (list): list of scalar scores for each ligand in the library
    is_active (list): binary vector with 1 if ligand is active or 0 else, one for each of the scores
    lower_is_better (bool): True if a lower score means higher binding likelihood, False v.v.

    Returns
    ---------
    int
        Mean rank of the active ligand [0, 1], 1 is the best score.
        

    >>> mean_active_rank([-1, -5, 1], [0, 1, 0], lower_is_better=True)
    >>> 1.0

    """
    is_active_sorted = sorted(zip(scores, is_active), reverse=lower_is_better)
    return (np.mean([rank for rank, (score, is_active) in enumerate(is_active_sorted) if is_active]) + 1) / len(scores)


raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinx3dna_raw.csv')
ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinx3dna.csv')

raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robin_raw.csv')
ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robin.csv')

raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinprebuilt_raw.csv')
ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_robinprebuilt.csv')

raw_df = pd.read_csv('outputs/final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin_x3dna_raw.csv')
ef_df = pd.read_csv('outputs/final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch0_robin_x3dna.csv')

raw_df_2 = pd.read_csv('outputs/final_chembl_native_graphligs_dim64_optimol1_robinprebuilt_raw.csv')
ef_df_2 = pd.read_csv('outputs/final_chembl_native_graphligs_dim64_optimol1_robinprebuilt.csv')

# raw_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0_raw.csv')
# ef_df = pd.read_csv('outputs/definitive_chembl_fp_dim64_simhungarian_prew0.csv')

fig, axs = plt.subplots(7, 3, sharex=False, sharey=True)

nt_key = 'nt_code'
colors = {'C': 'red', 'G': 'yellow', 'A': 'blue', 'U': 'green'}

for i, (name, df) in enumerate(raw_df.groupby('pocket_id')):
    legend = False if i != 6 else True
    sns.kdeplot(data=df, x='raw_score', hue='is_active', legend=legend, common_norm=False, ax=axs[i][2])
    g = load_json(f"data/robin_graphs_x3dna/{name}.json")
    g = g.subgraph([n for n,d in g.nodes(data=True) if d['in_pocket'] == True])
    print(g.nodes(data=True))
    rna_draw(g, 
             node_colors=[colors[d[nt_key]] for n,d in g.nodes(data=True)],
             ax=axs[i][1])
    ef = enrichment_factor(df['raw_score'], df['is_active'], lower_is_better=True, frac=0.05)
    mar = mean_active_rank(df['raw_score'], df['is_active'], lower_is_better=True)
    #ef = f"EF@1\% {list(ef_df.loc[ef_df['pocket_id'] == name]['score'])[0]:.3f}"
    axs[i][0].text(0, 0, f"{name} EF: {ef:.3} MAR: {mar:.3}")
    axs[i][0].axis("off")
    axs[i][1].axis("off")
    sns.despine()

plt.show()
