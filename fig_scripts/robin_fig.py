import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rnaglib.drawing import rna_draw
from rnaglib.utils import load_json


raw_df = pd.read_csv('outputs/final_chembl_fp_dim64_simhungarian_prew0_robin_raw.csv')
ef_df = pd.read_csv('outputs/final_chembl_fp_dim64_simhungarian_prew0_robin.csv')

fig, axs = plt.subplots(7, 3, sharex=False, sharey=True)

colors = {'C': 'red', 'G': 'yellow', 'A': 'blue', 'U': 'green'}

for i, (name, df) in enumerate(raw_df.groupby('pocket_id')):
    legend = False if i != 6 else True
    sns.kdeplot(data=df, x='raw_score', hue='is_active', legend=legend, common_norm=False, ax=axs[i][2])
    g = load_json(f"data/robin_graphs/{name}.json")
    print(g.nodes(data=True))
    rna_draw(g, 
             node_colors=[colors[d['nt']] for n,d in g.nodes(data=True)],
             ax=axs[i][1])
    ef = f"EF@1\% {list(ef_df.loc[ef_df['pocket_id'] == name]['score'])[0]:.3f}"
    axs[i][0].text(0, 0, f"{name} ({ef})")
    axs[i][0].axis("off")
    axs[i][1].axis("off")
    sns.despine()

plt.show()
