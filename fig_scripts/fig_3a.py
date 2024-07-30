""" Bar plot comparing VS methods on ChEMBL.
"""

import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

paths = {'RNAmigos1': 'outputs/fp_split_grouped1_raw.csv',
         'RNAmigos2': 'outputs/docknat_grouped_0_raw.csv',
         'RLDOCK': 'outputs/rldock_docking_consolidate_all_terms.csv',
         'rDock': 'outputs/rdock_raw.csv',
         'AnnapuRNA': 'outputs/annapurna_results_consolidate.csv',
         'AutoDock-Vina': 'outputs/vina_docking_consolidate.csv'
         }

score_to_use = {'RNAmigos1': 'raw_score',
                'RNAmigos2': 'docknat',
                'RLDOCK': 'Total_Energy',
                'AnnapuRNA': 'score_RNA-Ligand',
                'AutoDock-Vina': 'score',
                'rDock': 'raw_score'
                }

names_train, names_test, grouped_train, grouped_test = pickle.load(open("data/train_test_75.p", 'rb'))

def merge_raw_dfs():
    """ Combine all raw dfs into one df which is: pocket ID x method x AuROC  """
    dfs = []
    cols = ['raw_score', 'pocket_id', 'smiles', 'is_active', 'normed_score']
    for method, path in paths.items():
        df = pd.read_csv(path)
        if method in ['RNAmigos1', 'RNAmigos2', 'rDock']:
            df = df.loc[df['decoys'] == 'chembl']
        if method == 'RLDOCK':
            df['raw_score'] = (df['Total_Energy'] - (df['Self_energy_of_ligand'] + df['Self_energy_of_receptor']))
            df.loc[df['raw_score'] > 0, 'raw_score'] = 0
        else:
            df['raw_score'] = df[score_to_use[method]]


        if method in ['RNAmigos2']:
            df['normed_score'] = df.groupby(['pocket_id'])['raw_score'].rank(pct=True)
        else:
            df['normed_score'] = df.groupby(['pocket_id'])['raw_score'].rank(pct=True, ascending=False)


        df = df.loc[:,cols]
        df['method'] = method
        dfs.append(df)

    big_df = pd.concat(dfs)
    return big_df

def plot(df):
    df = df.loc[df['pocket_id'].isin(grouped_test)]
    df = df.loc[df['is_active'] > 0]

    print(df.groupby(['method'])['normed_score'].mean())

    custom_palette_bar = {method: '#e9e9f8' if method.startswith('RNAmigos') else '#d3d3d3' \
                        for method in df['method'].unique()}

    custom_palette_point = {method: '#b2b2ff' if method.startswith('RNAmigos') else '#a5a5a5' \
            for method in df['method'].unique()}

    order = ['RLDOCK', 'AutoDock-Vina', 'AnnapuRNA', 'rDock', 'RNAmigos1', 'RNAmigos2']
    g = sns.barplot(df, x='method', y='normed_score', order=order, palette=custom_palette_bar, alpha=0.7)
    sns.stripplot(df, x='method', y='normed_score', ax=g, order=order, palette=custom_palette_point)
    plt.show()
    pass

if __name__ == "__main__":
    df = merge_raw_dfs()
    print(df)
    plot(df)
    pass
