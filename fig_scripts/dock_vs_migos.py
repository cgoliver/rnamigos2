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

def get_fps(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    return fps

if __name__ == "__main__":

    robin_ids = {'2GDI_Y_TPP_100': 'TPP',
                 '2QWY_A_SAM_100': 'SAM_ll',
                 '3FU2_C_PRF_101': 'PreQ1',
                 '5BTP_A_AMZ_106': 'ZTP'}

    robins = ['2GDI_Y_TPP_100',
              '2QWY_A_SAM_100',
              '3FU2_C_PRF_101',
              '5BTP_A_AMZ_106']

    dock_df = pd.read_csv("outputs/robin/robin_targets_docking_consolidated.csv")

    fig, axarr = plt.subplots(2, 2)
    axs = axarr.flatten()

    for i, robin in enumerate(robins):
        dock_scores = dock_df.loc[dock_df['TARGET']  == robin_ids[robin]]

        actives = pd.read_csv(f"outputs/robin/{robin}_actives.txt", delimiter=' ')
        actives.columns = ['SMILE', 'dock', 'native', 'fp', 'mixed']
        inactives = pd.read_csv(f"outputs/robin/{robin}_inactives.txt", delimiter=' ')
        inactives.columns = ['SMILE', 'dock', 'native', 'fp', 'mixed']

        true_actives = set(actives['SMILE'])

        migos_df = pd.concat([actives, inactives])
        migos_df['is_active'] = ([1] * len(actives)) + ([0] * len(inactives))

        migos_df['random'] = [random.random() for _ in range(len(migos_df))]

        N_migos = len(migos_df)
        N_dock = len(dock_scores)
        
        hmap_lists = []
        accs = []

        modes = ['RDOCK', 'random', 'dock', 'native', 'fp', 'mixed']
        for mode in modes:
            if mode == 'RDOCK':
                dock_scores['rank'] = 1 - (ss.rankdata(dock_scores['INTER']) / N_dock)
                dock_actives_pred = set(dock_scores.loc[dock_scores['rank'] > 0.95]['SMILE'])
                dock_correct = true_actives & dock_actives_pred
                dock_list = [1 if lig in dock_correct else 0 for lig in true_actives]
                dock_acc = np.sum(dock_list) / len(true_actives)
                accs.append(dock_acc)
                hmap_lists.append(dock_list)
            else:
                migos_df[f'rank_{mode}'] = ss.rankdata(migos_df[mode]) / N_migos

                actives_pred = set(migos_df.loc[migos_df[f'rank_{mode}'] > 0.95]['SMILE'])
                migos_correct = true_actives & actives_pred
                
                migos_list = [1 if lig in migos_correct else 0 for lig in true_actives]
                migos_acc = np.sum(migos_list) / len(true_actives)
                hmap_lists.append(migos_list)
                accs.append(migos_acc)

        """
        sns.kdeplot(data=dock_scores, hue='TYPE', x='rank', common_norm=False)
        plt.show()

        sns.kdeplot(data=migos_df, hue='is_active', x='rank', common_norm=False)
        plt.title("RNAmigos")
        plt.show()
        """


        sns.heatmap(hmap_lists, yticklabels=[f"{m} ({a*100:.1f}%)" for m, a in zip(modes, accs)], ax=axs[i], cmap='inferno')
        axs[i].set_title(robin)

        """
        print("--"*20)
        fps = np.array(get_fps(true_actives))

        X_embedded = TSNE(n_components=2, learning_rate='auto',
                          init='pca').fit_transform(fps)

        legend = {'RNAmigos': 'green',
                  'RDOCK': 'red',
                  'both': 'blue',
                  'none': 'grey'}

        for j, (fp, sm) in enumerate(zip(fps, true_actives)):
            if sm in migos_correct and sm not in dock_correct:
                case = 'RNAmigos'
            elif sm in migos_correct and sm in dock_correct:
                case = 'both' 
            elif sm not in migos_correct and sm in dock_correct:
                case = 'RDOCK'
            else:
                case = 'none'
        
            axs[i].scatter(X_embedded[j,0], X_embedded[j,1], c=legend[case], marker='.', s=30, alpha=.9, label=case)
            axs[i].axis('off')
            axs[i].set_title(robin)

    legend_elements = [Line2D([0], [0], linestyle='none', marker='.', color='blue', markersize=20, label=f'both'),
                       Line2D([0], [0], linestyle='none', marker='.', color='red', markersize=20, label=f'RDOCK only'),
                       Line2D([0], [0], linestyle='none', marker='.', color='green', markersize=20, label=f'RNAmigos only'),
                       Line2D([0], [0], linestyle='none', marker='.', color='grey', markersize=20, label=f'none'),
                       ]
        """
    #fig.legend(handles=legend_elements, loc='lower center')
    # plt.title("Hits @ 1%")
    plt.show()
 
    pass
