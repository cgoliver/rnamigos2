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
        actives.columns = ['SMILE', 'score_1', 'score_2', 'score_3', 'score_4']
        inactives = pd.read_csv(f"outputs/robin/{robin}_inactives.txt", delimiter=' ')
        inactives.columns = ['SMILE', 'score_1', 'score_2', 'score_3', 'score_4']

        true_actives = set(actives['SMILE'])


        migos_df = pd.concat([actives, inactives])
        migos_df['is_active'] = ([1] * len(actives)) + ([0] * len(inactives))

        N_migos = len(migos_df)
        N_dock = len(dock_scores)
        
        migos_df['rank'] = ss.rankdata(migos_df['score_4']) / N_migos
        dock_scores['rank'] = 1 - (ss.rankdata(dock_scores['INTER']) / N_dock)

        migos_actives_pred = set(migos_df.loc[migos_df['rank'] > 0.95]['SMILE'])
        dock_actives_pred = set(dock_scores.loc[dock_scores['rank'] > 0.95]['SMILE'])

        """
        sns.kdeplot(data=dock_scores, hue='TYPE', x='rank', common_norm=False)
        plt.show()

        sns.kdeplot(data=migos_df, hue='is_active', x='rank', common_norm=False)
        plt.title("RNAmigos")
        plt.show()
        """

        migos_correct = true_actives & migos_actives_pred
        dock_correct = true_actives & dock_actives_pred

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
    fig.legend(handles=legend_elements, loc='lower center')
    plt.show()
 
    pass
