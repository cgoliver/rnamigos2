import os

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn import metrics

from rnaglib.drawing import rna_draw
from rnaglib.utils import load_json

from fig_scripts.plot_utils import PALETTE_DICT, group_df

def enrichment_factor(scores, is_active, lower_is_better=True, frac=0.01):
    n_actives = np.sum(is_active)
    n_screened = int(frac * len(scores))
    is_active_sorted = [a for _, a in sorted(zip(scores, is_active), reverse=not lower_is_better)]
    scores_sorted = [s for s,_ in sorted(zip(scores, is_active), reverse=not lower_is_better)]
    n_actives_screened = np.sum(is_active_sorted[:n_screened])
    ef = (n_actives_screened / n_screened) / (n_actives / len(scores))
    return ef, scores_sorted[n_screened]


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


pocket_names = [
    "2GDI_Y_TPP_100",
    "5BTP_A_AMZ_106",
    "2QWY_A_SAM_100",
    "3FU2_C_PRF_101",
]
ligand_names = [
    "TPP",
    "ZTP",
    "SAM_ll",
    "PreQ1",
]


def get_dfs_docking(ligand_name):
    out_dir = 'outputs/robin'

    # Get relevant mapping smiles : normalized score
    docking_df = pd.read_csv(os.path.join(out_dir, "robin_targets_docking.csv"))
    # docking_df = pd.read_csv(os.path.join(out_dir, "robin_targets_docking_consolidated.csv"))
    docking_df = docking_df[docking_df["TARGET"] == ligand_name]
    scores = -docking_df[["TOTAL"]].values.squeeze()

    # DEAL WITH NANS, ACTUALLY WHEN SORTING NANS, THEY GO THE END
    # count = np.count_nonzero(np.isnan(scores))
    # scores = np.sort(scores)
    # cropped_scores = np.concatenate((scores[:10], scores[-10:]))
    # print(cropped_scores)

    scores = np.nan_to_num(scores, nan=np.nanmin(scores))
    # mi = np.nanmin(scores)
    # ma = np.nanmax(scores)
    # print(ma, mi)
    # normalized_scores = (scores - np.nanmin(scores)) / (np.nanmax(scores) - np.nanmin(scores))
    # normalized_scores = scores
    normalized_scores = scores / 80
    mapping = {}
    for smiles, score in zip(docking_df[["SMILE"]].values, normalized_scores):
        mapping[smiles[0]] = score
    mapping = defaultdict(int, mapping)

    # Use this mapping to create our actives/inactives distribution dataframe
    active_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "actives.txt")
    # out_path = os.path.join(out_dir, f"{pocket_name}_actives.txt")
    smiles_list = [s.lstrip().rstrip() for s in list(open(active_ligands_path).readlines())]
    actives_df = pd.DataFrame({"docking_score": [mapping[sm] for sm in smiles_list]})
    actives_df['split'] = 'actives'

    # scores = actives_df[["docking_score"]].values.squeeze()
    # ma = np.nanmax(scores)
    # mi = np.nanmin(scores)
    # count = np.count_nonzero(np.isnan(scores))
    # print(f"actives max/min : {ma} {mi}, nancount : {count} "
    #       f"scores over 200 : {np.sum(scores > 200)} length : {len(scores)} ")

    inactives_ligands_path = os.path.join("data", "ligand_db", ligand_name, "robin", "decoys.txt")
    # out_path = os.path.join(out_dir, f"{pocket_name}_inactives.txt")
    smiles_list = [s.lstrip().rstrip() for s in list(open(inactives_ligands_path).readlines())]
    inactives_df = pd.DataFrame({"docking_score": [mapping[sm] for sm in smiles_list]})
    inactives_df['split'] = 'inactives'

    # scores = inactives_df[["docking_score"]].values.squeeze()
    # ma = np.nanmax(scores)
    # mi = np.nanmin(scores)
    # count = np.count_nonzero(np.isnan(scores))
    # print(f"inactives max/min : {ma} {mi}, nancount : {count} "
    #       f"scores over 200 : {np.sum(scores > 200)} length : {len(scores)} ")

    merged = pd.concat([actives_df, inactives_df]).reset_index()
    return merged


def get_dfs_migos(pocket_name, swap=False, swap_on='merged'):
    names = ['smiles', 'dock', 'is_native', 'native_fp', 'merged']
    out_dir = 'outputs/robin'
    actives_df = pd.read_csv(os.path.join(out_dir, f"{pocket_name}_actives.txt"), names=names, sep=' ')
    actives_df['split'] = 'actives'
    inactives_df = pd.read_csv(os.path.join(out_dir, f"{pocket_name}_inactives.txt"), names=names, sep=' ')
    inactives_df['split'] = 'inactives'
    # decoys_df = pd.read_csv(os.path.join(out_dir, f"{pocket_name}_decoys.txt"), names=names, sep=' ')
    # decoys_df['split'] = 'decoys'
    # merged = pd.concat([actives_df, inactives_df, decoys_df])
    # merged = pd.concat([actives_df, decoys_df])
    merged = pd.concat([actives_df, inactives_df])
    if swap:
        other_pocket = random.choice(list(set(pocket_names) - set([pocket_name])))
        other_actives = pd.read_csv(os.path.join(out_dir, f"{other_pocket}_actives.txt"), names=names, sep=' ')
        other_actives['split'] = 'actives'
        other_inactives = pd.read_csv(os.path.join(out_dir, f"{other_pocket}_inactives.txt"), names=names, sep=' ')
        other_inactives['split'] = 'inactives'
        merged_other = pd.concat([other_actives, other_inactives])
        both = merged.merge(merged_other, on='smiles', suffixes=('_orig', '_other'))
        if swap_on == 'merged':
            both['merged'] = both['merged_other']
            both['split'] = both['split_orig']
        if swap_on == 'split':
            both['split'] = both['split_other']
            both['merged'] = both['merged_orig']
        return both

    return merged


if __name__ == '__main__':

    all_efs = list()
    all_aurocs = list()

    score_to_use = 'dock_nat'
    # score_to_use = 'merged'
    # score_to_use = 'dock'
    # score_to_use = 'is_native'
    # score_to_use = 'native_fp'
    print(score_to_use)

    #plt.gca().set_yscale('custom')
    for i, (pocket_name, ligand_name) in enumerate(zip(pocket_names, ligand_names)):
        # FOR DOCKING
        # merged = get_dfs_docking(ligand_name=ligand_name)
        # score_to_use = 'docking_score'

        # FOR MIGOS

        swap = False
        merged = get_dfs_migos(pocket_name=pocket_name, swap=swap)
        merged['dock_nat'] = (merged['is_native'] + merged['dock'])/2
        merged['dock_nat_rank'] = merged['dock_nat'].rank(pct=True, ascending=True)
        print(merged)

        fig, ax = plt.subplots(figsize=(6,6))

        #ax.set_xscale('custom')

        # g = load_json(f"data/robin_graphs_x3dna/{name}.json")
        # g = g.subgraph([n for n,d in g.nodes(data=True) if d['in_pocket'] == True])
        # print(g.nodes(data=True))
        # nt_key = 'nt_code'
        # colors = {'C': 'red', 'G': 'yellow', 'A': 'blue', 'U': 'green'}
        # rna_draw(g,
        #          node_colors=[colors[d[nt_key]] for n,d in g.nodes(data=True)],
        #          ax=axs[i][1])
        scores = merged[score_to_use]
        actives = merged['split'].isin(['actives'])

        # GET EFS
        fracs = [0.01, 0.02, 0.05]
        linecolors = ['black', 'grey', 'lightgrey']


        colors = sns.color_palette("Paired", 4)
        default_frac = 0.01
        offset = 0
        legend_y = 0.9 
        legend_x = 0.7
        for _, frac in enumerate(fracs):
            ef, thresh = enrichment_factor(scores=scores, is_active=actives,
                                   lower_is_better=False, frac=frac)
            #ax.axvline(x=thresh, ymin=0, ymax=max(scores), color=linecolors[i])

            ax.fill_between(np.linspace(thresh, 1, 10), 0, 1,
                color='orange', alpha=0.05, transform=ax.get_xaxis_transform())

            arrow_to_y = legend_y - offset

            ax.text(legend_x, legend_y - offset, f"EF@{int(frac*100)}%: {ef:.2f}", fontsize=10, transform=ax.transAxes)

            offset += 0.05

        ef, thresh = enrichment_factor(scores=scores, is_active=actives,
                               lower_is_better=False, frac=default_frac)
        all_efs.append(ef)
        print(f'EF@{frac} : ', pocket_name, ef)
        
        g = sns.kdeplot(data=merged, x=score_to_use, fill=True, hue='split', alpha=.9, palette={'actives': colors[i], 'inactives': 'lightgrey'}, common_norm=False, ax=ax, log_scale=False)
        ax.set_title(pocket_name)
        g.legend().remove()
        sns.despine()
        plt.savefig(f"figs/panel_3_{i}.pdf", format="pdf")
        plt.show()


        # GET AUROC
        fpr, tpr, thresholds = metrics.roc_curve(actives, scores)
        auroc = metrics.auc(fpr, tpr)
        all_aurocs.append(auroc)
        # print('AuROC : ', pocket_name, auroc)
        # print()

        # mar = mean_active_rank(scores, actives, lower_is_better=False)
        # print(mar)
    #     #ef = f"EF@1\% {list(ef_df.loc[ef_df['pocket_id'] == name]['score'])[0]:.3f}"
    #     axs[i][0].text(0, 0, f"{name} EF: {ef:.3} MAR: {mar:.3}")
    #     axs[i][0].axis("off")
    #     axs[i][1].axis("off")
    #     sns.despine()
    #
    print(np.mean(all_efs))
    print(np.mean(all_aurocs))
    # plt.tight_layout()
    plt.savefig("figs/fig_3a.pdf", format="pdf")
    plt.show()

#
