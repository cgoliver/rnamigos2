"""
    Do embedding clustering and generate similar subgraphs.
"""
import sys
if __name__ == '__main__':
    sys.path.append('./')

import os
import pickle
import math
import random
from random import choice, shuffle
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn import metrics
from numpy.linalg import norm
from scipy.spatial.distance import jaccard,euclidean
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import seaborn as sns
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch
#from openbabel import pybel
import numpy as np

from learning.rgcn import Model
#from rna_classes import *
from post.utils import *
# from post.tree_grid_vincent import compute_clustering 

from learning.attn import get_attention_map
from learning.utils import dgl_to_nx
from tools.learning_utils import load_model
# from post.drawing import rna_draw
import time
from sklearn.preprocessing import QuantileTransformer

def mse(x,y):
    d = np.sum((x-y)**2) / len(x)
    return d

def get_decoys(mode='pdb', annots_dir='../data/annotated/pockets_nx_2'):
    """
    Build decoys set for validation.
    """
    if mode=='pdb':
        fp_dict = {}
        for g in os.listdir(annots_dir):
            try:
                lig_id = g.split(":")[2]
            except:
                print(f"failed on {g}")
            _,_,_,fp = pickle.load(open(os.path.join(annots_dir, g), 'rb'))
            fp_dict[lig_id] = fp
        decoy_dict = {k:(v, [f for lig,f in fp_dict.items() if lig != k]) for k,v in fp_dict.items()}
        return decoy_dict
    
    if mode=='sdf':
        fp_dict = {}
        fp_dict = pickle.load(open('./data/all_ligs_pdb_maccs_2022.p', 'rb'))
        decoy_dict = {k: (v, [f for lig, f in fp_dict.items() if lig != k]) for k, v in fp_dict.items()}
        return decoy_dict

    if mode == 'chembl':
        return pickle.load(open('./data/chembl_dict.p', 'rb'))
    
    if mode == 'dude':
        return pickle.load(open('./data/decoys_zinc.p', 'rb'))
    pass
def distance_rank(active, pred, decoys, dist_func=mse):
    """
        Get rank of prediction in `decoys` given a known active ligand.
    """

    #print(active)
    #print('pred')
    #print(pred)
    pred_dist = dist_func(active, pred)
    #print('Distance in rank initial')
    #print(pred_dist)
    rank = 0
    for decoy in decoys:
        #print('decoy')
        #print(decoy)
        d = dist_func(pred[0], decoy)
        #print(d)
        #if find a decoy closer to prediction, worsen the rank.
        if d < pred_dist:
            rank += 1
    return 1 - (rank / (len(decoys) + 1))

def decoy_test(model, decoys, edge_map, embed_dim,
                        test_graphlist=None,
                        shuffle=False,
                        nucs=False,
                        test_graph_path="../data/annotated/pockets_nx",
                        majority=False):
    """
        Check performance against decoy set.
        decoys --> {'ligand_id', ('expected_FP', [decoy_fps])}
        test_set --> [annot_graph_path,]
        :model trained model
        :test_set inputs for model to test (RNA graphs)
        :decoys dictionary with list of decoys for each input to test.
        :test_graphlist list of graph names to use in the test.

        :return: enrichment score
    """
    ranks = []
    sims = []

    if test_graphlist is None:
        test_graphlist = os.listdir(test_graph_path)
        
    ligs = list(decoys.keys())
    if majority:
        generic = generic_fp("./data/annotated/pockets_nx_symmetric_orig")
    # generic = generic_fp("./data/annotated/pockets_nx_symmetric_orig_withouterrors_docking")
    true_ids = []
    fp_dict = {}
    for g_path in test_graphlist:
        g,_,_,true_fp = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
        try:
            #true_id = g_path.split(":")[2]
            true_id = g_path.split("_")[3]
            #pocket_name_list = g_path.split('_')
            pocket_name_list = g_path.split(':')
            if len(pocket_name_list) == 5:
                true_id = pocket_name_list[3].split('.')[0]
            #print(true_id)
            fp_dict[true_id] = true_fp
            decoys[true_id]
        except:
            print(f">> failed on {g_path}")
            continue
        nx_graph, dgl_graph = nx_to_dgl(g, edge_map, nucs=nucs)
        with torch.no_grad():
            fp_pred, _ = model(dgl_graph)

        # fp_pred = fp_pred.detach().numpy()
        fp_pred = fp_pred.detach().numpy() > 0.5
        fp_pred = fp_pred.astype(int)
        #print('predicted_fingerprint')
        #print(fp_pred[0])
        #print(type(fp_pred))
        #fp_pred = fp_pred.astype(float)
        #print(fp_pred)
        if majority:
            fp_pred = generic
            #print('pf pred majority')
            #print(len(fp_pred))
            #print(type(fp_pred))
        # fp_pred = fp_pred.detach().numpy()
        #active = decoys[true_id][0]
        active = true_fp
        #print('active')
        #print(len(active))
        #print(type(active))
        decs = decoys[true_id][1]
        rank = distance_rank(active, fp_pred, decs, dist_func=mse)
        # print(rank)
        sim = mse(true_fp, fp_pred)
        true_ids.append(true_id)
        ranks.append(rank)
        sims.append(sim)
    return ranks, sims, true_ids, fp_dict

def wilcoxon_all_pairs(df, methods):
    """
        Compute pairwise wilcoxon on all runs.
    """
    from scipy.stats import wilcoxon
    wilcoxons = {'method_1': [], 'method_2':[], 'p-value': []}
    for method_1 in methods:
        for method_2 in methods:
            vals1 = df.loc[df['method'] == method_1]
            vals2 = df.loc[df['method'] == method_2]
            p_val = wilcoxon(vals1['rank'], vals2['rank'], correction=True)

            wilcoxons['method_1'].append(method_1)
            wilcoxons['method_2'].append(method_2)
            wilcoxons['p-value'].append(p_val[1])
            pass
    wil_df = pd.DataFrame(wilcoxons)
    wil_df.fillna(0)
    pvals = wil_df.pivot("method_1", "method_2", "p-value")
    pvals.fillna(0)
    print(pvals.to_latex())
    # mask = np.zeros_like(pvals)
    # mask[np.triu_indices_from(mask)] = True
    # g = sns.heatmap(pvals, cmap="Reds_r", annot=True, mask=mask, cbar=True)
    # g.set_facecolor('grey')
    # plt.show()
    pass
def generic_fp(annot_dir):
    """
        Compute generic fingerprint by majority over dimensions.
        TODO: Finish this
    """
    fps = []
    for g in os.listdir(annot_dir):
        _,_,_,fp = pickle.load(open(os.path.join(annot_dir, g), 'rb'))
        fps.append(fp)
    counts = np.sum(fps, axis=0)
    consensus = np.zeros(166)
    ones = counts > len(fps) / 2
    consensus[ones] = 1
    return consensus
    
def make_violins(df, x='method', y='rank', save=None, show=True):
    ax = sns.violinplot(x=x, y=y, data=df, color='0.8', bw=.1)
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)
    sns.stripplot(data=df, x=x, y=y, jitter=True, alpha=0.6)
    if not save is None:
        plt.savefig(save, format="pdf")
    if show:
        plt.show()

    pass

def make_ridge(df, x='method', y='rank', save=None, show=True):
    # Initialize the FacetGrid object
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=x, hue=x, aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, y, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, y, clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, x)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    if save:
        plt.savefig(save)
    if show:
        plt.show()


def make_tree_grid(df, fp_dict, method='htune'):
    lig_dict = {}
    df_tree = df.loc[df['method'] == method]
    means = df_tree.groupby('lig').mean()
    for row in means.itertuples():
        lig_dict[row.Index] = (fp_dict[row.Index], row.rank)
    compute_clustering(lig_dict)
    pass


def get_lig_decoys(decoy_mode='dude'):
    graph_dir = '/home/mcb/users/jcarva4/rnamigos/rnamigos1/rnamigos_gcn/data/annotated/pockets_nx_annot_fp_2022'
    decoys = get_decoys(mode=decoy_mode, annots_dir=graph_dir)
    print(decoys)
    pickle.dump(decoys, open('dude_decoys.p', 'wb'))
    pass


def ablation_results(run, graph_dir, mode, decoy_mode='pdb', num_folds=10):
    """
        Compute decoy and distances for a given run and ablation mode

        Returns:
            DataFrame: decoy results dataframe
    """
    ranks, methods, jaccards, ligs  = [], [], [], []
    graph_dir = './data/annotated/' + graph_dir
    decoys = get_decoys(mode=decoy_mode, annots_dir=graph_dir)
    # majority = mode == 'majority'
    fp_dict = {}
    for fold in range(int(num_folds)):
        model, meta = load_model(run +"_" + str(fold))
        # model, meta = load_model(run)
        edge_map = meta['edge_map']
        embed_dim = meta['embedding_dims'][-1]
        num_edge_types = len(edge_map)

        graph_ids = pickle.load(open(f'./results/trained_models/{run}_{fold}/splits_{fold}.p', 'rb'))

        ranks_this,sims_this, lig_ids, fp_dict_this  = decoy_test(model, decoys, edge_map, embed_dim,
            nucs=meta['nucs'],
            test_graphlist=graph_ids['test'],
            test_graph_path=graph_dir, majority=False)
        fp_dict.update(fp_dict_this)
        ranks.extend(ranks_this)
        jaccards.extend(sims_this)
        ligs.extend(lig_ids)
        methods.extend([mode]*len(ranks_this))


    df = pd.DataFrame({'rank': ranks, 'jaccard': jaccards, 'method': methods, 'lig': ligs})
    print(df.describe())
    df.to_csv(f'./results/{run}_{decoy_mode}.csv')
    return df


def summ_complete_dataset():
    rdock_scores = pd.DataFrame(columns=['POCKET_ID', 'INTER'])
    graph_dir = 'data/annotated/pockets_docking_annotated_inter'
    graphlist = os.listdir(graph_dir)
    i = 0
    for g in graphlist:
        i += 1
        print(i)
        name_pocket, graph, _, _, _, _, inter, _, _, _, _, _, _, _, _ = pickle.load(open(os.path.join(graph_dir, g), 'rb'))
        rdock_scores = rdock_scores.append({'BINDING_SITE_ID': g,
            'INTER': str(inter)}, ignore_index=True)
    rdock_scores.to_csv('desc_dataset_upper_intscore_0.csv')


def summ_train_data(run, graph_dir, folds=1):
    rdock_scores = pd.DataFrame(columns=['POCKET_ID', 'INTER'])
    graph_dir = 'data/annotated/pockets_docking_annotated_inter'
    for fold in range(int(folds)):
        model, meta = load_model(run + "_" + str(fold))
        graph_ids = pickle.load(open(f'results/trained_models/{run}_{fold}/splits_{fold}.p', 'rb'))
        test_graphlist=graph_ids['train']
        test_graph_path=graph_dir
        nm = 0
        for g_path in test_graphlist:
            nm += 1
            print('Ejemplo', str(nm))
            p = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
            _, graph, _, ring, fp_nat, fp, inter_score, label_native_lig, label_1std, label_2std, label_thr_min30, label_thr_min17, label_thr_min12, label_thr_min8, label_thr_0 = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
            try:
                true_id = g_path
                #print(true_id)
                #fp_dict[true_id] = inter_score
                #print(total_score)
                rdock_scores = rdock_scores.append({'POCKET_ID': g_path, 'INTER': str(inter_score)}, ignore_index=True)
            except:
                print(f">> failed on {g_path}")
                continue

    rdock_scores.to_csv('df_inter_score_train_mseloss_up0.csv')


def validate_robin_dataset(run, graph_dir, folds=1):
    compounds_info = pd.DataFrame(columns=['PDB_ID', 'SMILES', 'FINGERPRINT', 'TYPE','RNAMIGOS_SCORE'])
    df = pd.read_csv('data/annotated/consolidate_inter_labeled_dataset_qu_trans.csv')
    quantile_transformer = QuantileTransformer(output_distribution="normal")
    df['INTER_TRANS_2'] = pd.Series(quantile_transformer.fit_transform(np.array(df['INTER']).reshape(-1, 1))[:, 0])
    rdock_scores = pd.DataFrame(columns=['POCKET_ID', 'INTER_SCORE', 'PREDICTED_SCORE', 'INTER_SCORE_TRANS','PREDICTED_SCORE_TRANS','ELAPSED_TIME','ELAPSED_TIME_2'])
    true_label = []
    predicted_label = []
    dict_sm = pickle.load(open('/home/mcb/users/jcarva4/rnamigos2/rnamigos1/rnamigos_gcn/data/annotated/pockets_annotated_robin/compounds_fingerprints.p', 'rb'))
    #graph_dir = 'data/annotated/pockets_docking_annotated_cons_val'
    #graph_dir = 'data/annotated/pockets_docking_annotated_decoys'
    graph_dir = 'data/annotated/pockets_docking_annotated_inter_cleaned'
    fp_dict = {}
    pred_dict = {}
    res_model = []
    res_model_dict = {}
    for fold in range(int(folds)):
        model, meta = load_model(run + "_" + str(fold))
        # model, meta = load_model(run)
        edge_map = meta['edge_map']
        embed_dim = meta['embedding_dims'][-1]
        num_edge_types = len(edge_map)
        graph_ids = pickle.load(open(f'results/trained_models/{run}_{fold}/splits_{fold}.p', 'rb'))
        nucs=meta['nucs']
        test_graphlist=graph_ids['test']
        #test_graphlist = pickle.load(open('data/annotated/pockets_docking_annotated_decoys_list.p', 'rb'))
        #test_graphlist = pickle.load(open('data/annotated/pockets_docking_annotated_cons_val.p', 'rb'))
        test_graph_path=graph_dir
        nm = 0
        path = '/home/mcb/users/jcarva4/rnamigos2/rnamigos1/rnamigos_gcn/data/annotated/pockets_annotated_robin/'
        df = pd.read_csv(path + 'pockets_compounds_info.csv')
        graph, _, _, fp  = pickle.load(open(os.path.join(path, '7sxp_BIND.nx.p_annot.p'), 'rb'))
        #print(type(n1))
        #print(type(n2))
        #print(type(n3))
        #print(type(n4))
        for index, row in df.iterrows():
            nm += 1
            print('Ejemplo', str(nm))
            if row['PDB_ID'] == 'HIV_SL3_1bn0':
                graph, _, _, fp  = pickle.load(open(os.path.join(path, '1bn0_BIND.nx.p_annot.p'), 'rb'))
            if row['PDB_ID'] == 'HBV_2k5z':
                graph, _, _, fp  = pickle.load(open(os.path.join(path, '2k5z_BIND.nx.p_annot.p'), 'rb'))
            if row['PDB_ID'] == 'PreQ1_3gca':
                graph, _, _, fp  = pickle.load(open(os.path.join(path, '3gca_BIND.nx.p_annot.p'), 'rb'))
            if row['PDB_ID'] == 'NRAS_7sxp':
                graph, _, _, fp  = pickle.load(open(os.path.join(path, '7sxp_BIND.nx.p_annot.p'), 'rb'))
            if row['PDB_ID'] == 'TERRA_2m18':
                graph, _, _, fp  = pickle.load(open(os.path.join(path, '2m18_BIND.nx.p_annot.p'), 'rb'))
            
            nx_graph, dgl_graph = nx_to_dgl(graph, edge_map, nucs=nucs)
            with torch.no_grad():
                fp =dict_sm[row['SMILES']]
                #print('TYPE')
                #print(fp.shape)
                fp = torch.from_numpy(fp).float()
                fp = torch.unsqueeze(fp,0)
                #print(torch.from_numpy(fp).float().shape)
                fp_pred, _ = model(dgl_graph, fp)
                end_pred = time.time()
                #print('prediction ----')
                #print(fp_pred)
                fp_pred =fp_pred.detach().numpy()
                #end_pred_2 = time.time()
                #fp_pred = fp_pred.detach().numpy() > 0.5
                #fp_pred = fp_pred.astype(int)
                #pred_dict[true_id] = fp_pred
                #print(inter_score_trans)
                #print(fp_pred[0][0])
                ps = [fp_pred[0][0]]
                ps_ser = pd.Series(ps)
                pr_inver = pd.Series(quantile_transformer.inverse_transform(np.array(ps_ser).reshape(-1, 1))[:, 0])
                print(pr_inver[0])
                compounds_info = pd.concat([compounds_info, pd.DataFrame([{'PDB_ID': row['PDB_ID'], 'SMILES': row['SMILES'], 
                    'FINGERPRINT': row['FINGERPRINT'], 'TYPE': row['TYPE'], 'RNAMIGOS_SCORE': str(pr_inver[0])}])], ignore_index=True)
    
    compounds_info.to_csv(path + 'pockets_compounds_info_predictions.csv')




def validate_affinity_classification_model(run, graph_dir, folds=1):
    df = pd.read_csv('data/annotated/consolidate_inter_labeled_dataset_qu_trans.csv')
    quantile_transformer = QuantileTransformer(output_distribution="normal")
    df['INTER_TRANS_2'] = pd.Series(quantile_transformer.fit_transform(np.array(df['INTER']).reshape(-1, 1))[:, 0])
    rdock_scores = pd.DataFrame(columns=['POCKET_ID', 'INTER_SCORE', 'PREDICTED_SCORE', 'INTER_SCORE_TRANS','PREDICTED_SCORE_TRANS','ELAPSED_TIME','ELAPSED_TIME_2'])
    true_label = []
    predicted_label = []
    #graph_dir = 'data/annotated/pockets_docking_annotated_cons_val'
    #graph_dir = 'data/annotated/pockets_docking_annotated_decoys'
    graph_dir = 'data/annotated/pockets_docking_annotated_inter_cleaned'
    fp_dict = {}
    pred_dict = {}
    res_model = []
    res_model_dict = {}
    for fold in range(int(folds)):
        model, meta = load_model(run + "_" + str(fold))
        # model, meta = load_model(run)
        edge_map = meta['edge_map']
        embed_dim = meta['embedding_dims'][-1]
        num_edge_types = len(edge_map)

        graph_ids = pickle.load(open(f'results/trained_models/{run}_{fold}/splits_{fold}.p', 'rb'))
        nucs=meta['nucs']
        test_graphlist=graph_ids['test']
        #test_graphlist = pickle.load(open('data/annotated/pockets_docking_annotated_decoys_list.p', 'rb'))
        #test_graphlist = pickle.load(open('data/annotated/pockets_docking_annotated_cons_val.p', 'rb'))
        test_graph_path=graph_dir
        nm = 0            
        for g_path in test_graphlist:
            nm += 1
            print('Ejemplo', str(nm))
            p = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
            _, graph, _, ring, fp_nat, fp, inter_score, inter_score_trans, score_native_ligand, label_native_lig, label_1std, label_2std, label_thr_min30, label_thr_min17, label_thr_min12, label_thr_min8, label_thr_0, sample_type, is_native  = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
            #print(type(graph))
            try:
                true_id = g_path
                #print(true_id)
                fp_dict[true_id] = label_native_lig
                #print(total_score)
            except:
                print(f">> failed on {g_path}")
                continue

            start = time.time()
            nx_graph, dgl_graph = nx_to_dgl(graph, edge_map, nucs=nucs)
            with torch.no_grad():
                #if model.clustered:
                #    fp = fp.long()
                #fp = fp.to(device)
                #fp = np.array(fp)
                fp = torch.from_numpy(fp).float()
                fp = torch.unsqueeze(fp,0)
                #print(torch.from_numpy(fp).float().shape)
                fp_pred, _ = model(dgl_graph, fp)
                end_pred = time.time()
                #print('prediction ----')
                #print(fp_pred)
                
                fp_pred =fp_pred.detach().numpy()
                #end_pred_2 = time.time()
                #fp_pred = fp_pred.detach().numpy() > 0.5
                #fp_pred = fp_pred.astype(int)
                
                #pred_dict[true_id] = fp_pred

                #print(inter_score_trans)
                #print(fp_pred[0][0])
                ps = [fp_pred[0][0]]
                ps_ser = pd.Series(ps)
                pr_inver = pd.Series(quantile_transformer.inverse_transform(np.array(ps_ser).reshape(-1, 1))[:, 0])
                end_pred_2 = time.time()
                #print('--inver-pred---')
                #print(inter_score)
                #print(pr_inver[0])
                #r = [label_2std, fp_pred[0][0], total_score]
                #res_model.append(r)
                #res_model_dict[g_path] = res_model
                #true_label.append(label_native_lig)
                #print(fp_pred[0][0])
                #print(inter_score_trans)
                #predicted_label.append(fp_pred[0][0])
                #print(g_path)
                t1 = end_pred - start
                t2 = end_pred_2 - start
                #pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                rdock_scores = pd.concat([rdock_scores, pd.DataFrame([{'POCKET_ID': g_path,
                    'INTER_SCORE': str(inter_score),
                    'PREDICTED_SCORE': str(pr_inver[0]),
                    'INTER_SCORE_TRANS': str(inter_score_trans),
                    'PREDICTED_SCORE_TRANS': str(fp_pred[0][0]),
                    'ELAPSED_TIME':str(t1),
                    'ELAPSED_TIME_2':str(t2)}])], ignore_index=True)

    #print('true label')
    #print(true_label)
    #print('pred_label')
    #print(predicted_label)
    y = np.array(true_label)
    pred = np.array(predicted_label)
    #res_model_dict[g_path] = res_model
    #pickle.dump(true_label, open('true_inter_score_mseloss_up0.p', 'wb'))
    #pickle.dump(predicted_label, open('pred_inter_score_mseloss_up0.p', 'wb'))
    #pickle.dump(res_model_dict, open('label_2std_predictions_2.p', 'wb'))
    #pickle.dump(rdock_scores, open('df_inter_score_predictions_mseloss_up0.p', 'wb'))
    #rdock_scores.to_csv('df_inter_score_predictions_mseloss_cleaned_dataset.csv')
    #rdock_scores.to_csv('df_inter_score_predictions_msaloss_cleaned_dataset_elaptime_cc.csv')
    #rdock_scores.to_csv('pred_ABPN_affinity_prediction_model_final_1_msa_emb166_oplbce_testset.csv')
    rdock_scores.to_csv('pred_ABPN_affinity_prediction_model_inter_trans_msa_cons_ccan_testset.csv')
    #rdock_scores.to_csv('predictions_ABPN_affinity_class_model_consolidate_tetset.csv')
    #fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    #auc = metrics.auc(fpr, tpr)
    #print('auc')
    #print(auc)
    mse = ''
    return mse





def structure_scanning(pdb, ligname, graph, model, edge_map, embed_dim):
    """
        Given a PDB structure make a prediction for each residue in the structure:
            - chop the structure into candidate sites (for each residue get a sphere..)
            - convert residue neighbourhood into graph
            - get prediction from model for each
            - compare prediction to native ligand.
        :returns: `residue_preds` dictionary with residue id as key and fingerprint prediction as value.
    """
    from data_processor.build_dataset import get_pocket_graph

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("", pdb)[0]

    residue_preds = {}
    residues = list(structure.get_residues())
    for residue in tqdm(residues):
        if residue.resname in ['A', 'U', 'C', 'G', ligname]:
            res_info = ":".join(["_",residue.get_parent().id, residue.resname, str(residue.id[1])])
            pocket_graph = get_pocket_graph(pdb, res_info, graph)
            _,dgl_graph = nx_to_dgl(pocket_graph, edge_map, embed_dim)
            _,fp_pred= model(dgl_graph)
            fp_pred = fp_pred.detach().numpy() > 0.5
            residue_preds[(residue.get_parent().id, residue.id[1])] = fp_pred
        else:
            continue
    return residue_preds

def scanning_analyze():
    """
        Visualize results of scanning on PDB.
        Color residues by prediction score.
          1fmn_#0.1:A:FMN:36.nx_annot.p
    """
    from data_processor.build_dataset import find_residue,lig_center

    model, edge_map, embed_dim  = load_model('small_no_rec_2', '../data/annotated/pockets_nx')
    for f in os.listdir("../data/annotated/pockets_nx"):
        pdbid = f.split("_")[0]
        _,chain,ligname,pos = f.replace(".nx_annot.p", "").split(":")
        pos = int(pos)
        print(chain,ligname, pos)
        graph = pickle.load(open(f'../data/RNA_Graphs/{pdbid}.pickle', 'rb'))
        if len(graph.nodes()) > 100:
            continue
        try:
            fp_preds = structure_scanning(f'../data/all_rna_prot_lig_2019/{pdbid}.cif', ligname, graph, model, edge_map, embed_dim)
        except Exception as e:
            print(e)
            continue
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("", f"../data/all_rna_prot_lig_2019/{pdbid}.cif")[0]
        lig_res = find_residue(structure[chain], pos)
        lig_c = lig_center(lig_res.get_atoms())

        fp_dict = pickle.load(open("../data/all_ligs_maccs.p", 'rb'))
        true_fp = fp_dict[ligname]
        dists = []
        jaccards = []
        decoys = get_decoys()
        for res, fp in fp_preds.items():
            chain, pos = res
            r = find_residue(structure[chain], pos)
            r_center = lig_center(r.get_atoms())
            dists.append(euclidean(r_center, lig_c))
            jaccards.append(mse(true_fp, fp))
        plt.title(f)
        plt.distplot(dists, jaccards)
        plt.xlabel("dist to binding site")
        plt.ylabel("dist to fp")
        plt.show()
    pass

def structure_scanning(pdb, ligname, graph, model, edge_map, embed_dim):
    """
        Given a PDB structure make a prediction for each residue in the structure:
            - chop the structure into candidate sites (for each residue get a sphere..)
            - convert residue neighbourhood into graph
            - get prediction from model for each
            - compare prediction to native ligand.
        :returns: `residue_preds` dictionary with residue id as key and fingerprint prediction as value.
    """
    from data_processor.build_dataset import get_pocket_graph

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("", pdb)[0]

    residue_preds = {}
    residues = list(structure.get_residues())
    for residue in tqdm(residues):
        if residue.resname in ['A', 'U', 'C', 'G', ligname]:
            res_info = ":".join(["_",residue.get_parent().id, residue.resname, str(residue.id[1])])
            pocket_graph = get_pocket_graph(pdb, res_info, graph)
            _,dgl_graph = nx_to_dgl(pocket_graph, edge_map, embed_dim)
            _,fp_pred= model(dgl_graph)
            fp_pred = fp_pred.detach().numpy() > 0.5
            residue_preds[(residue.get_parent().id, residue.id[1])] = fp_pred
        else:
            continue
    return residue_preds

def scanning_analyze():
    """
        Visualize results of scanning on PDB.
        Color residues by prediction score.
          1fmn_#0.1:A:FMN:36.nx_annot.p
    """
    from data_processor.build_dataset import find_residue,lig_center

    model, edge_map, embed_dim  = load_model('small_no_rec_2', '../data/annotated/pockets_nx')
    for f in os.listdir("../data/annotated/pockets_nx"):
        pdbid = f.split("_")[0]
        _,chain,ligname,pos = f.replace(".nx_annot.p", "").split(":")
        pos = int(pos)
        print(chain,ligname, pos)
        graph = pickle.load(open(f'../data/RNA_Graphs/{pdbid}.pickle', 'rb'))
        if len(graph.nodes()) > 100:
            continue
        try:
            fp_preds = structure_scanning(f'../data/all_rna_prot_lig_2019/{pdbid}.cif', ligname, graph, model, edge_map, embed_dim)
        except Exception as e:
            print(e)
            continue
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("", f"../data/all_rna_prot_lig_2019/{pdbid}.cif")[0]
        lig_res = find_residue(structure[chain], pos)
        lig_c = lig_center(lig_res.get_atoms())

        fp_dict = pickle.load(open("../data/all_ligs_maccs.p", 'rb'))
        true_fp = fp_dict[ligname]
        dists = []
        jaccards = []
        decoys = get_decoys()
        for res, fp in fp_preds.items():
            chain, pos = res
            r = find_residue(structure[chain], pos)
            r_center = lig_center(r.get_atoms())
            dists.append(euclidean(r_center, lig_c))
            jaccards.append(mse(true_fp, fp))
        plt.title(f)
        plt.distplot(dists, jaccards)
        plt.xlabel("dist to binding site")
        plt.ylabel("dist to fp")
        plt.show()
    pass

if __name__ == "__main__":
    # scanning_analyze()
    # ablation_results()
    run = sys.argv[1]
    pocket_dir = sys.argv[2]
    mod = sys.argv[3]
    dec_mod = sys.argv[4]
    fol = sys.argv[5]
    #ablation_results(run, pocket_dir, mod, dec_mod, fol)
    #validate_affinity_classification_model(run, pocket_dir, fol)
    validate_robin_dataset(run, pocket_dir, fol)
    #summ_train_data(run, pocket_dir, fol)
    #summ_complete_dataset()
    #get_lig_decoys(decoy_mode='dude')
    """    
    y = pickle.load(open('true_score.p', 'rb'))
    pred = pickle.load(open('pred_score.p', 'rb'))
    print(len(y))
    print(type(y))

    sum1_y = 0
    ny = []
    for l in y:
        ny.append(l)
        if l == 1:
            sum1_y += 1
        
    r_pred = pred
    """
    """
    r_pred = []
    num_1 = 0
    
    for i in pred:
        if i[0] == 1:
            num_1 += 1
        r_pred.append(i[0])
    
    r_pred = np.array(r_pred)
    """
    """
    nr = 0
    sum_nr = 0
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for i in y:
        if y[nr] == 1 and  r_pred[nr] == 1:
            tp += 1
            
        if y[nr] == 1 and r_pred[nr] == 0:
            fn += 1

        if y[nr] == 0 and r_pred[nr] == 0:
            tn += 1

        if y[nr] == 0 and r_pred[nr] == 1:
            fp += 1

        nr +=1
    print(tp)
    print(fn)
    print(tn)
    print(fp)

    tpr = tp / (tp + fn)
    print(tpr)

    fpr = fp / (fp + tn)
    print(fpr)
    from sklearn.metrics import roc_auc_score, confusion_matrix
    auc = roc_auc_score(y, pred)
    #fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    #auc = metrics.auc(fpr, tpr)
    print(auc)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(tp)
    print(fn)
    print(tn)
    print(fp)
    """

