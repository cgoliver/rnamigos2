"""
    Cluster ligands in dictionary to obtain ligand classes for training.
"""

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


def ligands_cluster(bs_dict, fp_dict, n_clusters=8):
    """
        Assign cluster labels to each ligand in ligand_list.

        Create new fingerprint dictionary {'lig_id': cluster_id}

    """
    # get which ligands to use in clustering
    binding_sites = pickle.load(open(bs_dict, 'rb'))
    fingerprints = pickle.load(open(fp_dict, 'rb'))
    ligs_2_cluster = []
    for _, ligs in binding_sites.items():
        pocket_ids = [f.split(":")[2] for f in ligs]
        ligs_2_cluster.extend(pocket_ids)
    # ligs_2_cluster_unique = list(set(ligs_2_cluster))

    fps = []
    for l in ligs_2_cluster:
        try:
            fps.append(fingerprints[l])
        except:
            print(l)

    # do the clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    clusterer.fit(fps)
    labels = clusterer.labels_
    clustered_fp_dict = dict(zip(ligs_2_cluster, labels))
    sns.distplot(labels)
    plt.show()

    return clustered_fp_dict
    pass


if __name__ == "__main__":
    clustered_fp_dict = ligands_cluster("../data/lig_dict_c_8A_06rna.p", "../data/all_ligs_maccs.p")
    pickle.dump(clustered_fp_dict, open("../data/fp_dict_8clusters.p", 'wb'))
