from sklearn.decomposition import PCA
import scipy as sc
import scipy.spatial
import scipy.cluster
import numpy as np
from collections import defaultdict


def perform_PCA(data_centered, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data_centered)
    projection = pca.transform(data_centered)
    return projection, pca.components_, pca.explained_variance_ratio_


def ward_clustering(x, x_labels):
    dist_mat = sc.spatial.distance.pdist(x, metric='euclidean')
    linkage = sc.cluster.hierarchy.linkage(dist_mat, method='ward')
    sc.cluster.hierarchy.set_link_color_palette(['b', 'r', 'g'])
    dend = sc.cluster.hierarchy.dendrogram(linkage, labels=x_labels, above_threshold_color="grey")
    labels, _ = get_cluster_classes(dend, x_labels, label='ivl')
    return labels, dend


def get_cluster_classes(dend, cell_ids, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(dend['color_list'], dend['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    color_label_dict = {'b': 0, 'g': 1, 'r': 2}
    cluster_labels_dict = {}
    for c, l in cluster_idxs.items():
        for i in l:
            cluster_labels_dict[dend[label][i]] = color_label_dict[c]

    cluster_labels = np.array([cluster_labels_dict[cell_id] for cell_id in cell_ids])
    return cluster_labels, color_label_dict