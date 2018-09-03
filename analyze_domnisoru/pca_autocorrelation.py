import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_auto_corr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    max_lag = 50
    bin_size = 1.0  # ms
    auto_corr_cells = np.load(os.path.join(save_dir_auto_corr, cell_type, 'auto_corr_'+str(max_lag)+'.npy'))
    max_lag_idx = to_idx(max_lag, bin_size)
    t_auto_corr = np.concatenate((np.arange(-max_lag_idx, 0, 1), np.arange(0, max_lag_idx + 1, 1))) * bin_size

    # PCA
    auto_corr_cells_centered = auto_corr_cells - np.mean(auto_corr_cells, 0)
    pca = PCA(n_components=2)
    pca.fit(auto_corr_cells_centered)
    transformed = pca.transform(auto_corr_cells_centered)
    print('Explained variance: ', pca.explained_variance_)

    # #
    # Z = linkage(transformed, method='centroid')
    # dendrogram(Z)

    # k-means
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(transformed)
    labels = kmeans.labels_

    print('Bursty: ', cell_ids[labels])
    print('Non-bursty: ', cell_ids[~labels])

    # pl.figure()
    # ax = pl.subplot(projection='3d')
    # ax.plot(transformed[labels==0, 0], transformed[labels==0, 1], transformed[labels==0, 2], 'or')
    # ax.plot(transformed[labels==1, 0], transformed[labels==1, 1], transformed[labels==1, 2], 'ob')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')

    pl.figure()
    ax = pl.subplot()
    # for label in range(n_clusters):
    #     ax.plot(transformed[labels==label, 0], transformed[labels==label, 1], 'o')
    ax.plot(transformed[labels==0, 0], transformed[labels==0, 1], 'or')
    ax.plot(transformed[labels==1, 0], transformed[labels==1, 1], 'ob')
    for i in range(len(cell_ids)):
        ax.annotate(cell_ids[i], xy=(transformed[i, 0], transformed[i, 1]))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # pl.figure()
    # for i in np.where(labels)[0]:
    #     pl.bar(t_auto_corr, auto_corr_cells[i], bin_size, color='r', align='center', alpha=0.5)
    # for i in np.where(~labels)[0]:
    #     pl.bar(t_auto_corr, auto_corr_cells[i], bin_size, color='b', align='center', alpha=0.5)
    # # for label in range(n_clusters):
    # #     idx = np.where(labels==label)[0][0]
    # #     pl.bar(t_auto_corr, auto_corr_cells[idx], bin_size, align='center', alpha=0.5)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Spike-time auto-correlation')


    # plot for thesis
    fig, ax = pl.subplots()
    ax.plot(transformed[labels == 0, 0], transformed[labels == 0, 1], 'or', label='Non-bursty')
    ax.plot(transformed[labels == 1, 0], transformed[labels == 1, 1], 'ob', label='Bursty')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    left, bottom, width, height = [0.75, 0.7, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    i = np.where(cell_ids == 's79_0003')[0][0]
    axins.bar(t_auto_corr, auto_corr_cells[i], bin_size, color='b', align='center')
    ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(0.75, 0.7),
                   xycoords='data', textcoords='figure fraction',
                   arrowprops=dict(arrowstyle="-", color='0.5', alpha=0.5))
    ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(0.95, 0.7),
                   xycoords='data', textcoords='figure fraction',
                   arrowprops=dict(arrowstyle="-", color='0.5', alpha=0.5))
    axins.set_yticks([])
    axins.set_xticks([-50, 0, 50])
    axins.set_xticklabels([-50, 0, 50], fontsize=10)
    axins.set_xlabel('Lag (ms)', fontsize=10)
    axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)

    left, bottom, width, height = [0.25, 0.7, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    i = np.where(cell_ids == 's85_0007')[0][0]
    axins.bar(t_auto_corr, auto_corr_cells[i], bin_size, color='r', align='center')
    ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(0.25, 0.7),
                   xycoords='data', textcoords='figure fraction',
                   arrowprops=dict(arrowstyle="-", color='0.5', alpha=0.5))
    ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(0.45, 0.7),
                   xycoords='data', textcoords='figure fraction',
                   arrowprops=dict(arrowstyle="-", color='0.5', alpha=0.5))
    axins.set_yticks([])
    axins.set_xticks([-50, 0, 50])
    axins.set_xticklabels([-50, 0, 50], fontsize=10)
    axins.set_xlabel('Lag (ms)', fontsize=10)
    axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)

    ax.legend(loc='lower right')
    ax.set_ylim([-0.12, 0.18])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation.png'))

    pl.show()