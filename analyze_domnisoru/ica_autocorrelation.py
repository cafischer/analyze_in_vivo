import numpy as np
import matplotlib.pyplot as pl
import os
from sklearn.cluster import KMeans
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from matplotlib.pyplot import Line2D
from matplotlib.patches import Patch
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.spike_time_autocorrelation import plot_autocorrelation
from sklearn.decomposition import FastICA
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_auto_corr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    #save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    cell_type = 'grid_cells'
    save_dir_img = os.path.join(save_dir_auto_corr, cell_type)
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    max_lag = 50.0
    bin_width = 1.0  # ms
    sigma_smooth = None
    dt_kernel = 0.05
    if sigma_smooth is not None:
        auto_corr_cells = np.load(
            os.path.join(save_dir_auto_corr, cell_type,
                         'autocorr_' + str(max_lag) + '_' + str(bin_width) + '_' + str(sigma_smooth) + '.npy'))
    else:
        auto_corr_cells = np.load(os.path.join(save_dir_auto_corr, cell_type,
                                               'autocorr_' + str(max_lag) +'_' + str(bin_width) + '.npy'))
    max_lag_idx = to_idx(max_lag, bin_width)
    if sigma_smooth is not None:
        t_auto_corr = np.arange(-max_lag_idx, max_lag_idx + dt_kernel, dt_kernel)
    else:
        t_auto_corr = np.arange(-max_lag_idx, max_lag_idx + bin_width, bin_width)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()

    # ICA
    auto_corr_cells_centered = auto_corr_cells - np.mean(auto_corr_cells, 0)
    ica = FastICA(n_components=2, random_state=11)
    transformed = ica.fit_transform(auto_corr_cells_centered)  # Reconstruct signals

    # plot components
    fig, ax = pl.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(t_auto_corr, ica.components_[0, :], bin_width, color='k', align='center')
    ax[1].bar(t_auto_corr, ica.components_[1, :], bin_width, color='gray', align='center')
    ax[0].set_xlabel('Lag (ms)')
    ax[1].set_xlabel('Lag (ms)')
    ax[0].set_ylabel('IC1')
    ax[1].set_ylabel('IC2')
    ax[0].set_xticks([-max_lag, 0, max_lag])
    ax[1].set_xticks([-max_lag, 0, max_lag])
    pl.tight_layout()
    if sigma_smooth is not None:
        pl.savefig(os.path.join(save_dir_img, 'ica_autocorrelation_PCs_' + str(max_lag) + '_' + str(
            bin_width) + '_' + str(sigma_smooth) + '.png'))
    else:
        pl.savefig(
            os.path.join(save_dir_img, 'ica_autocorrelation_PCs_' + str(max_lag) + '_' + str(bin_width) + '.png'))

    # ica backtransform
    from analyze_in_vivo.load.load_domnisoru import get_cell_ids_bursty
    back_transformed = ica.inverse_transform(transformed)  #np.dot(transformed, ica.components_[:2, :])
    back_transformed += np.mean(auto_corr_cells, axis=0)
    plot_kwargs = dict(t_auto_corr=t_auto_corr, auto_corr_cells=back_transformed, bin_size=bin_width,
                       max_lag=max_lag)
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
                            save_dir_img=os.path.join(save_dir_img, 'ica_backtransformed_autocorr_' + str(max_lag) + '_' + str(
                                bin_width) + '_' + str(sigma_smooth) + '.png'))

    # k-means
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(transformed)
    labels = kmeans.labels_

    print('Bursty: ', cell_ids[labels.astype(bool)])
    print('Non-bursty: ', cell_ids[~labels.astype(bool)])

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


    # # plot for thesis
    # fig, ax = pl.subplots(figsize=(8, 5.5))
    #
    # left, bottom, width, height = [0.5, 0.75, 0.2, 0.2]
    # axins = fig.add_axes([left, bottom, width, height])
    # i = np.where(cell_ids == 's79_0003')[0][0]
    # axins.bar(t_auto_corr, auto_corr_cells[i], bin_width, color='r', align='center')
    # ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(left, bottom),
    #                xycoords='data', textcoords='figure fraction',
    #                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    # ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(left+width, bottom),
    #                xycoords='data', textcoords='figure fraction',
    #                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    # axins.set_yticks([])
    # axins.set_xticks([-50, 0, 50])
    # axins.set_xticklabels([-50, 0, 50], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    #
    # left, bottom, width, height = [0.2, 0.75, 0.2, 0.2]
    # axins = fig.add_axes([left, bottom, width, height])
    # i = np.where(cell_ids == 's85_0007')[0][0]
    # axins.bar(t_auto_corr, auto_corr_cells[i], bin_width, color='b', align='center')
    # ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(left, bottom),
    #                xycoords='data', textcoords='figure fraction',
    #                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    # ax.annotate('', xy=(transformed[i, 0], transformed[i, 1]), xytext=(left+width, bottom),
    #                xycoords='data', textcoords='figure fraction',
    #                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    # axins.set_yticks([])
    # axins.set_xticks([-50, 0, 50])
    # axins.set_xticklabels([-50, 0, 50], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    #
    # plot_with_markers(ax, transformed[labels == 0, 0], transformed[labels == 0, 1], cell_ids[labels == 0],
    #                   cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
    #                   DAP_cells_additional=DAP_cells_additional, legend=False)
    # handles = plot_with_markers(ax, transformed[labels == 1, 0], transformed[labels == 1, 1], cell_ids[labels == 1],
    #                   cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells,
    #                             DAP_cells_additional=DAP_cells_additional, legend=False)
    # handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    # legend1 = ax.legend(handles=handles+handles_bursty, bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    # ax.add_artist(legend1)
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    #
    # ax.set_ylim([-0.07, 0.19])
    # ax.set_xlim([-0.07, 0.135])
    # pl.tight_layout()
    # pl.subplots_adjust(top=0.7, bottom=0.08, left=0.1, right=0.77)
    # pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation.png'))


    # plot for Andreas
    fig, ax = pl.subplots(figsize=(12., 8.))
    plot_with_markers(ax, transformed[labels == 0, 0], transformed[labels == 0, 1], cell_ids[labels == 0],
                      cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
    handles = plot_with_markers(ax, transformed[labels == 1, 0], transformed[labels == 1, 1], cell_ids[labels == 1],
                      cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    #legend1 = ax.legend(handles=handles+handles_bursty, bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    legend1 = ax.legend(handles=handles + handles_bursty)
    ax.add_artist(legend1)
    ax.set_xlabel('IC1')
    ax.set_ylabel('IC2')

    for i in range(len(cell_ids)):
        ax.annotate(cell_ids[i], xy=(transformed[i, 0] + 0.002, transformed[i, 1] + 0.002), fontsize=7)

    #ax.set_ylim([-0.058, 0.17])
    #ax.set_xlim([-0.065, 0.133])
    pl.tight_layout()
    #pl.subplots_adjust(bottom=0.06, left=0.06, top=0.98, right=0.84)
    if sigma_smooth is not None:
        pl.savefig(os.path.join(save_dir_img,
                                'ica_autocorrelation_with_cell_ids_' + str(max_lag) + '_' + str(
                                    bin_width) + '_' + str(sigma_smooth) + '.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'ica_autocorrelation_with_cell_ids_' + str(max_lag) +'_' + str(
            bin_width) + '.png'))

    # # plot for slides
    # fig, ax = pl.subplots(figsize=(8, 5.5))
    # plot_with_markers(ax, transformed[labels == 0, 0], transformed[labels == 0, 1], cell_ids[labels == 0],
    #                   cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
    #                   DAP_cells_additional=DAP_cells_additional, legend=False)
    # handles = plot_with_markers(ax, transformed[labels == 1, 0], transformed[labels == 1, 1], cell_ids[labels == 1],
    #                             cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells,
    #                             DAP_cells_additional=DAP_cells_additional, legend=False)
    # handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    # legend1 = ax.legend(handles=handles+handles_bursty, loc='upper right')
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    #
    # # ax.set_ylim([-0.058, 0.17])
    # # ax.set_xlim([-0.065, 0.133])
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation_no_examples.png'))

    pl.show()