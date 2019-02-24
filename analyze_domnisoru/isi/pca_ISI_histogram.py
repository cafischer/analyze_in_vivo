import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.isi.ISI_hist import get_ISI_hist_peak_width
from matplotlib.patches import Patch
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pandas as pd
pl.style.use('paper_subplots')


def plot_ISI_hist_on_ax(ax, cell_idx, ISI_hist_cells, bin_width, max_ISI):
    bins = np.arange(0, max_ISI+bin_width, bin_width)
    ax.bar(bins[:len(ISI_hist_cells[cell_idx, :])],
           ISI_hist_cells[cell_idx, :] / (np.sum(ISI_hist_cells[cell_idx, :]) * bin_width),
           bins[1] - bins[0], color='0.5', align='edge')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    cell_type = 'grid_cells'
    save_dir_img = os.path.join(save_dir_ISI_hist, cell_type)
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    max_ISI = 200
    bin_width = 1  # ms
    sigma_smooth = 1
    dt_kernel = 0.05
    remove_cells = True
    remove_cells_dict = {True: 'removed', False: 'not_removed'}
    if sigma_smooth is not None:
        ISI_hist_cells = np.load(
            os.path.join(save_dir_ISI_hist, 'cut_ISIs_at_'+str(max_ISI), cell_type,
                         'ISI_hist_' + str(max_ISI) + '_' + str(bin_width) + '_' + str(sigma_smooth) + '.npy'))
    else:
        ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, 'cut_ISIs_at_'+str(max_ISI), cell_type,
                                               'ISI_hist_' + str(max_ISI) + '_' + str(bin_width) + '.npy'))
    for cell_idx in range(len(cell_ids)):  # normalize
        ISI_hist_cells[cell_idx] = ISI_hist_cells[cell_idx] / (np.sum(ISI_hist_cells[cell_idx]) * bin_width)
    max_ISI_idx = to_idx(max_ISI, bin_width)
    if sigma_smooth is not None:
        t_ISI_hist = np.arange(0, max_ISI_idx + dt_kernel, dt_kernel)
    else:
        t_ISI_hist = np.arange(0, max_ISI_idx, bin_width)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()

    # PCA
    if remove_cells:  # take out autocorrelation for cell s104_0007 and s110_0002
        idx_s104_0007 = np.where(np.array(cell_ids) == 's104_0007')[0][0]
        idx_s110_0002 = np.where(np.array(cell_ids) == 's110_0002')[0][0]
        idxs = range(len(cell_ids))
        idxs.remove(idx_s104_0007)
        idxs.remove(idx_s110_0002)
        ISI_hist_cells_for_pca = ISI_hist_cells[np.array(idxs)]
    else:
        ISI_hist_cells_for_pca = ISI_hist_cells
    ISI_hist_cells_centered = ISI_hist_cells - np.mean(ISI_hist_cells_for_pca, 0)
    pca = PCA(n_components=2)
    pca.fit(ISI_hist_cells_for_pca)
    transformed = pca.transform(ISI_hist_cells_centered)

    print 'Explained variance: %.2f, %.2f' % (pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1])

    fig, ax = pl.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(t_ISI_hist, pca.components_[0, :], bin_width, color='k', align='center',
              label='Explained var.: %i' % np.round(pca.explained_variance_ratio_[0]*100)+'%')
    ax[1].bar(t_ISI_hist, pca.components_[1, :], bin_width, color='gray', align='center',
              label='Explained var.: %i' % np.round(pca.explained_variance_ratio_[1]*100)+'%')
    ax[0].set_xlabel('ISI (ms)')
    ax[1].set_xlabel('ISI (ms)')
    ax[0].set_ylabel('PC1')
    ax[1].set_ylabel('PC2')
    ax[0].set_xticks([0, max_ISI])
    ax[1].set_xticks([0, max_ISI])
    ax[0].legend()
    ax[1].legend()
    pl.tight_layout()
    if sigma_smooth is not None:
        pl.savefig(os.path.join(save_dir_img, 'pca_ISI_hist_PCs_' + str(max_ISI) + '_' + str(
            bin_width) + '_' + str(sigma_smooth) + '_' + remove_cells_dict[remove_cells] + '.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'pca_ISI_hist_PCs_' + str(max_ISI) + '_' + str(
            bin_width) + '_' + remove_cells_dict[remove_cells] + '.png'))
    # pl.show()

    # pca backtransform
    back_transformed = np.dot(transformed, pca.components_[:2, :])
    back_transformed += np.mean(ISI_hist_cells_for_pca, axis=0)
    if sigma_smooth is not None:
        plot_kwargs = dict(ISI_hist_cells=back_transformed, bin_width=dt_kernel, max_ISI=max_ISI)
    else:
        plot_kwargs = dict(ISI_hist_cells=back_transformed, bin_width=bin_width, max_ISI=max_ISI)
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
                            xlabel='Time (ms)', ylabel='ISI histogram', colors_marker=colors_marker,
                            save_dir_img=os.path.join(save_dir_img, 'backtransformed_ISI_hist_' + str(
                                max_ISI) + '_' + str(bin_width) + '_' + str(
                                sigma_smooth) + '_' + remove_cells_dict[remove_cells] + '.png'))

    if sigma_smooth is not None:
        peak_ISI_hist = np.zeros(len(cell_ids))
        width_at_half_max = np.zeros(len(cell_ids))
        for cell_idx, cell_id in enumerate(cell_ids):
            peak_ISI_hist[cell_idx], width_at_half_max[cell_idx] = get_ISI_hist_peak_width(back_transformed[cell_idx],
                                                                                           np.arange(0, max_ISI+dt_kernel, dt_kernel))

        burst_row = ['B' if l else 'N-B' for l in burst_label]
        df = pd.DataFrame(data=np.vstack((peak_ISI_hist, width_at_half_max, burst_row)).T,
                          columns=['ISI peak', 'ISI width', 'burst behavior'], index=cell_ids)
        df.index.name = 'Cell ids'
        df.to_csv(os.path.join(save_dir_img, 'ISI_distribution_bt_' + str(max_ISI) + '_' + str(bin_width) + '_' + str(
            sigma_smooth) + '.csv'))


    # k-means
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(transformed)
    labels = kmeans.labels_
    #labels = burst_label  # TODO

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
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    for i in range(len(cell_ids)):
        ax.annotate(cell_ids[i], xy=(transformed[i, 0] + 0.003, transformed[i, 1] + 0.003), fontsize=7)

    #ax.set_ylim([-0.058, 0.17])
    #ax.set_xlim([-0.065, 0.133])

    # # inset
    # axins = inset_axes(ax, width='50%', height='50%', loc='center')
    # plot_with_markers(axins, transformed[labels == 0, 0], transformed[labels == 0, 1], cell_ids[labels == 0],
    #                   cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
    # plot_with_markers(axins, transformed[labels == 1, 0], transformed[labels == 1, 1], cell_ids[labels == 1],
    #                   cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
    # for i in range(len(cell_ids)):
    #     axins.annotate(cell_ids[i], xy=(transformed[i, 0] + 4, transformed[i, 1] + 2), fontsize=7)
    # axins.set_ylim(-230, 100)
    # axins.set_xlim(-450, -100)
    # axins.spines['top'].set_visible(True)
    # axins.spines['right'].set_visible(True)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    pl.tight_layout()
    #pl.subplots_adjust(bottom=0.06, left=0.06, top=0.98, right=0.84)
    if sigma_smooth is not None:
        pl.savefig(os.path.join(save_dir_img,
                                'pca_autocorrelation_with_cell_ids_' + str(max_ISI) + '_' + str(
                                    bin_width) + '_' + str(sigma_smooth) + '_' + remove_cells_dict[remove_cells] + '.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'pca_ISI_hist_with_cell_ids_' + str(max_ISI) + '_' + str(
            bin_width) + '_' + remove_cells_dict[remove_cells] + '.png'))

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