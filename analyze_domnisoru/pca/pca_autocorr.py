import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.autocorr.spiketime_autocorr import plot_autocorrelation
from analyze_in_vivo.analyze_domnisoru.pca import perform_PCA
#pl.style.use('paper_subplots')


def plot_PCs(n_components, x, PCs, explained_var, max_lag, bin_width, save_dir_img=None):
    fig, ax = pl.subplots(1, n_components, figsize=(10, 4))
    for n_component in range(n_components):
        ax[n_component].bar(x, PCs[n_component, :], bin_width, color='k', align='center',
                            label='Explained var.: %i' % np.round(explained_var[n_component] * 100) + '%')
        ax[n_component].set_xlabel('Lag (ms)')
        ax[n_component].set_ylabel('PC' + str(n_component + 1))
        ax[n_component].set_xticks([-max_lag, 0, max_lag])
        ax[n_component].legend()
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'PCs_' + str(max_lag) + '_' + str(bin_width) + '_' + str(
                 sigma_smooth) + '_' + remove_cells_dict[remove_cells] + '.png'))


def plot_pca_projection_for_thesis(save_dir_img=None):
    fig, ax = pl.subplots(figsize=(8, 5.5))
    left, bottom, width, height = [0.5, 0.75, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    i = np.where(cell_ids == 's79_0003')[0][0]
    axins.bar(t_autocorr, autocorr_cells[i], bin_width, color='r', align='center')
    ax.annotate('', xy=(projected[i, 0], projected[i, 1]), xytext=(left, bottom),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    ax.annotate('', xy=(projected[i, 0], projected[i, 1]), xytext=(left + width, bottom),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    axins.set_yticks([])
    axins.set_xticks([-50, 0, 50])
    axins.set_xticklabels([-50, 0, 50], fontsize=10)
    axins.set_xlabel('Lag (ms)', fontsize=10)
    axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    left, bottom, width, height = [0.2, 0.75, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    i = np.where(cell_ids == 's85_0007')[0][0]
    axins.bar(t_autocorr, autocorr_cells[i], bin_width, color='b', align='center')
    ax.annotate('', xy=(projected[i, 0], projected[i, 1]), xytext=(left, bottom),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    ax.annotate('', xy=(projected[i, 0], projected[i, 1]), xytext=(left + width, bottom),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    axins.set_yticks([])
    axins.set_xticks([-50, 0, 50])
    axins.set_xticklabels([-50, 0, 50], fontsize=10)
    axins.set_xlabel('Lag (ms)', fontsize=10)
    axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    plot_with_markers(ax, projected[labels == 0, 0], projected[labels == 0, 1], cell_ids[labels == 0],
                      cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    handles = plot_with_markers(ax, projected[labels == 1, 0], projected[labels == 1, 1], cell_ids[labels == 1],
                                cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                DAP_cells_additional=DAP_cells_additional, legend=False)
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    legend1 = ax.legend(handles=handles + handles_bursty, bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
    ax.add_artist(legend1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_ylim([-0.07, 0.19])
    ax.set_xlim([-0.07, 0.135])
    pl.tight_layout()
    pl.subplots_adjust(top=0.7, bottom=0.08, left=0.1, right=0.77)
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation.png'))


def plot_pca_projection_with_cell_ids(save_dir_img=None):
    fig, ax = pl.subplots(figsize=(12., 8.))
    plot_with_markers(ax, projected[labels == 0, 0], projected[labels == 0, 1], cell_ids[labels == 0],
                      cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
    handles = plot_with_markers(ax, projected[labels == 1, 0], projected[labels == 1, 1], cell_ids[labels == 1],
                                cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                legend=False)
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    legend1 = ax.legend(handles=handles + handles_bursty)
    ax.add_artist(legend1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    for i in range(len(cell_ids)):
        ax.annotate(cell_ids[i], xy=(projected[i, 0] + 0.002, projected[i, 1] + 0.002), fontsize=7)
    # ax.set_ylim([-0.058, 0.17])
    # ax.set_xlim([-0.065, 0.133])
    pl.tight_layout()
    # pl.subplots_adjust(bottom=0.06, left=0.06, top=0.98, right=0.84)
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img,
                                'pca_autocorrelation_with_cell_ids_' + str(max_lag) + '_' + str(
                                    bin_width) + '_' + str(sigma_smooth) + '_' + remove_cells_dict[
                                    remove_cells] + '.png'))


def plot_pca_projection_3d_with_cell_ids(save_dir_img=None):
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_with_markers(ax, projected[labels == 0, 0], projected[labels == 0, 1], cell_ids[labels == 0],
                      cell_type_dict, z=projected[labels == 0, 2], edgecolor='b', theta_cells=theta_cells,
                      DAP_cells=DAP_cells, legend=False)
    handles = plot_with_markers(ax, projected[labels == 1, 0], projected[labels == 1, 1], cell_ids[labels == 1],
                                cell_type_dict, z=projected[labels == 1, 2], edgecolor='r', theta_cells=theta_cells,
                                DAP_cells=DAP_cells, legend=False)
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    legend1 = ax.legend(handles=handles + handles_bursty)
    ax.add_artist(legend1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    for i in range(len(cell_ids)):
        ax.text(projected[i, 0] + 0.002, projected[i, 1] + 0.002, projected[i, 2] + 0.002, cell_ids[i], size=7,
                zorder=1, color='k')

    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img,
                                'pca_autocorrelation_with_cell_ids_3d_' + str(max_lag) + '_' + str(
                                    bin_width) + '_' + str(sigma_smooth) + '_' + remove_cells_dict[
                                    remove_cells] + '.png'))


def plot_pca_projection_slides(save_dir_img=None):
    fig, ax = pl.subplots(figsize=(8, 5.5))
    plot_with_markers(ax, projected[labels == 0, 0], projected[labels == 0, 1], cell_ids[labels == 0],
                      cell_type_dict, edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    handles = plot_with_markers(ax, projected[labels == 1, 0], projected[labels == 1, 1], cell_ids[labels == 1],
                                cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                DAP_cells_additional=DAP_cells_additional, legend=False)
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    ax.legend(handles=handles + handles_bursty, loc='upper right')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation_no_examples.png'))


def plot_pca_projection_for_paper(save_dir_img):
    fig = pl.figure(figsize=(7, 7))
    outer = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.27)

    # upper plot
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)

    # # example 1
    # axins = inset_axes(ax, width='20%', height='20%', loc='upper left', bbox_to_anchor=(0.13, 0, 1, 1),
    #                    bbox_transform=ax.transAxes)
    # i = np.where(cell_ids == 's84_0002')[0][0]
    # axins.bar(t_autocorr, autocorr_cells[i], bin_width, color='b', align='center')
    # #axins.set_yticks([])
    # #axins.set_ylim(0, 0.03)
    # axins.set_xticks([-max_lag_for_pca, 0, max_lag_for_pca])
    # axins.set_xticklabels([-max_lag_for_pca, 0, max_lag_for_pca], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    #
    # # example 2
    # axins = inset_axes(ax, width='20%', height='20%', loc='upper right')  # bbox_to_anchor=(0.7, 0.7, 1.0, 1.0)
    # i = np.where(cell_ids == 's109_0002')[0][0]
    # axins.bar(t_autocorr, autocorr_cells[i], bin_width, color='r', align='center')
    # #axins.set_yticks([])
    # #axins.set_ylim(0, 0.03)
    # axins.set_xticks([-max_lag_for_pca, 0, max_lag_for_pca])
    # axins.set_xticklabels([-max_lag_for_pca, 0, max_lag_for_pca], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    #
    # # example 3
    # axins = inset_axes(ax, width='20%', height='20%', loc='center right')  # bbox_to_anchor=(0.7, 0.7, 1.0, 1.0)
    # i = np.where(cell_ids == 's76_0002')[0][0]
    # axins.bar(t_autocorr, autocorr_cells[i], bin_width, color='r', align='center')
    # #axins.set_yticks([])
    # #axins.set_ylim(0, 0.1)
    # axins.set_xticks([-max_lag_for_pca, 0, max_lag_for_pca])
    # axins.set_xticklabels([-max_lag_for_pca, 0, max_lag_for_pca], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)

    # outside plot
    plot_with_markers(ax, projected[labels == 0, 0], projected[labels == 0, 1], cell_ids[labels == 0],
                      cell_type_dict, edgecolor='b', DAP_cells=DAP_cells_new, legend=False)
    handles = plot_with_markers(ax, projected[labels == 1, 0], projected[labels == 1, 1], cell_ids[labels == 1],
                                cell_type_dict, edgecolor='r', DAP_cells=DAP_cells_new, legend=False)
    plot_with_markers(ax, projected[labels == 2, 0], projected[labels == 2, 1], cell_ids[labels == 2],
                      cell_type_dict, edgecolor='g', DAP_cells=DAP_cells_new, legend=False)  # TODO
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    for cell_idx, cell_id in enumerate(cell_ids):
        ax.annotate(cell_id, xy=(projected[cell_idx, 0], projected[cell_idx, 1]), fontsize=8)

    # lower plot
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1, 1, 0.5], wspace=0.5)
    for n_component in range(n_components):
        ax = pl.Subplot(fig, inner[n_component])
        fig.add_subplot(ax)
        ax.bar(t_autocorr, components[n_component, :], bin_width, color='k', align='center')
        ax.annotate('Explained var.: %i' % np.round(
                                explained_var[n_component] * 100) + '%',
                    xy=(0.05, 1.1), xycoords='axes fraction', ha='left', va='top', fontsize=10,  # xy=((0.05, 0.96))
                    bbox=dict(boxstyle='round', fc='w', alpha=0.2))
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('PC' + str(n_component + 1))
        ax.set_xticks([-max_lag_for_pca, 0, max_lag_for_pca])

    # legend from upper plot
    ax = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax)
    legend1 = ax.legend(handles=handles + handles_bursty, loc='upper right')
    ax.add_artist(legend1)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # other stuff
    pl.tight_layout()
    pl.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.98)
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation_paper_' + str(max_lag) + '_' + str(
            max_lag_for_pca) + '_' + str(bin_width) + '_' + str(sigma_smooth) + '_' + remove_cells_dict[
            remove_cells] + '.png'))


def plot_backtransformed(save_dir_img=None):
    plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells=back_projected, bin_size=bin_width,
                       max_lag=max_lag)
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'
    if save_dir_img is not None:
        save_dir_img_file = os.path.join(save_dir_img, 'backtransformed_autocorr_' + str(max_lag) + '_' + str(
                                                          bin_width) + '_' + str(sigma_smooth) + '_' +
                                                          remove_cells_dict[remove_cells] + '.png')
    else:
        save_dir_img_file = None
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_autocorrelation, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
                            save_dir_img=save_dir_img_file)

    def plot_2autocorrelations(ax, cell_idx, t_auto_corr, auto_corr_cells1, auto_corr_cells2, error, bin_size, max_lag):
        ax.bar(t_auto_corr, auto_corr_cells1[cell_idx], bin_size, color='k', align='center', alpha=0.5)
        ax.bar(t_auto_corr, auto_corr_cells2[cell_idx], bin_size, color='orange', align='center', alpha=0.5)
        ax.set_xlim(-max_lag_for_pca, max_lag_for_pca)
        ax.annotate(error[cell_idx], xy=(1.0, 1.0), xycoords='axes fraction', horizontalalignment='right',
                    verticalalignment='top')

    plot_kwargs = dict(t_auto_corr=t_autocorr, auto_corr_cells2=back_projected, auto_corr_cells1=autocorr_cells,
                       bin_size=bin_width, max_lag=max_lag, error=error_backprojection_str)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_2autocorrelations, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', colors_marker=colors_marker,
                            save_dir_img=save_dir_img_file)

if __name__ == '__main__':
    #save_dir_img_paper = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    #save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    #save_dir_autocorr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'

    save_dir_img_paper = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_autocorr = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'

    if not os.path.exists(save_dir_img_paper):
        os.makedirs(save_dir_img_paper)

    cell_type = 'grid_cells'
    #save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    max_lag = 150  # ms
    max_lag_for_pca = max_lag  # ms
    bin_width = 1  # ms
    sigma_smooth = None
    dt_kde = 0.05  # ms
    n_components = 2
    remove_cells = True
    normalization = 'sum'

    max_lag_idx = to_idx(max_lag, bin_width)
    remove_cells_dict = {True: 'removed', False: 'not_removed'}
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    DAP_cells_new = get_cell_ids_DAP_cells(new=True)

    folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(
        sigma_smooth) + '_normalization_' + str(normalization)
    save_dir_img = os.path.join(save_dir_autocorr, folder, 'PCA')
    if max_lag_for_pca != max_lag:
        save_dir_img = os.path.join(save_dir_img, 'max_lag_for_pca_'+str(max_lag_for_pca))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    autocorr_cells = np.load(os.path.join(save_dir_autocorr, folder, 'autocorr.npy'))
    if sigma_smooth is not None:
        t_autocorr = np.arange(-max_lag, max_lag + dt_kde, dt_kde)
    else:
        t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)

    # PCA
    if remove_cells:  # take out autocorrelation for cell s104_0007 and s110_0002
        idx_s104_0007 = np.where(np.array(cell_ids) == 's104_0007')[0][0]
        idx_s110_0002 = np.where(np.array(cell_ids) == 's110_0002')[0][0]
        idxs = range(len(cell_ids))
        idxs.remove(idx_s104_0007)
        idxs.remove(idx_s110_0002)
        autocorr_cells_for_pca = autocorr_cells[np.array(idxs)]
    else:
        autocorr_cells_for_pca = autocorr_cells

    if max_lag_for_pca != max_lag:
        assert max_lag > max_lag_for_pca
        assert sigma_smooth is None  # not implemented with smoothing
        diff_lag_idx = to_idx(max_lag - max_lag_for_pca, bin_width)
        t_autocorr = np.arange(-max_lag_idx + diff_lag_idx, max_lag_idx - diff_lag_idx + bin_width, bin_width)
        autocorr_cells = autocorr_cells[:, diff_lag_idx:-diff_lag_idx]
        autocorr_cells_for_pca = autocorr_cells_for_pca[:, diff_lag_idx:-diff_lag_idx]
    projected_, components, explained_var = perform_PCA(autocorr_cells_for_pca, n_components)
    projected = np.dot(autocorr_cells - np.mean(autocorr_cells_for_pca, 0), components[:n_components, :].T)

    # PCA backtransform
    back_projected = np.dot(projected, components[:n_components, :])
    back_projected += np.mean(autocorr_cells_for_pca, axis=0)

    len_autocorr = np.shape(autocorr_cells)[1]
    error_backprojection = np.array([np.sqrt(np.sum((o-bp)**2)/len_autocorr) for o, bp in zip(autocorr_cells, back_projected)])
    error_backprojection_str = ['%.4f' % e for e in error_backprojection]

    # k-means
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(projected)
    labels = kmeans.labels_

    # dbscan = DBSCAN(eps=0.03, min_samples=3).fit(transformed)
    # labels = dbscan.labels_

    # specclus = SpectralClustering(n_clusters=n_clusters, random_state=3).fit(transformed)
    # labels = specclus.labels_

    # save
    np.save(os.path.join(save_dir_img, 'projected.npy'), projected)

    # plots
    plot_pca_projection_for_paper(save_dir_img=save_dir_img_paper)
    pl.show()

    plot_PCs(n_components, t_autocorr, components, explained_var, max_lag, bin_width,
             save_dir_img)

    plot_backtransformed(save_dir_img)

    plot_pca_projection_for_thesis(save_dir_img=save_dir_img)

    plot_pca_projection_with_cell_ids(save_dir_img=save_dir_img)
    if n_components == 3:
        plot_pca_projection_3d_with_cell_ids(save_dir_img=save_dir_img)

    # plot_pca_projection_slides(save_dir_img=None)

    pl.show()