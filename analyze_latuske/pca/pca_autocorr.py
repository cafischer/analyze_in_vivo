import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from sklearn.cluster import KMeans
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.autocorr.spiketime_autocorr import plot_autocorrelation
from analyze_in_vivo.analyze_domnisoru.pca import perform_PCA
pl.style.use('paper_subplots')


def plot_pca_projection_for_paper(save_dir_img):
    fig = pl.figure(figsize=(7, 7))
    outer = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.27)

    # upper plot
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)

    # # example 1
    # axins = inset_axes(ax, width='20%', height='20%', loc='upper left', bbox_to_anchor=(0.06, 0, 1, 1),
    #                    bbox_transform=ax.transAxes)
    # i = np.where(cell_ids == 's84_0002')[0][0]
    # axins.bar(t_autocorr, auto_corr_cells[i], bin_width, color='b', align='center')
    # axins.set_yticks([])
    # axins.set_xticks([-max_lag, 0, max_lag])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    #
    # # example 2
    # axins = inset_axes(ax, width='20%', height='20%', loc='upper right')  # bbox_to_anchor=(0.7, 0.7, 1.0, 1.0)
    # i = np.where(cell_ids == 's109_0002')[0][0]
    # axins.bar(t_autocorr, auto_corr_cells[i], bin_width, color='r', align='center')
    # axins.set_yticks([])
    # axins.set_xticks([-max_lag, 0, max_lag])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)
    #
    # # example 3
    # axins = inset_axes(ax, width='20%', height='20%', loc='center right')  # bbox_to_anchor=(0.7, 0.7, 1.0, 1.0)
    # i = np.where(cell_ids == 's76_0002')[0][0]
    # axins.bar(t_autocorr, auto_corr_cells[i], bin_width, color='r', align='center')
    # axins.set_yticks([])
    # axins.set_xticks([-max_lag, 0, max_lag])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    # axins.set_xlabel('Lag (ms)', fontsize=10)
    # axins.set_ylabel('Spike-time \nautocorrelation', fontsize=10)

    # outside plot
    ax.plot(projected[labels_latuske == 0, 0], projected[labels_latuske == 0, 1], 'r', marker='d', linestyle='',
            markerfacecolor='None')
    ax.plot(projected[labels_latuske == 1, 0], projected[labels_latuske == 1, 1], 'b', marker='d', linestyle='',
            markerfacecolor='None')
    if n_clusters == 3:
        ax.plot(projected[labels_latuske == 2, 0], projected[labels_latuske == 2, 1], 'g', marker='d', linestyle='')
    plot_with_markers(ax, projected_domnisoru[labels_domnisoru == 0, 0],
                      projected_domnisoru[labels_domnisoru == 0, 1], cell_ids[labels_domnisoru == 0],
                      get_celltype_dict(save_dir_domnisoru), edgecolor='r', legend=False)
    handles = plot_with_markers(ax, projected_domnisoru[labels_domnisoru == 1, 0],
                                projected_domnisoru[labels_domnisoru == 1, 1], cell_ids[labels_domnisoru == 1],
                      get_celltype_dict(save_dir_domnisoru), edgecolor='b', legend=False)
    if n_clusters == 3:
        plot_with_markers(ax, projected_domnisoru[labels_domnisoru == 2, 0],
                          projected_domnisoru[labels_domnisoru == 2, 1], cell_ids[labels_domnisoru == 2],
                          get_celltype_dict(save_dir_domnisoru), edgecolor='g', legend=False)
    #ax.plot(projected_domnisoru[:, 0], projected_domnisoru[:, 1], 'xk')
    #ax.plot(projected[-26:, 0], projected[-26:, 1], 'xk')
    fig_fake, ax_fake = pl.subplots()
    handle_latuske = [ax_fake.scatter(0, 0, marker='d', edgecolor='k', facecolor='k', label='Latuske')]
    pl.close(fig_fake)
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    handles += handle_latuske + handles_bursty
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

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
        ax.set_xticks([-max_lag, 0, max_lag])

    # legend from upper plot
    ax = pl.Subplot(fig, inner[2])
    fig.add_subplot(ax)
    legend1 = ax.legend(handles=handles, loc='upper right')
    ax.add_artist(legend1)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # other stuff
    pl.tight_layout()
    pl.subplots_adjust(left=0.12, right=0.98, bottom=0.07, top=0.98)
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'pca_autocorrelation.png'))


if __name__ == '__main__':
    save_dir_domnisoru = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_autocorr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/autocorr'
    save_dir_autocorr_domnisoru = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
    save_dir_img = os.path.join(save_dir_autocorr, 'PCA')
    max_lag = 50
    bin_width = 1  # ms
    sigma_smooth = None
    dt_kde = 0.05
    n_components = 2
    remove_cells = False
    use_all = False
    max_lag_idx = to_idx(max_lag, bin_width)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
    autocorr_cells = np.load(os.path.join(save_dir_autocorr, folder, 'autocorr.npy'))
    if sigma_smooth is not None:
        t_autocorr = np.arange(-max_lag_idx, max_lag_idx + dt_kde, dt_kde)
    else:
        t_autocorr = np.arange(-max_lag_idx, max_lag_idx + bin_width, bin_width)
    autocorr_cells_domnisoru = np.load(os.path.join(save_dir_autocorr_domnisoru, folder, 'autocorr.npy'))
    cell_ids = np.array(load_cell_ids(save_dir_domnisoru, 'grid_cells'))
    if remove_cells:  # take out autocorrelation for cell s104_0007 and s110_0002
        idx_s104_0007 = np.where(np.array(cell_ids) == 's104_0007')[0][0]
        idx_s110_0002 = np.where(np.array(cell_ids) == 's110_0002')[0][0]
        idxs = range(len(cell_ids))
        idxs.remove(idx_s104_0007)
        idxs.remove(idx_s110_0002)
        autocorr_cells_domnisoru = autocorr_cells_domnisoru[np.array(idxs)]

    # PCA
    if use_all:
        autocorr_cells = np.vstack((autocorr_cells, autocorr_cells_domnisoru))
    auto_corr_cells_centered = autocorr_cells - np.mean(autocorr_cells, 0)
    projected, components, explained_var = perform_PCA(auto_corr_cells_centered, n_components)

    # project autocorr from domnisoru cells onto components
    projected_domnisoru = np.dot(autocorr_cells_domnisoru - np.mean(autocorr_cells, 0), components[:n_components, :].T)

    # projected_idx_sort = np.argsort(projected[:, 0])
    # for autocorr in autocorr_cells[projected_idx_sort, :][::2, :]:
    #     pl.figure()
    #     pl.bar(t_autocorr, autocorr, bin_width, color='0.5', align='center')
    #     pl.show()


    # k-means
    n_clusters = 3
    projected_all = np.vstack((projected, projected_domnisoru))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(projected_all)
    labels = kmeans.labels_
    labels_latuske = labels[:len(autocorr_cells)]
    labels_domnisoru = labels[len(autocorr_cells):]

    # plots
    plot_pca_projection_for_paper(save_dir_img=save_dir_img)
    pl.show()