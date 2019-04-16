import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerTuple
import matplotlib.gridspec as gridspec
import os
from sklearn.decomposition import PCA
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict, \
    get_label_burstgroups, get_colors_burstgroups
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.pca import perform_PCA
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img_paper = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_autocorr = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'

    if not os.path.exists(save_dir_img_paper):
        os.makedirs(save_dir_img_paper)

    labels_burstgroups = get_label_burstgroups()
    colors_burstgroups = get_colors_burstgroups()
    cell_type = 'grid_cells'
    #save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    max_lag = 50  # ms
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
    DAP_cells = get_cell_ids_DAP_cells(new=True)

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
    #components[1, :] *= -1
    projected = np.dot(autocorr_cells - np.mean(autocorr_cells_for_pca, 0), components[:n_components, :].T)


    # plot
    fig = pl.figure(figsize=(5, 12))
    outer = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
    cell_examples = ['s84_0002', 's109_0002', 's76_0002']
    colors_examples = [colors_burstgroups['NB'], colors_burstgroups['B+D'], colors_burstgroups['B']]
    letters_examples = ['a', 'b', 'c']

    # A
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, :])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)
    plot_with_markers(ax, projected[labels_burstgroups['NB'], 0], projected[labels_burstgroups['NB'], 1],
                      cell_ids[labels_burstgroups['NB']], cell_type_dict,
                      edgecolor=colors_burstgroups['NB'], theta_cells=theta_cells, legend=False)
    handles = plot_with_markers(ax, projected[labels_burstgroups['B+D'], 0], projected[labels_burstgroups['B+D'], 1],
                                cell_ids[labels_burstgroups['B+D']], cell_type_dict,
                                edgecolor=colors_burstgroups['B+D'], theta_cells=theta_cells, legend=False)
    plot_with_markers(ax, projected[labels_burstgroups['B'], 0], projected[labels_burstgroups['B'], 1],
                      cell_ids[labels_burstgroups['B']], cell_type_dict,
                      edgecolor=colors_burstgroups['B'], theta_cells=theta_cells, legend=False)
    labels = [h.get_label() for h in handles]
    handles += [(Patch(color=colors_burstgroups['B+D']), Patch(color=colors_burstgroups['B'])),
                 Patch(color=colors_burstgroups['NB'])]
    labels += ['Bursty', 'Non-bursty']
    ax.legend(handles, labels, loc='upper right', handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_xlabel('1st Principal Component (PC1)')
    ax.set_ylabel('2nd Principal Component (PC2)')
    ax.yaxis.set_label_coords(-0.13, 0.5)
    ax.set_xticks(np.arange(-0.05, 0.151, 0.05))
    ax.set_yticks(np.arange(-0.05, 0.151, 0.05))

    for i, cell_id in enumerate(cell_examples):
        cell_idx = np.where(cell_ids == cell_id)[0][0]
        ax.annotate('('+letters_examples[i]+')', xy=(projected[cell_idx, 0], projected[cell_idx, 1]), verticalalignment='top')

    ax.text(-0.19, 1.0, 'A', transform=ax.transAxes, size=18)

    # B
    for i, cell_id in enumerate(cell_examples):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, i])
        ax = pl.Subplot(fig, inner[0])
        fig.add_subplot(ax)
        cell_idx = np.where(cell_ids == cell_id)[0][0]

        ax.bar(t_autocorr, autocorr_cells[cell_idx], bin_width, color=colors_examples[i], align='center')
        #ax.set_ylim(0, 0.05)
        ax.set_xticks([-max_lag_for_pca, 0, max_lag_for_pca])
        ax.set_xticklabels([-max_lag_for_pca, 0, max_lag_for_pca], fontsize=10)
        ax.set_xlabel('Lag (ms)')
        if i == 0:
            ax.set_ylabel('Autocorrelation')
            ax.yaxis.set_label_coords(-0.51, 0.5)
            ax.text(-0.75, 1.0, 'B', transform=ax.transAxes, size=18)
        ax.set_title(letters_examples[i]+'     '+cell_id, loc='left', fontsize=10)

    # C
    inner = gridspec.GridSpecFromSubplotSpec(1, n_components, subplot_spec=outer[2, :n_components])
    for n_component in range(n_components):
        ax = pl.Subplot(fig, inner[n_component])
        fig.add_subplot(ax)
        ax.bar(t_autocorr, components[n_component, :], bin_width, color='k', align='center')
        ax.annotate('Explained \nvariance: \n%i' % np.round(
            explained_var[n_component] * 100) + '%',
                    xy=(0.57, 1.31), xycoords='axes fraction', ha='left', va='top', fontsize=10)
        ax.set_xlabel('Lag (ms)')
        if n_component == 0:
            ax.set_ylabel('Size')
            ax.yaxis.set_label_coords(-0.51, 0.5)
            ax.text(-0.75, 1.0, 'C', transform=ax.transAxes, size=18)
        ax.set_title('PC' + str(n_component + 1), loc='left', fontsize=10)
        ax.set_xticks([-max_lag_for_pca, 0, max_lag_for_pca])

    # D
    max_lags = np.load(os.path.join(save_dir_autocorr, 'explained_var_for_max_lags', 'max_lags.npy'))
    explained_vars = np.load(os.path.join(save_dir_autocorr, 'explained_var_for_max_lags', 'explained_vars.npy'))
    linestyles = ['-', '--', '.']

    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2, 2])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)
    ax.text(-0.6, 1.0, 'D', transform=ax.transAxes, size=18)
    for i in range(n_components):
        pl.plot(max_lags, explained_vars[:, i], label='PC'+str(i+1), color='k', linestyle=linestyles[i])
    ax.plot(max_lags, explained_vars[:, 0] + explained_vars[:, 1], label='PC1+PC2', color='g', linestyle='--')
    ax.plot(max_lags, explained_vars[:, 0] + explained_vars[:, 1] + explained_vars[:, 2], label='PC1+PC2+PC3',
            color='g', linestyle='.')
    ax.set_ylabel('Explained variance')
    ax.set_xlabel('Maximal lag (ms)')
    ax.ylim(0, 1)
    ax.legend()

    # other stuff
    pl.tight_layout()
    pl.subplots_adjust(hspace=0.55, wspace=0.5, bottom=0.07, right=0.95, left=0.16, top=0.97)
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'autocorr_pca_' + str(max_lag) + '_' + str(
            max_lag_for_pca) + '_' + str(bin_width) + '_' + str(sigma_smooth) + '_' + remove_cells_dict[
                                    remove_cells] + '.png'))
    pl.show()