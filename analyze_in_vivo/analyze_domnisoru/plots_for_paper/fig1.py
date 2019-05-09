import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import matplotlib.gridspec as gridspec
import os
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_colors_burstgroups, get_label_burstgroups, get_celltype_dict
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_fig1 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig1'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/extra'

    grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    colors_burstgroups = get_colors_burstgroups()
    labels_burstgroups = get_label_burstgroups(save_dir)
    cell_type_dict = get_celltype_dict(save_dir)

    # load data
    max_lag = 50  # ms
    bin_width = 1  # ms
    n_components = 2
    autocorr_cells = np.load(os.path.join(save_dir_fig1, 'autocorr.npy'))
    t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)
    projected = np.load(os.path.join(save_dir_fig1, 'projected.npy'))
    components = np.load(os.path.join(save_dir_fig1, 'components.npy'))
    explained_var = np.load(os.path.join(save_dir_fig1, 'explained_var.npy'))

    # plot
    fig = pl.figure(figsize=(6, 7.5))
    outer = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 1])
    cell_examples = ['s84_0002', 's109_0002', 's118_0002']
    colors_examples = [colors_burstgroups['NB'], colors_burstgroups['B+D'], colors_burstgroups['B']]
    letters_examples = ['a', 'b', 'c']

    # A
    for i, cell_id in enumerate(cell_examples):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, i])
        ax = pl.Subplot(fig, inner[0])
        fig.add_subplot(ax)
        cell_idx = np.where(grid_cells == cell_id)[0][0]

        ax.bar(t_autocorr, autocorr_cells[cell_idx], bin_width, color=colors_examples[i], align='center')
        ax.set_xticks([-max_lag, 0, max_lag])
        ax.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
        ax.set_xlabel('Lag (ms)')
        if i == 0:
            ax.set_ylabel('Autocorrelation')
            ax.yaxis.set_label_coords(-0.43, 0.5)
            ax.text(-0.85, 1.0, 'A', transform=ax.transAxes, size=18)
        ax.set_title('('+letters_examples[i]+')'+'     '+cell_id, loc='left', fontsize=10)

    # B
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, :])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)
    plot_with_markers(ax, projected[labels_burstgroups['NB'], 0], projected[labels_burstgroups['NB'], 1],
                      grid_cells[labels_burstgroups['NB']], cell_type_dict,
                      edgecolor=colors_burstgroups['NB'], theta_cells=theta_cells, legend=False)
    handles = plot_with_markers(ax, projected[labels_burstgroups['B+D'], 0], projected[labels_burstgroups['B+D'], 1],
                                grid_cells[labels_burstgroups['B+D']], cell_type_dict,
                                edgecolor=colors_burstgroups['B+D'], theta_cells=theta_cells, legend=False)
    plot_with_markers(ax, projected[labels_burstgroups['B'], 0], projected[labels_burstgroups['B'], 1],
                      grid_cells[labels_burstgroups['B']], cell_type_dict,
                      edgecolor=colors_burstgroups['B'], theta_cells=theta_cells, legend=False)
    labels = [h.get_label() for h in handles]
    handles += [(Patch(color=colors_burstgroups['B+D']), Patch(color=colors_burstgroups['B'])),
                 Patch(color=colors_burstgroups['NB'])]
    labels += ['Bursty', 'Non-bursty']
    ax.legend(handles, labels, loc='upper right', handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_xlabel('1st Principal Component (PC1)')
    ax.set_ylabel('2nd Principal Component (PC2)')
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.set_xticks(np.arange(-0.15, 0.251, 0.05))
    ax.set_yticks(np.arange(-0.1, 0.251, 0.05))

    for i, cell_id in enumerate(cell_examples):
        cell_idx = np.where(grid_cells == cell_id)[0][0]
        ax.annotate('('+letters_examples[i]+')', xy=(projected[cell_idx, 0]+0.005, projected[cell_idx, 1]-0.006),
                    verticalalignment='top', fontsize=10)

    ax.text(-0.215, 1.0, 'B', transform=ax.transAxes, size=18)

    # C
    inner = gridspec.GridSpecFromSubplotSpec(1, n_components, subplot_spec=outer[2, :n_components])
    for n_component in range(n_components):
        ax = pl.Subplot(fig, inner[n_component])
        fig.add_subplot(ax)
        ax.bar(t_autocorr, components[n_component, :], bin_width, color='k', align='center')
        ax.annotate('Explained \nvariance: \n%i' % np.round(
            explained_var[n_component] * 100) + '%',
                    xy=(0.95, 1.15), xycoords='axes fraction', ha='right', va='top', fontsize=8)
        ax.set_xlabel('Lag (ms)')
        if n_component == 0:
            ax.set_ylabel('Size')
            ax.yaxis.set_label_coords(-0.43, 0.5)
            ax.text(-0.85, 1.05, 'C', transform=ax.transAxes, size=18)
        ax.set_title('PC' + str(n_component + 1), loc='left', fontsize=10)
        ax.set_xticks([-max_lag, 0, max_lag])

    # D
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2, 2])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)
    mean_autocorr = np.mean(autocorr_cells[np.logical_or(np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D']), labels_burstgroups['NB'])], 0)
    ax.bar(t_autocorr, mean_autocorr, bin_width, color='k', align='center')
    ax.set_xticks([-max_lag, 0, max_lag])
    ax.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    ax.set_xlabel('Lag (ms)')
    ax.set_title('Mean \nautocorrelation', fontsize=10, pad=-0.1)
    ax.text(-0.4, 1.05, 'D', transform=ax.transAxes, size=18)

    # other stuff
    pl.tight_layout()
    pl.subplots_adjust(hspace=0.55, wspace=0.45, bottom=0.07, right=0.98, left=0.18, top=0.97)
    pl.savefig(os.path.join(save_dir_fig1, 'fig1.png'))
    pl.show()