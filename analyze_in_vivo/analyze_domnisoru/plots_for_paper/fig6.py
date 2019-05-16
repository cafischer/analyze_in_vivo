import os
import numpy as np
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, \
    get_label_burstgroups, get_colors_burstgroups
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/extra'
    save_dir_fig6 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig6'

    cell_type = 'grid_cells'
    grid_cells = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    labels_burstgroups = get_label_burstgroups()
    colors_burstgroups = get_colors_burstgroups()
    NB_label = labels_burstgroups['NB']
    BD_label = labels_burstgroups['B+D']
    B_label = labels_burstgroups['B']

    # load
    peak_ISI_hist = np.load(os.path.join(save_dir_fig6, 'peak_ISI_hist.npy'))
    fraction_burst = np.load(os.path.join(save_dir_fig6, 'fraction_burst.npy'))
    CV_ISIs = np.load(os.path.join(save_dir_fig6, 'CV_ISIs.npy'))
    fraction_ISIs_8_25 = np.load(os.path.join(save_dir_fig6, 'fraction_ISIs_8_25.npy'))
    firing_rate = np.load(os.path.join(save_dir_fig6, 'firing_rate.npy'))
    v_onset_fAHP = np.load(os.path.join(save_dir_fig6, 'v_onset_fAHP.npy'))
    v_DAP_fAHP = np.load(os.path.join(save_dir_fig6, 'v_DAP_fAHP.npy'))

    # plot
    x_data = [v_onset_fAHP, v_DAP_fAHP]
    x_labels = ['$\Delta V_{fAHP}$ [mV]', '$\Delta V_{DAP}$ [mV]']
    y_data = [firing_rate, fraction_burst, fraction_ISIs_8_25,
              peak_ISI_hist, CV_ISIs]
    ylabels = ['Firing rate [Hz]', 'P(ISIs $\leq$ 8ms)', 'P(8 < ISI < 25)',
               'ISI peak [ms]', '$CV_{ISI}$']

    fig, axes = pl.subplots(5, 2, figsize=(5, 8), squeeze=False)
    for i in range(len(x_data)):
        for j in range(len(y_data)):
            plot_with_markers(axes[j, i], x_data[i][BD_label], y_data[j][BD_label], grid_cells[BD_label],
                              cell_type_dict, edgecolor=colors_burstgroups['B+D'], legend=False)
            plot_with_markers(axes[j, i], x_data[i][B_label], y_data[j][B_label], grid_cells[B_label],
                              cell_type_dict, edgecolor=colors_burstgroups['B'], legend=False)
            plot_with_markers(axes[j, i], x_data[i][NB_label], y_data[j][NB_label], grid_cells[NB_label],
                              cell_type_dict, edgecolor=colors_burstgroups['NB'], legend=False)
            axes[j, i].set_xlabel(x_labels[i], fontsize=9)
            axes[j, i].set_ylabel(ylabels[j], fontsize=9)
            axes[j, i].set_ylim([0, None])

    handles_bursty = [Patch(color=colors_burstgroups['B'], label='B-D'),
                      Patch(color=colors_burstgroups['B+D'], label='B+D'),
                      Patch(color=colors_burstgroups['NB'], label='NB')]
    axes[0, 1].legend(handles=handles_bursty, loc='upper right', fontsize=8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_fig6, 'fig6.png'))
    pl.show()