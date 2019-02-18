from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import numpy as np
import os
from scipy.stats import ttest_ind
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import horizontal_square_bracket, get_star_from_p_val
from cell_fitting.util import change_color_brightness
from matplotlib.colors import to_rgb
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/latuske'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist/cut_ISIs_at_200/grid_cells'
    save_dir_n_spikes_in_burst = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
    save_dir_ISI_return_map = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map/cut_ISIs_at_200/grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    save_dir_img = os.path.join(save_dir_img, cell_type)

    ISI_burst = 8  # ms
    fraction_burst = np.load(os.path.join(save_dir_ISI_hist, 'fraction_burst.npy'))
    fraction_single = np.load(os.path.join(save_dir_n_spikes_in_burst, 'fraction_single_' + str(ISI_burst) + '.npy'))
    fraction_ISI_or_ISI_next_burst = np.load(os.path.join(save_dir_ISI_return_map, 'fraction_ISI_or_ISI_next_burst.npy'))

    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    cell_ids_burst1 = ['s76_0002', 's117_0002', 's118_0002', 's120_0002']
    burst1_label = np.array([True if cell_id in cell_ids_burst1 else False for cell_id in cell_ids])
    burst2_label = np.logical_and(burst_label, ~burst1_label)

    # significance tests
    _, p_val_fraction_burst = ttest_ind(fraction_burst[burst_label], fraction_burst[~burst_label])
    _, p_val_fraction_single = ttest_ind(fraction_single[burst_label], fraction_single[~burst_label])
    _, p_val_fraction_ISI_or_ISI_next_burst = ttest_ind(fraction_ISI_or_ISI_next_burst[burst_label],
                                                        fraction_ISI_or_ISI_next_burst[~burst_label])

    # plot
    n_bursty = sum(burst_label)
    n_nonbursty = sum(~burst_label)

    fig, axes = pl.subplots(1, 4, figsize=(8, 4))

    plot_with_markers(axes[0], np.zeros(n_bursty), fraction_burst[burst1_label], cell_ids[burst1_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'darker'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[0], np.zeros(n_bursty), fraction_burst[burst2_label], cell_ids[burst2_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'brighter'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[0], np.ones(n_nonbursty), fraction_burst[~burst_label], cell_ids[~burst_label], cell_type_dict,
                      edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    horizontal_square_bracket(axes[0], get_star_from_p_val(p_val_fraction_burst), 0, 1, 1.03, 1.04, 0.0)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Bursty', 'Non-bursty'])

    axes[0].set_xlim([-0.5, 1.5])
    axes[0].set_ylim([0, 1.1])
    axes[0].set_ylabel('Fraction ISIs $\leq$ 8ms')
    axes[0].text(-0.5, 1.0, 'A', transform=axes[0].transAxes, size=18, weight='bold')

    plot_with_markers(axes[1], np.zeros(n_bursty), fraction_single[burst1_label], cell_ids[burst1_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'darker'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[1], np.zeros(n_bursty), fraction_single[burst2_label], cell_ids[burst2_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'brighter'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[1], np.ones(n_nonbursty), fraction_single[~burst_label], cell_ids[~burst_label], cell_type_dict,
                      edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    horizontal_square_bracket(axes[1], get_star_from_p_val(p_val_fraction_single), 0, 1, 1.03, 1.04, 0.0)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Bursty', 'Non-bursty'])
    axes[1].set_xlim([-0.5, 1.5])
    axes[1].set_ylim([0, 1.1])
    axes[1].set_ylabel('Fraction single spikes')
    axes[1].text(-0.5, 1.0, 'B', transform=axes[1].transAxes, size=18, weight='bold')

    plot_with_markers(axes[2], np.zeros(n_bursty), fraction_ISI_or_ISI_next_burst[burst1_label],
                      cell_ids[burst1_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'darker'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[2], np.zeros(n_bursty), fraction_ISI_or_ISI_next_burst[burst2_label],
                      cell_ids[burst2_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'brighter'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    handles = plot_with_markers(axes[2], np.ones(n_nonbursty), fraction_ISI_or_ISI_next_burst[~burst_label],
                                cell_ids[~burst_label], cell_type_dict,
                                edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                DAP_cells_additional=DAP_cells_additional, legend=False)
    horizontal_square_bracket(axes[2], get_star_from_p_val(p_val_fraction_ISI_or_ISI_next_burst), 0, 1, 1.03, 1.04, 0.0)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['Bursty', 'Non-bursty'])
    axes[2].set_xlim([-0.5, 1.5])
    axes[2].set_ylim([0, 1.1])
    axes[2].set_ylabel('Fraction ISI[n] or ISI[n+1] $\leq$ 8ms')
    axes[2].text(-0.5, 1.0, 'C', transform=axes[2].transAxes, size=18, weight='bold')

    handles += [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    axes[3].legend(handles=handles)
    axes[3].spines['left'].set_visible(False)
    axes[3].spines['bottom'].set_visible(False)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    pl.tight_layout()
    pl.subplots_adjust(top=0.95, left=0.08, right=1.0, bottom=0.08)
    pl.savefig(os.path.join(save_dir_img, 'difference_bursty_nonbursty.png'))

    # plot 2
    n_bursty = sum(burst_label)
    n_nonbursty = sum(~burst_label)

    fig, axes = pl.subplots(1, 4, figsize=(8, 4))

    plot_with_markers(axes[0], -np.ones(n_bursty), fraction_burst[burst1_label], cell_ids[burst1_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'darker'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[0], np.zeros(n_bursty), fraction_burst[burst2_label], cell_ids[burst2_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'brighter'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[0], np.ones(n_nonbursty), fraction_burst[~burst_label], cell_ids[~burst_label], cell_type_dict,
                      edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    axes[0].set_xticks([-1, 0, 1])
    axes[0].set_xticklabels(['B1', 'B2', 'N-B'])

    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([0, 1.1])
    axes[0].set_ylabel('Fraction ISIs $\leq$ 8ms')
    axes[0].text(-0.5, 1.0, 'A', transform=axes[0].transAxes, size=18, weight='bold')

    plot_with_markers(axes[1], -np.ones(n_bursty), fraction_single[burst1_label], cell_ids[burst1_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'darker'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[1], np.zeros(n_bursty), fraction_single[burst2_label], cell_ids[burst2_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'brighter'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[1], np.ones(n_nonbursty), fraction_single[~burst_label], cell_ids[~burst_label], cell_type_dict,
                      edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    axes[1].set_xticks([-1, 0, 1])
    axes[1].set_xticklabels(['B1', 'B2', 'N-B'])
    axes[1].set_xlim([-1.5, 1.5])
    axes[1].set_ylim([0, 1.1])
    axes[1].set_ylabel('Fraction single spikes')
    axes[1].text(-0.5, 1.0, 'B', transform=axes[1].transAxes, size=18, weight='bold')

    plot_with_markers(axes[2], -np.ones(n_bursty), fraction_ISI_or_ISI_next_burst[burst1_label],
                      cell_ids[burst1_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'darker'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    plot_with_markers(axes[2], np.zeros(n_bursty), fraction_ISI_or_ISI_next_burst[burst2_label],
                      cell_ids[burst2_label], cell_type_dict,
                      edgecolor=change_color_brightness(to_rgb('r'), 50, 'brighter'),
                      theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    handles = plot_with_markers(axes[2], np.ones(n_nonbursty), fraction_ISI_or_ISI_next_burst[~burst_label],
                                cell_ids[~burst_label], cell_type_dict,
                                edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                DAP_cells_additional=DAP_cells_additional, legend=False)
    axes[2].set_xticks([-1, 0, 1])
    axes[2].set_xticklabels(['B1', 'B2', 'N-B'])
    axes[2].set_xlim([-1.5, 1.5])
    axes[2].set_ylim([0, 1.1])
    axes[2].set_ylabel('Fraction ISI[n] or ISI[n+1] $\leq$ 8ms')
    axes[2].text(-0.5, 1.0, 'C', transform=axes[2].transAxes, size=18, weight='bold')

    handles += [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    axes[3].legend(handles=handles)
    axes[3].spines['left'].set_visible(False)
    axes[3].spines['bottom'].set_visible(False)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    pl.tight_layout()
    pl.subplots_adjust(top=0.95, left=0.08, right=1.0, bottom=0.08)
    pl.savefig(os.path.join(save_dir_img, 'difference_bursty_nonbursty_3groups.png'))
    pl.show()