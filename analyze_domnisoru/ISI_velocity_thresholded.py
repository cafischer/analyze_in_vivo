from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs, get_cumulative_ISI_hist
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells_grid
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
pl.style.use('paper')


def get_ISIs_for_groups(group_indicator, AP_max_idxs, t, max_ISI):
    starts, ends = get_starts_ends_group_of_ones(group_indicator.astype(int))
    ISIs_groups = []
    for start, end in zip(starts, ends):
        AP_max_idxs_inside = AP_max_idxs[np.logical_and(start < AP_max_idxs, AP_max_idxs < end)]

        ISIs = get_ISIs(AP_max_idxs_inside, t)
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]
        ISIs_groups.extend(ISIs)
    return ISIs_groups


if __name__ == '__main__':
    # TODO: this biases towards shorter ISIs
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map/velocity_thresholding'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']

    # parameters
    velocity_threshold = 1  # cm/sec
    filter_long_ISIs = True
    max_ISI = 200
    bin_size = 1.0  # ms
    steps = np.arange(0, max_ISI + bin_size, bin_size)
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_' + str(max_ISI))
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs_above_cells = np.zeros(len(cell_ids), dtype=object)
    ISIs_under_cells = np.zeros(len(cell_ids), dtype=object)
    cum_ISI_hist_y_above = np.zeros(len(cell_ids), dtype=object)
    cum_ISI_hist_x_above = np.zeros(len(cell_ids), dtype=object)
    cum_ISI_hist_y_under = np.zeros(len(cell_ids), dtype=object)
    cum_ISI_hist_x_under = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']
        AP_max_idxs = data['spiketimes']

        # get ISIs under and above velocity threshold
        ISIs_above_cells[cell_idx] = get_ISIs_for_groups(velocity >= velocity_threshold, AP_max_idxs, t, max_ISI)
        ISIs_under_cells[cell_idx] = get_ISIs_for_groups(velocity < velocity_threshold, AP_max_idxs, t, max_ISI)
        cum_ISI_hist_y_above[cell_idx], cum_ISI_hist_x_above[cell_idx] = get_cumulative_ISI_hist(
            ISIs_above_cells[cell_idx])
        cum_ISI_hist_y_under[cell_idx], cum_ISI_hist_x_under[cell_idx] = get_cumulative_ISI_hist(
            ISIs_under_cells[cell_idx])

    # plot ISI histograms
    def plot_ISI_hist(ax, cell_idx, subplot_idx, steps, ISIs_above_cells, cum_ISI_hist_y_above, cum_ISI_hist_x_above,
                      ISIs_under_cells, cum_ISI_hist_y_under, cum_ISI_hist_x_under, max_ISI):
        if subplot_idx == 0:
            freqs, _, _ = ax.hist(ISIs_above_cells[cell_idx], steps, color='0.5')

            #cumulative
            cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x_above[cell_idx], len(cum_ISI_hist_x_above[cell_idx]),
                                                max_ISI)
            cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y_above[cell_idx], len(cum_ISI_hist_y_above[cell_idx]),
                                                1.0)
            ax_twin = ax.twinx()
            ax_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
            ax_twin.set_xlim(0, max_ISI)
            ax_twin.set_ylim(0, 1)
            if (cell_idx + 1) % 9 == 0:
                ax_twin.set_yticks([0, 1])
            else:
                ax_twin.set_yticks([])
            ax.spines['right'].set_visible(True)

            ax.set_xticks([])
            ax.set_xlabel('')
            ax.annotate('$\geq$ vel. thresh.', xy=(1, np.max(freqs)), textcoords='data',
                        horizontalalignment='left', verticalalignment='top', fontsize=9)
        if subplot_idx == 1:
            freqs, _, _ = ax.hist(ISIs_under_cells[cell_idx], steps, color='0.5')

            # cumulative
            cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x_under[cell_idx], len(cum_ISI_hist_x_under[cell_idx]),
                                                max_ISI)
            cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y_under[cell_idx], len(cum_ISI_hist_y_under[cell_idx]),
                                                1.0)
            ax_twin = ax.twinx()
            ax_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
            ax_twin.set_xlim(0, max_ISI)
            ax_twin.set_ylim(0, 1)
            if (cell_idx + 1) % 9 == 0:
                ax_twin.set_yticks([0, 1])
                ax_twin.set_yticklabels([0, 1], fontsize=10)
            else:
                ax_twin.set_yticks([])
            ax.spines['right'].set_visible(True)

            ax.annotate('$<$ vel. thresh.', xy=(1, np.max(freqs)), textcoords='data',
                        horizontalalignment='left', verticalalignment='top', fontsize=9)

    plot_kwargs = dict(steps=steps,
                       ISIs_above_cells=ISIs_above_cells, cum_ISI_hist_y_above=cum_ISI_hist_y_above,
                       cum_ISI_hist_x_above=cum_ISI_hist_x_above,
                      ISIs_under_cells=ISIs_under_cells, cum_ISI_hist_y_under=cum_ISI_hist_y_under,
                       cum_ISI_hist_x_under=cum_ISI_hist_x_under,
                       max_ISI=max_ISI)
    plot_for_all_grid_cells_grid(cell_ids, cell_type_dict, plot_ISI_hist, plot_kwargs,
                            xlabel='ISI (ms)', ylabel='Frequency', n_subplots=2,
                            save_dir_img=os.path.join(save_dir_img, 'ISI_hist.png'))

    # plot return maps
    def plot_ISI_return_map(ax, cell_idx, subplot_idx, ISIs_above_cells, ISIs_under_cells, max_ISI, log_scale=False):
        if subplot_idx == 0:
            if log_scale:
                ax.loglog(ISIs_above_cells[cell_idx][:-1], ISIs_above_cells[cell_idx][1:], color='0.5',
                          marker='o', linestyle='', markersize=1, alpha=0.5)
                ax.set_xlim(1, max_ISI)
                ax.set_ylim(1, max_ISI)
                ax.set_xticks([])
                ax.set_xlabel('')
            else:
                ax.plot(ISIs_above_cells[cell_idx][:-1], ISIs_above_cells[cell_idx][1:], color='0.5',
                        marker='o', linestyle='', markersize=1, alpha=0.5)
                ax.set_xlim(0, max_ISI)
                ax.set_ylim(0, max_ISI)
                ax.set_xticks([])
                ax.set_xlabel('')
            ax.annotate('$\geq$ vel. thresh.', xy=(1, max_ISI), textcoords='data',
                        horizontalalignment='left', verticalalignment='top', fontsize=9)
        if subplot_idx == 1:
            if log_scale:
                ax.loglog(ISIs_under_cells[cell_idx][:-1], ISIs_under_cells[cell_idx][1:], color='0.5',
                          marker='o', linestyle='', markersize=1, alpha=0.5)
                ax.set_xlim(1, max_ISI)
                ax.set_ylim(1, max_ISI)
            else:
                ax.plot(ISIs_under_cells[cell_idx][:-1], ISIs_under_cells[cell_idx][1:], color='0.5',
                        marker='o', linestyle='', markersize=1, alpha=0.5)
                ax.set_xlim(0, max_ISI)
                ax.set_ylim(0, max_ISI)
            ax.annotate('$<$ vel. thresh.', xy=(1, max_ISI), textcoords='data',
                        horizontalalignment='left', verticalalignment='top', fontsize=9)

        ax.set_aspect('equal', adjustable='box-forced')

    plot_kwargs = dict(ISIs_above_cells=ISIs_above_cells, ISIs_under_cells=ISIs_under_cells, max_ISI=max_ISI)
    plot_for_all_grid_cells_grid(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                            xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', n_subplots=2,
                            save_dir_img=os.path.join(save_dir_img, 'return_map.png'))

    plot_kwargs = dict(ISIs_above_cells=ISIs_above_cells, ISIs_under_cells=ISIs_under_cells, max_ISI=max_ISI,
                       log_scale=True)
    plot_for_all_grid_cells_grid(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                            xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', n_subplots=2,
                            save_dir_img=os.path.join(save_dir_img, 'return_map_log_scale.png'))
    pl.show()