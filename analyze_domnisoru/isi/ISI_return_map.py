from __future__ import division
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = True
    max_ISI = 200
    ISI_burst = 8  # ms
    bin_size = 1.0  # ms
    steps = np.arange(0, max_ISI + bin_size, bin_size)
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_' + str(max_ISI))
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs_per_cell = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    median_cells = init_nan((len(cell_ids), len(steps)))
    prob_next_ISI_burst = init_nan((len(cell_ids), len(steps)))
    area_under_curve_cum_prob_next_ISI_burst = np.zeros(len(cell_ids))
    fraction_ISI_or_ISI_next_burst = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # ISIs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        ISIs = get_ISIs(AP_max_idxs, t)
        if filter_long_ISIs:
            ISIs = ISIs[ISIs <= max_ISI]
        n_ISIs[cell_idx] = len(ISIs)
        ISIs_per_cell[cell_idx] = ISIs

        fraction_ISI_or_ISI_next_burst[cell_idx] = float(sum(np.logical_or(ISIs_per_cell[cell_idx][:-1] < ISI_burst,
                                                                           ISIs_per_cell[cell_idx][1:] < ISI_burst))) \
                                                   / len(ISIs_per_cell[cell_idx][1:])

        # running median
        window_size = 5.0  # ms
        prev_ISI = ISIs_per_cell[cell_idx][:-1]
        next_ISI = ISIs_per_cell[cell_idx][1:]
        median = init_nan(len(steps))
        mean = init_nan(len(steps))
        for i, s in enumerate(steps):
            idx = np.logical_and(s - window_size / 2.0 <= prev_ISI, prev_ISI <= s + window_size / 2.0)
            median_cells[cell_idx, i] = np.median(next_ISI[idx])
            mean[i] = np.mean(next_ISI[idx])

            # probability next ISI < x ms
            prob_next_ISI_burst[cell_idx, i] = np.sum(next_ISI[idx] < ISI_burst) / float(np.sum(idx))

        # area under prob_next_ISI_burst
        prob_normed = prob_next_ISI_burst[cell_idx, :] / np.nansum(prob_next_ISI_burst[cell_idx, :])
        cum_prob = np.nancumsum(prob_normed)
        area_under_curve_cum_prob_next_ISI_burst[cell_idx] = np.sum(cum_prob * bin_size) / (len(cum_prob) * bin_size)

        # save and plot
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # 2d return
        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(ISIs_per_cell[cell_idx][:-1], ISIs_per_cell[cell_idx][1:], color='0.5', marker='o',
                linestyle='', markersize=3)
        pl.plot(steps, median_cells[cell_idx, :], 'k', label='median')
        #pl.plot(steps, mean, 'b')
        pl.xlabel('ISI[n] (ms)')
        pl.ylabel('ISI[n+1] (ms)')
        pl.xlim(0, max_ISI)
        pl.ylim(0, max_ISI)
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map.png'))
        #pl.show()

        pl.figure()
        pl.title(cell_id, fontsize=16)
        pl.plot(steps, prob_next_ISI_burst[cell_idx, :], 'k')
        pl.xlabel('ISI[n] (ms)')
        pl.ylabel('Prob. ISI[n+1] < %i ms' % ISI_burst)
        pl.tight_layout()
        #pl.show()

        # 3d return
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(cell_id, fontsize=16)
        ax.scatter(ISIs_per_cell[cell_idx][:-2], ISIs_per_cell[cell_idx][1:-1], ISIs_per_cell[cell_idx][2:],
                   color='k', marker='o') #, markersize=6)
        ax.set_xlabel('ISI[n] (ms)', fontsize=16)
        ax.set_ylabel('ISI[n+1] (ms)', fontsize=16)
        ax.set_zlabel('ISI[n+2] (ms)', fontsize=16)
        ax.set_xlim3d(0, max_ISI)
        ax.set_ylim3d(max_ISI, 0)
        ax.set_zlim3d(0, max_ISI)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ISI_return_map_3d.png'))
        #pl.show()
        pl.close('all')

    # # fraction single between bursty and non-bursty
    # from scipy.stats import ttest_ind
    # burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    # _, p_val = ttest_ind(fraction_ISI_or_ISI_next_burst[burst_label], fraction_ISI_or_ISI_next_burst[~burst_label])
    # print 'p_val: ', p_val
    # pl.figure()
    # pl.plot(np.zeros(sum(burst_label)), fraction_ISI_or_ISI_next_burst[burst_label], 'or')
    # pl.plot(np.ones(sum(~burst_label)), fraction_ISI_or_ISI_next_burst[~burst_label], 'ob')

    # save and plot
    np.save(os.path.join(save_dir_img, 'fraction_ISI_or_ISI_next_burst.npy'), fraction_ISI_or_ISI_next_burst)

    def plot_ISI_return_map(ax, cell_idx, ISIs_per_cell, max_ISI, median_cells=None, log_scale=False):
        if log_scale:
            ax.loglog(ISIs_per_cell[cell_idx][:-1], ISIs_per_cell[cell_idx][1:], color='0.5',
                    marker='o', linestyle='', markersize=1, alpha=0.5)
            ax.set_xlim(1, max_ISI)
            ax.set_ylim(1, max_ISI)
        else:
            ax.plot(ISIs_per_cell[cell_idx][:-1], ISIs_per_cell[cell_idx][1:], color='0.5',
                    marker='o', linestyle='', markersize=1, alpha=0.5)
            ax.set_xlim(0, max_ISI)
            ax.set_ylim(0, max_ISI)
            ax.set_xticks(np.arange(0, max_ISI + 50, 50))
            ax.set_yticks(np.arange(0, max_ISI + 50, 50))
        ax.set_aspect('equal', adjustable='box-forced')
        if median_cells is not None:
            ax.plot(steps, median_cells[cell_idx, :], 'k', label='median')


    if cell_type == 'grid_cells':
        burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
        colors_marker = np.zeros(len(burst_label), dtype=str)
        colors_marker[burst_label] = 'r'
        colors_marker[~burst_label] = 'b'

        params = {'legend.fontsize': 9}
        pl.rcParams.update(params)

        # plot return maps
        plot_kwargs = dict(ISIs_per_cell=ISIs_per_cell, max_ISI=max_ISI)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)',
                                save_dir_img=os.path.join(save_dir_img, 'return_map.png'))
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', colors_marker=colors_marker,
                                wspace=0.18, save_dir_img=os.path.join(save_dir_img2, 'ISI_return_map.png'))

        plot_kwargs = dict(ISIs_per_cell=ISIs_per_cell, max_ISI=max_ISI, log_scale=True)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', colors_marker=colors_marker,
                                save_dir_img=os.path.join(save_dir_img, 'return_map_log_scale.png'))

        plot_kwargs = dict(ISIs_per_cell=ISIs_per_cell, max_ISI=max_ISI, median_cells=median_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)',
                                save_dir_img=os.path.join(save_dir_img, 'return_map_with_median.png'))

        # plot prob. next ISI < x ms
        def plot_prob_next_ISI_burst(ax, cell_idx, steps, prob_next_ISI_short, max_ISI):
            ax.plot(steps, prob_next_ISI_short[cell_idx, :], 'k')
            ax.set_xlim(0, max_ISI)
            ax.set_ylim(0, 1)

        plot_kwargs = dict(steps=steps, prob_next_ISI_short=prob_next_ISI_burst, max_ISI=max_ISI)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_prob_next_ISI_burst, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='Prob. ISI[n+1] < %i ms' % ISI_burst,
                                save_dir_img=os.path.join(save_dir_img, 'prob_next_ISI_burst.png'))

        # plot cumulative prob. next ISI < x ms
        def plot_cum_prob_next_ISI_burst(ax, cell_idx, steps, prob_next_ISI_short, max_ISI):
            prob_normed = prob_next_ISI_short[cell_idx, :] / np.nansum(prob_next_ISI_short[cell_idx, :])
            ax.plot(steps, np.nancumsum(prob_normed), 'k', drawstyle='steps-post')
            ax.set_xlim(0, max_ISI)
            ax.set_ylim(0, 1)

        plot_kwargs = dict(steps=steps, prob_next_ISI_short=prob_next_ISI_burst, max_ISI=max_ISI)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_cum_prob_next_ISI_burst, plot_kwargs,
                                xlabel='ISI[n] (ms)', ylabel='Cum. prob. \nISI[n+1] < %i ms' % ISI_burst,
                                save_dir_img=os.path.join(save_dir_img, 'cum_prob_next_ISI_burst.png'))

        # plot area of cum. prob. next ISI < x ms
        def plot_area(ax, cell_idx, area_under_curve_cum_prob_next_ISI_burst):
            ax.bar(0.5, area_under_curve_cum_prob_next_ISI_burst[cell_idx], width=0.4, color='0.5')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.axhline(0.25, linestyle='--', color='0.5')
            ax.axhline(0.5, linestyle='--', color='0.5')
            ax.axhline(0.75, linestyle='--', color='0.5')

        plot_kwargs = dict(area_under_curve_cum_prob_next_ISI_burst=area_under_curve_cum_prob_next_ISI_burst)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_area, plot_kwargs,
                                xlabel='', ylabel='Area cum. prob. \nISI[n+1] < %i ms' % ISI_burst,
                                save_dir_img=os.path.join(save_dir_img, 'area_cum_prob_next_ISI_burst.png'))
        pl.show()