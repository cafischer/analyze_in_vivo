from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_all_grid_cells_grid
from scipy.interpolate import UnivariateSpline
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP'
    save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_AP_max_idxs_domnisoru = True
    do_detrend = False
    in_field = False
    out_field = False
    before_AP = 10
    after_AP = 25
    DAP_deflections = init_nan(len(cell_ids))
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], folder_field[(in_field, out_field)],
                                cell_type)

    # main
    sta_mean_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_diff_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_diff_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']

        if in_field:
            in_field_len_orig = np.load(
                os.path.join(save_dir_in_out_field, cell_type, cell_id, 'in_field_len_orig.npy'))
            AP_max_idxs_selected = AP_max_idxs[in_field_len_orig[AP_max_idxs]]
        elif out_field:
            out_field_len_orig = np.load(
                os.path.join(save_dir_in_out_field, cell_type, cell_id, 'out_field_len_orig.npy'))
            AP_max_idxs_selected = AP_max_idxs[out_field_len_orig[AP_max_idxs]]
        else:
            AP_max_idxs_selected = AP_max_idxs

        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs_selected, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt
        if v_APs is None:
            continue

        # get AP amp. and width
        AP_amps = np.zeros(len(v_APs))
        AP_widths = np.zeros(len(v_APs))
        for i, v_AP in enumerate(v_APs):
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
            AP_amps[i], AP_widths[i] = get_spike_characteristics(v_AP, t_AP, ['AP_amp', 'AP_width'],
                                                                 v_AP[before_AP - to_idx(1.0, dt)],
                                                                 AP_max_idx=before_AP_idx,
                                                                 AP_onset=before_AP_idx - to_idx(1.0, dt),
                                                                 std_idx_times=(0, 1), check=False,
                                                                 **spike_characteristics_dict)

        good_APs = np.logical_and(AP_amps > 55.1, AP_widths < 0.72)  # TODO  np.logical_and(AP_amps > 52.82, AP_widths < 0.72)
        v_APs_good = v_APs[good_APs]

        # STA
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs)
        sta_mean, sta_std = get_sta(v_APs_good)
        sta_diff_cells[cell_idx] = np.diff(sta_mean_cells[cell_idx])  # derivative
        if len(v_APs_good) > 5:
            sta_mean_good_APs_cells[cell_idx] = sta_mean
            sta_std_good_APs_cells[cell_idx] = sta_std

            # get DAP deflection
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=False)  # dont want interpolation
            DAP_deflections[cell_idx], DAP_max_idx = get_spike_characteristics(sta_mean_good_APs_cells[cell_idx], t_AP,
                                                                  ['DAP_deflection', 'DAP_max_idx'],
                                                                  sta_mean_good_APs_cells[cell_idx][before_AP - to_idx(1.0, dt)],
                                                                  AP_max_idx=before_AP_idx,
                                                                  AP_onset=before_AP_idx - to_idx(1.0, dt), check=False,
                                                                  **spike_characteristics_dict)
            if DAP_max_idx is not None:
                sem_at_DAP = sta_std_good_APs_cells[cell_idx][DAP_max_idx] / np.sqrt(len(v_APs_good))
                if DAP_deflections[cell_idx] > sem_at_DAP:
                    print ' DAP deflection > sem: ', cell_id

            sta_diff_good_APs_cells[cell_idx] = np.diff(sta_mean_good_APs_cells[cell_idx])  # derivative
        else:
            sta_mean_good_APs_cells[cell_idx] = init_nan(len(sta_mean))
            sta_std_good_APs_cells[cell_idx] = init_nan(len(sta_mean))
            sta_diff_good_APs_cells[cell_idx] = init_nan(len(sta_mean)-1)

        # save
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        np.save(os.path.join(save_dir_cell, 'sta_mean.npy'), sta_mean_good_APs_cells[cell_idx])
        np.save(os.path.join(save_dir_cell, 'sta_std.npy'), sta_std_good_APs_cells[cell_idx])

    print 'Cells with DAP deflection: ', np.array(cell_ids)[DAP_deflections > 0]


    def plot_sta(ax, cell_idx, t_AP, sta_mean_cells, sta_std_cells):
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')

    def plot_sta_grid(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
                      sta_std_good_APs_cells, ylim=(-75, -50)):
        if subplot_idx == 0:
            ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
            ax.fill_between(t_AP, sta_mean_good_APs_cells[cell_idx] - sta_std_good_APs_cells[cell_idx],
                            sta_mean_good_APs_cells[cell_idx] + sta_std_good_APs_cells[cell_idx], color='0.6')
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylim(ylim)
            ax.annotate('selected APs', xy=(25, ylim[0]), textcoords='data',
                        horizontalalignment='right', verticalalignment='bottom', fontsize=8)
        elif subplot_idx == 1:
            ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
            ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                            sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
            ax.set_ylim(ylim)
            ax.annotate('all APs', xy=(25, ylim[0]), textcoords='data',
                        horizontalalignment='right', verticalalignment='bottom', fontsize=8)

    def plot_sta_derivative_grid(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_mean_good_APs_cells,
                                 ylim=(-0.2, -0.2), diff_selected_all=None):

        if subplot_idx == 0:
            if ~np.any(np.isnan(sta_mean_good_APs_cells[cell_idx])):
                ax.fill_between((0, 3.5), ylim[0], ylim[1], color='0.8')
            ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylim(ylim)
            ax.annotate('selected APs', xy=(25, ylim[0]), textcoords='data',
                        horizontalalignment='right', verticalalignment='bottom', fontsize=8)
            ax.annotate('%.3f' % diff_selected_all[cell_idx], xy=(25, ylim[1]), textcoords='data',
                        horizontalalignment='right', verticalalignment='top', fontsize=8)

            # # smooth
            # std = np.std(sta_mean_good_APs_cells[cell_idx][to_idx(2, dt):to_idx(3, dt)])
            # #std = sta_std_cells[cell_idx][:-1]
            # w = np.ones(len(sta_mean_good_APs_cells[cell_idx])) / std
            # print 'w1', w[0]
            # splines = UnivariateSpline(t_AP, sta_mean_good_APs_cells[cell_idx], w=w, s=None, k=3)
            # smoothed = splines(t_AP)
            #
            # ax.plot(t_AP, smoothed, 'r')

        elif subplot_idx == 1:
            if ~np.any(np.isnan(sta_mean_cells[cell_idx])):
                ax.fill_between((0, 3.5), ylim[0], ylim[1], color='0.8')
            ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
            ax.set_ylim(ylim)
            ax.annotate('all APs', xy=(25, ylim[0]), textcoords='data',
                        horizontalalignment='right', verticalalignment='bottom', fontsize=8)

            # # smooth
            # std = np.std(sta_mean_cells[cell_idx][to_idx(2, dt):to_idx(3, dt)])
            # #std = sta_std_good_APs_cells[cell_idx][:-1]
            # w = np.ones(len(sta_mean_cells[cell_idx])) / std
            # print 'w2', w[0]
            # splines = UnivariateSpline(t_AP, sta_mean_cells[cell_idx], w=w, s=None, k=3)
            # smoothed = splines(t_AP)
            #
            # ax.plot(t_AP, smoothed, 'r')


    max_after_0 = np.array([np.max(sta[before_AP_idx:before_AP_idx+to_idx(3.5, dt)]) for sta in sta_diff_cells])
    max_after_0_good =  np.array([np.max(sta[before_AP_idx:before_AP_idx+to_idx(3.5, dt)]) for sta in sta_diff_good_APs_cells])
    comp_max = (max_after_0_good > max_after_0).astype(float)
    comp_max[np.isnan(max_after_0)] = np.nan
    diff_selected_all = max_after_0_good - max_after_0
    print comp_max
    print diff_selected_all
    # cell_ids_DAP_idx = np.array([np.where(id==np.array(cell_ids))[0][0] for id in cell_ids_DAP])

    if cell_type == 'grid_cells':
        t_AP = np.arange(-before_AP_idx, after_AP_idx + 1) * dt

        plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_good_APs_cells, sta_std_cells=sta_std_good_APs_cells)
        plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                                save_dir_img=os.path.join(save_dir_img, 'sta.png'))

        plot_kwargs = dict(t_AP=t_AP,
                           sta_mean_cells=sta_mean_cells,
                           sta_std_cells=sta_std_cells,
                           sta_mean_good_APs_cells=sta_mean_good_APs_cells,
                           sta_std_good_APs_cells=sta_std_good_APs_cells,
                           ylim=(-75, -45)  #(-75, -50)
                           )
        plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                save_dir_img=os.path.join(save_dir_img, 'sta_grid.png'))

        plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                save_dir_img=os.path.join(save_dir_img2, 'sta_selected_APs.png'))

        plot_kwargs = dict(t_AP=t_AP[:-1],
                           sta_mean_cells=sta_diff_cells,
                           sta_mean_good_APs_cells=sta_diff_good_APs_cells,
                           ylim=(-0.2, 0.2),
                           diff_selected_all=diff_selected_all
                           )
        plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_derivative_grid, plot_kwargs,
                                     xlabel='Time (ms)', ylabel=r'$\frac{Mem. pot.}{Time} \left(\frac{mV}{ms}\right)$',
                                     n_subplots=2,
                                     save_dir_img=os.path.join(save_dir_img2, 'sta_selected_APs_derivative.png'))
        pl.show()