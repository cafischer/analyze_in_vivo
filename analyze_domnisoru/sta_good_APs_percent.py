from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_all_grid_cells_grid, \
    plot_with_markers
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP/percent'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_threshold = 'AP width'  # 'AP amplitude', 'AP width', 'both'
    quantile = 20  # percent
    use_AP_max_idxs_domnisoru = True
    do_detrend = False
    in_field = False
    out_field = False
    before_AP = 10
    after_AP = 25
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, cell_type, 'quantile_'+str(quantile), use_threshold)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

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
                                                                 v_rest=v_AP[before_AP_idx - to_idx(5.0, dt)],
                                                                 AP_max_idx=before_AP_idx,
                                                                 AP_onset=before_AP_idx - to_idx(1.0, dt),
                                                                 std_idx_times=(0, 1), check=False,
                                                                 **spike_characteristics_dict)

        AP_amp_thresh = np.nanpercentile(AP_amps, 100 - quantile)
        AP_width_thresh = np.nanpercentile(AP_widths, quantile)
        if use_threshold == 'AP amplitude':
            good_APs = AP_amps >= AP_amp_thresh
        elif use_threshold == 'AP width':
            good_APs = AP_widths <= AP_width_thresh
        elif use_threshold == 'both':
            good_APs = np.logical_and(AP_amps >= AP_amp_thresh, AP_widths <= AP_width_thresh)
        v_APs_good = v_APs[good_APs]

        # STA
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs)
        sta_mean, sta_std = get_sta(v_APs_good)
        sta_diff_cells[cell_idx] = np.diff(sta_mean_cells[cell_idx])  # derivative
        if len(v_APs_good) > 2:
            sta_mean_good_APs_cells[cell_idx] = sta_mean
            sta_std_good_APs_cells[cell_idx] = sta_std
            sta_diff_good_APs_cells[cell_idx] = np.diff(sta_mean_good_APs_cells[cell_idx])  # derivative
        else:
            sta_mean_good_APs_cells[cell_idx] = init_nan(len(sta_mean))
            sta_std_good_APs_cells[cell_idx] = init_nan(len(sta_mean))
            sta_diff_good_APs_cells[cell_idx] = init_nan(len(sta_mean)-1)


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

        elif subplot_idx == 1:
            if ~np.any(np.isnan(sta_mean_cells[cell_idx])):
                ax.fill_between((0, 3.5), ylim[0], ylim[1], color='0.8')
            ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
            ax.set_ylim(ylim)
            ax.annotate('all APs', xy=(25, ylim[0]), textcoords='data',
                        horizontalalignment='right', verticalalignment='bottom', fontsize=8)


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
        if use_threshold == 'AP amplitude':
            quantile_str = str(100 - quantile)
        elif use_threshold == 'AP width':
            quantile_str = str(quantile)
        elif use_threshold == 'both':
            quantile_str = str(100 - quantile)+', '+str(quantile)

        plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_good_APs_cells, sta_std_cells=sta_std_good_APs_cells)
        plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                                fig_title='Threshold by '+use_threshold+' (quantile: ' + quantile_str+')',
                                save_dir_img=os.path.join(save_dir_img, 'sta_selected_APs.png'))

        plot_kwargs = dict(t_AP=t_AP,
                           sta_mean_cells=sta_mean_cells,
                           sta_std_cells=sta_std_cells,
                           sta_mean_good_APs_cells=sta_mean_good_APs_cells,
                           sta_std_good_APs_cells=sta_std_good_APs_cells,
                           ylim=(-75, -45)  #(-75, -50)
                           )
        plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                fig_title='Threshold by '+use_threshold+' (quantile: ' + quantile_str+')',
                                save_dir_img=os.path.join(save_dir_img, 'sta_selected_and_all_APs.png'))

        plot_kwargs = dict(t_AP=t_AP[:-1],
                           sta_mean_cells=sta_diff_cells,
                           sta_mean_good_APs_cells=sta_diff_good_APs_cells,
                           ylim=(-0.2, 0.2),
                           diff_selected_all=diff_selected_all
                           )
        plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_derivative_grid, plot_kwargs,
                                     xlabel='Time (ms)', ylabel=r'$\frac{Mem. pot.}{Time} \left(\frac{mV}{ms}\right)$',
                                     n_subplots=2,
                                     fig_title='Threshold by '+use_threshold+' (quantile: ' + quantile_str+')',
                                     save_dir_img=os.path.join(save_dir_img, 'sta_selected_APs_derivative.png'))
        pl.show()