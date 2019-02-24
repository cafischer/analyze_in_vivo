from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from analyze_in_vivo.analyze_domnisoru.sta import plot_sta_grid_on_ax
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_all_grid_cells_grid, \
    plot_with_markers
pl.style.use('paper_subplots')


def get_sta_criterion(do_detrend, before_AP, after_AP, AP_criterion):
    sta_mean_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        AP_max_idxs = data['spiketimes']

        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt
        if v_APs is None:
            continue

        # get AP amp. and width
        AP_amps = np.zeros(len(v_APs))
        AP_widths = np.zeros(len(v_APs))
        for i, v_AP in enumerate(v_APs):
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
            AP_amps[i], AP_widths[i] = get_spike_characteristics(v_AP, t_AP, ['AP_amp', 'AP_width'],
                                                                 v_rest=v_AP[before_AP_idx - to_idx(t_vref, dt)],
                                                                 AP_max_idx=before_AP_idx,
                                                                 AP_onset=before_AP_idx - to_idx(1.0, dt),
                                                                 std_idx_times=(0, 1), check=False,
                                                                 **spike_characteristics_dict)
            # if np.isnan(AP_widths[i]):
            #     get_spike_characteristics(v_AP, t_AP, ['AP_amp', 'AP_width'],
            #                               v_rest=v_AP[before_AP_idx - to_idx(t_v_ref, dt)],
            #                               AP_max_idx=before_AP_idx,
            #                               AP_onset=before_AP_idx - to_idx(1.0, dt),
            #                               std_idx_times=(0, 1), check=True,
            #                               **spike_characteristics_dict)

        # select APs
        if AP_criterion.keys()[0] == 'quantile':
            AP_amp_thresh = np.nanpercentile(AP_amps, 100 - AP_criterion.values()[0])
            AP_width_thresh = np.nanpercentile(AP_widths, AP_criterion.values()[0])
        elif AP_criterion.keys()[0] == 'AP_amp_and_width':
            AP_amp_thresh = AP_criterion.values()[0][0]
            AP_width_thresh = AP_criterion.values()[0][1]
        else:
            raise ValueError('AP criterion does not exist!')

        good_APs = np.logical_and(AP_amps >= AP_amp_thresh, AP_widths <= AP_width_thresh)
        v_APs_good = v_APs[good_APs]

        # STA
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs)
        if len(v_APs_good) > 5:
            sta_mean_good_APs_cells[cell_idx], sta_std_good_APs_cells[cell_idx] = get_sta(v_APs_good)
        else:
            sta_mean_good_APs_cells[cell_idx] = init_nan(len(sta_mean_cells[cell_idx]))
            sta_std_good_APs_cells[cell_idx] = init_nan(len(sta_mean_cells[cell_idx]))

    return sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells, sta_std_good_APs_cells


def plot_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
                        sta_std_good_APs_cells, before_AP, after_AP, ylims_upper=(None, None), ylims_lower=(None, None)):
    if subplot_idx == 0:
        ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_good_APs_cells[cell_idx] - sta_std_good_APs_cells[cell_idx],
                        sta_mean_good_APs_cells[cell_idx] + sta_std_good_APs_cells[cell_idx], color='0.6')
        ax.set_xlabel('')
        ax.set_ylim(*ylims_upper)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP + 5, 10))
        ax.set_xticklabels([])
        if cell_idx == 0 or cell_idx == 9 or cell_idx == 18:
            ax.set_ylabel('$r_k$')
    elif subplot_idx == 1:
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
        ax.set_ylim(*ylims_lower)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP+5, 10))
        ax.set_xticklabels(np.arange(-before_AP, after_AP+5, 10))
        if cell_idx == 0 or cell_idx == 9 or cell_idx == 18:
            ax.set_ylabel(r'$\frac{\Delta Mem. pot. (mV)}{\Delta Time (ms)}$')

if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    t_vref = 10  # ms
    dt = 0.05  # ms
    do_detrend = False
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], 't_vref_'+str(t_vref))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # AP_criterions = [{'quantile': 20}, {'quantile': 10}, {'AP_amp_and_width': (40, 1)}]
    # time_before_after_AP = [(20, 25), (25, 20), (25, 25), (25, 30), (30, 25), (30, 30)]  # (before_AP, after_AP)

    AP_criterions = [{'AP_amp_and_width': (40, 1)}]
    time_before_after_AP = [(25, 25)]  # (before_AP, after_AP)

    # AP_criterions = [{'AP_amp_and_width': (51.8, 0.72)}]
    # time_before_after_AP = [(10, 25)]  # (before_AP, after_AP)

    # main
    for AP_criterion in AP_criterions:
        for (before_AP, after_AP) in time_before_after_AP:
            print AP_criterion, (before_AP, after_AP)
            (sta_mean_cells, sta_std_cells,
             sta_mean_good_APs_cells, sta_std_good_APs_cells) = get_sta_criterion(do_detrend, before_AP, after_AP,
                                                                                  AP_criterion)

            t_AP = np.arange(-before_AP, after_AP + dt, dt)
            plot_kwargs = dict(t_AP=t_AP,
                               sta_mean_cells=sta_mean_cells,
                               sta_std_cells=sta_std_cells,
                               sta_mean_good_APs_cells=sta_mean_good_APs_cells,
                               sta_std_good_APs_cells=sta_std_good_APs_cells,
                               before_AP=before_AP,
                               after_AP=after_AP,
                               ylims=(-75, -45)
                               )

            fig_title = 'Criterion: ' + AP_criterion.keys()[0].replace('_', ' ') + ' ' + str(AP_criterion.values()[0]) \
                        + '  Time range (before AP, after AP): ' + str((before_AP, after_AP))
            fig_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                       + '_before_after_AP_' + str((before_AP, after_AP)) + '.png'

            # plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid_on_ax, plot_kwargs,
            #                              xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
            #                              fig_title=fig_title,
            #                              save_dir_img=os.path.join(save_dir_img, fig_name))

            #pl.show()
            sta_1derivative_cells = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0])-2))
            k = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0])-2))
            for cell_idx in range(len(cell_ids)):
                # osculating circle
                t_1derivative = np.diff(t_AP) / dt
                t_2derivative = np.diff(t_1derivative) / dt
                sta_1derivative = np.diff(sta_mean_good_APs_cells[cell_idx]) / dt
                sta_2derivative = np.diff(sta_1derivative) / dt
                t_1derivative = t_1derivative[:-1]
                sta_1derivative = sta_1derivative[:-1]

                k[cell_idx, :] = np.abs(((t_1derivative**2 + sta_1derivative**2)**(3./2)) / (t_1derivative*sta_2derivative - t_2derivative*sta_1derivative))
                sta_1derivative_cells[cell_idx, :] = sta_1derivative


            sta_mean_good_APs_cells = [a[:-2] for a in sta_mean_good_APs_cells]
            sta_std_good_APs_cells = [a[:-2] for a in sta_std_good_APs_cells]

            plot_kwargs = dict(t_AP=t_AP[:-2],
                               sta_mean_cells=sta_1derivative_cells,
                               sta_std_cells=np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0]))),
                               sta_mean_good_APs_cells=k,
                               sta_std_good_APs_cells=np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0]))),
                               before_AP=before_AP,
                               after_AP=after_AP,
                               ylims_upper=(0, 20),
                               ylims_lower=(-3.5, 3.5),
                               )
            plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_grid_on_ax, plot_kwargs,
                                         xlabel='Time (ms)', ylabel='', n_subplots=2,
                                         fig_title=fig_title,
                                         save_dir_img=os.path.join(save_dir_img, 'test.png'))
            pl.show()