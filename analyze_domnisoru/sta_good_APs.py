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
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP'
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
    DAP_deflections = {}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], folder_field[(in_field, out_field)],
                                cell_type)

    # main
    sta_mean_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
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

        # get DAP deflections
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

        good_APs = np.logical_and(AP_amps > 53, AP_widths < 0.73)
        v_APs_good = v_APs[good_APs]

        # STA
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs)
        sta_mean, sta_std = get_sta(v_APs_good)
        if len(v_APs_good) > 5:
            sta_mean_good_APs_cells[cell_idx] = sta_mean
            sta_std_good_APs_cells[cell_idx] = sta_std
        else:
            sta_mean_good_APs_cells[cell_idx] = init_nan(len(sta_mean))
            sta_std_good_APs_cells[cell_idx] = init_nan(len(sta_mean))

        # save
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        np.save(os.path.join(save_dir_cell, 'sta_mean.npy'), sta_mean_good_APs_cells[cell_idx])
        np.save(os.path.join(save_dir_cell, 'sta_std.npy'), sta_std_good_APs_cells[cell_idx])

        # plot
        # pl.figure()
        # pl.plot(t_AP, sta_mean, 'k')
        # pl.fill_between(t_AP, sta_mean + sta_std, sta_mean - sta_std,
        #                 facecolor='k', alpha=0.5)
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Membrane Potential (mV)')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'sta.png'))
        #
        # if len(v_APs_good) > 5:
        #     v_APs_plots_good = v_APs_good[np.random.randint(0, len(v_APs_good), 20)]  # reduce to lower number
        # else:
        #     v_APs_plots_good = v_APs_good
        #
        # pl.figure()
        # for cell_idx, v_AP in enumerate(v_APs_plots_good):
        #     pl.plot(t_AP, v_AP)
        # pl.ylabel('Membrane potential (mV)')
        # pl.xlabel('Time (ms)')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'v_APs.png'))
        # #pl.show()

    pl.close('all')


    def plot_sta(ax, cell_idx, t_AP, sta_mean_cells, sta_std_cells):
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')

    def plot_sta_grid(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
                      sta_std_good_APs_cells):
        if subplot_idx == 0:
            ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
            ax.fill_between(t_AP, sta_mean_good_APs_cells[cell_idx] - sta_std_good_APs_cells[cell_idx],
                            sta_mean_good_APs_cells[cell_idx] + sta_std_good_APs_cells[cell_idx], color='0.6')
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylim(-75, -50)
        elif subplot_idx == 1:
            ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
            ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                            sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
            ax.set_ylim(-75, -50)


    if cell_type == 'grid_cells':
        plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_good_APs_cells, sta_std_cells=sta_std_good_APs_cells)
        plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                                save_dir_img=os.path.join(save_dir_img, 'sta.png'))

        plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_cells, sta_std_cells=sta_std_cells,
                           sta_mean_good_APs_cells=sta_mean_good_APs_cells,
                           sta_std_good_APs_cells=sta_std_good_APs_cells
                           )
        plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                save_dir_img=os.path.join(save_dir_img, 'sta_grid.png'))
        pl.show()