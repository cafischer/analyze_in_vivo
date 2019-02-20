from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta
from grid_cell_stimuli import find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_schmidt_hieber import detrend

pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/conditioned_on_voltage'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    do_detrend = True
    use_mean_or_std = 'mean'
    before_AP_times = [25]  #np.logspace(1, 10, 10, base=2)
    cut_off = 5
    after_AP = 25
    percentile = 10  # [0, 100]
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    sta_mean_greater_cells = np.zeros((len(cell_ids), len(before_AP_times)), dtype=object)
    sta_std_greater_cells = np.zeros((len(cell_ids), len(before_AP_times)), dtype=object)
    sta_mean_lower_cells = np.zeros((len(cell_ids), len(before_AP_times)), dtype=object)
    sta_std_lower_cells = np.zeros((len(cell_ids), len(before_AP_times)), dtype=object)
    t_AP = np.zeros((len(before_AP_times)), dtype=object)
    DAP_deflection_greater_per_cell = np.zeros((len(cell_ids), len(before_AP_times)))
    DAP_width_greater_per_cell = np.zeros((len(cell_ids), len(before_AP_times)))
    DAP_time_greater_per_cell = np.zeros((len(cell_ids), len(before_AP_times)))
    DAP_deflection_lower_per_cell = np.zeros((len(cell_ids), len(before_AP_times)))
    DAP_width_lower_per_cell = np.zeros((len(cell_ids), len(before_AP_times)))
    DAP_time_lower_per_cell = np.zeros((len(cell_ids), len(before_AP_times)))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        if do_detrend:
            # # test detrend
            # pl.figure()
            # pl.plot(t, v, 'k')
            # pl.plot(t, detrend(v, t, cutoff_freq=5), 'r')
            # pl.show()

            v = detrend(v, t, cutoff_freq=5)

        # get APs
        AP_max_idxs = data['spiketimes']
        after_AP_idx = to_idx(after_AP, dt)

        for con_idx, before_AP in enumerate(before_AP_times):
            before_AP_idx = to_idx(before_AP, dt)
            v_APs = find_all_AP_traces(v, before_AP_idx+to_idx(cut_off, dt), after_AP_idx, AP_max_idxs, AP_max_idxs)
                # extend before_AP_idx to cut out AP leftovers in the beginning
            v_APs = v_APs[:, to_idx(cut_off, dt):]
            t_AP[con_idx] = np.arange(np.size(v_APs, 1)) * dt

            # mean voltage before AP
            v_condition = np.zeros(len(v_APs))
            for i, v_AP in enumerate(v_APs):
                if use_mean_or_std == 'mean':
                    v_condition[i] = np.mean(v_AP[to_idx(1, dt):before_AP_idx - to_idx(1, dt)])

                    # # to test cutting the right region
                    # pl.figure()
                    # t_AP = np.arange(len(v_AP)) * dt
                    # pl.plot(t_AP, v_AP, 'k')
                    # pl.plot(t_AP[to_idx(1, dt):before_AP_idx - to_idx(1, dt)],
                    #         v_AP[to_idx(1, dt):before_AP_idx - to_idx(1, dt)], 'r')
                    # pl.show()
                elif use_mean_or_std == 'std':
                    v_condition[i] = np.std(v_AP[to_idx(1, dt):to_idx(1, dt) + before_AP_idx])
            percentile_lower = np.percentile(v_condition, percentile)
            percentile_greater = np.percentile(v_condition, 100 - percentile)
            v_APs_greater = v_APs[v_condition > percentile_greater]
            v_APs_lower = v_APs[v_condition < percentile_lower]

            # STA
            sta_mean_greater_cells[cell_idx, con_idx], sta_std_greater_cells[cell_idx, con_idx] = get_sta(v_APs_greater)
            sta_mean_lower_cells[cell_idx, con_idx], sta_std_lower_cells[cell_idx, con_idx] = get_sta(v_APs_lower)
            # pl.figure()
            # pl.plot(t_AP[0], np.mean(v_APs_lower, 0), 'b')
            # pl.plot(t_AP[0], np.mean(v_APs_greater, 0), 'r')
            # # plot_APs(v_APs_lower, t_AP[con_idx], None)
            # # plot_APs(v_APs_greater, t_AP[con_idx], None)
            # pl.show()

            # get DAP characteristics
            spike_characteristics_dict = get_spike_characteristics_dict()
            spike_characteristics_dict['AP_max_idx'] = before_AP_idx
            spike_characteristics_dict['AP_onset'] = before_AP_idx + to_idx(1, dt)
            DAP_time_greater_per_cell[cell_idx, con_idx], DAP_deflection_greater_per_cell[cell_idx, con_idx], \
            DAP_width_greater_per_cell[cell_idx, con_idx] = get_spike_characteristics(sta_mean_greater_cells[cell_idx, con_idx],
                                                                                      t_AP[con_idx],
                                                                                      ['fAHP2DAP_time', 'DAP_deflection', 'DAP_width'],
                                                                                      sta_mean_greater_cells[cell_idx, con_idx][0],
                                                                                      check=False, **spike_characteristics_dict)
            DAP_time_lower_per_cell[cell_idx, con_idx], DAP_deflection_lower_per_cell[cell_idx, con_idx], \
            DAP_width_lower_per_cell[cell_idx, con_idx] = get_spike_characteristics(sta_mean_lower_cells[cell_idx, con_idx],
                                                                                    t_AP[con_idx],
                                                                                    ['fAHP2DAP_time', 'DAP_deflection', 'DAP_width'],
                                                                                    sta_mean_lower_cells[cell_idx, con_idx][0],
                                                                                    check=False, **spike_characteristics_dict)

    # plots
    if use_mean_or_std == 'mean':
        labels = ['$\mu$ > '+str(100-percentile)+'th percentile', '$\mu$ < '+str(percentile)+'th percentile']
    elif use_mean_or_std == 'std':
        labels = ['$\sigma$ > '+str(100-percentile)+'th percentile', '$\sigma$ < '+str(percentile)+'th percentile']

    if use_mean_or_std == 'mean':
        save_dir_img = os.path.join(save_dir_img, 'conditioned_on_mean')
    elif use_mean_or_std == 'std':
        save_dir_img = os.path.join(save_dir_img, 'conditioned_on_std')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    for con_idx, before_AP in enumerate(before_AP_times):
        save_dir_con = os.path.join(save_dir_img, 'before_AP_time_%i' % before_AP)
        if not os.path.exists(save_dir_con):
            os.makedirs(save_dir_con)

        def plot_sta(ax, cell_idx, t_AP, sta_mean_greater_cells, sta_std_greater_cells,
                     sta_mean_lower_cells, sta_std_lower_cells):
            offset = 10
            ax.plot(t_AP, sta_mean_greater_cells[cell_idx] + offset, 'r', label=labels[0])
            ax.fill_between(t_AP,
                            sta_mean_greater_cells[cell_idx] - sta_std_greater_cells[cell_idx] + offset,
                            sta_mean_greater_cells[cell_idx] + sta_std_greater_cells[cell_idx] + offset,
                            color='r', alpha=0.5)
            ax.plot(t_AP, sta_mean_lower_cells[cell_idx], 'b', label=labels[1])
            ax.fill_between(t_AP, sta_mean_lower_cells[cell_idx] - sta_std_lower_cells[cell_idx],
                            sta_mean_lower_cells[cell_idx] + sta_std_lower_cells[cell_idx],
                            color='b', alpha=0.5)
            if cell_idx == 8:
                legend = ax.legend(fontsize=10)
                ax.add_artist(legend)

        plot_kwargs = dict(t_AP=t_AP[con_idx], sta_mean_greater_cells=sta_mean_greater_cells[:, con_idx],
                           sta_std_greater_cells=sta_std_greater_cells[:, con_idx],
                           sta_mean_lower_cells=sta_mean_lower_cells[:, con_idx],
                           sta_std_lower_cells=sta_std_lower_cells[:, con_idx])

        plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. (mV)', wspace=0.1,
                                save_dir_img=os.path.join(save_dir_con, 'sta.png'))
        pl.show()

        # pl.figure()
        # lim_max = max(np.nanmax(DAP_deflection_lower_per_cell), np.nanmax(DAP_deflection_greater_per_cell)) + 0.1
        # pl.plot(np.arange(0, lim_max + 0.1, 0.1), np.arange(0, lim_max + 0.1, 0.1), color='0.5', linestyle='--')
        # pl.plot(DAP_deflection_lower_per_cell, DAP_deflection_greater_per_cell, 'ok')
        # for cell_idx in range(len(cell_ids)):
        #     pl.annotate(cell_ids[cell_idx], xy=(DAP_deflection_lower_per_cell[cell_idx] + 0.05,
        #                                         DAP_deflection_greater_per_cell[cell_idx] + 0.05))
        # pl.title('DAP deflection (mV)')
        # pl.xlim(0, lim_max)
        # pl.ylim(0, lim_max)
        # pl.xlabel(labels[1])
        # pl.ylabel(labels[0])
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'DAP_deflection.png'))
        #
        # pl.figure()
        # lim_min = min(np.nanmin(DAP_time_lower_per_cell), np.nanmin(DAP_time_greater_per_cell)) + 0.1
        # lim_max = max(np.nanmax(DAP_time_lower_per_cell), np.nanmax(DAP_time_greater_per_cell)) + 0.1
        # pl.plot(np.arange(lim_min - 0.1, lim_max + 0.1, 0.1), np.arange(lim_min - 0.1, lim_max + 0.1, 0.1),
        #         color='0.5', linestyle='--')
        # pl.plot(DAP_time_lower_per_cell, DAP_time_greater_per_cell, 'ok')
        # for cell_idx in range(len(cell_ids)):
        #     pl.annotate(cell_ids[cell_idx], xy=(DAP_time_lower_per_cell[cell_idx] + 0.05,
        #                                         DAP_time_greater_per_cell[cell_idx] + 0.05))
        # pl.title('Time fAHP - DAP')
        # pl.xlabel(labels[1])
        # pl.ylabel(labels[0])
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'DAP_time.png'))
        #
        # pl.figure()
        # lim_min = min(np.nanmin(DAP_width_lower_per_cell), np.nanmin(DAP_width_greater_per_cell)) + 0.1
        # lim_max = max(np.nanmax(DAP_width_lower_per_cell), np.nanmax(DAP_width_greater_per_cell)) + 0.1
        # pl.plot(np.arange(lim_min - 0.1, lim_max + 0.1, 0.1), np.arange(lim_min - 0.1, lim_max + 0.1, 0.1),
        #         color='0.5', linestyle='--')
        # pl.plot(DAP_width_lower_per_cell, DAP_width_greater_per_cell, 'ok')
        # for cell_idx in range(len(cell_ids)):
        #     pl.annotate(cell_ids[cell_idx], xy=(DAP_width_lower_per_cell[cell_idx] + 0.05,
        #                                         DAP_width_greater_per_cell[cell_idx] + 0.05))
        # pl.title('DAP width')
        # pl.xlabel(labels[1])
        # pl.ylabel(labels[0])
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'DAP_width.png'))
        # pl.show()

    # plot DAP deflection all cells, all times
    pl.close('all')
    pl.figure()
    for cell_idx in range(len(cell_ids)):
        pl.semilogx(before_AP_times, DAP_deflection_greater_per_cell[cell_idx, :], 'or', basex=2, label=labels[0] if cell_idx==0 else '')
        pl.semilogx(before_AP_times, DAP_deflection_lower_per_cell[cell_idx, :], 'ob', basex=2, label=labels[1] if cell_idx==0 else '')
    pl.xlabel('Time before AP')
    pl.ylabel('DAP deflection')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'time_before_vs_DAP_deflection_points.png'))

    pl.figure()
    pl.errorbar(before_AP_times, np.nanmean(DAP_deflection_greater_per_cell, 0),
                yerr=np.nanstd(DAP_deflection_greater_per_cell, 0), color='r', marker='o', capsize=2, label=labels[0])
    pl.errorbar(before_AP_times, np.nanmean(DAP_deflection_lower_per_cell, 0),
                yerr=np.nanstd(DAP_deflection_lower_per_cell, 0), color='b', marker='o', capsize=2, label=labels[1])
    pl.xscale("log", basex=2)
    pl.xlabel('Time before AP')
    pl.ylabel('DAP deflection')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'time_before_vs_DAP_deflection.png'))
    pl.show()