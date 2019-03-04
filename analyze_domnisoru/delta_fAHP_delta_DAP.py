from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_characteristics import to_idx
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from cell_fitting.util import init_nan
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)

    # parameters
    with_selection = True
    use_avg_times = True
    thresh = '1der'
    do_detrend = False
    dt = 0.05
    before_AP = 25  # ms
    after_AP = 25  # ms
    t_vref = 10  # ms
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    t_AP = np.arange(0, after_AP + before_AP + dt, dt) - before_AP
    AP_thresh_derivative = 3.0
    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms', 'fY_cm', 'fvel_100ms']
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                  + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    if with_selection:
        save_dir_img = os.path.join(save_dir_img, 'good_AP_criterion')
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], folder_name)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_DAP_cells(new=True)
    cell_type_dict = get_celltype_dict(save_dir)

    # main
    sta_mean_cells = np.load(os.path.join(save_dir_img, 'sta_mean.npy'))

    AP_max_idx_cells = init_nan(len(cell_ids))
    fAHP_min_idx_cells = init_nan(len(cell_ids))
    DAP_max_idx_cells = init_nan(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        if np.isnan(sta_mean_cells[cell_idx][0]):
            continue

        # get spike_characteristics
        sta_derivative = np.diff(sta_mean_cells[cell_idx]) / dt
        sta_2derivative = np.diff(sta_derivative) / dt
        if thresh == '1der':
            AP_thresh_idx = get_AP_onset_idxs(sta_derivative[:before_AP_idx], AP_thresh_derivative)[-1]
        elif thresh == '2der':
            AP_thresh_idx = np.argmax(sta_2derivative[:before_AP_idx])
        # pl.figure()
        # pl.plot(t_AP, sta_mean_cells[cell_idx])
        # pl.plot(t_AP[AP_thresh_idx:], sta_mean_cells[cell_idx][AP_thresh_idx:])
        # pl.show()

        # v_rest = sta_mean_cells[cell_idx][before_AP_idx - to_idx(time_rest_AP, dt)]
        v_rest = sta_mean_cells[cell_idx][AP_thresh_idx]
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx
        spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
        AP_max_idx_cells[cell_idx], fAHP_min_idx_cells[cell_idx], DAP_max_idx_cells[cell_idx] = np.array(get_spike_characteristics(sta_mean_cells[cell_idx], t_AP,
                                                                          ['AP_max_idx', 'fAHP_min_idx', 'DAP_max_idx'],
                                                                          v_rest, check=False,
                                                                          **spike_characteristics_dict)).astype(float)

    # compute average Time_AP-fAHP and Time_AP-DAP
    time_AP_fAHP_avg = np.nanmean((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    time_AP_DAP_avg = np.nanmean((DAP_max_idx_cells - AP_max_idx_cells) * dt)
    time_AP_fAHP_std = np.nanstd((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    time_AP_DAP_std = np.nanstd((DAP_max_idx_cells - AP_max_idx_cells) * dt)

    print 'Time_AP-fAHP: %.2f +- %.2f' % (time_AP_fAHP_avg, time_AP_fAHP_std)
    print 'Time_AP-fAHP: %.2f +- %.2f' % (time_AP_DAP_avg, time_AP_DAP_std)

    # compute v_rest_fAHP, delta_DAP
    v_rest_fAHP = np.zeros(len(cell_ids))
    v_DAP_fAHP = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        if np.isnan(sta_mean_cells[cell_idx][0]):
            v_rest_fAHP[cell_idx] = np.nan
            v_DAP_fAHP[cell_idx] = np.nan
            continue

        sta_derivative = np.diff(sta_mean_cells[cell_idx]) / dt
        if thresh == '1der':
            AP_thresh_idx = get_AP_onset_idxs(sta_derivative[:before_AP_idx], AP_thresh_derivative)[-1]
        elif thresh == '2der':
            AP_thresh_idx = np.argmax(sta_2derivative[:before_AP_idx])

        if not use_avg_times and cell_id in DAP_cells:
            fAHP_idx = int(fAHP_min_idx_cells[cell_idx])
            DAP_idx = int(DAP_max_idx_cells[cell_idx])
        else:
            time_AP_fAHP_avg_rounded = round(time_AP_fAHP_avg * 2.0 * 10.0) / 2.0 / 10.0
            time_AP_DAP_avg_rounded = round(time_AP_DAP_avg * 2.0 * 10.0) / 2.0 / 10.0
            fAHP_idx = to_idx(before_AP + time_AP_fAHP_avg_rounded, dt, 2)
            DAP_idx = to_idx(before_AP + time_AP_DAP_avg_rounded, dt, 2)
        v_rest_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx] - sta_mean_cells[cell_idx][AP_thresh_idx]
        v_DAP_fAHP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx] - sta_mean_cells[cell_idx][fAHP_idx]

        # pl.figure()
        # pl.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        # pl.plot(t_AP[fAHP_idx], sta_mean_cells[cell_idx][fAHP_idx], 'ob')
        # pl.plot(t_AP[DAP_idx], sta_mean_cells[cell_idx][DAP_idx], 'or')
        # pl.show()

    # plot
    fig, ax = pl.subplots(figsize=(10, 8))
    # pl.title('From STA with selection' if with_selection else 'From STA without selection',
    #          fontsize=12)
    plot_with_markers(ax, v_rest_fAHP, v_DAP_fAHP, cell_ids, cell_type_dict)
    ax.set_ylabel('$\Delta$ DAP', horizontalalignment='left', y=0.0)
    ax.set_xlabel('$\Delta$ fAHP', horizontalalignment='right', x=1.0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    # ax.set_xlim(-7, 2.0)
    # ax.set_ylim(-4.0, 2.5)
    for i in range(len(cell_ids)):
        ax.annotate(cell_ids[i], xy=(v_rest_fAHP[i] + 0.09, v_DAP_fAHP[i] + 0.06), fontsize=6)
    pl.tight_layout()
    name_add = 'with_selection' if with_selection else 'without_selection'
    name_add2 = 'avg_times' if use_avg_times else 'not_avg_times'
    pl.savefig(os.path.join(save_dir_img, 'delta_fAHP_delta_DAP_'+name_add+'_'+name_add2+'_'+thresh+'.png'))
    pl.show()