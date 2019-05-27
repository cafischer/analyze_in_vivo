from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_burstgroups, get_celltype_dict
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.spike_characteristics import get_spike_characteristics_dict
from cell_fitting.util import init_nan
from scipy.stats import pearsonr


if __name__ == '__main__':
    save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_fig2 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig2'

    # parameters
    with_selection = True
    use_avg_times = True
    thresh = '1der'
    AP_thresh_derivative = 15
    dt = 0.05
    before_AP = 25  # ms
    after_AP = 25  # ms
    t_vref = 10  # ms
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    t_AP = np.arange(0, after_AP + before_AP + dt, dt) - before_AP
    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)

    grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_burstgroups()['B+D']

    # load
    folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                  + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    sta_mean_cells = np.load(os.path.join(save_dir_sta, folder_name, 'sta_mean.npy'))
    t_sta = np.arange(-before_AP, after_AP+dt, dt)

    AP_max_idx_cells = init_nan(len(grid_cells))
    fAHP_min_idx_cells = init_nan(len(grid_cells))
    DAP_max_idx_cells = init_nan(len(grid_cells))
    AP_onset_idx_cells = np.zeros(len(grid_cells), dtype=int)
    for cell_idx, cell_id in enumerate(grid_cells):
        if np.isnan(sta_mean_cells[cell_idx][0]):
            continue

        # get spike_characteristics
        sta_derivative = np.diff(sta_mean_cells[cell_idx]) / dt
        sta_2derivative = np.diff(sta_derivative) / dt
        if thresh == '1der':
            AP_onset_idx_cells[cell_idx] = get_AP_onset_idxs(sta_derivative[:before_AP_idx], AP_thresh_derivative)[-1]
        elif thresh == '2der':
            AP_onset_idx_cells[cell_idx] = np.argmax(sta_2derivative[:before_AP_idx])
        # pl.figure()
        # pl.plot(t_AP, sta_mean_cells[cell_idx])
        # pl.plot(t_AP[AP_thresh_idx:], sta_mean_cells[cell_idx][AP_thresh_idx:])
        # pl.show()

        # v_rest = sta_mean_cells[cell_idx][before_AP_idx - to_idx(time_rest_AP, dt)]
        v_AP_onset = sta_mean_cells[cell_idx][AP_onset_idx_cells[cell_idx]]
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx
        spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
        (AP_max_idx_cells[cell_idx], fAHP_min_idx_cells[cell_idx],
         DAP_max_idx_cells[cell_idx]) = np.array(get_spike_characteristics(sta_mean_cells[cell_idx], t_AP,
                                                                                                     ['AP_max_idx', 'fAHP_min_idx', 'DAP_max_idx'],
                                                                                                     v_AP_onset, check=False,
                                                                                                     **spike_characteristics_dict)).astype(float)

    # compute average Time_AP-fAHP and Time_AP-DAP
    time_AP_fAHP_avg = 1.8  #np.nanmean((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    time_AP_DAP_avg = 4.6  #np.nanmean((DAP_max_idx_cells - AP_max_idx_cells) * dt)
    # time_AP_fAHP_std = np.nanstd((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    # time_AP_DAP_std = np.nanstd((DAP_max_idx_cells - AP_max_idx_cells) * dt)
    # time_AP_fAHP_avg_rounded = round(time_AP_fAHP_avg * 2.0 * 10.0) / 2.0 / 10.0
    # time_AP_DAP_avg_rounded = round(time_AP_DAP_avg * 2.0 * 10.0) / 2.0 / 10.0
    # print 'Time_AP-fAHP: %.2f +- %.2f' % (time_AP_fAHP_avg, time_AP_fAHP_std)
    # print 'Time_AP-fAHP: %.2f +- %.2f' % (time_AP_DAP_avg, time_AP_DAP_std)

    # compute v_rest_fAHP, delta_DAP
    v_onset_fAHP = np.zeros(len(grid_cells))
    v_DAP_fAHP = np.zeros(len(grid_cells))
    v_fAHP = np.zeros(len(grid_cells))
    v_DAP = np.zeros(len(grid_cells))
    v_onset = np.zeros(len(grid_cells))
    for cell_idx, cell_id in enumerate(grid_cells):
        print cell_id
        if np.isnan(sta_mean_cells[cell_idx][0]):
            v_onset_fAHP[cell_idx] = np.nan
            v_DAP_fAHP[cell_idx] = np.nan
            continue

        if not use_avg_times and cell_id in DAP_cells:
            fAHP_idx = int(fAHP_min_idx_cells[cell_idx])
            DAP_idx = int(DAP_max_idx_cells[cell_idx])
        else:
            fAHP_idx = to_idx(before_AP + time_AP_fAHP_avg, dt, 2)
            DAP_idx = to_idx(before_AP + time_AP_DAP_avg, dt, 2)
        v_onset_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx] - sta_mean_cells[cell_idx][AP_onset_idx_cells[cell_idx]]
        v_DAP_fAHP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx] - sta_mean_cells[cell_idx][fAHP_idx]
        v_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx]
        v_DAP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx]
        v_onset[cell_idx] = sta_mean_cells[cell_idx][AP_onset_idx_cells[cell_idx]]

        #pl.figure()
        #pl.title(cell_id)
        #pl.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        #pl.plot(t_AP[AP_thresh_idx], sta_mean_cells[cell_idx][AP_thresh_idx], 'oy')
        #pl.plot(t_AP[fAHP_idx], sta_mean_cells[cell_idx][fAHP_idx], 'ob')
        #pl.plot(t_AP[DAP_idx], sta_mean_cells[cell_idx][DAP_idx], 'or')
        #pl.show()

    # correlation between delta DAP and delta fAHP
    corr, p = pearsonr(v_onset_fAHP, v_DAP_fAHP)
    print 'Correlation: %.2f' % corr
    print 'p-val: %.5f' % p

    # save
    if not os.path.exists(save_dir_fig2):
        os.makedirs(save_dir_fig2)
    np.save(os.path.join(save_dir_fig2, 'sta_mean_cells.npy'), sta_mean_cells)
    np.save(os.path.join(save_dir_fig2, 'v_onset_fAHP.npy'), v_onset_fAHP)
    np.save(os.path.join(save_dir_fig2, 'v_DAP_fAHP.npy'), v_DAP_fAHP)
    np.save(os.path.join(save_dir_fig2, 'AP_onset_idx_cells.npy'), AP_onset_idx_cells)
    np.save(os.path.join(save_dir_fig2, 'DAP_max_idx_cells.npy'), DAP_max_idx_cells)
    np.save(os.path.join(save_dir_fig2, 'fAHP_min_idx_cells.npy'), fAHP_min_idx_cells)
    np.save(os.path.join(save_dir_fig2, 'AP_max_idx_cells.npy'), AP_max_idx_cells)
    np.save(os.path.join(save_dir_fig2, 'v_onset.npy'), v_onset)
    np.save(os.path.join(save_dir_fig2, 'v_fAHP.npy'), v_fAHP)
    np.save(os.path.join(save_dir_fig2, 'v_DAP.npy'), v_DAP)