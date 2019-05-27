from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_burstgroups, get_celltype_dict
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.spike_characteristics import get_spike_characteristics_dict
from cell_fitting.util import init_nan
import matplotlib.pyplot as pl
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_label_burstgroups, get_colors_burstgroups
from matplotlib.patches import Patch
pl.style.use('paper')


def compute_delta_fAHP_delta_DAP(sta_mean_cells):

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

        # v_AP_onset = sta_mean_cells[cell_idx][AP_onset_idx_cells[cell_idx]]
        # spike_characteristics_dict = get_spike_characteristics_dict()
        # spike_characteristics_dict['AP_max_idx'] = before_AP_idx
        # spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
        # (AP_max_idx_cells[cell_idx], fAHP_min_idx_cells[cell_idx],
        #  DAP_max_idx_cells[cell_idx]) = np.array(get_spike_characteristics(sta_mean_cells[cell_idx], t_AP,
        #                                                                    ['AP_max_idx', 'fAHP_min_idx',
        #                                                                     'DAP_max_idx'],
        #                                                                    v_AP_onset, check=False,
        #                                                                    **spike_characteristics_dict)).astype(float)

    # compute average Time_AP-fAHP and Time_AP-DAP
    time_AP_fAHP_avg = 1.8  # np.nanmean((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    time_AP_DAP_avg = 4.6  # np.nanmean((DAP_max_idx_cells - AP_max_idx_cells) * dt)
    # time_AP_fAHP_std = np.nanstd((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    # time_AP_DAP_std = np.nanstd((DAP_max_idx_cells - AP_max_idx_cells) * dt)
    # time_AP_fAHP_avg_rounded = round(time_AP_fAHP_avg * 2.0 * 10.0) / 2.0 / 10.0
    # time_AP_DAP_avg_rounded = round(time_AP_DAP_avg * 2.0 * 10.0) / 2.0 / 10.0

    # compute v_rest_fAHP, delta_DAP
    v_onset_fAHP = np.zeros(len(grid_cells))
    v_DAP_fAHP = np.zeros(len(grid_cells))
    v_fAHP = np.zeros(len(grid_cells))
    v_DAP = np.zeros(len(grid_cells))
    v_onset = np.zeros(len(grid_cells))
    for cell_idx, cell_id in enumerate(grid_cells):
        if np.isnan(sta_mean_cells[cell_idx][0]):
            v_onset_fAHP[cell_idx] = np.nan
            v_DAP_fAHP[cell_idx] = np.nan
            continue

        if not use_avg_times and cell_id in DAP_cells:
            if not np.isnan(fAHP_min_idx_cells[cell_idx]):
                fAHP_idx = int(fAHP_min_idx_cells[cell_idx])
            else:
                fAHP_idx = to_idx(before_AP + time_AP_fAHP_avg, dt, 2)  # TODO: this can highly increase variance
            if not np.isnan(DAP_max_idx_cells[cell_idx]):
                DAP_idx = int(DAP_max_idx_cells[cell_idx])
            else:
                DAP_idx = to_idx(before_AP + time_AP_DAP_avg, dt, 2)
        else:
            fAHP_idx = to_idx(before_AP + time_AP_fAHP_avg, dt, 2)
            DAP_idx = to_idx(before_AP + time_AP_DAP_avg, dt, 2)

        v_onset_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx] - sta_mean_cells[cell_idx][
            AP_onset_idx_cells[cell_idx]]
        v_DAP_fAHP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx] - sta_mean_cells[cell_idx][fAHP_idx]
        v_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx]
        v_DAP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx]
        v_onset[cell_idx] = sta_mean_cells[cell_idx][AP_onset_idx_cells[cell_idx]]

        # if cell_id in get_cell_ids_burstgroups()['B+D'][5]:
        #     pl.figure()
        #     pl.title(cell_id)
        #     pl.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        #     pl.plot(t_AP[AP_onset_idx_cells[cell_idx]], sta_mean_cells[cell_idx][AP_onset_idx_cells[cell_idx]], 'oy')
        #     pl.plot(t_AP[fAHP_idx], sta_mean_cells[cell_idx][fAHP_idx], 'ob')
        #     pl.plot(t_AP[DAP_idx], sta_mean_cells[cell_idx][DAP_idx], 'or')
        #     pl.show()

    return (v_onset_fAHP, v_DAP_fAHP, AP_onset_idx_cells, DAP_max_idx_cells, fAHP_min_idx_cells, AP_max_idx_cells,
            v_onset, v_fAHP, v_DAP)



if __name__ == '__main__':
    save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/bootstrap/not_detrended'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_fig2 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig2_add'

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
    sta_mean_bootstrap = np.load(os.path.join(save_dir_sta, folder_name, 'sta_mean_bootstrap.npy'))
    t_sta = np.arange(-before_AP, after_AP+dt, dt)
    n_bootstraps = np.shape(sta_mean_bootstrap)[2]

    v_onset_fAHP = np.zeros((len(grid_cells), n_bootstraps))
    v_DAP_fAHP = np.zeros((len(grid_cells), n_bootstraps))
    for i in range(n_bootstraps):
        (v_onset_fAHP[:, i], v_DAP_fAHP[:, i], AP_onset_idx_cells, DAP_max_idx_cells, fAHP_min_idx_cells,
         AP_max_idx_cells, v_onset, v_fAHP1, v_DAP) = compute_delta_fAHP_delta_DAP(sta_mean_bootstrap[:, :, i])

    v_onset_fAHP_mean = np.mean(v_onset_fAHP, 1)
    v_DAP_fAHP_mean = np.mean(v_DAP_fAHP, 1)
    v_onset_fAHP_std = np.std(v_onset_fAHP, 1)
    v_DAP_fAHP_std = np.std(v_DAP_fAHP, 1)

    # save
    np.save(os.path.join(save_dir_fig2, 'v_onset_fAHP_mean.npy'), v_onset_fAHP_mean)
    np.save(os.path.join(save_dir_fig2, 'v_onset_fAHP_std.npy'), v_onset_fAHP_std)
    np.save(os.path.join(save_dir_fig2, 'v_DAP_fAHP_mean.npy'), v_DAP_fAHP_mean)
    np.save(os.path.join(save_dir_fig2, 'v_DAP_fAHP_std.npy'), v_DAP_fAHP_std)

    # plot
    if not os.path.exists(save_dir_fig2):
        os.makedirs(save_dir_fig2)

    labels_burstgroups = get_label_burstgroups()
    colors_burstgroups = get_colors_burstgroups()

    fig, ax = pl.subplots()
    ax.errorbar(v_onset_fAHP_mean[labels_burstgroups['B']],
                v_DAP_fAHP_mean[labels_burstgroups['B']],
                xerr=v_onset_fAHP_std[labels_burstgroups['B']],
                yerr=v_DAP_fAHP_std[labels_burstgroups['B']],
                marker='o', linestyle='', markersize=4,
                color=colors_burstgroups['B'])
    ax.errorbar(v_onset_fAHP_mean[labels_burstgroups['B+D']],
                v_DAP_fAHP_mean[labels_burstgroups['B+D']],
                xerr=v_onset_fAHP_std[labels_burstgroups['B+D']],
                yerr=v_DAP_fAHP_std[labels_burstgroups['B+D']],
                marker='o', linestyle='', markersize=4,
                color=colors_burstgroups['B+D'])
    ax.errorbar(v_onset_fAHP_mean[labels_burstgroups['NB']],
                v_DAP_fAHP_mean[labels_burstgroups['NB']],
                xerr=v_onset_fAHP_std[labels_burstgroups['NB']],
                yerr=v_DAP_fAHP_std[labels_burstgroups['NB']],
                marker='o', linestyle='', markersize=4,
                color=colors_burstgroups['NB'])
    ax.set_ylabel(r'$\mathrm{\Delta V_{DAP}}$', horizontalalignment='left', y=0.0)
    ax.set_xlabel(r'$\mathrm{\Delta V_{fAHP}}$', horizontalalignment='right', x=1.0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    handles = [Patch(color=colors_burstgroups['B+D'], label='Bursty+DAP'),
               Patch(color=colors_burstgroups['B'], label='Bursty-DAP'),
               Patch(color=colors_burstgroups['NB'], label='Non-bursty')]
    ax.legend(handles=handles, loc='upper right', fontsize=10)
    pl.tight_layout()
    pl.show()