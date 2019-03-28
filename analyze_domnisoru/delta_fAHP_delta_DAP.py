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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
pl.style.use('paper')


if __name__ == '__main__':
    #save_dir_img_paper = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    #save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    #save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    save_dir_img_paper = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_data = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'
    save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))

    # parameters
    with_selection = True
    use_avg_times = False
    thresh = '1der'
    AP_thresh_derivative = 3.0
    do_detrend = False
    dt = 0.05
    before_AP = 25  # ms
    after_AP = 25  # ms
    t_vref = 10  # ms
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    t_AP = np.arange(0, after_AP + before_AP + dt, dt) - before_AP
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
    t_sta = np.arange(-before_AP, after_AP+dt, dt)

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

        #pl.figure()
        #pl.title(cell_id)
        #pl.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        #pl.plot(t_AP[AP_thresh_idx], sta_mean_cells[cell_idx][AP_thresh_idx], 'oy')
        #pl.plot(t_AP[fAHP_idx], sta_mean_cells[cell_idx][fAHP_idx], 'ob')
        #pl.plot(t_AP[DAP_idx], sta_mean_cells[cell_idx][DAP_idx], 'or')
        #pl.show()

    # load PCs
    #save_dir_autocorr = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
    #max_lag = 50
    #max_lag_for_pca = 50
    #bin_width = 1  # ms
    #sigma_smooth = None
    #folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
    #save_dir_pcs = os.path.join(save_dir_autocorr, folder, 'PCA')
    #projected = np.load(os.path.join(save_dir_pcs, 'projected.npy'))
    #cm = pl.get_cmap('viridis')
    #pmin = np.min(projected[:, 0])
    #pmax = np.max(projected[:, 0])
    #colors = cm((projected[:, 0] - pmin) / (pmax-pmin))

    # plot
    fig, ax = pl.subplots(figsize=(10, 8))
    # pl.title('From STA with selection' if with_selection else 'From STA without selection',
    #          fontsize=12)
    plot_with_markers(ax, v_rest_fAHP, v_DAP_fAHP, cell_ids, cell_type_dict)  # TODO remove edgecolor=colors
    ax.set_ylabel('$\Delta$ DAP', horizontalalignment='left', y=0.0)
    ax.set_xlabel('$\Delta$ fAHP', horizontalalignment='right', x=1.0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    # ax.set_xlim(-7, 2.0)
    # ax.set_ylim(-4.0, 2.5)
    for i in range(len(cell_ids)):
        ax.annotate(cell_ids[i], xy=(v_rest_fAHP[i] + 0.09, v_DAP_fAHP[i] + 0.06), fontsize=6)

    # example 1
    # axins = inset_axes(ax, width='20%', height='20%', loc='lower left', bbox_to_anchor=(0.05, 0.31, 1, 1),
    #                    bbox_transform=ax.transAxes)
    i = np.where(cell_ids == 's79_0003')[0][0]
    left, bottom, width, height = [0.06, 0.32, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left, bottom + height),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left + width, bottom + height),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    axins.plot(t_sta, sta_mean_cells[i], color='k')
    axins.set_ylim(-70, 0)
    # axins.set_xticks([])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.set_xlabel('Time (ms)', fontsize=10, labelpad=1)
    axins.set_ylabel('$STA_V$ (mV)', fontsize=10, labelpad=1)

    # example 2
    i = np.where(cell_ids == 's109_0002')[0][0]
    left, bottom, width, height = [0.06, 0.06, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left, bottom + height),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left + width, bottom + height),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    axins.plot(t_sta, sta_mean_cells[i], color='k')
    axins.set_ylim(-70, 0)
    # axins.set_xticks([])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.set_xlabel('Time (ms)', fontsize=10, labelpad=1)
    axins.set_ylabel('$STA_V$ (mV)', fontsize=10, labelpad=1)

    # example 3
    i = np.where(cell_ids == 's81_0004')[0][0]
    left, bottom, width, height = [0.33, 0.32, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left + width, bottom + height),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left + width, bottom),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    axins.plot(t_sta, sta_mean_cells[i], color='k')
    axins.set_ylim(-70, 0)
    # axins.set_xticks([])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.set_xlabel('Time (ms)', fontsize=10, labelpad=1)
    axins.set_ylabel('$STA_V$ (mV)', fontsize=10, labelpad=1)

    # example 4
    i = np.where(cell_ids == 's84_0002')[0][0]
    left, bottom, width, height = [0.33, 0.06, 0.2, 0.2]
    axins = fig.add_axes([left, bottom, width, height])
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left + width, bottom + height),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    ax.annotate('', xy=(v_rest_fAHP[i], v_DAP_fAHP[i]), xytext=(left + width, bottom),
                xycoords='data', textcoords='figure fraction',
                arrowprops=dict(arrowstyle="-", color='0.5', linewidth=0.75))
    axins.plot(t_sta, sta_mean_cells[i], color='k')
    axins.plot(t_sta, sta_mean_cells[i], color='k')
    axins.set_ylim(-70, 0)
    # axins.set_xticks([])
    # axins.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.set_xlabel('Time (ms)', fontsize=10, labelpad=1)
    axins.set_ylabel('$STA_V$ (mV)', fontsize=10, labelpad=1)


    pl.tight_layout()
    name_add = 'with_selection' if with_selection else 'without_selection'
    name_add2 = 'avg_times' if use_avg_times else 'not_avg_times'
    pl.savefig(os.path.join(save_dir_img_paper, 'delta_fAHP_delta_DAP_'+name_add+'_'+name_add2+'_'+thresh+'.png'))
    pl.show()

    if not os.path.exists(os.path.join(save_dir_data, name_add2)):
        os.makedirs(os.path.join(save_dir_data, name_add2))
    np.save(os.path.join(save_dir_data, name_add2, 'v_rest_fAHP.npy'), v_rest_fAHP)
    np.save(os.path.join(save_dir_data, name_add2, 'v_DAP_fAHP.npy'), v_DAP_fAHP)