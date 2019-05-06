from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict, get_label_burstgroups, get_colors_burstgroups
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.spike_characteristics import get_spike_characteristics_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from cell_fitting.util import init_nan
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img_paper = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_data = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    #save_dir_img_paper = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    #save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    #save_dir_data = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'
    #save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    if not os.path.exists(save_dir_img_paper):
        os.makedirs(save_dir_img_paper)

    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    labels_burstgroups = get_label_burstgroups()
    colors_burstgroups = get_colors_burstgroups()

    # parameters
    with_selection = True
    use_avg_times = False
    thresh = '1der'
    AP_thresh_derivative = 15
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
    DAP_time_cells = init_nan(len(cell_ids))
    time_AP_fAHP_cells = init_nan(len(cell_ids))
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
        v_AP_onset = sta_mean_cells[cell_idx][AP_thresh_idx]
        spike_characteristics_dict = get_spike_characteristics_dict()
        spike_characteristics_dict['AP_max_idx'] = before_AP_idx
        spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
        (AP_max_idx_cells[cell_idx], fAHP_min_idx_cells[cell_idx],
         DAP_max_idx_cells[cell_idx], DAP_time_cells[cell_idx]) = np.array(get_spike_characteristics(sta_mean_cells[cell_idx], t_AP,
                                                                                                     ['AP_max_idx', 'fAHP_min_idx', 'DAP_max_idx', 'DAP_time'],
                                                                                                     v_AP_onset, check=False,
                                                                                                     **spike_characteristics_dict)).astype(float)
        if not np.isnan(fAHP_min_idx_cells[cell_idx]):
            time_AP_fAHP_cells[cell_idx] = t_AP[int(fAHP_min_idx_cells[cell_idx])] - t_AP[int(AP_max_idx_cells[cell_idx])]

    # compute average Time_AP-fAHP and Time_AP-DAP
    time_AP_fAHP_avg = np.nanmean((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    time_AP_DAP_avg = np.nanmean((DAP_max_idx_cells - AP_max_idx_cells) * dt)
    time_AP_fAHP_std = np.nanstd((fAHP_min_idx_cells - AP_max_idx_cells) * dt)
    time_AP_DAP_std = np.nanstd((DAP_max_idx_cells - AP_max_idx_cells) * dt)

    print 'Time_AP-fAHP: %.2f +- %.2f' % (time_AP_fAHP_avg, time_AP_fAHP_std)
    print 'Time_AP-fAHP: %.2f +- %.2f' % (time_AP_DAP_avg, time_AP_DAP_std)

    # compute v_rest_fAHP, delta_DAP
    v_onset_fAHP = np.zeros(len(cell_ids))
    v_DAP_fAHP = np.zeros(len(cell_ids))
    v_fAHP = np.zeros(len(cell_ids))
    v_DAP = np.zeros(len(cell_ids))
    v_onset = np.zeros(len(cell_ids))
    AP_onset_idx = np.zeros(len(cell_ids), dtype=int)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        if np.isnan(sta_mean_cells[cell_idx][0]):
            v_onset_fAHP[cell_idx] = np.nan
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
        v_onset_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx] - sta_mean_cells[cell_idx][AP_thresh_idx]
        v_DAP_fAHP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx] - sta_mean_cells[cell_idx][fAHP_idx]
        v_fAHP[cell_idx] = sta_mean_cells[cell_idx][fAHP_idx]
        v_DAP[cell_idx] = sta_mean_cells[cell_idx][DAP_idx]
        v_onset[cell_idx] = sta_mean_cells[cell_idx][AP_thresh_idx]
        AP_onset_idx[cell_idx] = AP_thresh_idx

        #pl.figure()
        #pl.title(cell_id)
        #pl.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        #pl.plot(t_AP[AP_thresh_idx], sta_mean_cells[cell_idx][AP_thresh_idx], 'oy')
        #pl.plot(t_AP[fAHP_idx], sta_mean_cells[cell_idx][fAHP_idx], 'ob')
        #pl.plot(t_AP[DAP_idx], sta_mean_cells[cell_idx][DAP_idx], 'or')
        #pl.show()

    # plot
    fig = pl.figure(figsize=(6, 7.5))
    outer = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
    cell_examples = ['s84_0002', 's109_0002', 's118_0002']
    colors_examples = [colors_burstgroups['NB'], colors_burstgroups['B+D'], colors_burstgroups['B']]
    letters_examples = ['a', 'b', 'c']

    # A
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, :])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)
    plot_with_markers(ax, v_onset_fAHP[labels_burstgroups['B']], v_DAP_fAHP[labels_burstgroups['B']],
                      cell_ids[labels_burstgroups['B']], cell_type_dict, theta_cells=theta_cells,
                      edgecolor=colors_burstgroups['B'], legend=False)
    plot_with_markers(ax, v_onset_fAHP[labels_burstgroups['B+D']], v_DAP_fAHP[labels_burstgroups['B+D']],
                      cell_ids[labels_burstgroups['B+D']], cell_type_dict, theta_cells=theta_cells,
                      edgecolor=colors_burstgroups['B+D'], legend=False)
    handles = plot_with_markers(ax, v_onset_fAHP[labels_burstgroups['NB']], v_DAP_fAHP[labels_burstgroups['NB']],
                      cell_ids[labels_burstgroups['NB']], cell_type_dict, theta_cells=theta_cells,
                      edgecolor=colors_burstgroups['NB'], legend=False)
    ax.set_ylabel(r'$\mathrm{\Delta V_{DAP}}$', horizontalalignment='left', y=0.0)
    ax.set_xlabel(r'$\mathrm{\Delta V_{fAHP}}$', horizontalalignment='right', x=1.0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    # ax.set_xlim(-7, 2.0)
    # ax.set_ylim(-4.0, 2.5)
    handles += [Patch(color=colors_burstgroups['B+D'], label='Bursty+DAP'), Patch(color=colors_burstgroups['B'], label='Bursty-DAP'),
                Patch(color=colors_burstgroups['NB'], label='Non-bursty')]
    ax.legend(handles=handles, loc='upper right')

    for cell_idx, cell_id in cell_ids:
        ax.annotate(layer, xy=(v_onset_fAHP[cell_idx], v_DAP_fAHP[cell_idx]))

    # inset
    axin = inset_axes(ax, width='40%', height='45%', loc='lower left')
    cell_id = 's104_0007'
    cell_idx = np.where(cell_id == cell_ids)[0]
    idx1 = to_idx(12, dt)
    idx2 = to_idx(5, dt)
    axin.plot(t_sta[idx1:-idx2], sta_mean_cells[cell_idx][0][idx1:-idx2], color='k')
    axin.set_ylim(-68.5, -55)
    axin.set_xlim(-15, t_sta[-1])
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_xlabel('Time', horizontalalignment='right', x=1.0, fontsize=10)
    axin.set_ylabel('Voltage', horizontalalignment='right', y=1.0, fontsize=10)

    axin.plot([t_sta[0], t_sta[AP_onset_idx[cell_idx]]], [v_onset[cell_idx], v_onset[cell_idx]], color='g')  # AP onset line
    axin.plot([t_sta[0], t_sta[-1]+5], [v_fAHP[cell_idx], v_fAHP[cell_idx]], color='g')  # fAHP min. line
    axin.plot([t_sta[int(DAP_max_idx_cells[cell_idx])], t_sta[-1]+5], [v_DAP[cell_idx], v_DAP[cell_idx]], color='g')  # DAP max. line
    # annotate delta fAHP
    # axin.annotate('', xytext=(t_sta[to_idx(6.5, dt)], v_fAHP[cell_idx]),
    #               xy=(t_sta[to_idx(6.5, dt)], v_onset[cell_idx]),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta V_{fAHP}}$',
                  xy=(t_sta[to_idx(10.2, dt)], (v_fAHP[cell_idx] - v_onset[cell_idx])/2. + v_onset[cell_idx]),
                  ha='left', va='center')
    # annotate delta DAP
    # axin.annotate('', xytext=(t_sta[-to_idx(6.5, dt)], v_fAHP[cell_idx]),
    #               xy=(t_sta[-to_idx(6.5, dt)], v_DAP[cell_idx]),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta V_{DAP}}$',
                  xy=(t_sta[-to_idx(8.5, dt)], (v_DAP[cell_idx] - v_fAHP[cell_idx])/2. + v_fAHP[cell_idx]-0.3),
                  ha='left', va='center')

    axin.plot([t_sta[int(DAP_max_idx_cells[cell_idx])], t_sta[int(DAP_max_idx_cells[cell_idx])]],
              [axin.get_ylim()[0], v_DAP[cell_idx]], color='g', linestyle='--')  # t_DAP line
    axin.plot([t_sta[int(AP_max_idx_cells[cell_idx])], t_sta[int(AP_max_idx_cells[cell_idx])]],
              [-62, axin.get_ylim()[0]], color='g', linestyle='--')  # t_AP line
    axin.plot([t_sta[int(fAHP_min_idx_cells[cell_idx])], t_sta[int(fAHP_min_idx_cells[cell_idx])]],
              [-62, v_fAHP[cell_idx]], color='g', linestyle='--')  # t_fAHP line

    # annotate t_DAP
    # axin.annotate('', xytext=(t_sta[int(AP_max_idx_cells[cell_idx])], axin.get_ylim()[0]),
    #               xy=(t_sta[int(DAP_max_idx_cells[cell_idx])], axin.get_ylim()[0]),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta t_{DAP}}$',
                  xy=(t_sta[int(AP_max_idx_cells[cell_idx])]
                      + (t_sta[int(DAP_max_idx_cells[cell_idx])] - t_sta[int(AP_max_idx_cells[cell_idx])])/2.,
                      axin.get_ylim()[0]-0.05),
                  ha='center', va='bottom')
    # annotate t_fAHP
    # axin.annotate('', xytext=(t_sta[int(AP_max_idx_cells[cell_idx])], -62),
    #               xy=(t_sta[int(fAHP_min_idx_cells[cell_idx])], -62),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta t_{fAHP}}$',
                  xy=(t_sta[int(AP_max_idx_cells[cell_idx])]
                      + (t_sta[int(fAHP_min_idx_cells[cell_idx])] - t_sta[int(AP_max_idx_cells[cell_idx])])/2.,
                      -62),
                  ha='left', va='bottom')

    # B
    for i, cell_id in enumerate(cell_examples):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, i])
        ax = pl.Subplot(fig, inner[0])
        fig.add_subplot(ax)
        cell_idx = np.where(cell_id == cell_ids)[0]
        ax.plot(t_sta, sta_mean_cells[cell_idx][0], color=colors_examples[i])
        ax.set_ylim(-75, 0)
        # ax.set_xticklabels([-max_lag, 0, max_lag], fontsize=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(r'$\mathrm{V_{STA}}$ (mV)')
        ax.set_title('(' + letters_examples[i] + ')' + '     ' + cell_id, loc='left', fontsize=10)


    # C
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2, :])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img_paper, 'delta_fAHP_delta_DAP.png'))
    pl.show()