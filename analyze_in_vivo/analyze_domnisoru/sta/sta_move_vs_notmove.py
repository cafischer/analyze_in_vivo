from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_sta_median, plot_APs
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/move_notmove'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'giant_theta'  # 'stellate_layer2'  #pyramidal_layer2
    cell_ids = load_cell_ids(save_dir, cell_type)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']
    velocity_threshold = 1  # cm/sec

    # parameters
    use_AP_max_idxs_domnisoru = True
    do_detrend = False
    in_field = False
    out_field = False
    before_AP_sta = 25
    after_AP_sta = 25
    DAP_deflections = {}
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], folder_field[(in_field, out_field)],
                                cell_type)

    #
    sta_mean_per_cell_move = []
    sta_mean_per_cell_notmove = []
    sta_std_per_cell_move = []
    sta_std_per_cell_notmove = []
    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

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


        # divide APs for moving and not moving
        AP_max_idxs_notmove = []
        AP_max_idxs_move = []
        to_slow = np.where(velocity < velocity_threshold)[0]
        for AP_max_idx in AP_max_idxs_selected:
            if AP_max_idx in to_slow:
                AP_max_idxs_notmove.append(AP_max_idx)
            else:
                AP_max_idxs_move.append(AP_max_idx)


        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs_move = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs_move, AP_max_idxs)
        v_APs_notmove = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs_notmove, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt

        # STA
        sta_mean_move, sta_std_move = get_sta(v_APs_move)
        sta_mean_notmove, sta_std_notmove = get_sta(v_APs_notmove)
        sta_mean_per_cell_move.append(sta_mean_move)
        sta_mean_per_cell_notmove.append(sta_mean_notmove)
        sta_std_per_cell_move.append(sta_std_move)
        sta_std_per_cell_notmove.append(sta_std_notmove)

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        pl.figure()
        pl.plot(t_AP, sta_mean_move, 'r', label='>= velocity threshold')
        pl.fill_between(t_AP, sta_mean_move + sta_std_move, sta_mean_move - sta_std_move,
                        facecolor='r', alpha=0.5)
        pl.plot(t_AP, sta_mean_notmove, 'b', label='< velocity threshold')
        pl.fill_between(t_AP, sta_mean_notmove + sta_std_notmove, sta_mean_notmove - sta_std_notmove,
                        facecolor='b', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'sta.png'))

        v_APs_plots_move = v_APs_move[np.random.randint(0, len(v_APs_move), 20)]  # reduce to lower number
        v_APs_plots_notmove = v_APs_notmove[np.random.randint(0, len(v_APs_notmove), 20)]  # reduce to lower number

        pl.figure()
        pl.title('#APs >= vel. thresh.: %i, #APs < vel. thresh.: %i' % (len(v_APs_move), len(v_APs_notmove)))
        for i, v_AP in enumerate(v_APs_plots_move):
            pl.plot(t_AP, v_AP, 'r', alpha=0.5, label='>= velocity threshold' if i == 0 else '')
        for i, v_AP in enumerate(v_APs_plots_notmove):
            pl.plot(t_AP, v_AP, 'b', alpha=0.5, label='< velocity threshold' if i == 0 else '')
        pl.ylabel('Membrane potential (mV)')
        pl.xlabel('Time (ms)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'v_APs.png'))
        #pl.show()

    pl.close('all')
    n_rows = 1 if len(cell_ids) <= 3 else 2
    fig_height = 4.5 if len(cell_ids) <= 3 else 9
    fig, axes = pl.subplots(n_rows, int(round(len(cell_ids) / n_rows)), sharex='all', sharey='all',
                            figsize=(12, fig_height))
    if n_rows == 1:
        axes = np.array([axes])
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(int(round(len(cell_ids) / n_rows))):
            if cell_idx < len(cell_ids):
                # i1 = 0 if i < int(round(len(cell_ids)/2.)) else 1
                # i2 = i - i1 * int(round(len(cell_ids)/2.))
                axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
                axes[i1, i2].plot(t_AP, sta_mean_per_cell_move[cell_idx], 'r', label='>= velocity threshold')
                axes[i1, i2].fill_between(t_AP, sta_mean_per_cell_move[cell_idx] - sta_std_per_cell_move[cell_idx],
                                          sta_mean_per_cell_move[cell_idx] + sta_std_per_cell_move[cell_idx],
                                          color='r', alpha=0.5)
                axes[i1, i2].plot(t_AP, sta_mean_per_cell_notmove[cell_idx], 'b', label='< velocity threshold')
                axes[i1, i2].fill_between(t_AP, sta_mean_per_cell_notmove[cell_idx] - sta_std_per_cell_notmove[cell_idx],
                                          sta_mean_per_cell_notmove[cell_idx] + sta_std_per_cell_notmove[cell_idx],
                                          color='b', alpha=0.5)
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Membrane Potential (mV)')
    fig.text(0.01, 0.5, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.01, 'Time (ms)', ha='center', fontsize=16)
    pl.tight_layout()
    adjust_bottom = 0.12 if len(cell_ids) <= 3 else 0.08
    pl.subplots_adjust(left=0.08, bottom=adjust_bottom)
    pl.savefig(os.path.join(save_dir_img, 'sta.png'))
    pl.show()