from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, load_field_indices, get_celltype_dict
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_sta_median, plot_APs
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_cell_group
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import threshold_by_velocity, get_spike_train
pl.style.use('paper_subplots')


def get_in_or_out_field_AP_max_idxs(kind, AP_max_idxs, velocity, cell_id, save_dir):
    if kind == 'in_field':
        spike_train = init_nan(len(velocity))  # for velocity thresholding to match in field idxs
        spike_train[AP_max_idxs] = range(len(AP_max_idxs))
        [spike_train], velocity_thresh = threshold_by_velocity([spike_train], velocity, threshold=1)
        # assert np.array_equal(data['fvel_100ms']), velocity_thresh)  # make shure same velocity thresholding as Domnisoru

        in_field_idxs, _ = load_field_indices(cell_id, save_dir)
        in_field_bool_idxs = np.zeros(len(spike_train), dtype=bool)
        in_field_bool_idxs[in_field_idxs] = True

        spike_train[~in_field_bool_idxs] = np.nan
        AP_max_idxs_selected = AP_max_idxs[np.unique(spike_train[~np.isnan(spike_train)]).astype(int)]

        # test AP selection
        # pl.figure()
        # [v_t, t_t], _ = threshold_by_velocity([v, t], velocity, threshold=1)
        # pl.plot(t, v, 'k')
        # pl.plot(t_t[in_field_bool_idxs], v_t[in_field_bool_idxs], 'y')
        # pl.plot(t[AP_max_idxs_selected], v[AP_max_idxs_selected], 'or')
        # pl.show()
    elif kind == 'out_field':
        spike_train = init_nan(len(velocity))  # for velocity thresholding to match in field idxs
        spike_train[AP_max_idxs] = range(len(AP_max_idxs))
        [spike_train], velocity_thresh = threshold_by_velocity([spike_train], velocity, threshold=1)
        # assert np.array_equal(data['fvel_100ms']), velocity_thresh)  # make shure same velocity thresholding as Domnisoru

        _, out_field_idxs = load_field_indices(cell_id, save_dir)
        out_field_bool_idxs = np.zeros(len(spike_train), dtype=bool)
        out_field_bool_idxs[out_field_idxs] = True

        spike_train[~out_field_bool_idxs] = np.nan
        AP_max_idxs_selected = AP_max_idxs[np.unique(spike_train[~np.isnan(spike_train)]).astype(int)]
    elif kind == 'all':
        AP_max_idxs_selected = AP_max_idxs
    else:
        return ValueError('kind not correctly specified!')
    return AP_max_idxs_selected


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)

    # parameters
    use_AP_max_idxs_domnisoru = True
    do_detrend = False
    kind = 'all'
    before_AP_sta = 10
    after_AP_sta = 25
    bins_v = np.arange(-90, 40+0.5, 0.5)
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms', 'fY_cm', 'fvel_100ms']
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], kind, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    DAP_deflections = {}
    sta_mean_cells = []
    sta_std_cells = []
    v_hist_cells = []
    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)

        # # for testing
        # pl.figure()
        # pl.title(cell_id)
        # pl.plot(t, v)
        # #pl.xlim(1000, 11000)
        # pl.ylim(-90, 20)
        # pl.show()

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        AP_max_idxs_selected = get_in_or_out_field_AP_max_idxs(kind, AP_max_idxs, velocity, cell_id, save_dir)

        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx_sta, after_AP_idx_sta, AP_max_idxs_selected, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt - before_AP_sta
        if v_APs is None:
            sta_mean_cells.append(init_nan(after_AP_idx_sta + before_AP_idx_sta + 1))
            sta_std_cells.append(init_nan(after_AP_idx_sta + before_AP_idx_sta + 1))
            continue

        # STA
        sta_median, sta_mad = get_sta_median(v_APs)
        sta_mean, sta_std = get_sta(v_APs)
        sta_mean_cells.append(sta_mean)
        sta_std_cells.append(sta_std)

        # # histogram of voltage
        # v_hist = np.zeros((len(bins_v)-1, np.size(v_APs, 1)))
        # for c in range(np.size(v_APs, 1)):
        #     v_hist[:, c] = np.histogram(v_APs[:, c], bins_v)[0]
        # v_hist_cells.append(v_hist)

        # # plot
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        # np.save(os.path.join(save_dir_cell, 'sta_mean.npy'), sta_mean)
        # pl.close('all')
        # plot_sta(t_AP, sta_median, sta_mad, os.path.join(save_dir_cell, 'sta_median.png'))
        # plot_sta(t_AP, sta_mean, sta_std, os.path.join(save_dir_cell, 'sta_mean.png'))
        # plot_APs(v_APs, t_AP, os.path.join(save_dir_cell, 'v_APs.png'))
        # pl.figure()
        # x, y = np.meshgrid(t_AP, bins_v[:-1])
        # pl.pcolor(x, y, v_hist)
        # pl.ylabel('Mem. pot. distr. (mV)')
        # pl.xlabel('Time (ms)')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'v_hist.png'))

    pl.close('all')

    def plot_sta(ax, cell_idx, t_AP, sta_mean_cells, sta_std_cells):
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')

    plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_cells, sta_std_cells=sta_std_cells)

    if cell_type == 'grid_cells':
        plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                                save_dir_img=os.path.join(save_dir_img, 'sta.png'))
    else:
        plot_for_cell_group(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
                                xlabel='Time (ms)', ylabel='Mem. pot. (mV)', figsize=None,
                                save_dir_img=os.path.join(save_dir_img, 'sta.png'))

    # voltage histogram over time
    def plot_v_hist(ax, cell_idx, t_AP, bins_v, v_hist_cells, before_AP_sta, after_AP_sta):
        x, y = np.meshgrid(t_AP, bins_v[:-1])
        ax.pcolor(x, y, v_hist_cells[cell_idx])
        ax.set_xlim(t_AP[0], t_AP[-1])
        ax.set_ylim(-90, -30)

    plot_kwargs = dict(t_AP=t_AP, bins_v=bins_v, v_hist_cells=v_hist_cells, before_AP_sta=before_AP_sta,
                       after_AP_sta=after_AP_sta)
    if cell_type == 'grid_cells':
      plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_v_hist, plot_kwargs,
                              xlabel='Time (ms)', ylabel='Mem. pot. distr. (mV)',
                              save_dir_img=os.path.join(save_dir_img, 'v_hist.png'))

    pl.show()