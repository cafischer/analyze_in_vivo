from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from grid_cell_stimuli.spike_phase import get_spike_phases, get_spike_phases_by_min, plot_phase_hist, plot_phase_hist_on_axes
from scipy.stats import circmean, circstd
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones


def plot_phase_hist_all_cells(AP_max_phases_cells, title):
    n_rows = 1 if len(cell_ids) <= 3 else 2
    fig_height = 4.5 if len(cell_ids) <= 3 else 9
    fig, axes = pl.subplots(n_rows, int(round(len(cell_ids) / n_rows)), sharex='all', sharey='all',
                            figsize=(14, fig_height))
    fig.suptitle(title, fontsize=16)
    if n_rows == 1:
        axes = np.array([axes])
    if len(cell_ids) == 1:
        axes = np.array([axes])
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(int(round(len(cell_ids) / n_rows))):
            if cell_idx < len(cell_ids):
                plot_phase_hist_on_axes(axes[i1, i2], AP_max_phases_cells[cell_idx],
                                        mean_phase=circmean(AP_max_phases_cells[cell_idx], 360, 0),
                                        std_phase=circstd(AP_max_phases_cells[cell_idx], 360, 0))
                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('Phase')
                if i2 == 0:
                    axes[i1, i2].set_ylabel('Frequency')
                axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=14)
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    pl.tight_layout()
    adjust_bottom = 0.12 if len(cell_ids) <= 3 else 0.08
    pl.subplots_adjust(left=0.06, bottom=adjust_bottom, top=0.9)
    pl.savefig(os.path.join(save_dir_img, cell_type, 'spike_phase_'+title.replace('-', '_')+'.png'))


def plot_phase_hist_all_cells_with_bursts(AP_max_phases_cells, AP_max_phases_burst_cells, title):
    n_rows = 1 if len(cell_ids) <= 3 else 2
    fig_height = 4.5 if len(cell_ids) <= 3 else 9
    fig, axes = pl.subplots(n_rows, int(round(len(cell_ids) / n_rows)), sharex='all', sharey='all',
                            figsize=(14, fig_height))
    fig.suptitle(title, fontsize=16)
    if n_rows == 1:
        axes = np.array([axes])
    if len(cell_ids) == 1:
        axes = np.array([axes])
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(int(round(len(cell_ids) / n_rows))):
            if cell_idx < len(cell_ids):
                plot_phase_hist_on_axes(axes[i1, i2], AP_max_phases_cells[cell_idx],
                                        mean_phase=circmean(AP_max_phases_cells[cell_idx], 360, 0),
                                        std_phase=circstd(AP_max_phases_cells[cell_idx], 360, 0),
                                        color_hist='0.3', color_mean='0.3', label='all', alpha=0.6)
                plot_phase_hist_on_axes(axes[i1, i2], AP_max_phases_burst_cells[cell_idx],
                                        mean_phase=circmean(AP_max_phases_burst_cells[cell_idx], 360, 0),
                                        std_phase=circstd(AP_max_phases_burst_cells[cell_idx], 360, 0),
                                        color_hist='0.7', color_mean='0.7', label='bursts', alpha=0.6)
                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('Phase')
                if i2 == 0:
                    axes[i1, i2].set_ylabel('Frequency')
                axes[i1, i2].legend()
                axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
            cell_idx += 1
    pl.tight_layout()
    adjust_bottom = 0.12 if len(cell_ids) <= 3 else 0.08
    pl.subplots_adjust(left=0.06, bottom=adjust_bottom, top=0.88)
    pl.savefig(os.path.join(save_dir_img, cell_type, 'spike_phase_with_burst_'+title.replace('-', '_')+'.png'))


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_field/spike_phase'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir_theta_ramp = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/theta_ramp'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = '   pyramidal_layer2'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    velocity_threshold = 1  # cm/sec
    ISI_burst = 10
    use_AP_max_idxs_domnisoru = True

    AP_max_phases_theta_cells = []
    AP_max_phases_ramp_cells = []
    AP_max_phases_field_cells = []
    AP_max_phases_theta_burst_cells = []
    AP_max_phases_ramp_burst_cells = []
    AP_max_phases_field_burst_cells = []

    for cell_id in cell_ids:
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        in_field_len_orig = np.load(os.path.join(save_dir_in_out_field, cell_type, cell_id, 'in_field_len_orig.npy'))
        out_field_len_orig = np.load(os.path.join(save_dir_in_out_field, cell_type, cell_id, 'out_field_len_orig.npy'))

        # get theta and ramp
        theta = np.load(os.path.join(save_dir_theta_ramp, cell_type, cell_id, 'theta.npy'))
        ramp = np.load(os.path.join(save_dir_theta_ramp, cell_type, cell_id, 'ramp.npy'))

        # get phases
        start_in, end_in = get_start_end_group_of_ones(in_field_len_orig.astype(int))
        n_fields = len(start_in)

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        # find burst indices
        ISIs = get_ISIs(AP_max_idxs, t)
        starts_burst, ends_burst = get_start_end_group_of_ones(np.concatenate((ISIs <= ISI_burst, np.array([False]))).astype(int))
        AP_max_idxs_burst = AP_max_idxs[starts_burst]

        # remove low velocity spikes
        to_low = velocity < velocity_threshold
        AP_max_idxs = AP_max_idxs[np.array([~to_low[AP_max_idx] for AP_max_idx in AP_max_idxs], dtype=bool)]
        AP_max_idxs_burst = AP_max_idxs_burst[np.array([~to_low[AP_max_idx_burst]
                                                        for AP_max_idx_burst in AP_max_idxs_burst], dtype=bool)]

        AP_max_phases_theta = []
        AP_max_phases_ramp = []
        AP_max_phases_field = []
        AP_max_phases_theta_burst = []
        AP_max_phases_ramp_burst = []
        AP_max_phases_field_burst = []
        for i_field in range(n_fields):
            AP_max_idxs_in_field = AP_max_idxs[np.logical_and(AP_max_idxs > start_in[i_field],
                                                              AP_max_idxs < end_in[i_field])]
            AP_max_idxs_in_field_burst = AP_max_idxs_burst[np.logical_and(AP_max_idxs_burst > start_in[i_field],
                                                              AP_max_idxs_burst < end_in[i_field])]

            # respect to theta
            AP_max_phases_theta.append(get_spike_phases(AP_max_idxs_in_field, t, theta, order=int(round(20. / dt)),
                                                        dist_to_AP=int(round(200. / dt))))
            AP_max_phases_theta_burst.append(get_spike_phases(AP_max_idxs_in_field_burst, t, theta,
                                                              order=int(round(20. / dt)),
                                                              dist_to_AP=int(round(200. / dt))))

            # respect to ramp
            # pl.figure()
            # pl.plot(t, v, 'k')
            # #pl.xlim(t[start_in[i_field]], t[end_in[i_field]])
            # pl.ylabel('Membrane potential (mV)')
            # pl.xlabel('Time (ms)')
            # pl.tight_layout()
            # pl.show()
            AP_max_phases_ramp.append(get_spike_phases_by_min(AP_max_idxs_in_field, t, ramp, order=int(round(20. / dt)),
                                                        dist_to_AP=int(round(2000. / dt))))
            AP_max_phases_ramp_burst.append(get_spike_phases_by_min(AP_max_idxs_in_field_burst, t, ramp, order=int(round(20. / dt)),
                                                        dist_to_AP=int(round(2000. / dt))))

            # respect to field
            phases_field = np.linspace(0, 360, end_in[i_field] - start_in[i_field] + 1)
            AP_max_phases_field.append(phases_field[AP_max_idxs_in_field - start_in[i_field]])
            AP_max_phases_field_burst.append(phases_field[AP_max_idxs_in_field_burst - start_in[i_field]])

        # plots
        if not os.path.exists(os.path.join(save_dir_cell, 'single')):
            os.makedirs(os.path.join(save_dir_cell, 'single'))
        if not os.path.exists(os.path.join(save_dir_cell, 'burst')):
            os.makedirs(os.path.join(save_dir_cell, 'burst'))

        AP_max_phases_theta_all = [item for sublist in AP_max_phases_theta for item in sublist]
        AP_max_phases_theta_all = np.array(AP_max_phases_theta_all)[~np.isnan(AP_max_phases_theta_all)]
        AP_max_phases_theta_cells.append(AP_max_phases_theta_all)
        plot_phase_hist(AP_max_phases_theta_all, os.path.join(save_dir_cell, 'single', 'theta.png'),
                        mean_phase=circmean(AP_max_phases_theta_all, 360, 0),
                        std_phase=circstd(AP_max_phases_theta_all, 360, 0),
                        title='Theta')

        AP_max_phases_ramp_all = [item for sublist in AP_max_phases_ramp for item in sublist]
        AP_max_phases_ramp_all = np.array(AP_max_phases_ramp_all)[~np.isnan(AP_max_phases_ramp_all)]
        AP_max_phases_ramp_cells.append(AP_max_phases_ramp_all)
        plot_phase_hist(AP_max_phases_ramp_all, os.path.join(save_dir_cell, 'single', 'ramp.png'),
                        mean_phase=circmean(AP_max_phases_ramp_all, 360, 0),
                        std_phase=circstd(AP_max_phases_ramp_all, 360, 0),
                        title='Ramp')

        AP_max_phases_field_all = [item for sublist in AP_max_phases_field for item in sublist]
        AP_max_phases_field_cells.append(AP_max_phases_field_all)
        plot_phase_hist(AP_max_phases_field_all, os.path.join(save_dir_cell, 'single', 'field.png'),
                        mean_phase=circmean(AP_max_phases_field_all, 360, 0),
                        std_phase=circstd(AP_max_phases_field_all, 360, 0),
                        title='Field')

        AP_max_phases_theta_all_burst = [item for sublist in AP_max_phases_theta_burst for item in sublist]
        AP_max_phases_theta_all_burst = np.array(AP_max_phases_theta_all_burst)[~np.isnan(AP_max_phases_theta_all_burst)]
        AP_max_phases_theta_burst_cells.append(AP_max_phases_theta_all_burst)
        plot_phase_hist(AP_max_phases_theta_all_burst, os.path.join(save_dir_cell, 'burst', 'theta.png'),
                        mean_phase=circmean(AP_max_phases_theta_all_burst, 360, 0),
                        std_phase=circstd(AP_max_phases_theta_all_burst, 360, 0),
                        title='Theta')

        AP_max_phases_ramp_all_burst = [item for sublist in AP_max_phases_ramp_burst for item in sublist]
        AP_max_phases_ramp_all_burst = np.array(AP_max_phases_ramp_all_burst)[~np.isnan(AP_max_phases_ramp_all_burst)]
        AP_max_phases_ramp_burst_cells.append(AP_max_phases_ramp_all_burst)
        plot_phase_hist(AP_max_phases_ramp_all_burst, os.path.join(save_dir_cell, 'burst', 'ramp.png'),
                        mean_phase=circmean(AP_max_phases_ramp_all_burst, 360, 0),
                        std_phase=circstd(AP_max_phases_ramp_all_burst, 360, 0),
                        title='Ramp')

        AP_max_phases_field_all_burst = [item for sublist in AP_max_phases_field_burst for item in sublist]
        AP_max_phases_field_burst_cells.append(AP_max_phases_field_all_burst)
        plot_phase_hist(AP_max_phases_field_all_burst, os.path.join(save_dir_cell, 'burst', 'field.png'),
                        mean_phase=circmean(AP_max_phases_field_all_burst, 360, 0),
                        std_phase=circstd(AP_max_phases_field_all_burst, 360, 0),
                        title='Field')

    # plot all cells
    pl.close('all')
    plot_phase_hist_all_cells(AP_max_phases_theta_cells, 'Theta')
    plot_phase_hist_all_cells(AP_max_phases_ramp_cells, 'Ramp')
    plot_phase_hist_all_cells(AP_max_phases_field_cells, 'Field')
    plot_phase_hist_all_cells(AP_max_phases_theta_burst_cells, 'Theta-Bursts')
    plot_phase_hist_all_cells(AP_max_phases_ramp_burst_cells, 'Ramp-Bursts')
    plot_phase_hist_all_cells(AP_max_phases_field_burst_cells, 'Field-Bursts')

    pl.close('all')
    plot_phase_hist_all_cells_with_bursts(AP_max_phases_theta_cells, AP_max_phases_theta_burst_cells, 'Theta')
    plot_phase_hist_all_cells_with_bursts(AP_max_phases_ramp_cells, AP_max_phases_ramp_burst_cells, 'Ramp')
    plot_phase_hist_all_cells_with_bursts(AP_max_phases_field_cells, AP_max_phases_field_burst_cells, 'Field')
    pl.show()