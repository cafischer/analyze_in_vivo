from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, load_field_indices
from grid_cell_stimuli.spike_phase import get_spike_phases_by_min, plot_phase_hist_on_axes
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
from circular_statistics import circ_cmtest, circ_median
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import threshold_by_velocity
from analyze_in_vivo.analyze_domnisoru.burst_len_vs_preceding_silence import get_n_spikes_per_event, \
    get_ISI_idx_per_event
from cell_fitting.util import init_nan
from cell_characteristics import to_idx


def plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_cells, title):
    if cell_type == 'grid_cells':
        plot_kwargs = dict(phases=AP_max_phases_cells, title=None, color_hist='0.5', color_lines='k', alpha=1.0,
                           label='', y_max_vline=1, plot_mean=True, plot_median=False, plot_std=True)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_phase_hist_on_axes, plot_kwargs,
                                xlabel='Phase', ylabel='Frequency', fig_title=title, sharey='none',
                                save_dir_img=os.path.join(save_dir_img, 'spike_phase_'+title.replace('-', '_')+'.png'))


def horizontal_square_bracket(ax, star, x_l, x_r, y_d, y_u, y_text):
    ax.plot([x_l, x_l, (x_l + x_r) * 0.5, (x_l + x_r) * 0.5, (x_l + x_r) * 0.5, x_r, x_r],
            [y_d, (y_d + y_u) * 0.5, (y_d + y_u) * 0.5, y_u, (y_d + y_u) * 0.5, (y_d + y_u) * 0.5, y_d],
            lw=1.5, c='k')
    ax.text((x_l + x_r) * 0.5, y_text, star, ha='center', va='center', color='k', fontsize=10)


def plot_both_phase_hist_all_cells(AP_max_phases1_cells, AP_max_phases2_cells, title, labels, fig_name):

    def plot_both_phase_hist(ax, cell_idx, AP_max_phases1_cells, AP_max_phases2_cells):
        plot_phase_hist_on_axes(ax, cell_idx, AP_max_phases1_cells,
                                plot_median=True, color_hist='r', color_lines='r', label=labels[0], alpha=0.5,
                                y_max_vline=0.8)
        plot_phase_hist_on_axes(ax, cell_idx, AP_max_phases2_cells,
                                plot_median=True, color_hist='b', color_lines='b', label=labels[1], alpha=0.5,
                                y_max_vline=0.8)
        if cell_idx == 8:
            ax.legend(fontsize=10)

        # plotting significance
        ylim_max = ax.get_ylim()[1]
        pval, _, _ = circ_cmtest([AP_max_phases1_cells[cell_idx] / 360. * 2 * np.pi,
                                  AP_max_phases2_cells[cell_idx] / 360. * 2 * np.pi])
        med1 = circ_median(AP_max_phases1_cells[cell_idx] / 360. * 2 * np.pi) * 360. / (2 * np.pi)
        med2 = circ_median(AP_max_phases2_cells[cell_idx] / 360. * 2 * np.pi) * 360. / (2 * np.pi)
        stars = ['n.s.', '*', '**', '***']
        sig_levels = np.array([1, 0.1, 0.01, 0.001])
        idx = np.where(pval < sig_levels)[0]
        if len(idx) == 0:
            print pval
            idx = 0
        else:
            idx = idx[-1]
        sig_label = stars[idx]
        horizontal_square_bracket(ax, sig_label, x_l=med1, x_r=med2,
                                  y_d=ylim_max * 1.3, y_u=ylim_max * 1.38, y_text=ylim_max * 1.46)
        ax.set_ylim(None, ylim_max * 1.55)

    plot_kwargs = dict(AP_max_phases1_cells=AP_max_phases1_cells, AP_max_phases2_cells=AP_max_phases2_cells)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_both_phase_hist, plot_kwargs,
                            xlabel='Phase', ylabel='Frequency', fig_title=title, sharey='none',
                            save_dir_img=os.path.join(save_dir_img, fig_name))


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_phase'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes', 'fVm', 'dcVm_ljpc', 'fY_cm']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    velocity_threshold = 1  # cm/sec
    ISI_burst = 8
    n_spikes_variants = np.array(['all', 'single', 'burst', '2', '3', '4', '$\geq5$'])

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    AP_max_phases_theta_cells = np.zeros((len(cell_ids), len(n_spikes_variants)), dtype=object)
    AP_max_phases_ramp_cells = np.zeros((len(cell_ids), len(n_spikes_variants)), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']
        in_field_idxs, _ = load_field_indices(cell_id, save_dir)
        theta = data['fVm']
        ramp = data['dcVm_ljpc']
        position = data['Y_cm']

        # find AP_max_idx_per_event
        ISIs = get_ISIs(AP_max_idxs, t)
        burst_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        n_spikes_per_event = get_n_spikes_per_event(burst_ISI_indicator)
        AP_max_idx_per_event = get_ISI_idx_per_event(burst_ISI_indicator)
        AP_max_idxs = AP_max_idxs[AP_max_idx_per_event]  # only take first AP of event

        # pl.figure()
        # pl.plot(t, v, 'k')
        # pl.plot(t[AP_max_idxs], v[AP_max_idxs], 'or')
        # pl.show()

        # velocity threshold
        train_n_spikes = init_nan(len(v))
        train_n_spikes[AP_max_idxs] = n_spikes_per_event
        train_AP_max_idxs = init_nan(len(v))
        train_AP_max_idxs[AP_max_idxs] = AP_max_idxs
        [train_n_spikes, train_AP_max_idxs, t_thresh,
         v_thresh, position_thresh], velocity_thresh = threshold_by_velocity([train_n_spikes, train_AP_max_idxs, t,
                                                                              v, position], velocity,
                                                                             velocity_threshold)
        t = np.arange(len(v)) * dt
        AP_max_idxs_thresh = np.where(~np.isnan(train_n_spikes))[0]
        n_spikes_thresh = train_n_spikes[~np.isnan(train_n_spikes)]
        AP_max_idxs_not_thresh = train_AP_max_idxs[~np.isnan(train_AP_max_idxs)].astype(int)
        assert np.array_equal(t[AP_max_idxs_not_thresh], t_thresh[AP_max_idxs_thresh])

        # get field
        in_field_indicator = np.zeros(len(t_thresh), dtype=int)
        in_field_indicator[in_field_idxs] = 1
        start_in, end_in = get_start_end_group_of_ones(in_field_indicator)
        n_fields = len(start_in)

        AP_max_idxs_in_field = []
        n_spikes_in_field = []
        for i_field in range(n_fields):
            in_field = np.logical_and(AP_max_idxs_thresh > start_in[i_field],
                                      AP_max_idxs_thresh < end_in[i_field])
            AP_max_idxs_in_field.extend(AP_max_idxs_not_thresh[in_field])
            n_spikes_in_field.extend(n_spikes_thresh[in_field])
        AP_max_idxs_in_field = np.array(AP_max_idxs_in_field, dtype=int)
        n_spikes_in_field = np.array(n_spikes_in_field, dtype=int)

        # # for testing idxs
        # pl.figure()
        # pl.plot(position, v, 'k')
        # for s, e in zip(start_in, end_in):
        #     pl.plot(position_thresh[s:e], v_thresh[s:e], 'y')
        # pl.plot(position[AP_max_idxs_in_field], v[AP_max_idxs_in_field], 'or')
        # pl.show()

        # phase with respect to theta
        AP_max_phases_theta = get_spike_phases_by_min(AP_max_idxs_in_field, t, theta, order=to_idx(20, dt),
                                                      dist_to_AP=to_idx(200, dt), mode='nearest')

        # phase with respect to ramp
        AP_max_phases_ramp = get_spike_phases_by_min(AP_max_idxs_in_field, t, ramp, order=to_idx(20, dt),
                                                     dist_to_AP=to_idx(2000, dt), mode='lowest')

        # get rid of nans
        n_spikes_in_field_theta = n_spikes_in_field[~np.isnan(AP_max_phases_theta)]
        n_spikes_in_field_ramp = n_spikes_in_field[~np.isnan(AP_max_phases_ramp)]
        AP_max_phases_theta = AP_max_phases_theta[~np.isnan(AP_max_phases_theta)]
        AP_max_phases_ramp = AP_max_phases_ramp[~np.isnan(AP_max_phases_ramp)]

        # divide into different n spikes
        for i, n_spikes_variant in enumerate(n_spikes_variants):
            if n_spikes_variant == 'all':
                AP_max_phases_theta_cells[cell_idx, i] = AP_max_phases_theta
                AP_max_phases_ramp_cells[cell_idx, i] = AP_max_phases_ramp
            elif n_spikes_variant == 'single':
                AP_max_phases_theta_cells[cell_idx, i] = AP_max_phases_theta[n_spikes_in_field_theta == 1]
                AP_max_phases_ramp_cells[cell_idx, i] = AP_max_phases_ramp[n_spikes_in_field_ramp == 1]
            elif n_spikes_variant == 'burst':
                AP_max_phases_theta_cells[cell_idx, i] = AP_max_phases_theta[n_spikes_in_field_theta > 1]
                AP_max_phases_ramp_cells[cell_idx, i] = AP_max_phases_ramp[n_spikes_in_field_ramp > 1]
            elif n_spikes_variant == '$\geq5$':
                AP_max_phases_theta_cells[cell_idx, i] = AP_max_phases_theta[n_spikes_in_field_theta >= 5]
                AP_max_phases_ramp_cells[cell_idx, i] = AP_max_phases_ramp[n_spikes_in_field_ramp >= 5]
            else:
                AP_max_phases_theta_cells[cell_idx, i] = AP_max_phases_theta[
                    n_spikes_in_field_theta == int(n_spikes_variant)]
                AP_max_phases_ramp_cells[cell_idx, i] = AP_max_phases_ramp[
                    n_spikes_in_field_ramp == int(n_spikes_variant)]

    # stds of bursts on ramp  TODO
    from scipy.stats import circmean, circstd
    variants = ['2', '3', '4', '$\geq5$']
    stds = np.zeros(len(variants))
    means = np.zeros(len(variants))
    for i, variant in enumerate(variants):
        idx_n_spike_variants = np.where(n_spikes_variants == variant)[0][0]
        std_phase = np.zeros(len(cell_ids))
        mean_phase = np.zeros(len(cell_ids))
        for cell_idx, cell_id in enumerate(cell_ids):
            std_phase[cell_idx] = circstd(AP_max_phases_ramp_cells[cell_idx, idx_n_spike_variants], 360, 0)
            mean_phase[cell_idx] = circmean(AP_max_phases_ramp_cells[cell_idx, idx_n_spike_variants], 360, 0)
        stds[i] = np.nanmean(std_phase)
        means[i] = np.nanmean(mean_phase)
    print variants
    print stds
    print means

    print 'std'
    print 'single', np.nanmean(
        [circstd(AP_max_phases_ramp_cells[cell_idx, np.where(n_spikes_variants == 'single')[0][0]], 360, 0)
         for cell_idx in range(len(cell_ids))])
    print 'burst', np.nanmean(
        [circstd(AP_max_phases_ramp_cells[cell_idx, np.where(n_spikes_variants == 'burst')[0][0]], 360, 0)
         for cell_idx in range(len(cell_ids))])
    print 'mean'
    print 'single', np.nanmean(
        [circmean(AP_max_phases_ramp_cells[cell_idx, np.where(n_spikes_variants == 'single')[0][0]], 360, 0)
         for cell_idx in range(len(cell_ids))])
    print 'burst', np.nanmean(
        [circmean(AP_max_phases_ramp_cells[cell_idx, np.where(n_spikes_variants == 'burst')[0][0]], 360, 0)
         for cell_idx in range(len(cell_ids))])

    # plots
    for i, n_spikes_variant in enumerate(n_spikes_variants):
        plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_theta_cells[:, i], 'Theta-'+n_spikes_variant)
        plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_ramp_cells[:, i], 'Ramp-'+n_spikes_variant)
        pl.close('all')

    plot_both_phase_hist_all_cells(AP_max_phases_theta_cells[:, np.where(n_spikes_variants == 'single')[0][0]],
                                   AP_max_phases_theta_cells[:, np.where(n_spikes_variants == 'burst')[0][0]],
                                   'Theta', ['single', 'burst'], 'spike_phase_single+burst_theta.png')
    plot_both_phase_hist_all_cells(AP_max_phases_ramp_cells[:, np.where(n_spikes_variants == 'single')[0][0]],
                                   AP_max_phases_ramp_cells[:, np.where(n_spikes_variants == 'burst')[0][0]],
                                   'Ramp', ['single', 'burst'], 'spike_phase_single+burst_ramp.png')
    pl.show()