from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from grid_cell_stimuli.spike_phase import get_spike_phases, get_spike_phases_by_min, plot_phase_hist, plot_phase_hist_on_axes
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
from circular_statistics import circ_cmtest, circ_median
from analyze_in_vivo.analyze_domnisoru.n_spikes_in_burst import get_n_spikes_in_burst
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from scipy.stats import circmean, circstd


def plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_cells, title):
    if cell_type == 'grid_cells':
        plot_kwargs = dict(phases=AP_max_phases_cells, title=None, color_hist='0.5', color_lines='r',
                            alpha=0.5, label='', y_max_vline=1, plot_mean=True, plot_median=False, plot_std=True)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_phase_hist_on_axes, plot_kwargs,
                                xlabel='Phase', ylabel='Frequency', fig_title=title, sharey='none',
                                save_dir_img=os.path.join(save_dir_img, 'spike_phase_'+title.replace('-', '_')+'.png'))
    else:
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
                    plot_phase_hist_on_axes(axes[i1, i2], cell_idx, AP_max_phases_cells, plot_mean=True, plot_std=True)
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
        pl.savefig(os.path.join(save_dir_img, 'spike_phase_'+title.replace('-', '_')+'.png'))


def horizontal_square_bracket(ax, star, x_l, x_r, y_d, y_u, y_text):
    ax.plot([x_l, x_l, (x_l + x_r) * 0.5, (x_l + x_r) * 0.5, (x_l + x_r) * 0.5, x_r, x_r],
            [y_d, (y_d + y_u) * 0.5, (y_d + y_u) * 0.5, y_u, (y_d + y_u) * 0.5, (y_d + y_u) * 0.5, y_d],
            lw=1.5, c='k')
    ax.text((x_l + x_r) * 0.5, y_text, star, ha='center', va='center', color='k', fontsize=10)


def plot_both_phase_hist_all_cells(AP_max_phases1_per_cell, AP_max_phases2_per_cell, title, labels, fig_name):

    def plot_both_phase_hist(ax, cell_idx, AP_max_phases1_per_cell, AP_max_phases2_per_cell):
        plot_phase_hist_on_axes(ax, cell_idx, AP_max_phases1_per_cell,
                                plot_median=True, color_hist='0.3', color_lines='0.3', label=labels[0], alpha=0.6,
                                y_max_vline=0.9)
        plot_phase_hist_on_axes(ax, cell_idx, AP_max_phases2_per_cell,
                                plot_median=True, color_hist='0.7', color_lines='0.7', label=labels[1], alpha=0.6,
                                y_max_vline=0.9)
        if cell_idx == 8:
            ax.legend(fontsize=10)

        # plotting significance
        ylim_max = ax.get_ylim()[1]
        pval, _, _ = circ_cmtest([AP_max_phases1_per_cell[cell_idx] / 360. * 2 * np.pi,
                                  AP_max_phases2_per_cell[cell_idx] / 360. * 2 * np.pi])
        med1 = circ_median(AP_max_phases1_per_cell[cell_idx] / 360. * 2 * np.pi) * 360. / (2 * np.pi)
        med2 = circ_median(AP_max_phases2_per_cell[cell_idx] / 360. * 2 * np.pi) * 360. / (2 * np.pi)
        sig = pval < 0.05
        if sig:
            horizontal_square_bracket(ax, '*', x_l=med1, x_r=med2,
                                      y_d=ylim_max, y_u=ylim_max + 1, y_text=ylim_max + 2)
        else:
            horizontal_square_bracket(ax, 'n.s.', x_l=med1, x_r=med2,
                                      y_d=ylim_max, y_u=ylim_max + 1, y_text=ylim_max + 2)
        ax.set_ylim(None, ylim_max + 2)

    plot_kwargs = dict(AP_max_phases1_per_cell=AP_max_phases1_per_cell, AP_max_phases2_per_cell=AP_max_phases2_per_cell)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_both_phase_hist, plot_kwargs,
                            xlabel='Phase', ylabel='Frequency', fig_title=title, sharey='none',
                            save_dir_img=os.path.join(save_dir_img, fig_name))


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_phase'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field/vel_thresh_1'
    save_dir_theta_ramp = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/theta_ramp'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes', 'fVm', 'dcVm_ljpc']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    velocity_threshold = 1  # cm/sec
    ISI_burst = 10
    use_AP_max_idxs_domnisoru = True

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    AP_max_phases_theta_per_cell = []
    AP_max_phases_ramp_per_cell = []
    AP_max_phases_field_per_cell = []
    AP_max_phases_theta_burst_per_cell = []
    AP_max_phases_ramp_burst_per_cell = []
    AP_max_phases_field_burst_per_cell = []
    AP_max_phases_theta_single_per_cell = []
    AP_max_phases_ramp_single_per_cell = []
    AP_max_phases_field_single_per_cell = []

    for cell_id in cell_ids:
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        velocity = data['vel_100ms']
        dt = t[1] - t[0]
        in_field_len_orig = np.load(os.path.join(save_dir_in_out_field, cell_type, cell_id, 'in_field_len_orig.npy'))
        out_field_len_orig = np.load(os.path.join(save_dir_in_out_field, cell_type, cell_id, 'out_field_len_orig.npy'))

        # get theta and ramp
        theta = data['fVm']
        ramp = data['dcVm_ljpc']
        # theta = np.load(os.path.join(save_dir_theta_ramp, cell_type, cell_id, 'theta.npy'))
        # ramp = np.load(os.path.join(save_dir_theta_ramp, cell_type, cell_id, 'ramp.npy'))

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
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        starts_burst, ends_burst = get_start_end_group_of_ones(short_ISI_indicator.astype(int))
        n_spikes_in_bursts = get_n_spikes_in_burst(short_ISI_indicator)
        AP_max_idxs_burst = AP_max_idxs[starts_burst]
        AP_max_idxs_single = np.array(filter(lambda x: x not in AP_max_idxs[ends_burst+1],
                                             AP_max_idxs[~short_ISI_indicator]))

        # # for testing idxs
        # pl.figure()
        # pl.plot(t, v, 'k')
        # pl.plot(t[AP_max_idxs_burst], v[AP_max_idxs_burst], 'or')
        # pl.plot(t[AP_max_idxs_single], v[AP_max_idxs_single], 'ob')
        # pl.show()

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
        AP_max_phases_theta_single = []
        AP_max_phases_ramp_single = []
        AP_max_phases_field_single = []
        for i_field in range(n_fields):
            AP_max_idxs_in_field = AP_max_idxs[np.logical_and(AP_max_idxs > start_in[i_field],
                                                              AP_max_idxs < end_in[i_field])]
            AP_max_idxs_in_field_burst = AP_max_idxs_burst[np.logical_and(AP_max_idxs_burst > start_in[i_field],
                                                              AP_max_idxs_burst < end_in[i_field])]
            AP_max_idxs_in_field_single = AP_max_idxs_single[np.logical_and(AP_max_idxs_single > start_in[i_field],
                                                              AP_max_idxs_single < end_in[i_field])]

            # respect to theta
            AP_max_phases_theta.append(get_spike_phases(AP_max_idxs_in_field, t, theta, order=int(round(20. / dt)),
                                                        dist_to_AP=int(round(200. / dt))))
            AP_max_phases_theta_burst.append(get_spike_phases(AP_max_idxs_in_field_burst, t, theta,
                                                              order=int(round(20. / dt)),
                                                              dist_to_AP=int(round(200. / dt))))
            AP_max_phases_theta_single.append(get_spike_phases(AP_max_idxs_in_field_single, t, theta,
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
            AP_max_phases_ramp_burst.append(get_spike_phases_by_min(AP_max_idxs_in_field_burst, t, ramp,
                                                                    order=int(round(20. / dt)),
                                                                    dist_to_AP=int(round(2000. / dt))))
            AP_max_phases_ramp_single.append(get_spike_phases_by_min(AP_max_idxs_in_field_single, t, ramp,
                                                                     order=int(round(20. / dt)),
                                                                     dist_to_AP=int(round(2000. / dt))))
            
            # respect to field
            phases_field = np.linspace(0, 360, end_in[i_field] - start_in[i_field] + 1)
            AP_max_phases_field.append(phases_field[AP_max_idxs_in_field - start_in[i_field]])
            AP_max_phases_field_burst.append(phases_field[AP_max_idxs_in_field_burst - start_in[i_field]])
            AP_max_phases_field_single.append(phases_field[AP_max_idxs_in_field_single - start_in[i_field]])

        # # plots
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        #
        # if not os.path.exists(os.path.join(save_dir_cell, 'all')):
        #     os.makedirs(os.path.join(save_dir_cell, 'all'))
        # if not os.path.exists(os.path.join(save_dir_cell, 'burst')):
        #     os.makedirs(os.path.join(save_dir_cell, 'burst'))
        # if not os.path.exists(os.path.join(save_dir_cell, 'single')):
        #     os.makedirs(os.path.join(save_dir_cell, 'single'))
        #
        # pl.close('all')
        #
        # AP_max_phases_theta_all = [item for sublist in AP_max_phases_theta for item in sublist]
        # AP_max_phases_theta_all = np.array(AP_max_phases_theta_all)[~np.isnan(AP_max_phases_theta_all)]
        # AP_max_phases_theta_per_cell.append(AP_max_phases_theta_all)
        # plot_phase_hist(AP_max_phases_theta_all, os.path.join(save_dir_cell, 'single', 'theta.png'),
        #                 mean_phase=circmean(AP_max_phases_theta_all, 360, 0),
        #                 std_phase=circstd(AP_max_phases_theta_all, 360, 0),
        #                 title='Theta')
        #
        # AP_max_phases_ramp_all = [item for sublist in AP_max_phases_ramp for item in sublist]
        # AP_max_phases_ramp_all = np.array(AP_max_phases_ramp_all)[~np.isnan(AP_max_phases_ramp_all)]
        # AP_max_phases_ramp_per_cell.append(AP_max_phases_ramp_all)
        # plot_phase_hist(AP_max_phases_ramp_all, os.path.join(save_dir_cell, 'all', 'ramp.png'),
        #                 mean_phase=circmean(AP_max_phases_ramp_all, 360, 0),
        #                 std_phase=circstd(AP_max_phases_ramp_all, 360, 0),
        #                 title='Ramp')
        #
        # AP_max_phases_field_all = np.array([item for sublist in AP_max_phases_field for item in sublist])
        # AP_max_phases_field_per_cell.append(AP_max_phases_field_all)
        # plot_phase_hist(AP_max_phases_field_all, os.path.join(save_dir_cell, 'all', 'field.png'),
        #                 mean_phase=circmean(AP_max_phases_field_all, 360, 0),
        #                 std_phase=circstd(AP_max_phases_field_all, 360, 0),
        #                 title='Field')
        #
        # AP_max_phases_theta_all_burst = [item for sublist in AP_max_phases_theta_burst for item in sublist]
        # AP_max_phases_theta_all_burst = np.array(AP_max_phases_theta_all_burst)[~np.isnan(AP_max_phases_theta_all_burst)]
        # AP_max_phases_theta_burst_per_cell.append(AP_max_phases_theta_all_burst)
        # plot_phase_hist(AP_max_phases_theta_all_burst, os.path.join(save_dir_cell, 'all', 'theta.png'),
        #                 mean_phase=circmean(AP_max_phases_theta_all_burst, 360, 0),
        #                 std_phase=circstd(AP_max_phases_theta_all_burst, 360, 0),
        #                 title='Theta')
        #
        # AP_max_phases_ramp_all_burst = [item for sublist in AP_max_phases_ramp_burst for item in sublist]
        # AP_max_phases_ramp_all_burst = np.array(AP_max_phases_ramp_all_burst)[~np.isnan(AP_max_phases_ramp_all_burst)]
        # AP_max_phases_ramp_burst_per_cell.append(AP_max_phases_ramp_all_burst)
        # plot_phase_hist(AP_max_phases_ramp_all_burst, os.path.join(save_dir_cell, 'burst', 'ramp.png'),
        #                 mean_phase=circmean(AP_max_phases_ramp_all_burst, 360, 0),
        #                 std_phase=circstd(AP_max_phases_ramp_all_burst, 360, 0),
        #                 title='Ramp')
        #
        # AP_max_phases_field_all_burst = np.array([item for sublist in AP_max_phases_field_burst for item in sublist])
        # AP_max_phases_field_burst_per_cell.append(AP_max_phases_field_all_burst)
        # plot_phase_hist(AP_max_phases_field_all_burst, os.path.join(save_dir_cell, 'burst', 'field.png'),
        #                 mean_phase=circmean(AP_max_phases_field_all_burst, 360, 0),
        #                 std_phase=circstd(AP_max_phases_field_all_burst, 360, 0),
        #                 title='Field')
        #
        # AP_max_phases_theta_all_single = [item for sublist in AP_max_phases_theta_single for item in sublist]
        # AP_max_phases_theta_all_single = np.array(AP_max_phases_theta_all_single)[~np.isnan(AP_max_phases_theta_all_single)]
        # AP_max_phases_theta_single_per_cell.append(AP_max_phases_theta_all_single)
        # plot_phase_hist(AP_max_phases_theta_all_single, os.path.join(save_dir_cell, 'single', 'theta.png'),
        #                 mean_phase=circmean(AP_max_phases_theta_all_single, 360, 0),
        #                 std_phase=circstd(AP_max_phases_theta_all_single, 360, 0),
        #                 title='Theta')
        #
        # AP_max_phases_ramp_all_single = [item for sublist in AP_max_phases_ramp_single for item in sublist]
        # AP_max_phases_ramp_all_single = np.array(AP_max_phases_ramp_all_single)[~np.isnan(AP_max_phases_ramp_all_single)]
        # AP_max_phases_ramp_single_per_cell.append(AP_max_phases_ramp_all_single)
        # plot_phase_hist(AP_max_phases_ramp_all_single, os.path.join(save_dir_cell, 'single', 'ramp.png'),
        #                 mean_phase=circmean(AP_max_phases_ramp_all_single, 360, 0),
        #                 std_phase=circstd(AP_max_phases_ramp_all_single, 360, 0),
        #                 title='Ramp')
        #
        # AP_max_phases_field_all_single = np.array([item for sublist in AP_max_phases_field_single for item in sublist])
        # AP_max_phases_field_single_per_cell.append(AP_max_phases_field_all_single)
        # plot_phase_hist(AP_max_phases_field_all_single, os.path.join(save_dir_cell, 'single', 'field.png'),
        #                 mean_phase=circmean(AP_max_phases_field_all_single, 360, 0),
        #                 std_phase=circstd(AP_max_phases_field_all_single, 360, 0),
        #                 title='Field')

    # plot all cells
    pl.close('all')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_theta_per_cell, 'Theta')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_ramp_per_cell, 'Ramp')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_field_per_cell, 'Field')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_theta_burst_per_cell, 'Theta - Bursts')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_ramp_burst_per_cell, 'Ramp - Bursts')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_field_burst_per_cell, 'Field - Bursts')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_theta_single_per_cell, 'Theta - Single')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_ramp_single_per_cell, 'Ramp - Single')
    plot_phase_hist_all_cells(cell_type_dict, AP_max_phases_field_single_per_cell, 'Field - Single')

    plot_both_phase_hist_all_cells(AP_max_phases_theta_per_cell, AP_max_phases_theta_burst_per_cell, 'Theta',
                                   ['all', 'burst'], 'spike_phase_all+burst_theta.png')
    plot_both_phase_hist_all_cells(AP_max_phases_ramp_per_cell, AP_max_phases_ramp_burst_per_cell, 'Ramp',
                                   ['all', 'burst'], 'spike_phase_all+burst_ramp.png')
    plot_both_phase_hist_all_cells(AP_max_phases_field_per_cell, AP_max_phases_field_burst_per_cell, 'Field',
                                   ['all', 'burst'], 'spike_phase_all+burst_field.png')

    pl.close('all')
    plot_both_phase_hist_all_cells(AP_max_phases_theta_single_per_cell, AP_max_phases_theta_burst_per_cell,
                                          'Theta', ['single', 'burst'], 'spike_phase_single+burst_theta.png')
    plot_both_phase_hist_all_cells(AP_max_phases_ramp_single_per_cell, AP_max_phases_ramp_burst_per_cell,
                                          'Ramp', ['single', 'burst'], 'spike_phase_single+burst_ramp.png')
    plot_both_phase_hist_all_cells(AP_max_phases_field_single_per_cell, AP_max_phases_field_burst_per_cell,
                                          'Field', ['single', 'burst'], 'spike_phase_single+burst_field.png')
    pl.show()