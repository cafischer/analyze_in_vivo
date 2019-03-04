import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.util import init_nan
from analyze_in_vivo.load.load_domnisoru import load_data
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from grid_cell_stimuli import find_all_AP_traces
from cell_characteristics.analyze_APs import get_spike_characteristics
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_characteristics.sta_stc import get_sta


def get_sta_criterion(do_detrend, before_AP, after_AP, AP_criterion, t_vref, cell_ids, save_dir):
    sta_mean_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_good_APs_cells = np.zeros(len(cell_ids), dtype=object)
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    n_APs_good_cells = np.zeros(len(cell_ids))
    param_list = ['Vm_ljpc', 'spiketimes']

    for cell_idx, cell_id in enumerate(cell_ids):
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        AP_max_idxs = data['spiketimes']

        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)
        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt
        if v_APs is None:
            continue

        # get AP amp. and width
        AP_amps = np.zeros(len(v_APs))
        AP_widths = np.zeros(len(v_APs))
        for i, v_AP in enumerate(v_APs):
            spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
            AP_amps[i], AP_widths[i] = get_spike_characteristics(v_AP, t_AP, ['AP_amp', 'AP_width'],
                                                                 v_rest=v_AP[before_AP_idx - to_idx(t_vref, dt)],
                                                                 AP_max_idx=before_AP_idx,
                                                                 AP_onset=before_AP_idx - to_idx(1.0, dt),
                                                                 std_idx_times=(0, 1), check=False,
                                                                 **spike_characteristics_dict)

        # select APs
        if AP_criterion.keys()[0] == 'quantile':
            AP_amp_thresh = np.nanpercentile(AP_amps, 100 - AP_criterion.values()[0])
            AP_width_thresh = np.nanpercentile(AP_widths, AP_criterion.values()[0])
        elif AP_criterion.keys()[0] == 'AP_amp_and_width':
            AP_amp_thresh = AP_criterion.values()[0][0]
            AP_width_thresh = AP_criterion.values()[0][1]
        else:
            raise ValueError('AP criterion does not exist!')

        good_APs = np.logical_and(AP_amps >= AP_amp_thresh, AP_widths <= AP_width_thresh)
        v_APs_good = v_APs[good_APs]
        n_APs_good_cells[cell_idx] = len(v_APs_good)

        # STA
        sta_mean_cells[cell_idx], sta_std_cells[cell_idx] = get_sta(v_APs)
        if len(v_APs_good) > 5:
            sta_mean_good_APs_cells[cell_idx], sta_std_good_APs_cells[cell_idx] = get_sta(v_APs_good)
        else:
            sta_mean_good_APs_cells[cell_idx] = init_nan(len(sta_mean_cells[cell_idx]))
            sta_std_good_APs_cells[cell_idx] = init_nan(len(sta_mean_cells[cell_idx]))

    return sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells, sta_std_good_APs_cells, n_APs_good_cells


def plot_sta_on_ax(ax, cell_idx, t_AP, sta_mean_cells, sta_std_cells, before_AP=5, after_AP=25, ylims=(None, None)):
    ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
    ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                    sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
    ax.set_ylim(*ylims)
    ax.set_xlim(-before_AP, after_AP)
    ax.set_xticks(np.arange(-before_AP, after_AP+10, 10))


def plot_sta_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
                        sta_std_good_APs_cells, before_AP, after_AP, ylims=(None, None)):
    if subplot_idx == 0:
        ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_good_APs_cells[cell_idx] - sta_std_good_APs_cells[cell_idx],
                        sta_mean_good_APs_cells[cell_idx] + sta_std_good_APs_cells[cell_idx], color='0.6')
        ax.set_xlabel('')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP + 5, 10))
        ax.set_xticklabels([])
        ax.annotate('selected APs', xy=(25, ax.get_ylim()[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)
    elif subplot_idx == 1:
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP+5, 10))
        ax.annotate('all APs', xy=(25, ax.get_ylim()[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)


def plot_sta_derivative_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_mean_good_APs_cells,
                             before_AP, after_AP, time_for_max, ylims=(None, None), diff_selected_all=None):

    if subplot_idx == 0:
        if ~np.any(np.isnan(sta_mean_good_APs_cells[cell_idx])):
            ax.fill_between((0, time_for_max), ylims[0], ylims[1], color='0.8')
        ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.annotate('selected APs', xy=(25, ylims[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)
        if ~np.isnan(diff_selected_all[cell_idx]):
            ax.annotate('%.1f' % diff_selected_all[cell_idx], xy=(25, ylims[1]), textcoords='data',
                        horizontalalignment='right', verticalalignment='top', fontsize=8)

        # # smooth
        # std = np.std(sta_mean_good_APs_cells[cell_idx][to_idx(2, dt):to_idx(3, dt)])
        # #std = sta_std_cells[cell_idx][:-1]
        # w = np.ones(len(sta_mean_good_APs_cells[cell_idx])) / std
        # print 'w1', w[0]
        # splines = UnivariateSpline(t_AP, sta_mean_good_APs_cells[cell_idx], w=w, s=None, k=3)
        # smoothed = splines(t_AP)
        #
        # ax.plot(t_AP, smoothed, 'r')

    elif subplot_idx == 1:
        if ~np.any(np.isnan(sta_mean_cells[cell_idx])):
            ax.fill_between((0, time_for_max), ylims[0], ylims[1], color='0.8')
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks([-10, 0, 10, 20])
        ax.annotate('all APs', xy=(25, ylims[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        # # smooth
        # std = np.std(sta_mean_cells[cell_idx][to_idx(2, dt):to_idx(3, dt)])
        # #std = sta_std_good_APs_cells[cell_idx][:-1]
        # w = np.ones(len(sta_mean_cells[cell_idx])) / std
        # print 'w2', w[0]
        # splines = UnivariateSpline(t_AP, sta_mean_cells[cell_idx], w=w, s=None, k=3)
        # smoothed = splines(t_AP)
        #
        # ax.plot(t_AP, smoothed, 'r')