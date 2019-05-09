from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.stats import ttest_ind
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_DAP_cells, \
    get_label_burstgroups, get_colors_burstgroups
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import horizontal_square_bracket, get_star_from_p_val
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.stats import f_oneway, ttest_ind, kruskal
import pandas as pd
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
from scipy.stats import pearsonr
pl.style.use('paper')


save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_spike_events = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
save_dir_ISI_return_map = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_firing_rate = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
save_dir_ISI_hist_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
save_dir_spike_events_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
save_dir_ISI_return_map_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'
save_dir_firing_rate_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'
save_dir_delta_DAP = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'

# save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
# save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
# save_dir_ISI_hist = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
# save_dir_spike_events = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
# save_dir_ISI_return_map = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
# save_dir_firing_rate = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
# save_dir_sta = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
# save_dir_ISI_hist_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
# save_dir_spike_events_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
# save_dir_ISI_return_map_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'
# save_dir_firing_rate_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'
# save_dir_delta_DAP = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'

cell_type_dict = get_celltype_dict(save_dir)
theta_cells = load_cell_ids(save_dir, 'giant_theta')
DAP_cells = get_cell_ids_DAP_cells(new=True)

if not os.path.exists(save_dir_img):
    os.makedirs(save_dir_img)

max_ISI = 200
bin_width = 1  # ms
sigma_smooth = 1  # ms
burst_ISI = 8  # ms
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
remove_cells = True

folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)

# load domnisoru
peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy'))
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'width_at_half_ISI_peak.npy'))
fraction_burst = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_burst.npy'))
shortest_ISI = np.load(os.path.join(save_dir_ISI_hist, folder, 'shortest_ISI.npy'))
CV_ISIs = np.load(os.path.join(save_dir_ISI_hist, folder, 'CV_ISIs.npy'))
fraction_ISIs_8_16 = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_ISIs_8_16.npy'))
fraction_ISIs_8_25 = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_ISIs_8_25.npy'))
fraction_single = np.load(os.path.join(save_dir_spike_events, 'fraction_single_' + str(burst_ISI) + '.npy'))
fraction_ISI_or_ISI_next_burst = np.load(os.path.join(save_dir_ISI_return_map,
                                                      'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width),
                                                      'fraction_ISI_or_ISI_next_burst.npy'))
firing_rate = np.load(os.path.join(save_dir_firing_rate, 'firing_rate.npy'))
folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
              + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
sta_mean_cells = np.load(os.path.join(save_dir_sta, folder_name, 'sta_mean.npy'))
t_sta = np.arange(-before_AP, after_AP+dt, dt)
sta_std_before_AP = np.array([np.std(sta_mean[:to_idx(before_AP-t_vref, dt)]) for sta_mean in sta_mean_cells])
sta_derivative_cells = np.array([np.diff(sta_mean) / dt for sta_mean in sta_mean_cells])
AP_thresh_derivative = 15
AP_thresh_idx = np.array([get_AP_onset_idxs(sta_derivative[:to_idx(before_AP, dt)], AP_thresh_derivative)[-1] for sta_derivative in sta_derivative_cells])
v_onset = np.array([sta_mean_cells[i][AP_thresh_idx[i]] for i in range(len(sta_mean_cells))])
t_onset = np.array([t_sta[AP_thresh_idx[i]] + before_AP for i in range(len(sta_mean_cells))])
v_start = np.array([sta_mean_cells[i][0] for i in range(len(sta_mean_cells))])
vdiff_onset_start = (v_onset - v_start) / t_onset

v_onset_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_onset_fAHP.npy'))
v_DAP_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_DAP_fAHP.npy'))
v_onset = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_onset.npy'))
v_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_fAHP.npy'))
v_DAP = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_DAP.npy'))
DAP_time = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'DAP_time.npy'))

cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))
labels_burstgroups = get_label_burstgroups(save_dir)
colors_burstgroups = get_colors_burstgroups()
NB_label = labels_burstgroups['NB']
BD_label = labels_burstgroups['B+D']
B_label = labels_burstgroups['B']


# data = [fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst,  width_ISI_hist,
#         peak_ISI_hist, firing_rate, shortest_ISI, CV_ISIs, vdiff_onset_start]
# ylabels = ['Fraction ISIs $\leq$ 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] $\leq$ 8ms',
#            'Width of the ISI hist. (ms)', 'ISI hist. peak (ms)',
#            'Firing rate (Hz)', 'Mean 10% shortest ISIs', 'CV of ISIs', 'Linear slope before AP (mV/ms)']

data = [firing_rate, fraction_burst, fraction_ISIs_8_25,
        peak_ISI_hist, CV_ISIs]
ylabels = ['Firing rate (Hz)', 'P(ISIs $\leq$ 8ms)', 'P(8 < ISI < 25)',
           'ISI hist. peak (ms)', 'CV of ISIs']

def plot_against(x, x_label):
    fig, axes = pl.subplots(5, 1, figsize=(3.5, 8), squeeze=False) # pl.subplots(3, 3, figsize=(10, 8))
    i_row = 0
    i_col = 0
    for i in range(5):  #9
        #if i_col == 3:
        #    i_row += 1
        #    i_col = 0
        corr_BD = pearsonr(x[BD_label], data[i][BD_label])[0]
        corr_B = pearsonr(x[B_label], data[i][B_label])[0]
        corr_BD_B = pearsonr(x[np.logical_or(B_label, BD_label)], data[i][np.logical_or(B_label, BD_label)])[0]
        axes[i_row, i_col].annotate('corr (B+D): %.2f' % corr_BD, xy=(0.6, 0.9), xycoords='axes fraction')
        axes[i_row, i_col].annotate('corr (B-D): %.2f' % corr_B, xy=(0.6, 0.7), xycoords='axes fraction')
        axes[i_row, i_col].annotate('corr (B): %.2f' % corr_BD_B, xy=(0.6, 0.5), xycoords='axes fraction')

        plot_with_markers(axes[i_row, i_col], x[BD_label], data[i][BD_label], cell_ids[BD_label],
                          cell_type_dict, edgecolor=colors_burstgroups['B+D'], legend=False)
        plot_with_markers(axes[i_row, i_col], x[B_label], data[i][B_label], cell_ids[B_label],
                          cell_type_dict, edgecolor=colors_burstgroups['B'], legend=False)
        plot_with_markers(axes[i_row, i_col], x[NB_label], data[i][NB_label], cell_ids[NB_label],
                          cell_type_dict, edgecolor=colors_burstgroups['NB'], legend=False)
        axes[i_row, i_col].set_xlabel(x_label, fontsize=9)
        axes[i_row, i_col].set_ylabel(ylabels[i], fontsize=9)
        #axes[i_row, i_col].set_xlim(0, None)

        #if i < 3:
        #    axes[i_row, i_col].set_ylim([0, 1.1])
        #else:
        #    axes[i_row, i_col].set_ylim([0, None])
        axes[i_row, i_col].set_ylim([0, None])

        #i_col += 1
        i_row += 1
    handles_bursty = [Patch(color=colors_burstgroups['B'], label='B-D'),
                      Patch(color=colors_burstgroups['B+D'], label='B+D'),
                      Patch(color=colors_burstgroups['NB'], label='NB')]
    #axes[0, 0].legend(handles=handles_bursty, loc='upper right', fontsize=8)
    pl.tight_layout()
    pl.show()


# plot
# plot_against(DAP_time, '$Time_{AP-DAP}$')
plot_against(v_onset_fAHP, 'delta fAHP')
plot_against(v_DAP_fAHP, 'delta DAP')
# plot_against(v_onset, 'V at AP onset')
# plot_against(v_fAHP, 'V fAHP')
# plot_against(v_DAP, 'V DAP')
# pl.show()

fig, ax = pl.subplots()
#plot_with_markers(ax, v_onset_fAHP, peak_ISI_hist, cell_ids, cell_type_dict, legend=False)
plot_with_markers(ax, v_onset_fAHP[BD_label], peak_ISI_hist[BD_label], cell_ids[BD_label],
                  cell_type_dict, edgecolor=colors_burstgroups['B+D'], legend=False)
plot_with_markers(ax, v_onset_fAHP[B_label], peak_ISI_hist[B_label], cell_ids[B_label],
                  cell_type_dict, edgecolor=colors_burstgroups['B'], legend=False)
plot_with_markers(ax, v_onset_fAHP[NB_label], peak_ISI_hist[NB_label], cell_ids[NB_label],
                  cell_type_dict, edgecolor=colors_burstgroups['NB'], legend=False)
pl.semilogy()
pl.xlabel('Delta fAHP')
pl.ylabel('Peak ISI hist.')
pl.tight_layout()
pl.show()

fig, ax = pl.subplots()
#plot_with_markers(ax, v_DAP_fAHP, peak_ISI_hist, cell_ids, cell_type_dict, legend=False)
plot_with_markers(ax, v_DAP_fAHP[BD_label], peak_ISI_hist[BD_label], cell_ids[BD_label],
                  cell_type_dict, edgecolor=colors_burstgroups['B+D'], legend=False)
plot_with_markers(ax, v_DAP_fAHP[B_label], peak_ISI_hist[B_label], cell_ids[B_label],
                  cell_type_dict, edgecolor=colors_burstgroups['B'], legend=False)
plot_with_markers(ax, v_DAP_fAHP[NB_label], peak_ISI_hist[NB_label], cell_ids[NB_label],
                  cell_type_dict, edgecolor=colors_burstgroups['NB'], legend=False)
pl.semilogy()
pl.xlabel('Delta DAP')
pl.ylabel('Peak ISI hist.')
pl.tight_layout()
pl.show()

