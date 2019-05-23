from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_label_burstgroups
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.stats import ttest_ind, kruskal
import pandas as pd


save_dir_table = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_n_spikes_in_burst = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_events'
save_dir_ISI_return_map = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
cell_type_dict = get_celltype_dict(save_dir)
labels_burstgroups = get_label_burstgroups(save_dir)

if not os.path.exists(save_dir_table):
    os.makedirs(save_dir_table)

max_ISI = None
bin_width = 1
sigma_smooth = 1
ISI_burst = 8  # ms
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
remove_cells = True

grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))

folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy'))
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'width_at_half_ISI_peak.npy'))
fraction_burst = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_burst.npy'))
fraction_single = np.load(os.path.join(save_dir_n_spikes_in_burst, 'fraction_single_' + str(ISI_burst) + '.npy'))
fraction_ISI_or_ISI_next_burst = np.load(os.path.join(save_dir_ISI_return_map,
                                                      'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width),
                                                      'fraction_ISI_or_ISI_next_burst.npy'))
folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
              + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
sta_mean_cells = np.load(os.path.join(save_dir_sta, folder_name, 'sta_mean.npy'))
t_sta = np.arange(-before_AP, after_AP+dt, dt)
sta_std_before_AP = np.array([np.std(sta_mean[:to_idx(before_AP-t_vref, dt)]) for sta_mean in sta_mean_cells])
sta_derivative_cells = np.array([np.diff(sta_mean) / dt for sta_mean in sta_mean_cells])
AP_thresh_derivative = 3.0
AP_thresh_idx = np.array([get_AP_onset_idxs(sta_derivative[:to_idx(before_AP, dt)], AP_thresh_derivative)[-1] for sta_derivative in sta_derivative_cells])
v_onset = np.array([sta_mean_cells[i][AP_thresh_idx[i]] for i in range(len(sta_mean_cells))])
t_onset = np.array([t_sta[AP_thresh_idx[i]] + before_AP for i in range(len(sta_mean_cells))])
v_start = np.array([sta_mean_cells[i][AP_thresh_idx[i] - to_idx(10, dt)] for i in range(len(sta_mean_cells))])
vdiff_onset_start = (v_onset - v_start) / t_onset
print 'mean onset_slope B+D: %.2f' % np.mean(vdiff_onset_start[labels_burstgroups['B+D']])
print 'mean onset_slope B-D: %.2f' % np.mean(vdiff_onset_start[labels_burstgroups['B']])
print 'mean onset_slope NB: %.2f' % np.mean(vdiff_onset_start[labels_burstgroups['NB']])
print 'mean onset_slope B: %.2f' % np.mean(vdiff_onset_start[np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])])
print 'std onset_slope B+D: %.2f' % np.std(vdiff_onset_start[labels_burstgroups['B+D']])
print 'std onset_slope B-D: %.2f' % np.std(vdiff_onset_start[labels_burstgroups['B']])
print 'std onset_slope NB: %.2f' % np.std(vdiff_onset_start[labels_burstgroups['NB']])
print 'std onset_slope B: %.2f' % np.std(vdiff_onset_start[np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])])


# significance tests
_, p_val_fraction_burst = ttest_ind(fraction_burst[burst_label], fraction_burst[~burst_label])
_, p_val_fraction_single = ttest_ind(fraction_single[burst_label], fraction_single[~burst_label])
_, p_val_fraction_ISI_or_ISI_next_burst = ttest_ind(fraction_ISI_or_ISI_next_burst[burst_label],
                                                    fraction_ISI_or_ISI_next_burst[~burst_label])

# significance tests
data = [fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst, vdiff_onset_start]
ylabels = ['Fraction ISIs $\leq$ 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] $\leq$ 8ms',
           'Linear slope before AP (mV/ms)']

for i in range(len(data)):
    _, p_val1 = ttest_ind(data[i][burst1_label], data[i][burst2_label])
    _, p_val2 = ttest_ind(data[i][np.logical_or(burst1_label, burst2_label)], data[i][~burst_label])
    print ylabels[i]
    print 't-test'
    print 'p-val(B+D, B): %.5f' % (p_val1)
    print 'p-val(B(all), N-B): %.5f' % (p_val2)
    _, p_val1 = kruskal(data[i][burst1_label], data[i][burst2_label])
    _, p_val2 = kruskal(data[i][np.logical_or(burst1_label, burst2_label)], data[i][~burst_label])
    print ''
    print 'kruskal wallis'
    print 'p-val(B+D, B): %.5f' % (p_val1)
    print 'p-val(B(all), N-B): %.5f' % (p_val2)
    print ''


df = pd.DataFrame(data=np.vstack((fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst, vdiff_onset_start,
                                  width_ISI_hist, peak_ISI_hist)).T,
                  columns=['Fraction ISIs < 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] < 8ms',
                           'Linear slope before AP', 'Width ISI hist.', 'ISI hist. peak'], index=grid_cells)
df.index.name = 'Cell ID'
df = df.astype(float).round(2)
df.to_csv(os.path.join(save_dir_table, 'spike_characteristics.csv'))