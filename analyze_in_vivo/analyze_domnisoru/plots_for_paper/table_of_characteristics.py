from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_label_burstgroups
from scipy.stats import ttest_ind, kruskal
import pandas as pd


save_dir_table = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_autocorr = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr/kde'
save_dir_ISI_return_map = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_spike_events = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_events'
save_dir_firing_rate = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
cell_type_dict = get_celltype_dict(save_dir)
labels_burstgroups = get_label_burstgroups(save_dir)

if not os.path.exists(save_dir_table):
    os.makedirs(save_dir_table)

max_ISI = None
max_lag = 50
bin_width = 1
sigma_smooth = 1
burst_ISI = 8  # ms
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
t_vref = 10
before_AP = 25
after_AP = 25

grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))

peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'sigma_smooth_' + str(sigma_smooth), 'peak_ISI_hist.npy'))
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'sigma_smooth_' + str(sigma_smooth), 'width_ISI_hist.npy'))

peak_autocorr = np.load(os.path.join(save_dir_autocorr, 'max_lag_' + str(max_lag) + '_sigma_smooth_'+str(sigma_smooth),
                                     'peak_autocorr.npy'))
width_autocorr = np.load(os.path.join(save_dir_autocorr, 'max_lag_' + str(max_lag) + '_sigma_smooth_'+str(sigma_smooth),
                                     'width_autocorr.npy'))

folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
fraction_burst_ISIs = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_burst_ISIs.npy'))
fraction_ISIs_8_25 = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_ISIs_8_25.npy'))
shortest_ISI = np.load(os.path.join(save_dir_ISI_hist, folder, 'shortest_ISI.npy'))
CV_ISIs = np.load(os.path.join(save_dir_ISI_hist, folder, 'CV_ISIs.npy'))

fraction_ISI_or_ISI_next_burst = np.load(os.path.join(save_dir_ISI_return_map, 'max_ISI_' + str(max_ISI),
                                                      'fraction_ISI_or_ISI_next_burst.npy'))

fraction_single = np.load(os.path.join(save_dir_spike_events, 'burst_ISI_' + str(burst_ISI), 'fraction_single.npy'))

firing_rate = np.load(os.path.join(save_dir_firing_rate, 'firing_rate.npy'))

folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
         + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
v_onset = np.load(os.path.join(save_dir_sta, folder, 'v_onset.npy'))
linear_slope_APonset = np.load(os.path.join(save_dir_sta, folder, 'linear_slope_APonset.npy'))


# significance tests
data = [peak_ISI_hist, peak_autocorr, width_ISI_hist, width_autocorr, fraction_burst_ISIs, fraction_ISIs_8_25,
        shortest_ISI, CV_ISIs, fraction_ISI_or_ISI_next_burst,
        fraction_single, firing_rate, v_onset, linear_slope_APonset]
columns = ['ISI peak', 'SI peak', 'ISI width', 'SI width', 'ISI<8', '8<ISI<25',
           '10% shortest ISI', 'CV ISI', 'ISI[n]/[n+1]<8',
           'Frac. single', 'Firing rate', 'V onset', 'Linear slope']

p_BpD_BmD_ttest = np.zeros(len(data))
p_B_NB_ttest = np.zeros(len(data))
p_BpD_BmD_kruskal = np.zeros(len(data))
p_B_NB_kruskal = np.zeros(len(data))
mean_BpD = np.zeros(len(data))
mean_BmD = np.zeros(len(data))
mean_NB = np.zeros(len(data))
mean_B = np.zeros(len(data))
std_BpD = np.zeros(len(data))
std_BmD = np.zeros(len(data))
std_NB = np.zeros(len(data))
std_B = np.zeros(len(data))

for i in range(len(data)):
    _, p_BpD_BmD_ttest[i] = ttest_ind(data[i][labels_burstgroups['B+D']], data[i][labels_burstgroups['B']])
    _, p_B_NB_ttest[i] = ttest_ind(data[i][np.logical_or(labels_burstgroups['B+D'], labels_burstgroups['B'])],
                                data[i][labels_burstgroups['NB']])
    _, p_BpD_BmD_kruskal[i] = kruskal(data[i][labels_burstgroups['B+D']], data[i][labels_burstgroups['B']])
    _, p_B_NB_kruskal[i] = kruskal(data[i][np.logical_or(labels_burstgroups['B+D'], labels_burstgroups['B'])],
                              data[i][labels_burstgroups['NB']])
    _, p_BpD_NB_kruskal = kruskal(data[i][labels_burstgroups['B+D']], data[i][labels_burstgroups['NB']])
    _, p_BmD_NB_kruskal = kruskal(data[i][labels_burstgroups['B']], data[i][labels_burstgroups['NB']])
    print columns[i]
    print 'p(B+D,NB) kruskal: %.5f' % p_BpD_NB_kruskal
    print 'p(B-D,NB) kruskal: %.5f' % p_BmD_NB_kruskal

    mean_BpD[i] = np.mean(data[i][labels_burstgroups['B+D']])
    mean_BmD[i] = np.mean(data[i][labels_burstgroups['B']])
    mean_NB[i] = np.mean(data[i][labels_burstgroups['NB']])
    mean_B[i] = np.mean(data[i][np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])])
    std_BpD[i] = np.std(data[i][labels_burstgroups['B+D']])
    std_BmD[i] =  np.std(data[i][labels_burstgroups['B']])
    std_NB[i] =  np.std(data[i][labels_burstgroups['NB']])
    std_B[i] = np.std(data[i][np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])])


df_statistics1 = pd.DataFrame(data=np.vstack((mean_BpD, mean_BmD, mean_B, mean_NB,
                                              std_BpD, std_BmD, std_B, std_NB)),
                             columns=columns,
                            index=['mean(B+D)', 'mean(B-D)', 'mean(B)', 'mean(NB)',
                                   'std(B+D)', 'std(B-D)', 'std(B)', 'std(NB)'])
df_statistics1 = df_statistics1.astype(float).round(2)

df_statistics2 = pd.DataFrame(data=np.vstack((p_BpD_BmD_ttest, p_B_NB_ttest, p_BpD_BmD_kruskal, p_B_NB_kruskal)),
                             columns=columns,
                             index=['p(B+D,B-D) t-test', 'p(B,NB) t-test', 'p(B+D,B-D) kruskal', 'p(B,NB) kruskal'])
df_statistics2 = df_statistics2.astype(float).round(5)

df = pd.DataFrame(data=np.vstack(data).T,
                  columns=columns, index=grid_cells)
df.index.name = 'Cell ID'
df = df.astype(float).round(2)
df = pd.concat([df, df_statistics1, df_statistics2])
df.to_csv(os.path.join(save_dir_table, 'spike_characteristics.csv'))