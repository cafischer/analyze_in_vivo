from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, \
    get_label_burstgroups, get_colors_burstgroups
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from scipy.stats import pearsonr
pl.style.use('paper')


def check_sig_corr(x, y, fun, n_shuffles=10000):
    statistic = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        x_shuffle = np.random.choice(x, len(x))
        y_shuffle = np.random.choice(y, len(y))
        statistic[i] = np.abs(fun(x_shuffle, y_shuffle)[0])
        while np.isnan(statistic[i]):
            x_shuffle = np.random.choice(x, len(x))
            y_shuffle = np.random.choice(y, len(y))
            statistic[i] = np.abs(fun(x_shuffle, y_shuffle)[0])

    p_val = np.mean(np.abs(fun(x, y)[0]) <= statistic)

    # print p_val
    # pl.figure()
    # pl.hist(statistic, bins=50, color='0.5')
    # pl.axvline(np.abs(fun(x, y)[0]), color='r')
    # pl.axvline(np.quantile(statistic, .95), color='g')
    # pl.show()
    return p_val


save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_ISI_return_map = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_spike_events = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_events'
save_dir_firing_rate = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'

save_dir_delta_DAP = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig2'
    #'/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'

max_ISI = None
bin_width = 1  # ms
sigma_smooth = 1  # ms
burst_ISI = 8  # ms
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
remove_cells = True

# load
peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'sigma_smooth_' + str(sigma_smooth), 'peak_ISI_hist.npy'))
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, 'sigma_smooth_' + str(sigma_smooth), 'width_ISI_hist.npy'))
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

v_onset_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'v_onset_fAHP.npy'))
v_DAP_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'v_DAP_fAHP.npy'))
#v_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'v_fAHP.npy'))
#v_DAP = np.load(os.path.join(save_dir_delta_DAP, 'v_DAP.npy'))

cell_type_dict = get_celltype_dict(save_dir)
cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))
labels_burstgroups = get_label_burstgroups(save_dir)
colors_burstgroups = get_colors_burstgroups()
NB_label = labels_burstgroups['NB']
BD_label = labels_burstgroups['B+D']
B_label = labels_burstgroups['B']

# data
data_x = [v_onset_fAHP, v_DAP_fAHP]
data_y = [firing_rate, fraction_burst_ISIs, fraction_ISIs_8_25,
          peak_ISI_hist, CV_ISIs]
ylabels = ['Firing rate (Hz)', 'P(ISIs $\leq$ 8ms)', 'P(8 < ISI < 25)',
           'ISI hist. peak (ms)', 'CV of ISIs']

# significance tests
for x in data_x:
    for i in range(len(data_y)):
        print ylabels[i]
        p_val = check_sig_corr(x[BD_label], data_y[i][BD_label], pearsonr)
        print 'B+D %.3f' % p_val
        p_val = check_sig_corr(x[B_label], data_y[i][B_label], pearsonr)
        print 'B-D %.3f' % p_val
        p_val = check_sig_corr(x[np.logical_or(B_label, BD_label)], data_y[i][np.logical_or(B_label, BD_label)], pearsonr)
        print 'B %.3f' % p_val

# plot
def plot_against(x, x_label):
    print x_label
    fig, axes = pl.subplots(5, 1, figsize=(3.5, 8), squeeze=False)
    i_row = 0
    i_col = 0
    for i in range(5):
        plot_with_markers(axes[i_row, i_col], x[BD_label], data_y[i][BD_label], cell_ids[BD_label],
                          cell_type_dict, edgecolor=colors_burstgroups['B+D'], legend=False)
        plot_with_markers(axes[i_row, i_col], x[B_label], data_y[i][B_label], cell_ids[B_label],
                          cell_type_dict, edgecolor=colors_burstgroups['B'], legend=False)
        plot_with_markers(axes[i_row, i_col], x[NB_label], data_y[i][NB_label], cell_ids[NB_label],
                          cell_type_dict, edgecolor=colors_burstgroups['NB'], legend=False)
        axes[i_row, i_col].set_xlabel(x_label, fontsize=9)
        axes[i_row, i_col].set_ylabel(ylabels[i], fontsize=9)
        axes[i_row, i_col].set_ylim([0, None])

        i_row += 1
    pl.tight_layout()


# plot
plot_against(v_onset_fAHP, 'delta fAHP')
plot_against(v_DAP_fAHP, 'delta DAP')
# plot_against(DAP_time, '$Time_{AP-DAP}$')
# plot_against(v_onset, 'V at AP onset')
# plot_against(v_fAHP, 'V fAHP')
# plot_against(v_DAP, 'V DAP')
pl.show()

fig, ax = pl.subplots()
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


# delta fAHP
# Firing rate (Hz)
# B+D 0.881
# B-D 0.238
# B 0.914
# P(ISIs $\leq$ 8ms)
# B+D 0.356
# B-D 0.248
# B 0.017
# P(8 < ISI < 25)
# B+D 0.994
# B-D 0.104
# B 0.078
# ISI hist. peak (ms)
# B+D 0.498
# B-D 0.730
# B 0.041
# CV of ISIs
# B+D 0.517
# B-D 0.908
# B 0.075

# Delta DAP
# Firing rate (Hz)
# B+D 0.584
# B-D 0.895
# B 0.609
# P(ISIs $\leq$ 8ms)
# B+D 0.223
# B-D 0.019
# B 0.035
# P(8 < ISI < 25)
# B+D 0.511
# B-D 0.864
# B 0.263
# ISI hist. peak (ms)
# B+D 0.038
# B-D 0.088
# B 0.009
# CV of ISIs
# B+D 0.250
# B-D 0.971
# B 0.075