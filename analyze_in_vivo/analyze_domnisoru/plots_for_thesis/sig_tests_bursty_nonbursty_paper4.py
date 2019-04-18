from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.stats import ttest_ind
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_DAP_cells, get_label_burstgroups, get_colors_burstgroups
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import horizontal_square_bracket, get_star_from_p_val
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.stats import ttest_ind, kruskal, ks_2samp
import pandas as pd
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
pl.style.use('paper')


save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_spike_events = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
save_dir_ISI_return_map = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
save_dir_ISI_hist_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
save_dir_spike_events_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
save_dir_ISI_return_map_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'
save_dir_deltafAHP_deltaDAP = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP/not_avg_times'
save_dir_firing_rate = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
save_dir_firing_rate_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'

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

cell_type_dict = get_celltype_dict(save_dir)
theta_cells = load_cell_ids(save_dir, 'giant_theta')
DAP_cells = get_cell_ids_DAP_cells(new=True)

if not os.path.exists(save_dir_img):
    os.makedirs(save_dir_img)

labels_burstgroups = get_label_burstgroups()
colors_burstgroups = get_colors_burstgroups()
max_ISI = 200
bin_width = 1
sigma_smooth = 1
burst_ISI = 8  # ms
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
remove_cells = True

folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)

# load domnisoru
peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy')).astype(float)
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'width_at_half_ISI_peak.npy')).astype(float)
fraction_burst = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_burst.npy'))
shortest_ISI = np.load(os.path.join(save_dir_ISI_hist, folder, 'shortest_ISI.npy'))
CV_ISIs = np.load(os.path.join(save_dir_ISI_hist, folder, 'CV_ISIs.npy'))
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
AP_thresh_derivative = 3.0
AP_thresh_idx = np.array([get_AP_onset_idxs(sta_derivative[:to_idx(before_AP, dt)], AP_thresh_derivative)[-1] for sta_derivative in sta_derivative_cells])
v_onset = np.array([sta_mean_cells[i][AP_thresh_idx[i]] for i in range(len(sta_mean_cells))])
t_onset = np.array([t_sta[AP_thresh_idx[i]] + before_AP for i in range(len(sta_mean_cells))])
v_start = np.array([sta_mean_cells[i][0] for i in range(len(sta_mean_cells))])
vdiff_onset_start = (v_onset - v_start) / t_onset

DAP_time_cells = np.load(os.path.join(save_dir_deltafAHP_deltaDAP, 'DAP_time.npy'))
time_AP_fAHP_cells = np.load(os.path.join(save_dir_deltafAHP_deltaDAP, 'time_AP_fAHP.npy'))

cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))

# load latuske
peak_ISI_hist_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'peak_ISI_hist.npy')).astype(float)
width_ISI_hist_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'width_at_half_ISI_peak.npy')).astype(float)
fraction_burst_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'fraction_burst.npy')).astype(float)
shortest_ISI_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'shortest_ISI.npy'))
CV_ISIs_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'CV_ISIs.npy'))
fraction_ISI_or_ISI_next_burst_latuske = np.load(os.path.join(save_dir_ISI_return_map_latuske,
                                                              'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width),
                                                              'fraction_ISI_or_ISI_next_burst.npy')).astype(float)
fraction_single_latuske = np.load(os.path.join(save_dir_spike_events_latuske, 'burst_ISI_' + str(burst_ISI),
                                               'fraction_single.npy')).astype(float)
firing_rate_latuske = np.load(os.path.join(save_dir_firing_rate_latuske, 'firing_rate.npy'))
cell_ids_latuske = [str(i) for i in range(len(peak_ISI_hist_latuske))]
cell_ids_nonbursty_latuske = ['10', '11', '12', '13', '14', '21', '29', '30', '31', '33', '34', '35', '52', '57']
burst_label_latuske = np.logical_not(np.array([True if cell_id in cell_ids_nonbursty_latuske else False for cell_id in cell_ids_latuske]))

# plot
fig = pl.figure(figsize=(12, 8))
n_cols = 8
outer = gridspec.GridSpec(3, int(np.ceil((n_cols+1)*3/3)))

data = [fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst,  width_ISI_hist,
        peak_ISI_hist, firing_rate, shortest_ISI, CV_ISIs]
data_latuske = [fraction_burst_latuske, fraction_single_latuske, fraction_ISI_or_ISI_next_burst_latuske,
                width_ISI_hist_latuske, peak_ISI_hist_latuske, firing_rate_latuske, shortest_ISI_latuske,
                CV_ISIs_latuske]
ylabels = ['Fraction ISIs $\leq$ 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] $\leq$ 8ms',
           'Width of the ISI hist. (ms)', 'Location of the ISI hist. peak (ms)',
           'Firing rate (Hz)', 'Mean 10% shortest ISIs', 'CV of ISIs']
sigmas = [0.05, 0.05, 0.05, 1.5, 1.5, 0.5, 0.5, 0.05]
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

i_row = 0
i_col = 0
for i in range(n_cols):
    if i_col >= 9:
        i_row += 1
        i_col = 0
    ax = pl.Subplot(fig, outer[i_row, i_col])
    i_col += 1
    fig.add_subplot(ax)
    plot_with_markers(ax, -np.ones(np.sum(labels_burstgroups['B'])), data[i][labels_burstgroups['B']],
                      cell_ids[labels_burstgroups['B']], cell_type_dict,
                      edgecolor=colors_burstgroups['B'], theta_cells=theta_cells, legend=False)
    plot_with_markers(ax, np.zeros(np.sum(labels_burstgroups['B+D'])), data[i][labels_burstgroups['B+D']],
                      cell_ids[labels_burstgroups['B+D']], cell_type_dict,
                      edgecolor=colors_burstgroups['B+D'], theta_cells=theta_cells, legend=False)
    handles = plot_with_markers(ax, np.ones(np.sum(labels_burstgroups['NB'])), data[i][labels_burstgroups['NB']],
                                cell_ids[labels_burstgroups['NB']], cell_type_dict,
                      edgecolor=colors_burstgroups['NB'], theta_cells=theta_cells, legend=False)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['B', 'B+D', 'NB'])
    ax.yaxis.set_label_coords(-0.5, 0.5)

    ax.set_xlim([-1.5, 1.5])
    if i < 3:
        ax.set_ylim([0, 1.1])
    else:
        ax.set_ylim([0, None])
    ax.set_ylabel(ylabels[i], fontsize=10)
    ax.text(-0.8, 1.0, letters[i], transform=ax.transAxes, size=18, weight='bold')
    ylims = ax.get_ylim()

    # domnisoru data
    if data[i] is not None:
        ax = pl.Subplot(fig, outer[i_row, i_col])
        fig.add_subplot(ax)
        data_bursty = data[i][np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])]
        data_nonbursty = data[i][labels_burstgroups['NB']]
        kde_bursty = perform_kde(data_bursty, sigmas[i])
        kde_nonbursty = perform_kde(data_nonbursty, sigmas[i])
        x_kde = np.arange(ylims[0], ylims[1], 0.001)
        y_kde_bursty = evaluate_kde(x_kde, kde_bursty)
        y_kde_nonbursty = evaluate_kde(x_kde, kde_nonbursty)
        ax.plot(y_kde_bursty, x_kde, 'r')
        ax.plot(y_kde_bursty, x_kde, 'y', linestyle='--')
        ax.plot(y_kde_nonbursty, x_kde, 'b')
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_title('Domnisoru', fontsize=10)
        #ax.set_yticks([])
    i_col += 1

    # latuske data
    if data_latuske[i] is not None:
        ax = pl.Subplot(fig, outer[i_row, i_col])
        fig.add_subplot(ax)
        kde_bursty = perform_kde(data_latuske[i][burst_label_latuske], sigmas[i])
        kde_nonbursty = perform_kde(data_latuske[i][~burst_label_latuske], sigmas[i])
        #kde = perform_kde(data_latuske[i], sigmas[i])
        x_kde = np.arange(ylims[0], ylims[1], 0.001)
        #y_kde = evaluate_kde(x_kde, kde)
        y_kde_bursty = evaluate_kde(x_kde, kde_bursty)
        y_kde_nonbursty = evaluate_kde(x_kde, kde_nonbursty)
        #ax.plot(y_kde, x_kde, 'k')
        ax.plot(y_kde_bursty, x_kde, 'r')
        ax.plot(y_kde_nonbursty, x_kde, 'b')
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_title('Latuske', fontsize=10)
        #ax.set_yticks([])
    i_col += 1


ax = pl.Subplot(fig, outer[2, -1])
fig.add_subplot(ax)
handles += [Patch(color=colors_burstgroups['B+D'], label='Bursty+DAP'),
            Patch(color=colors_burstgroups['B'], label='Bursty'),
            Patch(color=colors_burstgroups['NB'], label='Non-bursty')]
ax.legend(handles=handles)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

pl.tight_layout()
pl.subplots_adjust(wspace=0.65, hspace=0.33)
pl.savefig(os.path.join(save_dir_img, 'difference_bursty_nonbursty_3groups_with_latuske_domnisoru.png'))
#pl.show()

df = pd.DataFrame(data=np.vstack((fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst,
                                 width_ISI_hist, peak_ISI_hist, firing_rate,
                                 shortest_ISI, CV_ISIs, vdiff_onset_start, DAP_time_cells, time_AP_fAHP_cells)).T,
                 columns=['Fraction ISIs <= 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] <= 8ms',
                          'Width ISI hist.', 'Location of ISI hist. peak', 'Firing rate (Hz)',
                          'Mean 10% shortest ISIs', 'CV of ISIs', 'Linear slope before AP',
                          'Time AP-DAP', 'Time AP-fAHP'], index=cell_ids)
df.index.name = 'Cell ID'
df.to_csv(os.path.join(save_dir_img, 'spike_characteristics.csv'), float_format='%.2f')

# statistics
p_domnisoru_B_BD = np.zeros(n_cols)
p_domnisoru_B_NB = np.zeros(n_cols)
p_latuske_B_NB = np.zeros(n_cols)
p_domnisoru_latuske_B = np.zeros(n_cols)
p_domnisoru_latuske_NB = np.zeros(n_cols)
for i in range(n_cols):
    # statistics Domnisoru
    _, p_domnisoru_B_BD[i] = ttest_ind(data[i][labels_burstgroups['B']], data[i][labels_burstgroups['B+D']])
    _, p_domnisoru_B_NB[i] = ttest_ind(data[i][np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])], data[i][labels_burstgroups['NB']])
    #_, p_val1 = kruskal(data[i][labels_burstgroups['B']], data[i][labels_burstgroups['B+D']])
    #_, p_val2 = kruskal(data[i][np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])], data[i][labels_burstgroups['NB']])

    # statistics Latuske
    _, p_latuske_B_NB[i] = ttest_ind(data_latuske[i][burst_label_latuske], data_latuske[i][~burst_label_latuske])
    #_, p_val2 = kruskal(data_latuske[i][burst_label_latuske],
    #                    data_latuske[i][~burst_label_latuske])

    # Kolmogorov-Smirnov Domnisoru - Latuske
    _, p_domnisoru_latuske_B[i] = ks_2samp(data[i][np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])],
                                           data_latuske[i][burst_label_latuske])
    _, p_domnisoru_latuske_NB[i] = ks_2samp(data[i][labels_burstgroups['NB']],
                                            data_latuske[i][~burst_label_latuske])

df = pd.DataFrame(data=np.vstack((p_domnisoru_B_BD, p_domnisoru_B_NB, p_latuske_B_NB, p_domnisoru_latuske_B,
                                  p_domnisoru_latuske_NB)).T,
                  index=['Fraction ISIs <= 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] <= 8ms',
                           'Width ISI hist.', 'Location of ISI hist. peak', 'Firing rate (Hz)',
                           'Mean 10% shortest ISIs', 'CV of ISIs'],
                  columns=['p(B, B+D) Domnisoru', 'p(B(all), NB) Domnisoru', 'p(B, NB) Latuske',
                           'p(B-Domnisoru, B-Latuske)', 'p(NB-Domnisoru, NB-Latuske)'])
df.index.name = 'Characteristic'
df.to_csv(os.path.join(save_dir_img, 'p_vals_spike_characteristics.csv'), float_format='%.5f')