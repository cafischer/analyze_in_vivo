from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import numpy as np
import os
from scipy.stats import ttest_ind
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import horizontal_square_bracket, get_star_from_p_val
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.stats import f_oneway, ttest_ind, kruskal
import pandas as pd
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
pl.style.use('paper')


#save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
#save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
#save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
#save_dir_spike_events = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
#save_dir_ISI_return_map = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map/cut_ISIs_at_200/grid_cells'
#save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
#save_dir_ISI_hist_latuske = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
#save_dir_spike_events_latuske = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
#save_dir_ISI_return_map_latuske = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'

save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
save_dir_ISI_hist = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_spike_events = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
save_dir_ISI_return_map = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_firing_rate = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
save_dir_sta = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
save_dir_ISI_hist_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
save_dir_spike_events_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
save_dir_ISI_return_map_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'
save_dir_firing_rate_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'

cell_type_dict = get_celltype_dict(save_dir)
theta_cells = load_cell_ids(save_dir, 'giant_theta')
DAP_cells = get_cell_ids_DAP_cells(new=True)

if not os.path.exists(save_dir_img):
    os.makedirs(save_dir_img)

color_burst1 = 'y'
color_burst2 = 'r'
color_nonburst = 'b'
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
peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy'))
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'width_at_half_ISI_peak.npy'))
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

cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))
burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
cell_ids_burst1 = DAP_cells + ['s43_0003']  # ['s76_0002', 's117_0002', 's118_0002', 's120_0002']
burst1_label = np.array([True if cell_id in cell_ids_burst1 else False for cell_id in cell_ids])
burst2_label = np.logical_and(burst_label, ~burst1_label)

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
n_bursty = sum(burst_label)
n_nonbursty = sum(~burst_label)

fig = pl.figure(figsize=(10, 8))
n_cols = 8
outer = gridspec.GridSpec(3, int(np.ceil(n_cols*2/3)))

data = [fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst,  width_ISI_hist,
        peak_ISI_hist, firing_rate, shortest_ISI, CV_ISIs]
data_latuske = [fraction_burst_latuske, fraction_single_latuske, fraction_ISI_or_ISI_next_burst_latuske,
                width_ISI_hist_latuske, peak_ISI_hist_latuske, firing_rate_latuske, shortest_ISI_latuske,
                CV_ISIs_latuske]
ylabels = ['Fraction ISIs $\leq$ 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] $\leq$ 8ms',
           'Width of the ISI hist. at half max. (ms)', 'Location of the ISI hist. peak (ms)',
           'Firing rate (Hz)', 'Mean 10% shortest ISIs', 'CV of ISIs']
sigmas = [0.05, 0.05, 0.05, 1, 1, 0.5, 0.5, 0.05]
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

i_row = 0
i_col = 0
for i in range(n_cols):
    # ANOVA and post-hoc t-test
    # f_stat, p_val = f_oneway(data[i][burst1_label], data[i][burst2_label], data[i][~burst_label])
    # #print 'p-val: %.5f' % p_val
    # bonferroni_correction = 3
    _, p_val1 = ttest_ind(data[i][burst1_label], data[i][burst2_label])
    _, p_val2 = ttest_ind(data[i][burst_label], data[i][~burst_label])
    print ylabels[i]
    print 't-test'
    print 'p-val(B+D, B): %.5f' % (p_val1)
    print 'p-val(B(all), N-B): %.5f' % (p_val2)
    _, p_val1 = kruskal(data[i][burst1_label], data[i][burst2_label])
    _, p_val2 = kruskal(data[i][burst_label], data[i][~burst_label])
    print ''
    print 'kruskal wallis'
    print 'p-val(B+D, B): %.5f' % (p_val1)
    print 'p-val(B(all), N-B): %.5f' % (p_val2)
    print ''

    if i_col >= 6:
        i_row += 1
        i_col = 0
    ax = pl.Subplot(fig, outer[i_row, i_col])
    i_col += 1
    fig.add_subplot(ax)
    plot_with_markers(ax, -np.ones(n_bursty), data[i][burst2_label], cell_ids[burst2_label], cell_type_dict,
                      edgecolor=color_burst2, theta_cells=theta_cells, legend=False)
    plot_with_markers(ax, np.zeros(n_bursty), data[i][burst1_label], cell_ids[burst1_label], cell_type_dict,
                      edgecolor=color_burst1, theta_cells=theta_cells, legend=False)
    handles = plot_with_markers(ax, np.ones(n_nonbursty), data[i][~burst_label], cell_ids[~burst_label], cell_type_dict,
                      edgecolor=color_nonburst, theta_cells=theta_cells, legend=False)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['B', 'B+D', 'N-B'])
    ax.yaxis.set_label_coords(-0.4, 0.5)

    ax.set_xlim([-1.5, 1.5])
    if i < 3:
        ax.set_ylim([0, 1.1])
    else:
        ax.set_ylim([0, None])
    ax.set_ylabel(ylabels[i], fontsize=10)
    ax.text(-0.9, 1.0, letters[i], transform=ax.transAxes, size=18, weight='bold')
    ylims = ax.get_ylim()

    # latuske data
    if data_latuske[i] is not None:
        ax = pl.Subplot(fig, outer[i_row, i_col])
        fig.add_subplot(ax)
        data_bursty = data_latuske[i][burst_label_latuske]
        data_nonbursty = data_latuske[i][~burst_label_latuske]
        kde_bursty = perform_kde(data_bursty, sigmas[i])
        kde_nonbursty = perform_kde(data_nonbursty, sigmas[i])
        #kde = perform_kde(data_latuske[i], sigmas[i])
        x_kde = np.arange(ylims[0], ylims[1], 0.001)
        #y_kde = evaluate_kde(x_kde, kde)
        y_kde_bursty = evaluate_kde(x_kde, kde_bursty)
        y_kde_nonbursty = evaluate_kde(x_kde, kde_nonbursty)
        #ax.plot(y_kde, x_kde, 'k')
        ax.plot(y_kde_bursty, x_kde, 'r')
        ax.plot(y_kde_nonbursty, x_kde, 'b')
        ax.set_ylim(ylims[0], ylims[1])
    i_col += 1


ax = pl.Subplot(fig, outer[2, -1])
fig.add_subplot(ax)
handles += [Patch(color=color_burst1, label='Bursty+DAP'), Patch(color=color_burst2, label='Bursty'),
            Patch(color=color_nonburst, label='Non-bursty')]
ax.legend(handles=handles)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

pl.tight_layout()
#pl.subplots_adjust(top=0.97, left=0.09, right=0.99, bottom=0.07)
pl.savefig(os.path.join(save_dir_img, 'difference_bursty_nonbursty_3groups_with_latuske.png'))
pl.show()


df = pd.DataFrame(data=np.vstack((fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst,
                                  width_ISI_hist, peak_ISI_hist, firing_rate,
                                  shortest_ISI, CV_ISIs, vdiff_onset_start)).T,
                  columns=['Fraction ISIs <= 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] <= 8ms',
                           'Width ISI hist.', 'Location of ISI hist. peak', 'Firing rate (Hz)',
                           'Mean 10% shortest ISIs', 'CV of ISIs', 'Linear slope before AP'], index=cell_ids)
df.index.name = 'Cell ID'
df = df.astype(float).round(2)
df.to_csv(os.path.join(save_dir_img, 'spike_characteristics.csv'))