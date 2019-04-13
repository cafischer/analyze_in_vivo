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
#pl.style.use('paper_subplots')


#save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
#save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
#save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
#save_dir_ISI_hist_latuske = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
#save_dir_n_spikes_in_burst = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
#save_dir_ISI_return_map = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
#save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'

save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
save_dir_ISI_hist = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_n_spikes_in_burst = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
save_dir_ISI_return_map = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
save_dir_sta = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
save_dir_ISI_hist_latuske = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'

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
ISI_burst = 8  # ms
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
remove_cells = True

folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)

peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy'))
width_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'width_at_half_ISI_peak.npy'))
peak_ISI_hist_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'peak_ISI_hist.npy'))
width_ISI_hist_latuske = np.load(os.path.join(save_dir_ISI_hist_latuske, folder, 'width_at_half_ISI_peak.npy'))
cell_ids_latuske = [str(i) for i in range(len(peak_ISI_hist_latuske))]
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
v_start = np.array([sta_mean_cells[i][0] for i in range(len(sta_mean_cells))])
vdiff_onset_start = (v_onset - v_start) / t_onset

cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))
burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
cell_ids_burst1 = DAP_cells + ['s43_0003']  # ['s76_0002', 's117_0002', 's118_0002', 's120_0002']
burst1_label = np.array([True if cell_id in cell_ids_burst1 else False for cell_id in cell_ids])
burst2_label = np.logical_and(burst_label, ~burst1_label)

# plot
n_bursty = sum(burst_label)
n_nonbursty = sum(~burst_label)

fig = pl.figure(figsize=(12, 4))
n_cols = 6
outer = gridspec.GridSpec(1, n_cols + 1)

data = [fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst, vdiff_onset_start,  width_ISI_hist, peak_ISI_hist]
ylabels = ['Fraction ISIs $\leq$ 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] $\leq$ 8ms',
           'Linear slope before AP (mV/ms)',
           'Width of the ISI hist. at half max. (ms)', 'Location of the ISI hist. peak (ms)']
letters = ['A', 'B', 'C', 'D', 'E', 'F']

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

    ax = pl.Subplot(fig, outer[0, i])
    fig.add_subplot(ax)
    plot_with_markers(ax, -np.ones(n_bursty), data[i][burst1_label], cell_ids[burst1_label], cell_type_dict,
                      edgecolor=color_burst1, theta_cells=theta_cells, legend=False)
    plot_with_markers(ax, np.zeros(n_bursty), data[i][burst2_label], cell_ids[burst2_label], cell_type_dict,
                      edgecolor=color_burst2, theta_cells=theta_cells, legend=False)
    handles = plot_with_markers(ax, np.ones(n_nonbursty), data[i][~burst_label], cell_ids[~burst_label], cell_type_dict,
                      edgecolor=color_nonburst, theta_cells=theta_cells, legend=False)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['B+D', 'B', 'N-B'])

    ax.set_xlim([-1.5, 1.5])
    if i < 3:
        ax.set_ylim([0, 1.1])
    else:
        ax.set_ylim([0, None])
    ax.set_ylabel(ylabels[i])
    ax.text(-0.5, 1.0, letters[i], transform=ax.transAxes, size=18, weight='bold')

ax = pl.Subplot(fig, outer[0, -1])
fig.add_subplot(ax)
fig_fake, ax_fake = pl.subplots()
pl.close(fig_fake)
handles += [Patch(color=color_burst1, label='Bursty+DAP'), Patch(color=color_burst2, label='Bursty'),
            Patch(color=color_nonburst, label='Non-bursty')]
ax.legend(handles=handles)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

pl.tight_layout()
#pl.subplots_adjust(top=0.97, left=0.09, right=0.99, bottom=0.07)
pl.savefig(os.path.join(save_dir_img, 'difference_bursty_nonbursty_3groups_new.png'))
pl.show()


df = pd.DataFrame(data=np.vstack((fraction_burst, fraction_single, fraction_ISI_or_ISI_next_burst, vdiff_onset_start,
                                  width_ISI_hist, peak_ISI_hist)).T,
                  columns=['Fraction ISIs <= 8ms', 'Fraction single spikes', 'Fraction ISI[n] or ISI[n+1] <= 8ms',
                           'Linear slope before AP', 'Width ISI hist.', 'Location of ISI hist. peak'], index=cell_ids)
df.index.name = 'Cell ID'
df = df.astype(float).round(2)
df.to_csv(os.path.join(save_dir_img, 'spike_characteristics.csv'))