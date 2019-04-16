import os
import numpy as np
import matplotlib.pyplot as pl
from analyze_in_vivo.load.load_domnisoru import get_label_burstgroups, get_colors_burstgroups
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_hist_on_ax


# mean + std for
# - Spike-time autocorrelation
# - STA_V
# - ISI probability density und cumulative probability (wie bisher normiert)
# - ISI return map (Hierbei gibt es die Herausforderung, dass die verschiedenen Zellen
# unterschiedlich viele Events haben. Ein Losungsweg: Fur jede Zelle zuerst die Dichte
# im ISI_{n+1} vs ISI_n Plot schatzen (KDE), das 2D-Integral normieren und dann lokal
# uber die verschiedenen Zellen aufsummieren).

# burst groups
groups = ['NB', 'B', 'B+D']
label_burstgroups = get_label_burstgroups()
colors_burstgroups = get_colors_burstgroups()

# spiketime autocorrelation
max_lag = 50  # ms
bin_width = 1  # ms
sigma_smooth = None
normalization = 'sum'
folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(
    sigma_smooth) + '_normalization_' + str(normalization)
save_dir_autocorr = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
autocorr_cells = np.load(os.path.join(save_dir_autocorr, folder, 'autocorr.npy'))
t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)

# STA
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
         + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
sta_mean_cells = np.load(os.path.join(save_dir_sta, folder, 'sta_mean.npy'))
t_sta = np.arange(-before_AP, after_AP+dt, dt)

# ISI hist
max_ISI = 200
bin_width = 1
sigma_smooth = None
folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, folder, 'ISI_hist.npy'))
cum_ISI_hist_y = np.load(os.path.join(save_dir_ISI_hist, folder, 'cum_ISI_hist_y.npy'))
cum_ISI_hist_x = np.load(os.path.join(save_dir_ISI_hist, folder, 'cum_ISI_hist_x.npy'))
bins = np.arange(0, max_ISI + bin_width, bin_width)


# plot
fig, axes = pl.subplots(4, 4)

# spiketime autocorrelation
for i, group in enumerate(groups):
    axes[0, i].bar(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), bin_width,
                   yerr=[np.zeros(np.shape(autocorr_cells)[1]), np.std(autocorr_cells[label_burstgroups[group]], 0)],
                   color=colors_burstgroups[group],
                   align='center', alpha=0.5)

    axes[0, 3].bar(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), bin_width,
                   color=colors_burstgroups[group],
                   align='center', alpha=0.5)

# STA
for i, group in enumerate(groups):
    axes[1, i].plot(t_sta, np.mean(sta_mean_cells[label_burstgroups[group]], 0), color=colors_burstgroups[group])
    axes[1, i].fill_between(t_sta,
                            np.mean(sta_mean_cells[label_burstgroups[group]], 0) -
                            np.std(sta_mean_cells[label_burstgroups[group]], 0),
                            np.mean(sta_mean_cells[label_burstgroups[group]], 0) +
                            np.std(sta_mean_cells[label_burstgroups[group]], 0),
                            color=colors_burstgroups[group], alpha=0.5)
    axes[1, i].set_xlim(-5, None)
    axes[1, i].set_ylim(-75, -30)

    axes[1, 3].plot(t_sta, np.mean(sta_mean_cells[label_burstgroups[group]], 0), color=colors_burstgroups[group])
    # axes[1, 3].fill_between(t_sta,
    #                         np.mean(sta_mean_cells[label_burstgroups[group]], 0) -
    #                         np.std(sta_mean_cells[label_burstgroups[group]], 0),
    #                         np.mean(sta_mean_cells[label_burstgroups[group]], 0) +
    #                         np.std(sta_mean_cells[label_burstgroups[group]], 0),
    #                         color=colors_burstgroups[group], alpha=0.5)
    axes[1, 3].set_xlim(-5, None)
    axes[1, 3].set_ylim(-75, -30)

# ISI hist
for i, group in enumerate(groups):
    axes[2, i].bar(bins[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
                   bins[1] - bins[0],
                   yerr=[np.zeros(np.shape(ISI_hist_cells)[1]), np.std(ISI_hist_cells[label_burstgroups[group]], 0)],
                   color=colors_burstgroups[group], align='edge')
    ax_twin = axes[2, i].twinx()
    ax_twin.plot(cum_ISI_hist_x, cum_ISI_hist_y, color='k', drawstyle='steps-post')

    axes[2, 3].bar(bins[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
                   bins[1] - bins[0],
                   color=colors_burstgroups[group], align='edge', alpha=0.5)


pl.tight_layout()
pl.show()