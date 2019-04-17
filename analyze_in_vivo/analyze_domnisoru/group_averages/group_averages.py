import os
import numpy as np
import matplotlib.pyplot as pl
from analyze_in_vivo.load.load_domnisoru import get_label_burstgroups, get_colors_burstgroups, get_cell_ids_burstgroups, load_cell_ids
from cell_fitting.util import init_nan


# TODO
#- ISI hist nicht genormt?!
# - Kannst Du bitte zusatzlich zu den mean STA_V Plots noch eine
# Variante anfertigen, bei der das Membranpotential beim AP-Onset zu
# Null gesetzt wird, so dass man einen Plot erhält, der unabhangig von
# Zell-zu-Zell Fluktionen im AP-Onset ist.

# - Wie gross sind die Membranpotentiale zum AP-Onset in den drei
#  Gruppen (jeweils Mittelwert und Standardabweichung)?


save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/group_averages'
if not os.path.exists(save_dir_img):
    os.makedirs(save_dir_img)
cell_ids_remove = ['s104_0007', 's110_0002']

# burst groups
groups = ['NB', 'B', 'B+D']
cell_ids = np.array(load_cell_ids('/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru',
                         'grid_cells'))
cell_ids_burstgroups = get_cell_ids_burstgroups()
label_burstgroups = get_label_burstgroups()
colors_burstgroups = get_colors_burstgroups()

# remove s104, s110
for cell_id in cell_ids_remove:
    idx = np.where(cell_id == cell_ids)[0]
    label_burstgroups['B+D'][idx] = False

# spiketime autocorrelation
max_lag = 50  # ms
bin_width = 1  # ms
sigma_smooth = None
normalization = 'sum'
folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(
    sigma_smooth) + '_normalization_' + str(normalization)
save_dir_autocorr = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
autocorr_cells = np.load(os.path.join(save_dir_autocorr, folder, 'autocorr.npy'))
autocorr_cells = 2 * autocorr_cells  # *2 to normalize it to the positive half (instead of the whole) autocorrelation
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

# ISI return maps
sigma_smooth = 5  # ms
dt_kde = 1  # ms
folder1 = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
folder2 = 'sigma_smooth_' + str(sigma_smooth) + '_dt_kde_' + str(dt_kde)
save_dir_ISI_return_map = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
ISI_return_map_kde_cells = np.load(os.path.join(save_dir_ISI_return_map, folder1, folder2, 'ISI_return_map_kde.npy'))
t_kde = np.arange(0, max_ISI + dt_kde, dt_kde)
X_kde, Y_kde = np.meshgrid(t_kde, t_kde)
for i, ISI_return_map in enumerate(ISI_return_map_kde_cells):  # norm again because of discretization
    ISI_return_map_kde_cells[i] = ISI_return_map[i] / (np.sum(ISI_return_map[i]) * dt_kde ** 2)

# plot
fig, axes = pl.subplots(4, 3, figsize=(10, 10))

# spiketime autocorrelation
for i, group in enumerate(groups):
    axes[0, i].bar(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), bin_width,
                   #yerr=[np.zeros(np.shape(autocorr_cells)[1]), np.std(autocorr_cells[label_burstgroups[group]], 0)],
                   color=colors_burstgroups[group],
                   align='center')
    axes[0, i].bar(t_autocorr, np.std(autocorr_cells[label_burstgroups[group]], 0), bin_width,
                   bottom=np.mean(autocorr_cells[label_burstgroups[group]], 0),
                   color=colors_burstgroups[group],
                   align='center', alpha=0.3)
    axes[0, i].plot(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), color='k', drawstyle='steps-mid',
                    linewidth=0.5)
    axes[0, i].set_xlabel('Lag (ms)')
    axes[0, i].set_ylabel('Autocorrelation (norm.)')

    # axes[0, 3].bar(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), bin_width,
    #                color=colors_burstgroups[group],
    #                align='center', alpha=0.5)
    # axes[0, 3].set_xlabel('Lag (ms)')
    # axes[0, 3].set_ylabel('Autocorrelation (norm.)')

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
    axes[1, i].set_xlabel('Time (ms)')
    axes[1, i].set_ylabel('Mem. pot. (mV)')

    # axes[1, 3].plot(t_sta, np.mean(sta_mean_cells[label_burstgroups[group]], 0), color=colors_burstgroups[group])
    # # axes[1, 3].fill_between(t_sta,
    # #                         np.mean(sta_mean_cells[label_burstgroups[group]], 0) -
    # #                         np.std(sta_mean_cells[label_burstgroups[group]], 0),
    # #                         np.mean(sta_mean_cells[label_burstgroups[group]], 0) +
    # #                         np.std(sta_mean_cells[label_burstgroups[group]], 0),
    # #                         color=colors_burstgroups[group], alpha=0.5)
    # axes[1, 3].set_xlim(-5, None)
    # axes[1, 3].set_ylim(-75, -30)
    # axes[1, 3].set_xlabel('Time (ms)')
    # axes[1, 3].set_ylabel('Mem. pot. (mV)')

# ISI hist
for i, group in enumerate(groups):
    axes[2, i].bar(bins[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
                   bins[1] - bins[0],
                   color=colors_burstgroups[group], align='edge')
    axes[2, i].bar(bins[:-1], np.std(ISI_hist_cells[label_burstgroups[group]], 0), bins[1] - bins[0],
                   bottom=np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
                   color=colors_burstgroups[group], align='edge', alpha=0.3)
    axes[2, i].plot(bins[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0), color='k', drawstyle='steps-pre',
                    linewidth=0.5)
    axes[2, i].set_xlabel('ISI (ms)')
    axes[2, i].set_ylabel('Frequency (norm.)')

    # axes[2, 3].bar(bins[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
    #                bins[1] - bins[0],
    #                color=colors_burstgroups[group], align='edge', alpha=0.5)
    # axes[2, 3].set_xlabel('ISI (ms)')
    # axes[2, 3].set_ylabel('Frequency (norm.)')


# ISI return map
for i, group in enumerate(groups):
    pcol = axes[3, i].pcolor(X_kde, Y_kde, np.mean(ISI_return_map_kde_cells[label_burstgroups[group]], 0))
    fig.colorbar(pcol, ax=axes[3, i])
    axes[3, i].set_xlabel('ISI[n] (ms)')
    axes[3, i].set_ylabel('ISI[n+1] (ms)')
    axes[3, i].set_aspect('equal', adjustable='box-forced')

# ISI_return_map_mat = init_nan(np.shape(ISI_return_map_kde_cells[0]))
# for i, group in enumerate(groups):
#     ISI_return_map_mat[np.mean(ISI_return_map_kde_cells[label_burstgroups[group]], 0) > 0.00015] = i + 1
#
# pcol = axes[3, 3].pcolor(X_kde, Y_kde, ISI_return_map_mat, cmap='hsv')
# fig.colorbar(pcol, ax=axes[3, 3])


pl.tight_layout()
pl.subplots_adjust(top=0.97, bottom=0.07, right=0.98, left=0.07)
pl.savefig(os.path.join(save_dir_img, 'group_avg.png'))
pl.show()