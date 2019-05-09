import os
import numpy as np
import matplotlib.pyplot as pl
from analyze_in_vivo.load.load_domnisoru import get_label_burstgroups, get_colors_burstgroups, get_cell_ids_burstgroups, load_cell_ids
from cell_fitting.util import init_nan
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
pl.style.use('paper')


save_dir_fig3 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig3'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/extra'
if not os.path.exists(save_dir_fig3):
    os.makedirs(save_dir_fig3)

# burst groups
groups = ['NB', 'B', 'B+D']
group_names = ['Non-bursty', 'Bursty with DAP', 'Bursty without DAP']
cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))
cell_ids_burstgroups = get_cell_ids_burstgroups()
label_burstgroups = get_label_burstgroups()
colors_burstgroups = get_colors_burstgroups()

# remove s104, s110  TODO
cell_ids_remove = ['s104_0007', 's110_0002']
for cell_id in cell_ids_remove:
    idx = np.where(cell_id == cell_ids)[0]
    label_burstgroups['B+D'][idx] = False

# load
bin_width = 1
max_lag = 50
autocorr_cells = np.load(os.path.join(save_dir_fig3, 'autocorr_cells.npy'))
t_autocorr = np.load(os.path.join(save_dir_fig3, 't_autocorr.npy'))
max_ISI = 200
ISI_hist_cells = np.load(os.path.join(save_dir_fig3, 'ISI_hist_cells.npy'))
bins_ISI_hist = np.load(os.path.join(save_dir_fig3, 'bins_ISI_hist.npy'))
ISI_return_map_kde_cells = np.load(os.path.join(save_dir_fig3, 'ISI_return_map_kde_cells.npy'))
X_kde = np.load(os.path.join(save_dir_fig3, 'X_kde.npy'))
Y_kde = np.load(os.path.join(save_dir_fig3, 'Y_kde.npy'))

# plot
fig, axes = pl.subplots(3, 3, figsize=(6.05, 5))

# A
for i, group in enumerate(groups):
    axes[0, i].bar(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), bin_width,
                   #yerr=[np.zeros(np.shape(autocorr_cells)[1]), np.std(autocorr_cells[label_burstgroups[group]], 0)],
                   color=colors_burstgroups[group],
                   align='center')
    axes[0, i].bar(t_autocorr, np.std(autocorr_cells[label_burstgroups[group]], 0), bin_width,
                   bottom=np.mean(autocorr_cells[label_burstgroups[group]], 0),
                   color=colors_burstgroups[group],
                   align='center', alpha=0.4)
    axes[0, i].plot(t_autocorr, np.mean(autocorr_cells[label_burstgroups[group]], 0), color='k', drawstyle='steps-mid',
                    linewidth=0.4)
    axes[0, i].set_xlabel('Lag (ms)')
    axes[0, i].set_xlim(-max_lag, max_lag)
    if i == 0:
        axes[0, i].yaxis.set_label_coords(-0.46, 0.5)
        axes[0, i].set_ylabel('Autocorr. (norm.)')
        axes[0, i].annotate('', xy=(4.0, 0.04), xytext=(27, 0.075),
                            arrowprops=dict(facecolor='black', shrink=0.1, width=0.5, headwidth=5, headlength=5))
        axes[0, i].text(-0.8, 1.0, 'A', transform=axes[0, i].transAxes, size=18)

# B
for i, group in enumerate(groups):
    axes[1, i].bar(bins_ISI_hist[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
                   bins_ISI_hist[1] - bins_ISI_hist[0],
                   color=colors_burstgroups[group], align='edge')
    axes[1, i].bar(bins_ISI_hist[:-1], np.std(ISI_hist_cells[label_burstgroups[group]], 0), bins_ISI_hist[1] - bins_ISI_hist[0],
                   bottom=np.mean(ISI_hist_cells[label_burstgroups[group]], 0),
                   color=colors_burstgroups[group], align='edge', alpha=0.4)
    axes[1, i].plot(bins_ISI_hist[:-1], np.mean(ISI_hist_cells[label_burstgroups[group]], 0), color='k', drawstyle='steps-post',
                    linewidth=0.4)
    axes[1, i].set_xlabel('ISI (ms)')
    axes[1, i].set_xlim(0, max_ISI)
    axes[1, i].set_ylim(0, np.max(np.mean(ISI_hist_cells[label_burstgroups[group]], 0)
                        + np.std(ISI_hist_cells[label_burstgroups[group]], 0))+0.005)
    if i == 0:
        axes[1, i].yaxis.set_label_coords(-0.46, 0.5)
        axes[1, i].set_ylabel('Freq. (norm.)')
        axes[1, i].annotate('', xy=(4., 0.008), xytext=(47, 0.0145),
                            arrowprops=dict(facecolor='black', shrink=0.1, width=0.5, headwidth=5, headlength=5))
        axes[1, i].text(-0.82, 1.0, 'B', transform=axes[1, i].transAxes, size=18)

# C
for i, group in enumerate(groups):
    Z = np.mean(ISI_return_map_kde_cells[label_burstgroups[group]], 0)
    pcol = axes[2, i].pcolor(X_kde, Y_kde, Z, norm=colors.LogNorm(vmin=0.00001, vmax=0.001))
    #pcol = axes[2, i].contour(X_kde, Y_kde, np.mean(ISI_return_map_kde_cells[label_burstgroups[group]], 0),
    #                          levels=np.logspace(-5, -2, 4))
    axes[2, i].set_xlabel('ISI[n] (ms)')
    axes[2, i].set_aspect('equal', adjustable='box-forced')
    axes[2, i].annotate(group_names[i], xy=(0.5, -0.7), xycoords='axes fraction', fontsize=12, ha='center')
    if i == 0:
        axes[2, i].yaxis.set_label_coords(-0.46, 0.5)
        axes[2, i].set_ylabel('ISI[n+1] (ms)')
        axes[2, i].text(-0.91, 1.0, 'C', transform=axes[2, i].transAxes, size=18)

    divider = make_axes_locatable(axes[2, i])
    cax = divider.append_axes("right", size="8%", pad=0.05)
    fig.colorbar(pcol, cax=cax)  # ax=axes[2, i]

pl.tight_layout()
pl.subplots_adjust(top=0.96, bottom=0.12, right=0.92, left=0.14, wspace=0.78, hspace=0.43)
pl.savefig(os.path.join(save_dir_fig3, 'fig3.png'))
pl.show()