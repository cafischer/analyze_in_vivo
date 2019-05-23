import os
import numpy as np
import matplotlib.pyplot as pl
pl.style.use('paper')

save_dir_fig3 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig3'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

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

# ISI hist
max_ISI = 200
bin_width = 1
sigma_smooth = None
folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)
save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, folder, 'ISI_hist.npy'))
bins_ISI_hist = np.arange(0, max_ISI + bin_width, bin_width)

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
    ISI_return_map_kde_cells[i] = ISI_return_map / (np.sum(ISI_return_map) * dt_kde ** 2)

# save
if not os.path.exists(save_dir_fig3):
    os.makedirs(save_dir_fig3)

np.save(os.path.join(save_dir_fig3, 'autocorr_cells.npy'), autocorr_cells)
np.save(os.path.join(save_dir_fig3, 't_autocorr.npy'), t_autocorr)
np.save(os.path.join(save_dir_fig3, 'ISI_hist_cells.npy'), ISI_hist_cells)
np.save(os.path.join(save_dir_fig3, 'bins_ISI_hist.npy'), bins_ISI_hist)
np.save(os.path.join(save_dir_fig3, 'ISI_return_map_kde_cells.npy'), ISI_return_map_kde_cells)
np.save(os.path.join(save_dir_fig3, 'X_kde.npy'), X_kde)
np.save(os.path.join(save_dir_fig3, 'Y_kde.npy'), Y_kde)

