import numpy as np
import os
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids
from analyze_in_vivo.analyze_domnisoru.pca import perform_PCA


save_dir_autocorr = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
save_dir_fig1 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig1'

save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

max_lag = 50  # ms
bin_width = 1  # ms
sigma_smooth = None
dt_kde = 0.05  # ms
n_components = 2
remove_cells = True
normalization = 'sum'
cells_to_remove = ['s104_0007', 's110_0002', 's81_0004', 's115_0030']
max_lag_idx = to_idx(max_lag, bin_width)

grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))

folder = 'max_lag_' + str(max_lag) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(
    sigma_smooth) + '_normalization_' + str(normalization)
save_dir_img = os.path.join(save_dir_autocorr, folder, 'PCA')
if not os.path.exists(save_dir_img):
    os.makedirs(save_dir_img)

# load
autocorr_cells = np.load(os.path.join(save_dir_autocorr, folder, 'autocorr.npy'))
if sigma_smooth is not None:
    t_autocorr = np.arange(-max_lag, max_lag + dt_kde, dt_kde)
else:
    t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)

# PCA
if remove_cells:
    idxs = range(len(grid_cells))
    for cell_id in cells_to_remove:
        idx_remove = np.where(grid_cells == cell_id)[0][0]
        idxs.remove(idx_remove)
    autocorr_cells_for_pca = autocorr_cells[np.array(idxs)]
else:
    autocorr_cells_for_pca = autocorr_cells

projected_, components, explained_var = perform_PCA(autocorr_cells_for_pca, n_components)
projected = np.dot(autocorr_cells - np.mean(autocorr_cells_for_pca, 0), components[:n_components, :].T)

# save autocorr and projected
if not os.path.exists(save_dir_fig1):
    os.makedirs(save_dir_fig1)
np.save(os.path.join(save_dir_fig1, 'autocorr.npy'), autocorr_cells)
np.save(os.path.join(save_dir_fig1, 'projected.npy'), projected)
np.save(os.path.join(save_dir_fig1, 'components.npy'), components)
np.save(os.path.join(save_dir_fig1, 'explained_var.npy'), explained_var)