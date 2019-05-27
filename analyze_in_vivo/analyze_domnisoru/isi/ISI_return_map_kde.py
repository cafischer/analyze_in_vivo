from __future__ import division
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_return_map
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
pl.style.use('paper')


if __name__ == '__main__':
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_ids = load_cell_ids(save_dir, 'grid_cells')
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    max_ISI = None  # None if you want to take all ISIs
    ISI_burst = 8  # ms
    sigma_smooth = 5  # ms
    dt_kde = 1

    folder = 'max_ISI_' + str(max_ISI) + '_sigma_smooth_' + str(sigma_smooth)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs_cells = [0] * len(cell_ids)
    ISI_return_map_kde_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # ISIs
        ISIs = get_ISIs(AP_max_idxs, t)
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]
        ISIs_cells[cell_idx] = ISIs


        # compute KDE
        kde = perform_kde(np.vstack([ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:]]), sigma_smooth)
        t_kde = np.arange(0, max_ISI + dt_kde, dt_kde)
        X, Y = np.meshgrid(t_kde, t_kde)
        kde_mat = evaluate_kde(np.vstack([X.flatten(), Y.flatten()]), kde)
        kde_mat = kde_mat.reshape(len(t_kde), len(t_kde))
        ISI_return_map_kde_cells[cell_idx] = kde_mat

        # pl.figure()
        # pl.pcolor(X, Y, kde_mat)
        # pl.scatter(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='k', s=3, alpha=0.3)
        # pl.show()

    # save and plot
    np.save(os.path.join(save_dir_img, folder, 'ISI_return_map_kde.npy'), ISI_return_map_kde_cells)