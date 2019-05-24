from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.isi import get_ISI_hist_peak_and_width
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    param_list = ['Vm_ljpc', 'spiketimes']
    burst_ISI = 8  # ms
    sigma_smooth = 1  # ms  has to be a number
    max_ISI = None
    dt_kde = 0.05  # ms
    max_ISI_plot = 200  # ms

    folder = 'sigma_smooth_' + str(sigma_smooth)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    peak_ISI_hist = np.zeros(len(cell_ids), dtype=object)
    width_ISI_hist = np.zeros(len(cell_ids))
    kde_ISI_dist_cells = np.zeros((len(cell_ids), int(max_ISI_plot / dt_kde) + 1))
    t_kde = np.arange(0, max_ISI_plot + dt_kde, dt_kde)

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

        # compute KDE
        if sigma_smooth is not None:
            kde = perform_kde(ISIs, sigma_smooth)
            kde_ISI_dist_cells[cell_idx] = evaluate_kde(t_kde, kde)

            peak_ISI_hist[cell_idx], width_ISI_hist[cell_idx] = get_ISI_hist_peak_and_width(kde_ISI_dist_cells[cell_idx], t_kde)

        # save and plot
        # print peak_ISI_hist[cell_idx]
        # print width_ISI_hist[cell_idx]
        # pl.figure()
        # pl.plot(t_kde, kde_ISI_dist_cells[cell_idx], 'k')
        # pl.show()
        # pl.close('all')

    # save
    if sigma_smooth is not None:
        np.save(os.path.join(save_dir_img, 'peak_ISI_hist.npy'), peak_ISI_hist)
        np.save(os.path.join(save_dir_img, 'width_ISI_hist.npy'), width_ISI_hist)
        np.save(os.path.join(save_dir_img, 'kde_ISI_dist.npy'), kde_ISI_dist_cells)
        np.save(os.path.join(save_dir_img, 't_kde.npy'), t_kde)