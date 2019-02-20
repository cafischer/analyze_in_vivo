from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_hist_on_ax
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # parameters
    ISI_burst = 15  # ms
    max_ISI = 15  # for plotting
    bin_width = 0.5
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    # main
    first_ISIs_burst = np.zeros(len(cell_ids), dtype=object)
    ISI_hist_cells = np.zeros((len(cell_ids), len(bins) - 1))
    cum_ISI_hist_y = np.zeros(len(cell_ids), dtype=object)
    cum_ISI_hist_x = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # find burst indices
        ISIs = get_ISIs(AP_max_idxs, t)
        short_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        first_ISIs_burst[cell_idx] = ISIs[np.where(np.diff(short_ISI_indicator.astype(int)) == 1)[0] + 1]

        # histogram
        ISI_hist_cells[cell_idx, :] = get_ISI_hist(first_ISIs_burst[cell_idx], bins)
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(first_ISIs_burst[cell_idx])


    # plot
    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if cell_id in cell_ids_bursty else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'

    plot_kwargs = dict(ISI_hist=ISI_hist_cells, cum_ISI_hist_x=cum_ISI_hist_x, cum_ISI_hist_y=cum_ISI_hist_y,
                           max_ISI=max_ISI, bin_width=bin_width)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
                            wspace=0.18, xlabel='ISI (ms)', ylabel='Rel. frequency', colors_marker=colors_marker,
                            save_dir_img=os.path.join(save_dir_img, 'first_ISI_in_burst_' + str(max_ISI) + '_' + str(
                                bin_width) + '_' + str(ISI_burst) + '.png'))
    pl.show()