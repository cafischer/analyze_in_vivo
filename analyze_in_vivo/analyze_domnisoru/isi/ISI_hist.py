from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist, get_cumulative_ISI_hist, \
    plot_ISI_hist, plot_cumulative_ISI_hist, plot_cumulative_ISI_hist_all_cells
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_hist_on_ax
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper')


if __name__ == '__main__':
    # Note: not all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies
    #save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    param_list = ['Vm_ljpc', 'spiketimes']
    max_ISI = 200  # None if you want to take all ISIs
    max_ISI_plot = 200
    burst_ISI = 8  # ms
    bin_width = 1  # ms
    bins = np.arange(0, max_ISI_plot+bin_width, bin_width)

    folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISIs_cells = [0] * len(cell_ids)
    n_ISIs = [0] * len(cell_ids)
    ISI_hist_cells = np.zeros((len(cell_ids), len(bins) - 1))
    cum_ISI_hist_y = [0] * len(cell_ids)
    cum_ISI_hist_x = [0] * len(cell_ids)
    fraction_ISIs_filtered = np.zeros(len(cell_ids))
    fraction_burst_ISIs = np.zeros(len(cell_ids))
    shortest_ISI = np.zeros(len(cell_ids))
    CV_ISIs = np.zeros(len(cell_ids))
    fraction_ISIs_8_16 = np.zeros(len(cell_ids))
    fraction_ISIs_8_25 = np.zeros(len(cell_ids))

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
            fraction_ISIs_filtered[cell_idx] = np.sum(ISIs <= max_ISI) / float(len(ISIs))
            ISIs = ISIs[ISIs <= max_ISI]
        n_ISIs[cell_idx] = len(ISIs)
        ISIs_cells[cell_idx] = ISIs
        fraction_burst_ISIs[cell_idx] = np.sum(ISIs < burst_ISI) / float(len(ISIs))
        fraction_ISIs_8_16[cell_idx] = np.sum(np.logical_and(8 < ISIs, ISIs < 16)) / float(len(ISIs))
        fraction_ISIs_8_25[cell_idx] = np.sum(np.logical_and(8 < ISIs, ISIs < 25)) / float(len(ISIs))
        shortest_ISI[cell_idx] = np.mean(np.sort(ISIs)[:int(round(len(ISIs)*0.1))])
        CV_ISIs[cell_idx] = np.std(ISIs) / np.mean(ISIs)

        # ISI histograms
        ISI_hist_cells[cell_idx, :] = get_ISI_hist(ISIs, bins, norm='sum')
        cum_ISI_hist_y[cell_idx], cum_ISI_hist_x[cell_idx] = get_cumulative_ISI_hist(ISIs, max_ISI)

        # save and plot
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        #
        # plot_cumulative_ISI_hist(cum_ISI_hist_x[i], cum_ISI_hist_y[i], xlim=(0, 200), title=cell_id,
        #                          save_dir=save_dir_cell)
        # print peak_ISI_hist[cell_idx]
        # plot_ISI_hist(ISI_hist[cell_idx, :], bins, title=cell_id, save_dir=save_dir_cell)
        # pl.show()
        # pl.close('all')

    # save
    np.save(os.path.join(save_dir_img, 'ISI_hist.npy'), ISI_hist_cells)
    np.save(os.path.join(save_dir_img, 'cum_ISI_hist_y.npy'), cum_ISI_hist_y)
    np.save(os.path.join(save_dir_img, 'cum_ISI_hist_x.npy'), cum_ISI_hist_x)
    np.save(os.path.join(save_dir_img, 'ISIs.npy'), ISIs_cells)
    np.save(os.path.join(save_dir_img, 'shortest_ISI.npy'), shortest_ISI)
    np.save(os.path.join(save_dir_img, 'CV_ISIs.npy'), CV_ISIs)
    np.save(os.path.join(save_dir_img, 'fraction_burst_ISIs.npy'), fraction_burst_ISIs)
    np.save(os.path.join(save_dir_img, 'fraction_ISIs_8_16.npy'), fraction_ISIs_8_16)
    np.save(os.path.join(save_dir_img, 'fraction_ISIs_8_25.npy'), fraction_ISIs_8_25)

    # plot all cumulative ISI histograms in one
    # ISIs_all = np.array([item for sublist in ISIs_per_cell for item in sublist])
    # cum_ISI_hist_y_avg, cum_ISI_hist_x_avg = get_cumulative_ISI_hist(ISIs_all)
    # plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
    #                                                cell_ids, max_ISI, os.path.join(save_dir_img))

    # cumulative ISI histogram for bursty and non-bursty group
    #ISIs_all_bursty =  np.array([item for sublist in np.array(ISIs_per_cell)[burst_label] for item in sublist])
    #ISIs_all_nonbursty = np.array([item for sublist in np.array(ISIs_per_cell)[~burst_label] for item in sublist])
    #cum_ISI_hist_y_avg_bursty, cum_ISI_hist_x_avg_bursty = get_cumulative_ISI_hist(ISIs_all_bursty)
    #cum_ISI_hist_y_avg_nonbursty, cum_ISI_hist_x_avg_nonbursty = get_cumulative_ISI_hist(ISIs_all_nonbursty)
    #plot_cumulative_ISI_hist_all_cells_with_bursty(cum_ISI_hist_y, cum_ISI_hist_x,
    #                                               cum_ISI_hist_y_avg_bursty, cum_ISI_hist_x_avg_bursty,
    #                                               cum_ISI_hist_y_avg_nonbursty, cum_ISI_hist_x_avg_nonbursty,
    #                                               cell_ids, burst_label, max_ISI, os.path.join(save_dir_img2))

    # plot all ISI hists
    params = {'legend.fontsize': 9}
    pl.rcParams.update(params)

    plot_kwargs = dict(ISI_hist=ISI_hist_cells, cum_ISI_hist_x=cum_ISI_hist_x, cum_ISI_hist_y=cum_ISI_hist_y,
                       max_ISI=max_ISI_plot, bin_width=bin_width)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
                            wspace=0.18, xlabel='ISI (ms)', ylabel='Rel. frequency',
                            save_dir_img=os.path.join(save_dir_img, 'ISI_hist.png'))
    # plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_on_ax, plot_kwargs,
    #                         xlabel='ISI (ms)', ylabel='Rel. frequency', colors_marker=colors_marker,
    #                         wspace=0.18, save_dir_img=os.path.join(save_dir_img2, 'ISI_hist.png'))