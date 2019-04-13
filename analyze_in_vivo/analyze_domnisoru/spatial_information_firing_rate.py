from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, \
    get_last_bin_edge, get_track_len
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers, plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train, get_spatial_firing_rate_per_run
from analyze_in_vivo.analyze_domnisoru.spatial_information import get_MI
from sklearn import linear_model
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)

    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'Y_cm']

    # parameters
    bin_size = 5.0  # cm
    bins_rate = np.arange(0, 100, 5)  # mV
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type, 'bin_size_'+str(bin_size))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    MI_rate = np.zeros(len(cell_ids))
    position_cells = []
    firing_rate_cells = []
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        position = data['Y_cm']
        dt = t[1] - t[0]
        bins_pos = np.arange(0, get_last_bin_edge(cell_id) + bin_size, bin_size)
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']

        # firing rate for entire recording
        spike_train = get_spike_train(AP_max_idxs, len(v))
        firing_rate_per_run = get_spatial_firing_rate_per_run(spike_train, position, bins_pos, dt,
                                                              get_track_len(cell_id))
        n_runs = len(firing_rate_per_run)

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # Mutual information: firing rate and position
        MI_rate[cell_idx] = get_MI(firing_rate_per_run.flatten(), np.tile(bins_pos[:-1], n_runs),
                                   bins_rate, bins_pos, len(bins_rate) - 1, len(bins_pos) - 1,
                                   save_dir_cell, 'MI_rate.png')


    # plots
    np.save(os.path.join(save_dir_img, 'MI_rate.npy'), MI_rate)
    spatial_info_skaggs = np.load(os.path.join(save_dir_img, 'spatial_info.npy'))

    pl.close('all')
    if cell_type == 'grid_cells':
        def plot_spatial_info(ax, cell_idx, spatial_info):
            ax.bar(1, (spatial_info[cell_idx]), width=0.8, color='0.5')

            ax.set_xlim(0, 2)
            ax.set_ylim(0, np.max(spatial_info))
            ax.set_xticks([])

        plot_kwargs = dict(spatial_info=MI_rate)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_spatial_info, plot_kwargs,
                                xlabel='', ylabel='MI (spat. firing rate)',
                                save_dir_img=os.path.join(save_dir_img, 'MI_rate.png'))

        # MI(spat. firing rate) vs Skaggs
        regression = linear_model.LinearRegression()
        regression.fit(np.array([MI_rate]).T, np.array([spatial_info_skaggs]).T)

        theta_cells = load_cell_ids(save_dir, 'giant_theta')
        DAP_cells = ['s79_0003', 's104_0007', 's109_0002', 's110_0002', 's119_0004']
        fig, ax = pl.subplots()
        plot_with_markers(ax, MI_rate, spatial_info_skaggs, cell_ids, cell_type_dict, theta_cells, DAP_cells)
        pl.plot(MI_rate, regression.coef_[0, 0] * MI_rate + regression.intercept_[0], 'r')
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('MI (spat. firing rate)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'mi_rate_vs_skaggs.png'))
        pl.show()