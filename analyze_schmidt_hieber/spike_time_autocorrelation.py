from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_schmidt_hieber import load_full_runs, get_cell_type_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli import get_AP_max_idxs
from cell_characteristics import to_idx
import warnings
from analyze_in_vivo.analyze_domnisoru.spike_time_autocorrelation import get_autocorrelation
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/schmidthieber/whole_trace/spike_time_auto_corr'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/schmidt-hieber'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    cell_type_dict = get_cell_type_dict()
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    bin_size = 1.0  # ms
    max_lag = 50
    use_AP_max_idxs_domnisoru = True
    max_lag_idx = to_idx(max_lag, bin_size)
    t_auto_corr = np.concatenate((np.arange(-max_lag_idx, 0, 1), np.arange(0, max_lag_idx + 1, 1))) * bin_size

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    auto_corr_cells = []
    peak_auto_corr = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(save_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5

        # get APs
        AP_max_idxs = get_AP_max_idxs(v, AP_threshold, dt)

        # get spike-trains
        spike_train = get_spike_train(AP_max_idxs, len(v))

        # change to bin size
        bin_change = to_idx(bin_size, dt)
        spike_train_new = np.zeros(int(round(len(spike_train) / bin_change)))
        for i in range(len(spike_train_new)):
            if sum(spike_train[i * int(bin_change):(i + 1) * int(bin_change)] == 1) == 1:
                spike_train_new[i] = 1
            elif sum(spike_train[i * int(bin_change):(i + 1) * int(bin_change)] == 1) == 0:
                spike_train_new[i] = 0
            else:
                warnings.warn('More than one spike in bin!')

        # pl.figure()
        # pl.plot(t[spike_train==1], spike_train[spike_train==1], 'ok')
        # pl.plot((np.arange(len(spike_train_new)) * bin_size)[spike_train_new==1], spike_train_new[spike_train_new==1], 'ob')
        # pl.show()

        # spike-time autocorrelation
        auto_corr = get_autocorrelation(spike_train_new, max_lag_idx)
        auto_corr[max_lag_idx] = 0  # for better plotting
        auto_corr /= (np.sum(auto_corr) * bin_size)  # normalize
        auto_corr_cells.append(auto_corr)
        peak_auto_corr[cell_idx] = t_auto_corr[max_lag_idx:][np.argmax(auto_corr[max_lag_idx:])]  # start at pos. half
                                                                                                  # as 1st max is taken

        # # plot
        # save_dir_cell = os.path.join(save_dir_img, cell_id)
        # if not os.path.exists(save_dir_cell):
        #     os.makedirs(save_dir_cell)
        # print peak_auto_corr[cell_idx]
        # pl.close('all')
        # pl.figure()
        # pl.bar(t_auto_corr, auto_corr, bin_size, color='0.5', align='center')
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Spike-time autocorrelation')
        # pl.xlim(-max_lag, max_lag)
        # pl.tight_layout()
        # #pl.savefig(os.path.join(save_dir_cell, 'auto_corr_'+str(max_lag)+'.png'))
        # pl.show()

    np.save(os.path.join(save_dir_img, 'peak_auto_corr_'+str(max_lag)+'.npy'), peak_auto_corr)
    def plot_auto_corr(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag):
        ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
        ax.set_xlim(-max_lag, max_lag)
    plot_kwargs = dict(t_auto_corr=t_auto_corr, auto_corr_cells=auto_corr_cells, bin_size=bin_size, max_lag=max_lag)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_auto_corr, plot_kwargs,
                            xlabel='Time (ms)', ylabel='Spike-time \nautocorrelation', sharey='none',
                            save_dir_img=os.path.join(save_dir_img, 'auto_corr_'+str(max_lag)+'.png'))
    pl.show()