from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype, get_track_len
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import threshold_by_velocity
from scipy.ndimage.filters import convolve
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/velocity'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']  # TODO ['Vm_ljpc', 'Y_cm', 'fY_cm', 'vel_100ms', 'spiketimes']
    threshold = 1  # cm/s

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    time_lost = np.zeros(len(cell_ids))
    spikes_lost = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = data['dt']
        position = data['Y_cm']
        velocity_domnisoru = data['vel_100ms']

        # spike train
        AP_max_idxs = data['spiketimes']
        spike_train = get_spike_train(AP_max_idxs, len(v))

        # # velocity from position
        # velocity = np.concatenate((np.array([0]), np.diff(position) / (np.diff(t)/1000.)))
        #
        # # put velocity at switch from end of track to the beginning to 0
        # run_start_idxs = np.where(np.diff(position) < -get_track_len(cell_id)/2.)[0] + 1  # +1 because diff shifts one to front
        # velocity[run_start_idxs] = 0
        #
        # # smoothed by a 100 ms uniform sliding window
        # window = np.ones(int(round(100. / data['dt'])))
        # window /= np.sum(window)
        # velocity_smoothed = convolve(velocity, window, mode='nearest')
        #
        # # threshold by velocity
        # [position_thresholded], _ = threshold_by_velocity([position], velocity)
        #
        # # check same length
        # print 'Length Domnisoru - me (s): ', (len(data['fY_cm']) - len(position_thresholded)) * data['dt'] / 1000
        #
        # pl.figure()
        # pl.plot(np.arange(len(position_thresholded)) * data['dt'], position_thresholded)
        # pl.plot(np.arange(len(data['fY_cm'])) * data['dt'], data['fY_cm'])
        # pl.show()

        # threshold by velocity
        [t_thresholded, spike_train_thresholded], vel = threshold_by_velocity([t, spike_train], velocity_domnisoru, threshold)
        time_lost[cell_idx] = (len(t) - len(t_thresholded)) / float(len(t)) * 100  # %
        spikes_lost[cell_idx] = (np.sum(spike_train) - np.sum(spike_train_thresholded)) / float(np.sum(spike_train)) * 100  # %

        # print time_lost[cell_idx]
        # print spikes_lost[cell_idx]
        # pl.figure()
        # pl.plot(t, velocity_domnisoru, 'k')
        # pl.plot(t[velocity_domnisoru < threshold], velocity_domnisoru[velocity_domnisoru < threshold], 'ro', markersize=2)
        # pl.figure()
        # pl.plot(t, spike_train, 'k')
        # pl.plot(t[velocity_domnisoru < threshold], spike_train[velocity_domnisoru < threshold], 'ro',
        #         markersize=2)
        # pl.figure()
        # pl.plot(np.arange(0, len(vel))*dt, vel, 'k')
        # pl.show()

    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)

                    axes[i1, i2].bar(0, time_lost[cell_idx], color='0.5')
                    axes[i1, i2].bar(1, spikes_lost[cell_idx], color='0.5')
                    axes[i1, i2].set_xlim(-1, 2)
                    axes[i1, i2].set_ylim(0, 100)
                    axes[i1, i2].set_xticks([0, 1])
                    axes[i1, i2].set_xticklabels(['Time \nlost', '#APs \nlost'], fontsize=12)
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Percentage')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'loss.png'))
        pl.show()