from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from scipy.ndimage.filters import convolve
from sklearn.metrics import mean_squared_error
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/velocity'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_names = load_cell_ids(save_dir, 'stellate_layer2')
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms']
    track_len = 400

    for cell_name in cell_names:
        print cell_name

        save_dir_cell = os.path.join(save_dir_img, cell_name)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_name, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        position = data['Y_cm']

        # velocity from position
        velocity = np.concatenate((np.array([0]), np.diff(position) / (np.diff(t)/1000.)))

        # put velocity at switch from end of track to the beginning to 0
        run_start_idxs = np.where(np.diff(position) < -track_len/2.)[0] + 1  # +1 because diff shifts one to front
        velocity[run_start_idxs] = 0

        # smoothed by a 100 ms uniform sliding window
        window = np.ones(int(round(100. / data['dt'])))
        window /= np.sum(window)
        velocity_smoothed = convolve(velocity, window, mode='nearest')

        # plot
        print 'RMS: %.4f cm/sec' % np.sqrt(mean_squared_error(data['vel_100ms'], velocity_smoothed))

        t /= 1000
        pl.figure()
        pl.plot(t, data['vel_100ms'], 'r', label='domnisoru')
        pl.plot(t, velocity, 'k', label='raw')
        pl.plot(t, velocity_smoothed, 'b', label='smoothed')
        pl.ylabel('Velocity (cm/sec)')
        pl.xlabel('Time (s)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'velocity.png'))

        pl.figure()
        pl.plot(t, data['vel_100ms'], 'r', label='domnisoru')
        pl.plot(t, velocity, 'k', label='raw')
        pl.plot(t, velocity_smoothed, 'b', label='smoothed')
        pl.ylabel('Velocity (cm/sec)')
        pl.xlabel('Time (s)')
        idx = np.where(velocity > 20)[0][0]
        idx2 = int(idx + 1./data['dt']*1000)
        pl.xlim(t[idx], t[idx2])
        pl.ylim(np.min(velocity[idx:idx2]), np.max(velocity[idx:idx2]))
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'velocity_zoom.png'))
        #pl.show()