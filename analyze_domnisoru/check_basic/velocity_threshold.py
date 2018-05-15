from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import threshold_by_velocity
from scipy.ndimage.filters import convolve

pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/velocity'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_names = load_cell_ids(save_dir, 'stellate_layer2')
    param_list = ['Vm_ljpc', 'Y_cm', 'fY_cm', 'vel_100ms']
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

        # threshold by velocity
        [v, t, position], velocity = threshold_by_velocity([v, t, position], velocity)

        # check same length
        print 'Length Domisoru - me (s): ', (len(data['fY_cm']) - len(position)) * data['dt'] / 1000

        pl.figure()
        pl.plot(np.arange(len(position)) * data['dt'], position)
        pl.plot(np.arange(len(data['fY_cm'])) * data['dt'], data['fY_cm'])
        #pl.show()