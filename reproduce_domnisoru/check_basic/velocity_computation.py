from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from analyze_in_vivo.load.load_domnisoru import load_cell_names, load_data
from scipy.ndimage.filters import convolve


if __name__ == '__main__':
    # Note: domnisoru Bandpass filters before spike detection between 500-10000 Hz, this leads to slightly different
    # spike times

    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_names = load_cell_names(save_dir, 'stellate_layer2')
    cell_name = cell_names[0]
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_threshold = -60 #-45  # save dict for each cell
    track_len = 400

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
    velocity_smoothed = convolve(velocity, window, mode='reflect')

    pl.figure()
    pl.plot(t, data['vel_100ms'], 'k', label='domnisoru')
    pl.plot(t, velocity, '0.5', label='me raw')
    pl.plot(t, velocity_smoothed, 'lightblue', label='me smoothed')
    pl.ylabel('Velocity (cm/sec)')
    pl.xlabel('Time (ms)')
    pl.legend()
    pl.tight_layout()
    pl.show()