from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import firwin, freqz, kaiserord
from grid_cell_stimuli import get_nyquist_rate
from analyze_in_vivo.load.load_domnisoru import load_cell_names, load_data
from analyze_in_vivo.reproduce_domnisoru.in_out_field import get_spike_train


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)  # forward and backward, therefore no lag
    return y


def butter_lowpass_filter(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)  # forward and backward, therefore no lag
    return y


def butter_highpass_filter(data, lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)  # forward and backward, therefore no lag
    return y


if __name__ == '__main__':
    # Note: domnisoru Bandpass filters before spike detection between 500-10000 Hz, this leads to slightly different
    # spike times

    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_names = load_cell_names(save_dir, 'stellate_layer2')
    cell_name = cell_names[0]
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_threshold = -60 #-45  # save dict for each cell

    # load
    data = load_data(cell_name, param_list, save_dir)
    v = data['Vm_ljpc']
    t = np.arange(0, len(v)) * data['dt']
    position = data['Y_cm']
    velocity = data['vel_100ms']

    # filter between 500 and 10000 Hz
    v_filtered = butter_bandpass_filter(v, 500, 10000 - 5, 1./data['dt']*1000, order=5)  # 10000
    #v_filtered = butter_highpass_filter(v, 500, 1. / data['dt'] * 1000, order=5)

    # compute spike train
    _, AP_max_idxs_filtered = get_spike_train(v_filtered, 7, data['dt'])
    _, AP_max_idxs = get_spike_train(v, AP_threshold, data['dt'])


    # check spike times
    spike_idxs = data['spiketimes']

    pl.figure()
    pl.plot(t, v, '0.7', label='v raw')
    pl.plot(t, v_filtered[:len(t)], '0.3', label='v filtered')
    pl.plot(t[spike_idxs], v[spike_idxs], 'or', label='domnisoru')
    pl.plot(t[AP_max_idxs_filtered], v[AP_max_idxs_filtered], 'ob', label='me filtered')
    pl.plot(t[AP_max_idxs], v[AP_max_idxs], 'og', label='me unfiltered')
    pl.legend()
    pl.tight_layout()

    pl.figure()
    pl.plot(t, v_filtered[:len(t)], '0.3', label='v filtered')
    pl.legend()
    pl.tight_layout()
    pl.show()