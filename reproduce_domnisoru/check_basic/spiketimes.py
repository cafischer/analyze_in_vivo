from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.signal import butter, filtfilt
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.reproduce_domnisoru.check_basic.in_out_field import get_spike_train
pl.style.use('paper')


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
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/spiketimes'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'stellate_layer2'  #'pyramidal_layer2' #
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s73_0004': 2.5, 's90_0006': 6, 's82_0002': 6,
                              's117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}

    for cell_id in cell_ids:
        print cell_id
        save_dir_cell = os.path.join(save_dir_img, cell_type, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        position = data['Y_cm']
        velocity = data['vel_100ms']

        # filter between 500 and 10000 Hz
        v_filtered = butter_bandpass_filter(v, 500, 10000 - 5, 1./data['dt']*1000, order=5)
        #v_filtered = butter_highpass_filter(v, 500, 1. / data['dt'] * 1000, order=5)

        # compute spike train
        _, AP_max_idxs_filtered = get_spike_train(v_filtered, AP_thresholds_filtered[cell_id], data['dt'],
                                                  v_diff_onset_max=0)
        _, AP_max_idxs = get_spike_train(v, AP_thresholds[cell_id], data['dt'], v_diff_onset_max=5)

        # spike times Domnisoru
        AP_max_idxs_domnisoru = data['spiketimes']

        # check number of spikes
        similar_spikes = 0
        domnisoru_not_me_spikes = 0
        me_not_domnisoru_spikes = 0
        for spike_domnisoru in AP_max_idxs_domnisoru:
            if np.any(np.abs(spike_domnisoru - AP_max_idxs) < 2):
                similar_spikes += 1
            else:
                domnisoru_not_me_spikes += 1

        for spike_me in AP_max_idxs:
            if np.any(np.abs(spike_me - AP_max_idxs_domnisoru) < 2):
                pass
            else:
                me_not_domnisoru_spikes += 1

        print '# In Domnisoru and mine (within 2 time steps): ', similar_spikes
        print '# In Domnisorus not mine: ', domnisoru_not_me_spikes
        print '# In mine not Domnisoru: ', me_not_domnisoru_spikes

        # plots
        t /= 1000

        fig, axes = pl.subplots(2, 1, sharex='all')
        axes[0].plot(t, v, 'k')
        axes[0].plot(t[AP_max_idxs_domnisoru], v[AP_max_idxs_domnisoru], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[0].plot(t[AP_max_idxs], v[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[0].plot(t[AP_max_idxs_filtered], v[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].legend(fontsize=12)
        axes[1].plot(t, v_filtered, 'k')
        axes[1].plot(t[AP_max_idxs_domnisoru], v_filtered[AP_max_idxs_domnisoru], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[1].plot(t[AP_max_idxs], v_filtered[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[1].plot(t[AP_max_idxs_filtered], v_filtered[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].set_ylabel('Membrane \nPotential (mV)')
        axes[1].set_ylabel('Membrane \nPotential (mV)')
        axes[1].set_xlabel('Time (s)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'spiketimes.png'))

        fig, axes = pl.subplots(2, 1, sharex='all')
        axes[0].plot(t, v, 'k')
        axes[0].plot(t[AP_max_idxs_domnisoru], v[AP_max_idxs_domnisoru], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[0].plot(t[AP_max_idxs], v[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[0].plot(t[AP_max_idxs_filtered], v[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].legend(fontsize=12)
        axes[1].plot(t, v_filtered, 'k')
        axes[1].plot(t[AP_max_idxs_domnisoru], v_filtered[AP_max_idxs_domnisoru], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[1].plot(t[AP_max_idxs], v_filtered[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[1].plot(t[AP_max_idxs_filtered], v_filtered[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].set_ylabel('Membrane \nPotential (mV)')
        axes[1].set_ylabel('Membrane \nPotential (mV)')
        axes[1].set_xlabel('Time (s)')
        axes[0].set_xlim((t[AP_max_idxs[int(round(len(AP_max_idxs)/2.))]] - int(round(1./data['dt']))/1000.,
                          t[AP_max_idxs[int(round(len(AP_max_idxs)/2.))]] + int(round(1./data['dt']))/1000.))
        axes[1].set_xlim((t[AP_max_idxs[int(round(len(AP_max_idxs)/2.))]] - int(round(1./data['dt']))/1000.,
                          t[AP_max_idxs[int(round(len(AP_max_idxs)/2.))]] + int(round(1./data['dt']))/1000.))
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'spiketimes_zoom.png'))
        #pl.show()