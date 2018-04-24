from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.signal import butter, filtfilt
from analyze_in_vivo.load.load_domnisoru import load_cell_names, load_data
from analyze_in_vivo.reproduce_domnisoru.in_out_field import get_spike_train
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
    cell_names = load_cell_names(save_dir, 'stellate_layer2')
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_thresholds = {'s117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    AP_thresholds_filtered = {'s117_0002': 7, 's119_0004': 9, 's104_0007': 8, 's79_0003': 8, 's76_0002': 6.5, 's101_0009': 7}

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
        velocity = data['vel_100ms']

        # filter between 500 and 10000 Hz
        v_filtered = butter_bandpass_filter(v, 500, 10000 - 5, 1./data['dt']*1000, order=5)
        #v_filtered = butter_highpass_filter(v, 500, 1. / data['dt'] * 1000, order=5)

        # compute spike train
        _, AP_max_idxs_filtered = get_spike_train(v_filtered, AP_thresholds_filtered[cell_name], data['dt'])
        _, AP_max_idxs = get_spike_train(v, AP_thresholds[cell_name], data['dt'])

        # spike times Domnisoru
        spike_idxs = data['spiketimes']

        # check number of spikes
        similar_spikes = []
        domnisoru_not_me_spikes = []
        me_not_domnisoru_spikes = []
        for spike_domnisoru in spike_idxs:
            found_spike = False
            for spike in AP_max_idxs:
                if np.abs(spike_domnisoru - spike) < 2:
                    similar_spikes.append(spike_domnisoru)
                    found_spike = True
                    break
            if not found_spike:
                domnisoru_not_me_spikes.append(spike_domnisoru)

        for spike in AP_max_idxs:
            found_spike = False
            for spike_domnisoru in spike_idxs:
                if np.abs(spike_domnisoru - spike) < 2:
                    found_spike = True
                    break
            if not found_spike:
                me_not_domnisoru_spikes.append(spike_domnisoru)
        print '# In Domnisoru and mine (within 2 time steps): ', len(similar_spikes)
        print '# In Domnisorus not mine: ', len(domnisoru_not_me_spikes)
        print '# In mine not Domnisoru: ', len(domnisoru_not_me_spikes)

        # plots
        t /= 1000

        fig, axes = pl.subplots(2, 1, sharex='all')
        axes[0].plot(t, v, 'k')
        axes[0].plot(t[spike_idxs], v[spike_idxs], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[0].plot(t[AP_max_idxs], v[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[0].plot(t[AP_max_idxs_filtered], v[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].legend(fontsize=12)
        axes[1].plot(t, v_filtered, 'k')
        axes[1].plot(t[spike_idxs], v_filtered[spike_idxs], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[1].plot(t[AP_max_idxs], v_filtered[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[1].plot(t[AP_max_idxs_filtered], v_filtered[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].set_ylabel('Membrane \nPotential (mV)')
        axes[1].set_ylabel('Membrane \nPotential (mV)')
        axes[1].set_xlabel('Time (s)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'spiketimes.png'))

        fig, axes = pl.subplots(2, 1, sharex='all')
        axes[0].plot(t, v, 'k')
        axes[0].plot(t[spike_idxs], v[spike_idxs], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
        axes[0].plot(t[AP_max_idxs], v[AP_max_idxs], 'ob', markersize=4.0, alpha=0.5, label='unfiltered')
        axes[0].plot(t[AP_max_idxs_filtered], v[AP_max_idxs_filtered], 'oy', markersize=4.0, alpha=0.5, label='filtered')
        axes[0].legend(fontsize=12)
        axes[1].plot(t, v_filtered, 'k')
        axes[1].plot(t[spike_idxs], v_filtered[spike_idxs], 'or', markersize=4.0, alpha=0.5, label='domnisoru')
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