from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import pandas as pd
import json
import random
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_names, load_data, load_field_indices
import scipy.signal
from scipy.ndimage.filters import convolve
from cell_fitting.util import init_nan


def get_spike_idxs(v, AP_threshold, dt):
    """
    At max of AP
    :return:
    """
    AP_onsets = get_AP_onset_idxs(v, AP_threshold)
    AP_onsets = np.concatenate((AP_onsets, np.array([len(v)])))
    AP_max_idxs_ = np.array([get_AP_max_idx(v, AP_onsets[i], AP_onsets[i + 1], interval=int(round(2. / dt))) for i in
                             range(len(AP_onsets) - 1)])
    idxs_not_none = ~np.array([x is None for x in AP_max_idxs_], dtype=bool)
    AP_max_idxs = np.array(AP_max_idxs_[idxs_not_none], dtype=int)

    # # for testing:
    # t = np.arange(len(v)) * dt
    # pl.figure()
    # pl.plot(t, v, 'k')
    # pl.plot(t[AP_onsets[:-1]], v[AP_onsets[:-1]], 'or')
    # pl.plot(t[AP_onsets[:-1][~idxs_not_none]], v[AP_onsets[:-1][~idxs_not_none]], 'ob')
    # pl.show()
    return AP_max_idxs


def get_spike_train(v, AP_threshold, dt):
    AP_max_idxs = get_spike_idxs(v, AP_threshold, dt)

    spike_train = np.zeros(len(v))
    spike_train[AP_max_idxs] = 1
    return spike_train, AP_max_idxs


def shuffle_spike_train(spike_train, random_generator):
    """
    Generation of the shuffled spike train: Cut original spike train at a randomly chosen index in the interval
    [0.05 * len(spike_train), 0.95 * len(spiketrain] and concatenate the 2nd part of the spike train to the 1st part
    of the spike train.
    :param spike_train: Array with the same length as the time array. In each time bin there is a 1 when in this time
    bin an AP (its max) occured, 0 else.
    :param random_generator: Random number generator (random.Random()) for generating random integers.
    :return: shuffled_spike_trains: Shuffled spike train.
    """
    cut_idx = random_generator.randint(int(np.ceil(0.05 * len(spike_train))), int(np.floor(0.95 * len(spike_train))))
    shuffled_spike_train = np.concatenate((spike_train[cut_idx:], spike_train[:cut_idx]))
    return shuffled_spike_train


def get_firing_rate_per_bin(spike_train, t, position, bins, dt):
    """
    Computes the firing rate in the given bins.
    :param spike_train: Spike train.
    :param t: Time.
    :param position: Position of the animal on the track.
    :param bins: Bins for the position of the animal.
    :return: firing_rate: mean firing rate. firing_rate_per_run: Firing rate for each run through the track.
    """
    # # for testing
    # i = int(round(len(APs)/16.))
    # APs = APs[:i]
    # t = t[:i]
    # position = position[:i]
    #
    # pl.figure()
    # pl.plot(t, position)
    # pl.show()

    run_start_idxs = np.where(np.diff(position) < -track_len/2.)[0] + 1  # +1 because diff shifts one to front
    APs_runs = np.split(spike_train, run_start_idxs)
    pos_runs = np.split(position, run_start_idxs)
    t_runs = np.split(t, run_start_idxs)
    n_bins = len(bins) - 1  # -1 for last edge

    # # for testing
    # pl.figure()
    # for i_run in range(len(t_runs)):
    #     pl.plot(t_runs[i_run], pos_runs[i_run])
    # pl.show()

    firing_rate_per_run = np.zeros((len(run_start_idxs) + 1, n_bins))
    firing_rate_per_run[:] = np.nan
    max_diff = lambda x: np.max(x) - np.min(x)
    for i_run, (APs_run, pos_run, t_run) in enumerate(zip(APs_runs, pos_runs, t_runs)):
        pos_binned = np.digitize(pos_run, bins) - 1
        AP_count_per_bin = pd.Series(APs_run).groupby(pos_binned).sum()
        seconds_per_bin = (pd.Series(t_run).groupby(pos_binned).size() - 1) * dt / 1000.
        pos_in_track = np.unique(pos_binned)[np.unique(pos_binned) <= n_bins-1]  # to ignore data higher than track_len
        firing_rate_per_run[i_run, pos_in_track] = AP_count_per_bin[pos_in_track] / seconds_per_bin[pos_in_track]

    firing_rate = np.nanmean(firing_rate_per_run, 0)

    # # for testing: print AP_max_idx[0] in np.where(pos_binned == AP_count_per_bin.index[AP_count_per_bin > 0][0])[0]
    # pl.figure()
    # for i_run in range(len(t_runs)):
    #     pl.plot(firing_rate_per_run[i_run, :])
    # pl.plot(firing_rate, 'k', linewidth=2.0)
    # pl.show()

    return firing_rate, firing_rate_per_run


def get_in_out_field(p_value, firing_rate_per_run, n_bins):
    out_field = np.zeros(n_bins)
    out_field[1 - p_value <= 0.05] = 1  # 1 - P value <= 0.05
    idx1 = np.where(out_field)[0]
    groups = np.split(idx1, np.where(np.diff(idx1) > 1)[0] + 1)
    for g in groups:
        if len(g) <= 1:  # more than 1 adjacent bins
            out_field[g] = 0
    in_field = np.zeros(n_bins)
    in_field[1 - p_value >= 0.85] = 1  # 1 - P value >= 0.85
    idx1 = np.where(in_field)[0]
    groups = np.split(idx1, np.where(np.diff(idx1) > 1)[0] + 1)
    for g in groups:
        if len(g) <= 2:  # more than 2 adjacent bins
            in_field[g] = 0
        else:
            n_runs = np.shape(firing_rate_per_run)[0]
            spiked_per_run = [np.sum(firing_rate_per_run[i, g]) > 0 for i in range(n_runs)]
            if np.sum(spiked_per_run) / n_runs < 0.2:  # spikes on at least 20 % of all runs
                in_field[g] = 0
                continue
            if g[0] - 1 > 0:  # extend by 1 bin left and right if: 1 - P value >= 0.70
                if 1 - p_value[g[0] - 1] >= 0.7:
                    in_field[g[0] - 1] = 1
            if g[-1] + 1 < n_bins:
                if 1 - p_value[g[-1] + 1] >= 0.7:
                    in_field[g[-1] + 1] = 1
    return in_field, out_field


def get_start_end_group_of_ones(x):
    start = np.where(np.diff(x) == 1)[0] + 1
    end = np.where(np.diff(x) == -1)[0]
    if x[0] == 1:
        start = np.concatenate((np.array([0]), start))
    if x[-1] == 1:
        end = np.concatenate((end, np.array([len(x) - 1])))
    return start, end


def get_v_and_t_per_run(v, t, position):
    run_start_idx = np.where(np.diff(position) < -track_len/2.)[0]
    v_runs = np.split(v, run_start_idx)
    t_runs = np.split(t, run_start_idx)
    return v_runs, t_runs


def threshold_by_velocity(v, t, position, velocity, threshold=1):
    """
    Remove regions where velocity < threshold from the data. Note: Data will contain discontinuities.
    :param v: (mV)
    :param t: (ms)
    :param position: (cm).
    :param velocity: (cm/sec).
    :param threshold: Threshold (cm/sec) below which to cut out the data.
    :return: v, t, position, velocity with regions removed where velocity < threshold.
    """
    to_low = velocity < threshold
    v = v[~to_low]
    t = t[~to_low]
    position = position[~to_low]
    velocity = velocity[~to_low]
    return v, t, position, velocity


def smooth_firing_rate(firing_rate, std=1):
    window = scipy.signal.gaussian(3, std)
    firing_rate_smoothed = convolve(firing_rate, window/window.sum(), mode='nearest')

    # # for testing:
    # pl.figure()
    # pl.plot(firing_rate, label='normal')
    # pl.plot(firing_rate_smoothed, label='smoothed')
    # pl.legend()
    # pl.show()
    return firing_rate_smoothed


def get_in_out_field_idxs_domnisoru(cell_name, save_dir, bins):
    run_start_idxs = np.where(np.diff(position) < -track_len / 2.)[0] + 1  # +1 because diff shifts one to front
    n_runs = len(run_start_idxs) + 1  # +1 for first start at 0
    n_bins = len(bins) - 1

    in_field, out_field = load_field_indices(cell_name, save_dir)
    position_thresholded = load_data(cell_name, ['fY_cm'], save_dir)['fY_cm']
    in_field_bool = np.zeros(len(position_thresholded))
    in_field_bool[in_field] = 1
    out_field_bool = np.zeros(len(position_thresholded))
    out_field_bool[out_field] = 1
    in_field_per_run = np.split(in_field_bool, run_start_idxs)
    out_field_per_run = np.split(out_field_bool, run_start_idxs)
    pos_per_run = np.split(position_thresholded, run_start_idxs)
    in_field_per_run_binned = init_nan((n_runs, n_bins))
    out_field_per_run_binned = init_nan((n_runs, n_bins))
    for run_i in range(n_runs):
        pos_binned = np.digitize(pos_per_run[run_i], bins) - 1
        in_field_grouped = pd.Series(in_field_per_run[run_i]).groupby(pos_binned).apply(lambda x: [x.values]).values
        check = [np.all(a == a[0]) for a in in_field_grouped]
        assert np.all(check)
        out_field_grouped = pd.Series(out_field_per_run[run_i]).groupby(pos_binned).apply(lambda x: [x.values]).values
        check = [np.all(a == a[0]) for a in out_field_grouped]
        assert np.all(check)
        pos_in_track = np.unique(pos_binned)[np.unique(pos_binned) <= n_bins - 1]  # to ignore data higher than track_len
        in_field_per_run_binned[run_i, pos_in_track] = pd.Series(in_field_per_run[run_i]).groupby(pos_binned).apply(lambda x:
                                                                    x.values[0]).values[np.unique(pos_binned) <= n_bins - 1]
        out_field_per_run_binned[run_i, pos_in_track] = pd.Series(out_field_per_run[run_i]).groupby(pos_binned).apply(lambda x:
                                                                    x.values[0]).values[np.unique(pos_binned) <= n_bins - 1]
    # in_field_grouped = pd.Series(in_field_bool).groupby(pos_binned).apply(lambda x: [x.values]).values
    # out_field_grouped = pd.Series(out_field_bool).groupby(pos_binned).apply(lambda x: [x.values]).values
    # TODO: did Domnisoru determine fields per run? or for average over runs? if second than grouped values should always be the same
    return in_field_per_run_binned, out_field_per_run_binned


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_names = load_cell_names(save_dir, 'stellate_layer2')
    cell_name = cell_names[0]
    param_list = ['Vm_ljpc', 'Y_cm', 'vel_100ms', 'spiketimes']
    AP_threshold = -60  # -45  # save dict for each cell

    # parameters
    seed = 1
    n_shuffles = 1000  # TODO 1000
    bin_size = 5  # cm
    params = {'seed': seed, 'n_shuffles': n_shuffles, 'bin_size': bin_size}
    track_len = 400  # cm

    random_generator = random.Random()
    random_generator.seed(seed)

    # load
    data = load_data(cell_name, param_list, save_dir)
    v = data['Vm_ljpc']
    t = np.arange(0, len(v)) * data['dt']
    position = data['Y_cm']
    velocity = data['vel_100ms']
    dt = t[1] - t[0]

    # velocity threshold the data
    v, t, position, velocity = threshold_by_velocity(v, t, position, velocity)

    # compute spike train
    spike_train, AP_max_idxs = get_spike_train(v, AP_threshold, dt)

    # bin according to position and compute firing rate
    bins = np.arange(0, track_len + bin_size, bin_size)
    n_bins = len(bins) - 1  # -1 for last edge

    # compute firing rate of original spike train
    firing_rate_real_tmp, firing_rate_per_run = get_firing_rate_per_bin(spike_train, t, position, bins, dt)
    firing_rate_real = smooth_firing_rate(firing_rate_real_tmp)

    mean_firing_rate_shuffled = np.zeros(n_bins)
    p_value = np.zeros(n_bins)
    p_value_per_run = np.zeros((len(firing_rate_per_run), n_bins))
    for i in range(n_shuffles):
        # shuffle
        shuffled_spiketrain = shuffle_spike_train(spike_train, random_generator)

        # compute firing rate
        firing_rate_shuffled_tmp, _ = get_firing_rate_per_bin(shuffled_spiketrain, t, position, bins, dt)
        firing_rate_shuffled = smooth_firing_rate(firing_rate_shuffled_tmp)
        mean_firing_rate_shuffled += firing_rate_shuffled

        # compute P-value: percent of shuffled firing rates that were higher than the cells real firing rate
        p_value += (firing_rate_shuffled > firing_rate_real).astype(int)

    mean_firing_rate_shuffled /= n_shuffles
    p_value /= n_shuffles

    # get in-field and out-field
    in_field, out_field = get_in_out_field(p_value, firing_rate_per_run, n_bins)

    # for comparison Domnisorus fields
    in_field_domnisoru, out_field_domnisoru = get_in_out_field_idxs_domnisoru(cell_name, save_dir, bins)
    pl.figure()
    pl.title('Domnisoru')
    con = np.vstack((in_field_domnisoru, np.array([in_field])))
    pl.imshow(con)
    pl.figure()
    pl.title('Domnisoru')
    con = np.vstack((out_field_domnisoru, np.array([out_field])))
    pl.imshow(con)
    pl.show()

    # save and plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'in_field.npy'), in_field)
    np.save(os.path.join(save_dir_img, 'out_field.npy'), out_field)
    np.save(os.path.join(save_dir_img, 'firing_rate_real.npy'), firing_rate_real)

    with open(os.path.join(save_dir_img, 'params.json'), 'w') as f:
        json.dump(params, f)

    start_in, end_in = get_start_end_group_of_ones(in_field)
    start_out, end_out = get_start_end_group_of_ones(out_field)


    fig, axes = pl.subplots(2, 1, sharex='all')
    axes[0].plot(position, t / 1000., '0.5')
    axes[0].plot(position[AP_max_idxs], (t / 1000.)[AP_max_idxs], 'or', markersize=1.0)
    axes[0].set_ylabel('Time (s)')
    axes[1].plot(bins[:-1], firing_rate_real, 'k')
    axes[1].set_ylabel('Firing rate (Hz)')
    axes[1].set_xlabel('Position (cm)')
    for i, (s, e) in enumerate(zip(start_in, end_in)):
        axes[1].hlines(-0.01, bins[s], bins[e], 'r', label='In field' if i == 0 else None, linewidth=3)
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        axes[1].hlines(-0.01, bins[s], bins[e], 'b', label='Out field' if i == 0 else None, linewidth=3)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'position_vs_firing_rate.png'))
    pl.show()

    pl.figure()
    pl.plot(firing_rate_real, 'k', label='Real')
    pl.plot(mean_firing_rate_shuffled, 'r', label='Shuffled mean')
    pl.xticks(np.arange(0, n_bins+n_bins/4, n_bins/4), np.arange(0, track_len+track_len/4, track_len/4))
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir_img, 'firing_rate_binned.png'))
    pl.show()

    pl.figure()
    pl.plot(1 - p_value, 'g', label='1 - P value')
    for i, (s, e) in enumerate(zip(start_in, end_in)):
        pl.hlines(-0.01, s, e, 'r', label='In field' if i == 0 else None, linewidth=3)
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(-0.01, s, e, 'b', label='Out field' if i == 0 else None, linewidth=3)
    pl.xticks(np.arange(0, n_bins + n_bins / 4, n_bins / 4), np.arange(0, track_len + track_len / 4, track_len / 4))
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir_img, 'p_value.png'))
    pl.show()

    pl.figure()
    pl.plot(firing_rate_real, 'k', label='')
    for i, (s, e) in enumerate(zip(start_in, end_in)):
        pl.hlines(-1, s, e, 'r', label='In field' if i == 0 else None, linewidth=3)
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(-1, s, e, 'b', label='Out field' if i == 0 else None, linewidth=3)
    pl.xticks(np.arange(0, n_bins + n_bins / 4, n_bins / 4), np.arange(0, track_len + track_len / 4, track_len / 4))
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Firing rate (Hz)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir_img, 'firing_rate_and_fields.png'))
    pl.show()

    # v_per_run, t_per_run = get_v_and_t_per_run(v, t, position)
    # i_run = 0
    # start_out = start_out / (n_bins-1) * t_per_run[i_run][-1]
    # end_out = end_out / (n_bins-1) * t_per_run[i_run][-1]
    # start_in = start_in / (n_bins-1) * t_per_run[i_run][-1]
    # end_in = end_in / (n_bins-1) * t_per_run[i_run][-1]
    #
    # pl.figure()
    # pl.plot(t_per_run[i_run], v_per_run[i_run], 'k', label='')
    # for i, (s, e) in enumerate(zip(start_in, end_in)):
    #     pl.hlines(np.min(v_per_run[i_run])-1, s, e, 'r', label='In field' if i == 0 else None, linewidth=3)
    # for i, (s, e) in enumerate(zip(start_out, end_out)):
    #     pl.hlines(np.min(v_per_run[i_run])-1, s, e, 'b', label='Out field' if i == 0 else None, linewidth=3)
    # pl.xlabel('Time (ms)', fontsize=16)
    # pl.ylabel('Membrane potential (mV)', fontsize=16)
    # pl.legend(fontsize=16)
    # pl.savefig(os.path.join(save_dir_img, 'v_and_fields.png'))
    # pl.show()