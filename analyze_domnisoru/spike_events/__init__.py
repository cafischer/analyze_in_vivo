import numpy as np
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
import time



def get_burst_lengths_and_n_single(burst_ISI_indicator):
    starts_burst, ends_burst = get_starts_ends_burst(burst_ISI_indicator)
    return get_burst_lengths(starts_burst, ends_burst), get_n_single(burst_ISI_indicator, ends_burst)


def get_starts_ends_burst(burst_ISI_indicator):
    starts_burst, ends_ones = get_starts_ends_group_of_ones(burst_ISI_indicator.astype(int))
    ends_burst = ends_ones + 1  # to change from idx of ISIs to index of AP_max_idxs
    return starts_burst, ends_burst


def get_burst_lengths(starts_burst, ends_burst):
    burst_lengths = ends_burst - starts_burst + 1
    return burst_lengths


def get_n_single(burst_ISI_indicator, ends_burst):
    return len(get_idxs_single(burst_ISI_indicator, ends_burst))


def get_idxs_single(burst_ISI_indicator, ends_burst):
    burst_ISI_indicator = np.concatenate((burst_ISI_indicator, np.array([False])))
    burst_ISI_indicator[ends_burst] = True
    return np.where(~burst_ISI_indicator.astype(bool))[0]



if __name__ == '__main__':
    burst_ISI_indicator = np.array([0, 1, 1])
    print get_burst_lengths_and_n_single(burst_ISI_indicator)  # [3], 1

    burst_ISI_indicator = np.array([0, 0, 1])  # [2], 2
    print get_burst_lengths_and_n_single(burst_ISI_indicator)

    burst_ISI_indicator = np.array([1, 1, 0])  # [3], 1
    print get_burst_lengths_and_n_single(burst_ISI_indicator)

    burst_ISI_indicator = np.array([1, 1, 0, 0])  # [3], 2
    print get_burst_lengths_and_n_single(burst_ISI_indicator)

    burst_ISI_indicator = np.array([1, 1, 0, 0, 1, 1, 1])  # [3, 4], 1
    print get_burst_lengths_and_n_single(burst_ISI_indicator)
