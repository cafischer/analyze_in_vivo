


def get_spike_characteristics_dict(for_data=False):
    spike_characteristics_dict = {
        'AP_threshold': -10,  # mV
        'AP_interval': 2.5,  # ms
        'fAHP_interval': 4.0,
        'AP_width_before_onset': 2.0,  # ms
        'DAP_interval': 10.0,  # ms
        'order_fAHP_min': 1.0,  # ms (how many points to consider for the minimum)
        'order_DAP_max': 1.0,  # ms (how many points to consider for the minimum)
        'min_dist_to_DAP_max': 0.5,  # ms
        'k_splines': 3,
        's_splines': 0  # 0 means no interpolation, use for models
    }
    if for_data:
        spike_characteristics_dict['s_splines'] = None
    return spike_characteristics_dict