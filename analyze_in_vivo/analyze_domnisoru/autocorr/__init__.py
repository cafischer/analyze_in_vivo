import numpy as np
from cell_characteristics import to_idx


def get_crosscorrelation(x, y, max_lag=0):
    assert len(x) == len(y)
    cross_corr = np.zeros(2 * max_lag + 1)
    for lag in range(max_lag, 0, -1):
        cross_corr[max_lag - lag] = np.correlate(x[:-lag], y[lag:], mode='valid')[0]
    for lag in range(1, max_lag + 1, 1):
        cross_corr[max_lag + lag] = np.correlate(x[lag:], y[:-lag], mode='valid')[0]
        cross_corr[max_lag] = np.correlate(x, y, mode='valid')[0]

    assert np.all(cross_corr[:max_lag] == cross_corr[max_lag + 1:][::-1])
    return cross_corr


def get_autocorrelation(x, max_lag=50):
    auto_corr_lag = np.zeros(max_lag)
    for lag in range(1, max_lag+1, 1):
        auto_corr_lag[lag-1] = np.correlate(x[:-lag], x[lag:], mode='valid')[0]
    auto_corr_no_lag = np.array([np.correlate(x, x, mode='valid')[0]])
    auto_corr = np.concatenate((np.flipud(auto_corr_lag), auto_corr_no_lag, auto_corr_lag))
    return auto_corr


def get_autocorrelation_by_ISIs(ISIs, max_lag=50, bin_width=1, remove_zero=True, normalization=None):
    """
    Computes the autocorrelation of some spike train by means of the ISIs.
    :param ISIs: All ISIs obtained from the spike train. They need to be kept in the same order!
    :type ISIs: array-like
    :param max_lag: Up to which time the auto-correlation should be computed.
    :type max_lag: float
    :param bin_width: Width of the bins for the auto-correlation.
    :type bin_width: float
    :return: Spike-time autocorrelation, center of the bins, bins.
    """
    ISIs_cum = np.cumsum(ISIs)
    SIs = get_all_SIs(ISIs_cum)
    SIs = SIs[np.abs(SIs) <= max_lag]

    bins_half = np.arange(bin_width / 2., max_lag + bin_width / 2. + bin_width, bin_width)  # bins are centered
    bins_half += np.spacing(bins_half)  # without spacing the histogram would not be symmetric as comparison operation is different for both sides of the bin edges
    bins = np.concatenate((-bins_half[::-1], bins_half))
    autocorr = np.histogram(SIs, bins=bins)[0]
    t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)

    # control: autocorr is symmetric
    # half_len = int((len(autocorr) - 1) / 2)
    # assert np.all(autocorr[half_len+1:] == autocorr[:half_len][::-1])

    if remove_zero:
        autocorr[to_idx(max_lag, bin_width)] = 0
    if normalization == 'sum':
        if np.sum(autocorr) > 0:
            autocorr = autocorr / float(np.sum(autocorr) * bin_width) * 2  # normalize to just one half of the autocorrelation
    elif normalization == 'max':
        if np.max(autocorr) > 0:
            autocorr = autocorr / float(np.max(autocorr))
    return autocorr, t_autocorr, bins


def get_all_SIs(ISIs_cum):
    ISIs_cum = np.insert(ISIs_cum, 0, 0)  # add distance to 1st spike
    SI_mat = np.tile(ISIs_cum, (len(ISIs_cum), 1)) - np.array([ISIs_cum]).T
    return SI_mat.flatten()


def get_all_SIs_lower_max_lag_except_zero(ISIs, max_lag):
    SIs = get_all_SIs(np.cumsum(ISIs))
    SIs = SIs[np.abs(SIs) <= max_lag]
    SIs = SIs[SIs != 0]
    return SIs


def plot_autocorrelation(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag):
    ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
    ax.set_xlim(-max_lag, max_lag)


def plot_autocorrelation_with_kde(ax, cell_idx, t_auto_corr, auto_corr_cells, bin_size, max_lag, kernel_cells,
                                  dt_kernel=0.05):
    ax.bar(t_auto_corr, auto_corr_cells[cell_idx], bin_size, color='0.5', align='center')
    t_kernel = np.arange(-max_lag, max_lag+dt_kernel, dt_kernel)
    ax.plot(t_kernel, kernel_cells[cell_idx].pdf(t_kernel), color='k')
    ax.set_xlim(-max_lag, max_lag)