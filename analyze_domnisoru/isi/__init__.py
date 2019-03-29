import numpy as np
import matplotlib.pyplot as pl


def get_ISI_hist_peak_and_width(ISI_hist, t_hist):
    peak_idx = np.argmax(ISI_hist)
    peak_ISI_hist = t_hist[peak_idx]
    half_max = ISI_hist[peak_idx] / 2.

    root1_idx = np.nonzero(np.diff(np.sign(ISI_hist[:peak_idx] - half_max
                                           + np.spacing(ISI_hist[:peak_idx] - half_max))) == 2)[0][-1]
    root2_idx = np.nonzero(np.diff(np.sign(ISI_hist[peak_idx:] - half_max
                                           + np.spacing(ISI_hist[peak_idx:] - half_max))) == -2)[0][0] + peak_idx
    root1 = t_hist[root1_idx]
    root2 = t_hist[root2_idx]
    width_at_half_max = root2 - root1

    # for visualization
    # pl.figure()
    # pl.plot(t_hist, ISI_hist, 'k')
    # pl.plot(peak_ISI_hist, ISI_hist[peak_idx], 'or')
    # pl.plot([root1, root2], [ISI_hist[root1_idx], ISI_hist[root2_idx]], 'r', linewidth=2)
    # pl.show()
    return peak_ISI_hist, width_at_half_max


def plot_ISI_hist_on_ax(ax, cell_idx, ISI_hist, cum_ISI_hist_x, cum_ISI_hist_y, max_ISI, bin_width):
    bins = np.arange(0, max_ISI + bin_width, bin_width)
    ax.bar(bins[:-1], ISI_hist[cell_idx, :] / (np.sum(ISI_hist[cell_idx, :]) * bin_width),
           bins[1] - bins[0], color='0.5', align='edge')
    cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[cell_idx], len(cum_ISI_hist_x[cell_idx]), max_ISI)
    cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[cell_idx], len(cum_ISI_hist_y[cell_idx]), 1.0)
    ax_twin = ax.twinx()
    ax_twin.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, color='k', drawstyle='steps-post')
    ax_twin.set_xlim(0, max_ISI)
    ax_twin.set_ylim(0, 1)
    ax_twin.set_yticks([0, 0.5, 1])
    if (cell_idx + 1) % 9 == 0 or (cell_idx+1) == 26:
        ax_twin.set_yticklabels([0, 0.5, 1])
        ax_twin.set_ylabel('Cum. frequency')
    else:
        ax_twin.set_yticklabels([])
    ax.spines['right'].set_visible(True)


def plot_ISI_hist_on_ax_with_kde(ax, cell_idx, ISI_hist, max_ISI, bin_width,
                                 kernel_cells, dt_kernel=0.01):
    bins = np.arange(0, max_ISI+bin_width, bin_width)
    ax.bar(bins[:-1], ISI_hist[cell_idx, :] / (np.sum(ISI_hist[cell_idx, :]) * bin_width),
           bins[1] - bins[0], color='0.5', align='edge')
    t_kernel = np.arange(0, max_ISI+dt_kernel, dt_kernel)
    ax.plot(t_kernel, kernel_cells[cell_idx].pdf(t_kernel), color='k')


def plot_ISI_return_map(ax, cell_idx, ISIs_cells, max_ISI, median_cells=None, steps_median=None, log_scale=False):
    if log_scale:
        ax.loglog(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5',
                  marker='o', linestyle='', markersize=1, alpha=0.5)
        ax.set_xlim(1, max_ISI)
        ax.set_ylim(1, max_ISI)
    else:
        ax.plot(ISIs_cells[cell_idx][:-1], ISIs_cells[cell_idx][1:], color='0.5',
                marker='o', linestyle='', markersize=1, alpha=0.5)
        ax.set_xlim(0, max_ISI)
        ax.set_ylim(0, max_ISI)
        ax.set_xticks(np.arange(0, max_ISI + 50, 50))
        ax.set_yticks(np.arange(0, max_ISI + 50, 50))
    ax.set_aspect('equal', adjustable='box-forced')
    if median_cells is not None:
        ax.plot(steps_median, median_cells[cell_idx, :], 'k', label='median')