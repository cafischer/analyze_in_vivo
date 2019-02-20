import numpy as np
import matplotlib.pyplot as pl


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
