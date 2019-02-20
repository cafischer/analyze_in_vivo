import numpy as np
import matplotlib.pyplot as pl


def plot_sta_on_ax(ax, cell_idx, t_AP, sta_mean_cells, sta_std_cells, before_AP=5, after_AP=25, ylims=(None, None)):
    ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
    ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                    sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
    ax.set_ylim(*ylims)
    ax.set_xlim(-before_AP, after_AP)
    ax.set_xticks(np.arange(-before_AP, after_AP+10, 10))


def plot_sta_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
                        sta_std_good_APs_cells, before_AP, after_AP, ylims=(None, None)):
    if subplot_idx == 0:
        ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_good_APs_cells[cell_idx] - sta_std_good_APs_cells[cell_idx],
                        sta_mean_good_APs_cells[cell_idx] + sta_std_good_APs_cells[cell_idx], color='0.6')
        ax.set_xlabel('')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP + 5, 10))
        ax.set_xticklabels([])
        ax.annotate('selected APs', xy=(25, ylims[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)
    elif subplot_idx == 1:
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP+5, 10))
        ax.annotate('all APs', xy=(25, ylims[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)


def plot_sta_derivative_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_mean_good_APs_cells,
                             before_AP, after_AP, time_for_max, ylims=(None, None), diff_selected_all=None):

    if subplot_idx == 0:
        if ~np.any(np.isnan(sta_mean_good_APs_cells[cell_idx])):
            ax.fill_between((0, time_for_max), ylims[0], ylims[1], color='0.8')
        ax.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.annotate('selected APs', xy=(25, ylims[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)
        if ~np.isnan(diff_selected_all[cell_idx]):
            ax.annotate('%.1f' % diff_selected_all[cell_idx], xy=(25, ylims[1]), textcoords='data',
                        horizontalalignment='right', verticalalignment='top', fontsize=8)

        # # smooth
        # std = np.std(sta_mean_good_APs_cells[cell_idx][to_idx(2, dt):to_idx(3, dt)])
        # #std = sta_std_cells[cell_idx][:-1]
        # w = np.ones(len(sta_mean_good_APs_cells[cell_idx])) / std
        # print 'w1', w[0]
        # splines = UnivariateSpline(t_AP, sta_mean_good_APs_cells[cell_idx], w=w, s=None, k=3)
        # smoothed = splines(t_AP)
        #
        # ax.plot(t_AP, smoothed, 'r')

    elif subplot_idx == 1:
        if ~np.any(np.isnan(sta_mean_cells[cell_idx])):
            ax.fill_between((0, time_for_max), ylims[0], ylims[1], color='0.8')
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks([-10, 0, 10, 20])
        ax.annotate('all APs', xy=(25, ylims[0]), textcoords='data',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=8)

        # # smooth
        # std = np.std(sta_mean_cells[cell_idx][to_idx(2, dt):to_idx(3, dt)])
        # #std = sta_std_good_APs_cells[cell_idx][:-1]
        # w = np.ones(len(sta_mean_cells[cell_idx])) / std
        # print 'w2', w[0]
        # splines = UnivariateSpline(t_AP, sta_mean_cells[cell_idx], w=w, s=None, k=3)
        # smoothed = splines(t_AP)
        #
        # ax.plot(t_AP, smoothed, 'r')