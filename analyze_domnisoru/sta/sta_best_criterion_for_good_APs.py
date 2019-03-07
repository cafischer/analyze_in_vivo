from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.sta import get_sta_criterion, plot_sta_grid_on_ax
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells, plot_for_all_grid_cells_grid, \
    plot_with_markers
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from scipy.signal import gaussian
pl.style.use('paper_subplots')


def smooth(y, box_pts):
    box = gaussian(box_pts, 1)
    box /= np.sum(box)
    #box = np.ones(box_pts)/box_pts
    box_pts_half = int(np.floor(box_pts/2))
    y_padded = np.concatenate((np.array([y[0]]*box_pts_half), y, np.array([y[-1]]*box_pts_half)))
    y_smooth = np.convolve(y_padded, box, mode='valid')

    # visualize
    # pl.figure()
    # pl.plot(y)
    # pl.plot(y_smooth)
    # pl.show()
    return y_smooth


def plot_grid_on_ax(ax, cell_idx, t_AP, sta_1der, sta_1der_smooth, sta_2der, sta_2der_smooth, k, k_smooth,
                    sta_mean_cells, before_AP, after_AP, ylims=(None, None)):
    ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')

    argmax = np.argmax(k_smooth[cell_idx][:to_idx(before_AP, 0.05)])
    #ax.plot(t_AP[argmax], sta_mean_cells[cell_idx][argmax], 'b', marker='^', markersize=4)
    ax.axhline(sta_mean_cells[cell_idx][argmax], 0, 1, color='b', linewidth=0.6)
    ax.axvline(t_AP[argmax], 0, 1, color='b', linewidth=0.6)
    # argmin = np.argmin(k_smooth[cell_idx][to_idx(before_AP-1, 0.05):to_idx(before_AP, 0.05)]) + to_idx(before_AP-1, 0.05)
    # ax.axhline(sta_mean_cells[cell_idx][argmin], 0, 1, color='lightblue')
    # ax.axvline(t_AP[argmin], 0, 1, color='lightblue')

    # argmax = np.argmax(sta_2der[cell_idx][:to_idx(before_AP, 0.05)])
    # ax.plot(t_AP[argmax], sta_mean_cells[cell_idx][argmax], color='r', marksize=3)
    # ax.axhline(sta_mean_cells[cell_idx][argmax], 0, 1, color='r', linewidth=0.5)
    # ax.axvline(t_AP[argmax], 0, 1, color='r', linewidth=0.5)
    # argmax = np.argmax(sta_2der_smooth[cell_idx][:to_idx(before_AP, 0.05)])
    # ax.axhline(sta_mean_cells[cell_idx][argmax], 0, 1, color='orange')
    # ax.axvline(t_AP[argmax], 0, 1, color='orange')

    AP_thresh_derivative = 5  # TODO 3
    AP_thresh_idx = get_AP_onset_idxs(sta_1der[cell_idx][:to_idx(before_AP, 0.05)], AP_thresh_derivative)[-1]
    # ax.plot(t_AP[AP_thresh_idx], sta_mean_cells[cell_idx][AP_thresh_idx], 'y', marker='v', markersize=4)
    ax.axhline(sta_mean_cells[cell_idx][AP_thresh_idx], 0, 1, color='y', linewidth=0.6)
    ax.axvline(t_AP[AP_thresh_idx], 0, 1, color='y', linewidth=0.6)

    # pl.figure()
    # pl.plot(t_AP, sta_mean_cells[cell_idx]/np.max(np.abs(sta_mean_cells[cell_idx])), 'k')
    # argmax = np.argmax(sta_2der[cell_idx][:to_idx(before_AP, 0.05)])
    # pl.plot(t_AP[argmax], (sta_mean_cells[cell_idx] / np.max(np.abs(sta_mean_cells[cell_idx])))[argmax], 'ob')
    # AP_thresh_idx = get_AP_onset_idxs(sta_1der[cell_idx][:to_idx(before_AP, 0.05)], AP_thresh_derivative)[-1]
    # pl.plot(t_AP[AP_thresh_idx], (sta_mean_cells[cell_idx] / np.max(np.abs(sta_mean_cells[cell_idx])))[AP_thresh_idx], 'og')
    # pl.plot(t_AP, sta_1der[cell_idx]/np.max(sta_1der[cell_idx]), 'r')
    # pl.plot(t_AP, sta_2der[cell_idx]/np.max(sta_2der[cell_idx]), 'b')
    # pl.show()

    # pl.figure()
    # pl.plot(t_AP, sta_mean_cells[cell_idx]/np.max(np.abs(sta_mean_cells[cell_idx])), 'k')
    # argmin = np.argmin(k[cell_idx][to_idx(before_AP - 1, 0.05):to_idx(before_AP, 0.05)]) + to_idx(before_AP - 1, 0.05)
    # pl.plot(t_AP[argmin], (sta_mean_cells[cell_idx] / np.max(np.abs(sta_mean_cells[cell_idx])))[argmin], 'ob')
    # pl.plot(t_AP, (k[cell_idx] / np.max(np.abs(k[cell_idx]))), 'g')
    # #pl.show()
    #
    # pl.figure()
    # pl.plot(t_AP, k[cell_idx], 'g')
    # pl.show()

    ax.set_ylim(*ylims)
    ax.set_xlim(-5, 3)
    ax.set_xticks(np.arange(-5, 3+5, 10))
    if cell_idx == 0 or cell_idx == 9 or cell_idx == 18:
        ax.set_ylabel('Mem. pot.')


def get_osculating_circle(x, y, dx):
    x_1deriv = np.diff(x) / dx
    x_2deriv = np.diff(x_1deriv) / dx
    y_1deriv = np.diff(y) / dx
    y_2deriv = np.diff(y_1deriv) / dx
    x_1deriv = x_1deriv[:-1]
    y_1deriv = y_1deriv[:-1]
    k = init_nan(len(y_2deriv))
    numerator = x_1deriv * y_2deriv - x_2deriv * y_1deriv
    denominator = ((x_1deriv ** 2 + y_1deriv ** 2) ** (3. / 2))
    k[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    return k


def get_osculating_circle_with_scaling(x, y, dx, scaling):
    x_1deriv = np.diff(x) / dx
    x_2deriv = np.diff(x_1deriv) / dx
    y_1deriv = np.diff(y) / dx
    y_2deriv = np.diff(y_1deriv) / dx
    x_1deriv = x_1deriv[:-1]
    y_1deriv = y_1deriv[:-1]
    k = init_nan(len(y_2deriv))
    numerator = (x_1deriv * y_2deriv - x_2deriv * y_1deriv) / scaling**2
    denominator = ((x_1deriv ** 2 + (y_1deriv / scaling) ** 2) ** (3. / 2))
    k[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    return k


def test_osculating_circle(scale=1):
    x = np.arange(-5, 5, 0.01)
    y = scale * np.exp(x) #+ np.random.rand(len(x)) * 0.001  # (x-1)**2 #
    k = get_osculating_circle(x, y, x[1]-x[0])
    print 'min: ', x[np.argmin(k)]

    pl.figure()
    pl.plot(x[:-2], k, 'm')
    pl.plot(x, y, 'k')
    pl.ylim(-200, 200)
    pl.show()


if __name__ == '__main__':
    # test_osculating_circle(1)
    # test_osculating_circle(8)
    # test_osculating_circle(0.05)
    # pl.show()

    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    t_vref = 10  # ms
    dt = 0.05  # ms
    do_detrend = False
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend])
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # AP_criterions = [{'quantile': 20}, {'quantile': 10}, {'AP_amp_and_width': (40, 1)}]
    # time_before_after_AP = [(20, 25), (25, 20), (25, 25), (25, 30), (30, 25), (30, 30)]  # (before_AP, after_AP)

    AP_criterions = [{'AP_amp_and_width': (40, 1)}]
    time_before_after_AP = [(25, 25)]  # (before_AP, after_AP)

    # AP_criterions = [{'AP_amp_and_width': (51.8, 0.72)}]
    # time_before_after_AP = [(10, 25)]  # (before_AP, after_AP)

    # main
    for AP_criterion in AP_criterions:
        for (before_AP, after_AP) in time_before_after_AP:
            print AP_criterion, (before_AP, after_AP)
            (sta_mean_cells, sta_std_cells, sta_mean_good_APs_cells,
             sta_std_good_APs_cells, _) = get_sta_criterion(do_detrend, before_AP, after_AP,
                                                            AP_criterion, t_vref, cell_ids, save_dir)

            # save
            folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                       + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
            if not os.path.exists(os.path.join(save_dir_img, folder_name)):
                os.makedirs(os.path.join(save_dir_img, folder_name))
            np.save(os.path.join(save_dir_img, folder_name, 'sta_mean.npy'), sta_mean_good_APs_cells)

            t_AP = np.arange(-before_AP, after_AP + dt, dt)
            plot_kwargs = dict(t_AP=t_AP,
                               sta_mean_cells=sta_mean_cells,
                               sta_std_cells=sta_std_cells,
                               sta_mean_good_APs_cells=sta_mean_good_APs_cells,
                               sta_std_good_APs_cells=sta_std_good_APs_cells,
                               before_AP=before_AP,
                               after_AP=after_AP,
                               ylims=(-75, -45)
                               )

            fig_title = 'Criterion: ' + AP_criterion.keys()[0].replace('_', ' ') + ' ' + str(AP_criterion.values()[0]) \
                        + '  Time range (before AP, after AP): ' + str((before_AP, after_AP))
            plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid_on_ax, plot_kwargs,
                                         xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                         fig_title=fig_title,
                                         save_dir_img=os.path.join(save_dir_img, folder_name, 'sta.png'))

            #pl.show()
            sta_1derivative_cells = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0])-2))
            sta_2derivative_cells = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0])-2))
            k = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0])-2))
            sta_1derivative_cells_smooth = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0]) - 2))
            sta_2derivative_cells_smooth = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0]) - 2))
            k_smooth = np.zeros((len(cell_ids), len(sta_mean_good_APs_cells[0]) - 2))
            for cell_idx in range(len(cell_ids)):
                k[cell_idx, :] = get_osculating_circle(t_AP, sta_mean_good_APs_cells[cell_idx], dt)
                sta_1derivative_cells[cell_idx, :] = (np.diff(sta_mean_good_APs_cells[cell_idx]) / dt)[:-1]
                sta_2derivative_cells[cell_idx, :-1] = (np.diff(sta_1derivative_cells[cell_idx]) / dt)

                sta_mean_good_APs_cells_smooth = smooth(sta_mean_good_APs_cells[cell_idx], 5)
                k_smooth_noscale = get_osculating_circle_with_scaling(t_AP, sta_mean_good_APs_cells_smooth, dt, scaling=1)
                k_smooth[cell_idx, :] = get_osculating_circle_with_scaling(t_AP, sta_mean_good_APs_cells_smooth, dt, scaling=20)
                sta_1derivative_cells_smooth[cell_idx, :] = (np.diff(sta_mean_good_APs_cells_smooth) / dt)[:-1]
                sta_2derivative_cells_smooth[cell_idx, :-1] = (np.diff(sta_1derivative_cells_smooth[cell_idx]) / dt)

                # pl.figure()
                # pl.plot(t_AP, sta_mean_good_APs_cells[cell_idx], 'k')
                # pl.plot(t_AP, sta_mean_good_APs_cells_smooth, 'r')
                # pl.show()
                # pl.figure()
                # pl.plot(t_AP[:-2], k_smooth[cell_idx, :], 'k')
                # pl.plot(t_AP[:-2], k_smooth_noscale, 'r')
                # pl.show()

            sta_mean_good_APs_cells = [a[:-2] for a in sta_mean_good_APs_cells]
            sta_std_good_APs_cells = [a[:-2] for a in sta_std_good_APs_cells]

            plot_kwargs = dict(t_AP=t_AP[:-2],
                               k=k,
                               k_smooth=k_smooth,
                               sta_1der=sta_1derivative_cells,
                               sta_1der_smooth=sta_1derivative_cells_smooth,
                               sta_2der=sta_2derivative_cells,
                               sta_2der_smooth=sta_2derivative_cells_smooth,
                               sta_mean_cells=sta_mean_good_APs_cells,
                               before_AP=before_AP,
                               after_AP=after_AP,
                               ylims=(-80, -20),
                               )
            plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_grid_on_ax, plot_kwargs,
                                         xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
                                         fig_title=fig_title,
                                         save_dir_img=os.path.join(save_dir_img, 'test.png'))
            pl.show()