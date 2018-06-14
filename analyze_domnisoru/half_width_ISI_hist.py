from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs, get_ISI_hist
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
from scipy.ndimage.filters import convolve
from scipy.optimize import curve_fit, brentq
pl.style.use('paper')


def fun_fit(x, a, b, c, tau):
    res = c * (x - b) ** a * (np.exp(-(x - b) / tau))
    if isinstance(res, (float, int)):
        res = res if res > 0 else 0
    else:
        res[(x - b) <= 0] = 0
    return res


if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies

    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    all_cells = True
    if all_cells:
        cell_ids = load_cell_ids(save_dir, cell_type)
    else:
        cell_ids = ['s43_0003', 's66_0003', 's67_0000', 's73_0004', 's76_0002', 's79_0003', 's82_0002', 's95_0006',
                   's101_0009', 's109_0002', 's117_0002', 's118_0002', 's119_0004', 's120_0002']
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = True
    max_ISI = 200
    max_ISI_fit = 100
    n_smooth = 5
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(max_ISI))

    # parameter
    bin_widths = [1.0]  #np.arange(0.5, 2.0+0.1, 0.1)  TODO

    # main
    half_width = np.zeros((len(cell_ids), len(bin_widths)))
    half_width_smoothed = np.zeros((len(cell_ids), len(bin_widths)))
    ISI_at_peak = np.zeros((len(cell_ids), len(bin_widths)))
    ISI_at_peak_smoothed = np.zeros((len(cell_ids), len(bin_widths)))
    half_width_fit = np.zeros((len(cell_ids)))
    ISI_at_peak_fit = np.zeros((len(cell_ids)))
    ISI_hist_fit = []
    root1_fit = np.zeros((len(cell_ids)))
    root2_fit = np.zeros((len(cell_ids)))
    p_opt_fit = []
    for cell_idx, cell_id in enumerate(cell_ids):
        for bin_idx, bin_width in enumerate(bin_widths):
            print cell_id
            bins = np.arange(0, max_ISI + bin_width, bin_width)

            # load
            data = load_data(cell_id, param_list, save_dir)
            v = data['Vm_ljpc']
            t = np.arange(0, len(v)) * data['dt']
            dt = t[1] - t[0]

            # ISIs
            if use_AP_max_idxs_domnisoru:
                AP_max_idxs = data['spiketimes']
            else:
                AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
            ISIs = get_ISIs(AP_max_idxs, t)
            if filter_long_ISIs:
                ISIs = ISIs[ISIs <= max_ISI]

            # ISI histograms
            ISI_hist = get_ISI_hist(ISIs, bins)

            # estimate width at half peak of ISI hist
            bin_idx_peak = np.argmax(ISI_hist)
            half_height = ISI_hist[bin_idx_peak] / 2.0
            bins_greater = ISI_hist >= half_height
            starts, ends = get_start_end_group_of_ones(bins_greater.astype(int))
            idx_group = np.array([True if s <= bin_idx_peak <= e else False for s, e in zip(starts, ends)])
            len_group = ends[idx_group] - starts[idx_group] + 1
            half_width[cell_idx, bin_idx] = len_group * bin_width

            # plot_ISI_hist(ISI_hist, bins, title=cell_id)
            # pl.plot([bins[starts[idx_group]], bins[ends[idx_group]]], [half_height, half_height], 'r')
            # pl.show()

            # max_val
            ISI_at_peak[cell_idx, bin_idx] = bins[bin_idx_peak]

            # fit
            if bin_width == 1.0:
                bins_fit = np.arange(0, max_ISI_fit+bin_width, bin_width)
                ISI_hist_fit.append(ISI_hist[(bins <= max_ISI_fit)[:-1]][:-1])

                try:
                    p_opt_ = curve_fit(fun_fit, bins_fit[:-1], ISI_hist_fit[cell_idx],
                                       bounds=([0, 0, 0, 1e-8], [np.inf, np.inf, np.inf, np.inf]))[0]
                    p_opt_fit.append(p_opt_)
                except RuntimeError:
                    p_opt_fit.append([np.nan, np.nan, np.nan, np.nan])

                # half width of fit
                ISI_at_peak_fit[cell_idx] = p_opt_fit[cell_idx][0] * p_opt_fit[cell_idx][3] + p_opt_fit[cell_idx][1]
                half_height_fit = fun_fit(np.array([ISI_at_peak_fit[cell_idx]]), *p_opt_fit[cell_idx]) / 2.0
                def fun_roots(x):
                    return fun_fit(x, *p_opt_fit[cell_idx]) - half_height_fit
                root1_fit[cell_idx] = brentq(fun_roots, 0, ISI_at_peak_fit[cell_idx])
                if fun_roots(max_ISI_fit) >= 0:
                    root2_fit[cell_idx] = brentq(fun_roots, ISI_at_peak_fit[cell_idx], max_ISI_fit+10)
                else:
                    root2_fit[cell_idx] = brentq(fun_roots, ISI_at_peak_fit[cell_idx], max_ISI_fit)
                half_width_fit[cell_idx] = root2_fit[cell_idx] - root1_fit[cell_idx]

                # pl.figure()
                # pl.bar(bins_fit[:-1], ISI_hist_fit[cell_idx], bins_fit[1] - bins_fit[0], color='0.5')
                # x = np.arange(0, 100, 0.01)
                # pl.plot(x, fun_fit(x, *p_opt_fit[cell_idx]), '-', color='k', linewidth=1.5)
                # pl.plot(ISI_at_peak_fit[cell_idx],
                #         fun_fit(np.array([ISI_at_peak_fit[cell_idx]]), *p_opt_fit[cell_idx]), 'or')
                # pl.plot([root1_fit[cell_idx], root2_fit[cell_idx]],
                #         [fun_fit(root1_fit[cell_idx], *p_opt_fit[cell_idx]),
                #          fun_fit(root2_fit[cell_idx], *p_opt_fit[cell_idx])], 'r', linewidth=2)
                # pl.show()

            # # fit Gamma distribution to hist
            # import scipy.stats as stats
            # max_ISI_fit = 50
            # ISIs_fit = ISIs_per_cell[cell_idx][ISIs_per_cell[cell_idx] <= max_ISI_fit]
            # #bins_fit = bins[bins <= 50]
            # bin_width = 0.5
            # bins_fit = np.arange(0, max_ISI_fit+bin_width, bin_width)
            # #fit_alpha, fit_loc, fit_beta = stats.gamma.fit(ISIs_fit)
            # fit_a, fit_loc, fit_scale = stats.invgauss.fit(ISIs_fit)  # gamma, lognorm, invgauss
            # #print fit_alpha, fit_loc, fit_beta
            # pl.figure()
            # pl.hist(ISIs_fit, bins=bins_fit, weights=np.ones(len(ISIs_fit))/(len(ISIs_fit)*bin_width), color='0.5')
            # #pl.plot(bins_fit, stats.gamma.pdf(bins_fit, fit_alpha, fit_loc, fit_beta), 'r')
            # pl.plot(bins_fit, stats.invgauss.pdf(bins_fit, fit_a, fit_loc, fit_scale), 'r')
            # pl.show()

        # smooth 
        half_width_smoothed[cell_idx, :] = convolve(half_width[cell_idx, :], np.ones(n_smooth) / float(n_smooth),
                                                    mode='nearest')
        ISI_at_peak_smoothed[cell_idx, :] = convolve(ISI_at_peak[cell_idx, :], np.ones(n_smooth) / float(n_smooth),
                                                    mode='nearest')

    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    half_width_smoothed_labels = {}
    for cell_idx in range(len(cell_ids)):
        if half_width_smoothed_labels.get(half_width_smoothed[cell_idx, -1], None) is None:
            half_width_smoothed_labels[half_width_smoothed[cell_idx, -1]] = cell_ids[cell_idx]
        else:
            if len(half_width_smoothed_labels[half_width_smoothed[cell_idx, -1]].split(',')) == 3:
                half_width_smoothed_labels[half_width_smoothed[cell_idx, -1]] += ',\n' + cell_ids[cell_idx]
            else:
                half_width_smoothed_labels[half_width_smoothed[cell_idx, -1]] += ', ' + cell_ids[cell_idx]
    pl.figure()
    for cell_idx in range(len(cell_ids)):
        # pl.plot(bin_widths, half_width[cell_idx, :], 'o-', color=colors[cell_idx], label=cell_ids[cell_idx])
        pl.plot(bin_widths, half_width_smoothed[cell_idx, :], 'o-', color=colors[cell_idx], label=cell_ids[cell_idx],
                markersize=4)

    for label in half_width_smoothed_labels.keys():
        pl.annotate(half_width_smoothed_labels[label], xy=(bin_widths[-1] + 0.05, label),
                    horizontalalignment='left', verticalalignment='center', fontsize=8)
    # pl.legend(loc='upper left', fontsize=8)
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Bin width (ms)')
    pl.ylabel('ISI hist. width at half peak (ms)')
    pl.tight_layout()
    if all_cells:
        pl.savefig(os.path.join(save_dir_img, cell_type, 'half_width_smooth_'+str(n_smooth)+'_all_cells.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, cell_type, 'half_width_smooth_' + str(n_smooth) + '.png'))

    # ISI at peak
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    ISI_at_peak_smoothed_labels = {}
    for cell_idx in range(len(cell_ids)):
        if ISI_at_peak_smoothed_labels.get(ISI_at_peak_smoothed[cell_idx, -1], None) is None:
            ISI_at_peak_smoothed_labels[ISI_at_peak_smoothed[cell_idx, -1]] = cell_ids[cell_idx]
        else:
            if len(ISI_at_peak_smoothed_labels[ISI_at_peak_smoothed[cell_idx, -1]].split(',')) == 3:
                ISI_at_peak_smoothed_labels[ISI_at_peak_smoothed[cell_idx, -1]] += ',\n' + cell_ids[cell_idx]
            else:
                ISI_at_peak_smoothed_labels[ISI_at_peak_smoothed[cell_idx, -1]] += ', ' + cell_ids[cell_idx]
    pl.figure()
    for cell_idx in range(len(cell_ids)):
        pl.plot(bin_widths, ISI_at_peak_smoothed[cell_idx, :], 'o-', color=colors[cell_idx], label=cell_ids[cell_idx],
                markersize=4)

    for label in ISI_at_peak_smoothed_labels.keys():
        pl.annotate(ISI_at_peak_smoothed_labels[label], xy=(bin_widths[-1] + 0.05, label),
                    horizontalalignment='left', verticalalignment='center', fontsize=8)
    # pl.legend(loc='upper left', fontsize=8)
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Bin width (ms)')
    pl.ylabel('ISI at peak (ms)')
    pl.tight_layout()
    if all_cells:
        pl.savefig(os.path.join(save_dir_img, cell_type, 'ISI_at_peak_all_cells.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, cell_type, 'ISI_at_peak.png'))

    # fitting
    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)

                axes[i1, i2].bar(1, (half_width_fit[cell_idx]), width=0.8, color='0.5')
                axes[i1, i2].set_xlim(0, 2)
                axes[i1, i2].set_xticks([])

                if i2 == 0:
                    axes[i1, i2].set_ylabel('Half width (ms)')
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'half_width.png'))

        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)

                axes[i1, i2].bar(1, (ISI_at_peak_fit[cell_idx]), width=0.8, color='0.5')
                axes[i1, i2].set_xlim(0, 2)
                axes[i1, i2].set_xticks([])

                if i2 == 0:
                    axes[i1, i2].set_ylabel('ISI at peak (ms)')
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'ISI_at_peak.png'))


        fig, axes = pl.subplots(n_rows, n_columns, sharex='all', figsize=(14, 8.5))
        cell_idx = 0
        for i1 in range(n_rows):
            for i2 in range(n_columns):
                if cell_idx < len(cell_ids):
                    if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                    elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                        axes[i1, i2].set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                    else:
                        axes[i1, i2].set_title(cell_ids[cell_idx], fontsize=12)

                axes[i1, i2].bar(bins_fit[:-1], ISI_hist_fit[cell_idx], bins_fit[1] - bins_fit[0], color='0.5')
                x = np.arange(0, 100, 0.01)
                axes[i1, i2].plot(x, fun_fit(x, *p_opt_fit[cell_idx]), '-', color='k', linewidth=1.5)
                axes[i1, i2].plot(ISI_at_peak_fit[cell_idx],
                                  fun_fit(np.array([ISI_at_peak_fit[cell_idx]]), *p_opt_fit[cell_idx]), 'or')
                axes[i1, i2].plot([root1_fit[cell_idx], root2_fit[cell_idx]],
                                  [fun_fit(root1_fit[cell_idx], *p_opt_fit[cell_idx]),
                                   fun_fit(root2_fit[cell_idx], *p_opt_fit[cell_idx])], 'r', linewidth=2)

                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel('ISI (ms)')
                if i2 == 0:
                    axes[i1, i2].set_ylabel('# ISIs')
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'fit_ISI_hist.png'))
        pl.show()

    # pl.figure()
    # pl.plot(range(len(cell_ids)), [p_opt_cells[c][0] for c in range(len(cell_ids))])
    # pl.xticks(range(len(cell_ids)), cell_ids)
    #
    # pl.figure()
    # pl.plot(range(len(cell_ids)), [p_opt_cells[c][1] for c in range(len(cell_ids))])
    # pl.xticks(range(len(cell_ids)), cell_ids)
    #
    # pl.figure()
    # pl.plot(range(len(cell_ids)), [p_opt_cells[c][2] for c in range(len(cell_ids))])
    # pl.xticks(range(len(cell_ids)), cell_ids)
    #
    # pl.figure()
    # pl.plot(range(len(cell_ids)), [p_opt_cells[c][3] for c in range(len(cell_ids))])
    # pl.xticks(range(len(cell_ids)), cell_ids)
    # pl.show()