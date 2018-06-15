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


def fun_fit_with_b(x, a, b, c, tau):
    res = c * (x - b) * (np.exp(-(x - b) / tau))
    if isinstance(res, (float, int)):
        res = res if res > 0 else 0
    else:
        res[(x - b) <= 0] = 0
    return res


def fun_fit_with_a(x, a, b, c, tau):
    res = c * x ** a * (np.exp(-x / tau))
    if isinstance(res, (float, int)):
        res = res if res > 0 else 0
    else:
        res[(x) <= 0] = 0
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
    max_ISI = 100
    use_fit_fun = 'fun_fit'
    save_dir_img = os.path.join(save_dir_img, cell_type, use_fit_fun)
    n_smooth = 5
    bin_widths = np.arange(0.5, 2.0+0.1, 0.1)  # TODO
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    half_width = np.zeros((len(cell_ids), len(bin_widths)))
    half_width_smoothed = np.zeros((len(cell_ids), len(bin_widths)))
    ISI_at_peak = np.zeros((len(cell_ids), len(bin_widths)))
    ISI_at_peak_smoothed = np.zeros((len(cell_ids), len(bin_widths)))
    half_width_fit = np.zeros((len(cell_ids), len(bin_widths)))
    ISI_at_peak_fit = np.zeros((len(cell_ids), len(bin_widths)))
    ISI_hist_fit = np.zeros((len(cell_ids), len(bin_widths)), dtype=object)
    bins_fit = np.zeros((len(cell_ids), len(bin_widths)), dtype=object)
    root1_fit = np.zeros((len(cell_ids), len(bin_widths)))
    root2_fit = np.zeros((len(cell_ids), len(bin_widths)))
    p_opt_fit = np.zeros((len(cell_ids), len(bin_widths)), dtype=object)
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
            bins_fit[cell_idx, bin_idx] = bins
            ISI_hist_fit[cell_idx, bin_idx] = ISI_hist

            if use_fit_fun == 'fun_fit':
                fit_fun = fun_fit
            elif use_fit_fun == 'fun_fit_with_a':
                fit_fun = fun_fit_with_a
            elif use_fit_fun == 'fun_fit_with_b':
                fit_fun = fun_fit_with_b
            else:
                raise ValueError

            try:
                p_opt_fit[cell_idx, bin_idx] = curve_fit(fit_fun, bins_fit[cell_idx, bin_idx][:-1],
                                                         ISI_hist_fit[cell_idx, bin_idx],
                                                         bounds=([0, 0, 0, 1e-5], [np.inf, np.inf, np.inf, np.inf]))[0]
            except RuntimeError:
                p_opt_fit[cell_idx, bin_idx] = np.array([np.nan, np.nan, np.nan, np.nan])

            # compute ISI at peak and half width for fit
            if use_fit_fun == 'fun_fit':
                ISI_at_peak_fit[cell_idx, bin_idx] = p_opt_fit[cell_idx, bin_idx][0] * p_opt_fit[cell_idx, bin_idx][3] \
                                                     + p_opt_fit[cell_idx, bin_idx][1]
            elif use_fit_fun == 'fun_fit_with_a':
                ISI_at_peak_fit[cell_idx, bin_idx] = p_opt_fit[cell_idx, bin_idx][0] * p_opt_fit[cell_idx, bin_idx][3]
            elif use_fit_fun == 'fun_fit_with_b':
                ISI_at_peak_fit[cell_idx, bin_idx] = p_opt_fit[cell_idx, bin_idx][3] + p_opt_fit[cell_idx, bin_idx][1]

            half_height_fit = fit_fun(ISI_at_peak_fit[cell_idx, bin_idx], *p_opt_fit[cell_idx, bin_idx]) / 2.0

            def fun_roots(x):
                return fit_fun(x, *p_opt_fit[cell_idx, bin_idx]) - half_height_fit

            root1_fit[cell_idx, bin_idx] = brentq(fun_roots, 0, ISI_at_peak_fit[cell_idx, bin_idx])
            add_range = 0
            while fun_roots(max_ISI + add_range) >= 0:
                add_range += 10
            root2_fit[cell_idx, bin_idx] = brentq(fun_roots, ISI_at_peak_fit[cell_idx, bin_idx], max_ISI+add_range)

            half_width_fit[cell_idx, bin_idx] = root2_fit[cell_idx, bin_idx] - root1_fit[cell_idx, bin_idx]

            # pl.figure()
            # pl.bar(bins_fit[cell_idx, bin_idx][:-1], ISI_hist_fit[cell_idx, bin_idx],
            #        bins_fit[cell_idx, bin_idx][1] - bins_fit[cell_idx, bin_idx][0], color='0.5')
            # x = np.arange(0, 100, 0.01)
            # pl.plot(x, fit_fun(x, *p_opt_fit[cell_idx, bin_idx]), '-', color='k', linewidth=1.5)
            # pl.plot(ISI_at_peak_fit[cell_idx, bin_idx],
            #         fit_fun(np.array([ISI_at_peak_fit[cell_idx, bin_idx]]), *p_opt_fit[cell_idx, bin_idx]), 'or')
            # pl.plot([root1_fit[cell_idx, bin_idx], root2_fit[cell_idx, bin_idx]],
            #         [fit_fun(root1_fit[cell_idx, bin_idx], *p_opt_fit[cell_idx, bin_idx]),
            #          fit_fun(root2_fit[cell_idx, bin_idx], *p_opt_fit[cell_idx, bin_idx])], 'r', linewidth=2)
            # pl.show()

        # smooth 
        half_width_smoothed[cell_idx, :] = convolve(half_width[cell_idx, :], np.ones(n_smooth) / float(n_smooth),
                                                    mode='nearest')
        ISI_at_peak_smoothed[cell_idx, :] = convolve(ISI_at_peak[cell_idx, :], np.ones(n_smooth) / float(n_smooth),
                                                    mode='nearest')

    # plots for estimate from ISI histogram
    # half-width
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
        pl.plot(bin_widths, half_width_smoothed[cell_idx, :], 'o-', color=colors[cell_idx], label=cell_ids[cell_idx],
                markersize=4)

    for label in half_width_smoothed_labels.keys():
        pl.annotate(half_width_smoothed_labels[label], xy=(bin_widths[-1] + 0.05, label),
                    horizontalalignment='left', verticalalignment='center', fontsize=8)
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Bin width (ms)')
    pl.ylabel('ISI hist. width at half peak (ms)')
    pl.tight_layout()
    if all_cells:
        pl.savefig(os.path.join(save_dir_img, 'half_width_smooth_'+str(n_smooth)+'_all_cells.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'half_width_smooth_' + str(n_smooth) + '.png'))

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
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Bin width (ms)')
    pl.ylabel('ISI at peak (ms)')
    pl.tight_layout()
    if all_cells:
        pl.savefig(os.path.join(save_dir_img, 'ISI_at_peak_smooth_'+str(n_smooth)+'_all_cells.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'ISI_at_peak_smooth_'+str(n_smooth)+'.png'))

    # plots for fitted
    # half-width
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    half_width_fit_labels = {}
    for cell_idx in range(len(cell_ids)):
        if half_width_smoothed_labels.get(half_width_fit[cell_idx, -1], None) is None:
            half_width_fit_labels[half_width_fit[cell_idx, -1]] = cell_ids[cell_idx]
        else:
            if len(half_width_fit_labels[half_width_fit[cell_idx, -1]].split(',')) == 3:
                half_width_fit_labels[half_width_fit[cell_idx, -1]] += ',\n' + cell_ids[cell_idx]
            else:
                half_width_fit_labels[half_width_fit[cell_idx, -1]] += ', ' + cell_ids[cell_idx]
    pl.figure()
    for cell_idx in range(len(cell_ids)):
        pl.plot(bin_widths, half_width_fit[cell_idx, :], 'o-', color=colors[cell_idx], label=cell_ids[cell_idx],
                markersize=4)

    for label in half_width_fit_labels.keys():
        pl.annotate(half_width_fit_labels[label], xy=(bin_widths[-1] + 0.05, label),
                    horizontalalignment='left', verticalalignment='center', fontsize=8)
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Bin width (ms)')
    pl.ylabel('ISI hist. width at half peak (ms)')
    pl.tight_layout()
    if all_cells:
        pl.savefig(os.path.join(save_dir_img, 'half_width_fit_all_cells.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'half_width_fit.png'))

    # ISI at peak
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    ISI_at_peak_fit_labels = {}
    for cell_idx in range(len(cell_ids)):
        if ISI_at_peak_fit_labels.get(ISI_at_peak_fit[cell_idx, -1], None) is None:
            ISI_at_peak_fit_labels[ISI_at_peak_fit[cell_idx, -1]] = cell_ids[cell_idx]
        else:
            if len(ISI_at_peak_fit_labels[ISI_at_peak_fit[cell_idx, -1]].split(',')) == 3:
                ISI_at_peak_fit_labels[ISI_at_peak_fit[cell_idx, -1]] += ',\n' + cell_ids[cell_idx]
            else:
                ISI_at_peak_fit_labels[ISI_at_peak_fit[cell_idx, -1]] += ', ' + cell_ids[cell_idx]
    pl.figure()
    for cell_idx in range(len(cell_ids)):
        pl.plot(bin_widths, ISI_at_peak_fit[cell_idx, :], 'o-', color=colors[cell_idx], label=cell_ids[cell_idx],
                markersize=4)

    for label in ISI_at_peak_fit_labels.keys():
        pl.annotate(ISI_at_peak_fit_labels[label], xy=(bin_widths[-1] + 0.05, label),
                    horizontalalignment='left', verticalalignment='center', fontsize=8)
    # pl.legend(loc='upper left', fontsize=8)
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Bin width (ms)')
    pl.ylabel('ISI at peak (ms)')
    pl.tight_layout()
    if all_cells:
        pl.savefig(os.path.join(save_dir_img, 'ISI_at_peak_fit_all_cells.png'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'ISI_at_peak_fit.png'))

    # comparison half-width
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    pl.figure()
    for cell_idx in range(len(cell_ids)):
        pl.plot(half_width_smoothed[cell_idx, :], half_width_fit[cell_idx, :], 'o-', color=colors[cell_idx],
                label=cell_ids[cell_idx], markersize=4)
    pl.xlim(bin_widths[0] - 0.05, bin_widths[-1] + 0.8)
    pl.xlabel('Width at half peak from histogram (ms)')
    pl.ylabel('Width at half peak from fit (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'comparison_half_width.png'))


    # other plots for fit
    for bin_idx, bin_width in enumerate(bin_widths):
        save_dir_img_bin = os.path.join(save_dir_img, 'bin_width_%.2f' % bin_width)
        if not os.path.exists(save_dir_img_bin):
            os.makedirs(save_dir_img_bin)
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

                    axes[i1, i2].bar(1, (half_width_fit[cell_idx, bin_idx]), width=0.8, color='0.5')
                    axes[i1, i2].set_xlim(0, 2)
                    axes[i1, i2].set_xticks([])

                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Half width (ms)')
                    cell_idx += 1
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img_bin, 'half_width_fit.png'))

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

                    axes[i1, i2].bar(1, (ISI_at_peak_fit[cell_idx, bin_idx]), width=0.8, color='0.5')
                    axes[i1, i2].set_xlim(0, 2)
                    axes[i1, i2].set_xticks([])

                    if i2 == 0:
                        axes[i1, i2].set_ylabel('ISI at peak (ms)')
                    cell_idx += 1
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img_bin, 'ISI_at_peak_fit.png'))


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

                    axes[i1, i2].bar(bins_fit[cell_idx, bin_idx][:-1], ISI_hist_fit[cell_idx, bin_idx],
                                     bins_fit[cell_idx, bin_idx][1] - bins_fit[cell_idx, bin_idx][0], color='0.5')
                    x = np.arange(0, 100, 0.01)
                    axes[i1, i2].plot(x, fun_fit(x, *p_opt_fit[cell_idx, bin_idx]), '-', color='k', linewidth=1.5)
                    axes[i1, i2].plot(ISI_at_peak_fit[cell_idx, bin_idx],
                                      fun_fit(np.array([ISI_at_peak_fit[cell_idx, bin_idx]]), *p_opt_fit[cell_idx, bin_idx]), 'or')
                    axes[i1, i2].plot([root1_fit[cell_idx, bin_idx], root2_fit[cell_idx, bin_idx]],
                                      [fun_fit(root1_fit[cell_idx, bin_idx], *p_opt_fit[cell_idx, bin_idx]),
                                       fun_fit(root2_fit[cell_idx, bin_idx], *p_opt_fit[cell_idx, bin_idx])],
                                      'r', linewidth=2)

                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xlabel('ISI (ms)')
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('# ISIs')
                    cell_idx += 1
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img_bin, 'ISI_hist_fit.png'))
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