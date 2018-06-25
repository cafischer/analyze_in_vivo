from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.burst_len_vs_preceding_silence import get_n_spikes_per_event, \
    get_ISI_idx_per_event
from cell_fitting.util import init_nan
pl.style.use('paper')


def groupby_apply(array, idx_array, n_bins, fun):
    applied_array = np.zeros(n_bins)
    for i in range(n_bins):
        applied_array[i] = fun(array[idx_array == i])
    return applied_array


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/Harris/preceding_silence_vs_burst_prob'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    ISI_burst = 8  # ms
    bins_silence = np.logspace(0, 4, 20, base=10)

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    prob_burst_after_single_cells = [0] * len(cell_ids)
    prob_burst_after_burst_cells = [0] * len(cell_ids)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # compute preceding silence and next burst
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        ISIs = get_ISIs(AP_max_idxs, t)
        burst_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        n_spikes_per_event = get_n_spikes_per_event(burst_ISI_indicator)
        ISI_idx_per_event = get_ISI_idx_per_event(burst_ISI_indicator)

        if n_spikes_per_event[-1] == 1:  # shorten by 2 to be able to index ISIs, plus want to know if next is a burst
            n_spikes_per_event = n_spikes_per_event[:-2]
            ISI_idx_per_event = ISI_idx_per_event[:-2]
        elif n_spikes_per_event[-1] == 2:
            n_spikes_per_event = n_spikes_per_event[:-1]
            ISI_idx_per_event = ISI_idx_per_event[:-1]

        silence_prec_single = ISIs[ISI_idx_per_event[n_spikes_per_event == 1] - 1]
        silence_prec_burst = ISIs[ISI_idx_per_event[n_spikes_per_event > 1] - 1]
        burst_after_single = ISIs[ISI_idx_per_event[n_spikes_per_event == 1] + 1] <= ISI_burst
        burst_after_burst = ISIs[ISI_idx_per_event[n_spikes_per_event > 1] + 1] <= ISI_burst

        # bin
        bin_idxs = np.digitize(silence_prec_single, bins_silence) - 1
        prob_burst_after_single_cells[cell_idx] = groupby_apply(burst_after_single, bin_idxs, len(bins_silence) - 1, np.sum) \
                                         / groupby_apply(np.ones(len(burst_after_single)), bin_idxs, len(bins_silence) - 1, np.sum)

        bin_idxs = np.digitize(silence_prec_burst, bins_silence) - 1
        prob_burst_after_burst_cells[cell_idx] = groupby_apply(burst_after_burst, bin_idxs, len(bins_silence) - 1, np.sum) \
                                        / groupby_apply(np.ones(len(burst_after_burst)), bin_idxs, len(bins_silence) - 1, np.sum)


    # save and plot
    if cell_type == 'grid_cells':
        def plot_burst_len_vs_preceding_silence(ax, cell_idx, bins_prec_silence, prob_next_burst_single_cells,
                                                prob_next_burst_burst_cells):

            ax.semilogx(bins_prec_silence[:-1], prob_next_burst_single_cells[cell_idx], color='k',
                    label='after single')
            ax.semilogx(bins_prec_silence[:-1], prob_next_burst_burst_cells[cell_idx], color='k',
                    linestyle='--', label='after burst')
            if cell_idx == 8:
                ax.legend(fontsize=10)

        plot_kwargs = dict(bins_prec_silence=bins_silence,
                           prob_next_burst_single_cells=prob_burst_after_single_cells,
                           prob_next_burst_burst_cells=prob_burst_after_burst_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_burst_len_vs_preceding_silence, plot_kwargs,
                                xlabel='Prec. \nsilence (ms)', ylabel='Prob. next burst',
                                save_dir_img=os.path.join(save_dir_img, 'preceding_silence_vs_burst_prob.png'))
        pl.show()