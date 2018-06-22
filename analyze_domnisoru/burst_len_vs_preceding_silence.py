from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from cell_fitting.util import init_nan
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_start_end_group_of_ones
pl.style.use('paper')


def get_n_spikes_per_event(burst_ISI_indicator):
    groups = np.split(burst_ISI_indicator, np.where(np.abs(np.diff(burst_ISI_indicator)) == 1)[0] + 1)
    n_spikes_per_event = []
    for group in groups:
        if False in group:
            n_spikes_per_event.extend(np.ones(len(group)))
        else:
            n_spikes_per_event.append(len(group)+1)
    return np.array(n_spikes_per_event)


def get_ISI_idx_per_event(burst_ISI_indicator):
    counter = 0
    ISI_idx_per_event = []
    for i in range(len(burst_ISI_indicator)):
        if burst_ISI_indicator[i] == 0 or (burst_ISI_indicator[i-1] == 0 and burst_ISI_indicator[i] == 1):
            ISI_idx_per_event.append(counter)
        counter += 1
    return np.array(ISI_idx_per_event)


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    max_ISI = 200
    ISI_burst = 8  # ms
    n_spikes_variants = [1, 2, 3, 4, 5]
    n_spikes_variants_labels = ['1', '2', '3', '4', '$\geq5$']

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    med_preceding_silence_cells = [0] * len(cell_ids)
    std_preceding_silence_cells = [0] * len(cell_ids)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # compute median preceding silence
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        ISIs = get_ISIs(AP_max_idxs, t)
        burst_ISI_indicator = np.concatenate((ISIs <= ISI_burst, np.array([False])))
        n_spikes_per_event = get_n_spikes_per_event(burst_ISI_indicator)
        ISI_idx_per_event = get_ISI_idx_per_event(burst_ISI_indicator)

        if n_spikes_per_event[-1] == 1:  # shorten by 1 to be able to index ISIs (just neccesary if last spike is single)
            n_spikes_per_event = n_spikes_per_event[:-1]
            ISI_idx_per_event = ISI_idx_per_event[:-1]

        med_preceding_silence = np.zeros(len(n_spikes_variants))
        std_preceding_silence = np.zeros(len(n_spikes_variants))
        for i, n_spikes in enumerate(n_spikes_variants):
            if n_spikes == 5:
                med_preceding_silence[i] = np.median(ISIs[ISI_idx_per_event[n_spikes_per_event >= n_spikes] - 1])
                std_preceding_silence[i] = np.std(ISIs[ISI_idx_per_event[n_spikes_per_event >= n_spikes] - 1])
            else:
                med_preceding_silence[i] = np.median(ISIs[ISI_idx_per_event[n_spikes_per_event == n_spikes] - 1])
                std_preceding_silence[i] = np.std(ISIs[ISI_idx_per_event[n_spikes_per_event == n_spikes] - 1])
        med_preceding_silence_cells[cell_idx] = med_preceding_silence
        std_preceding_silence_cells[cell_idx] = std_preceding_silence

    # save and plot
    pl.close('all')
    if cell_type == 'grid_cells':
        # plot all return maps
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

                    axes[i1, i2].bar(n_spikes_variants, med_preceding_silence_cells[cell_idx],
                                     yerr=std_preceding_silence_cells[cell_idx], color='0.5', capsize=2)

                    axes[i1, i2].set_ylim(0, 500)
                    axes[i1, i2].set_xticks(n_spikes_variants)
                    if i1 == (n_rows - 1):
                        axes[i1, i2].set_xticklabels(n_spikes_variants_labels)
                    if i2 == 0:
                        axes[i1, i2].set_ylabel('Med. prec. \nsilence (ms)')
                else:
                    axes[i1, i2].spines['left'].set_visible(False)
                    axes[i1, i2].spines['bottom'].set_visible(False)
                    axes[i1, i2].set_xticks([])
                    axes[i1, i2].set_yticks([])
                cell_idx += 1
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'burst_len_vs_preceding_silence.png'))
        pl.show()