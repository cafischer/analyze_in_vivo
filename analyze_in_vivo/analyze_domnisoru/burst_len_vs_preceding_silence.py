from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_groups
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper')


def get_n_spikes_per_event(burst_ISI_indicator):
    groups = np.split(burst_ISI_indicator, np.where(np.abs(np.diff(burst_ISI_indicator)) == 1)[0] + 1)
    n_spikes_per_event = []
    if False in groups[0]:
        n_spikes_per_event.extend(np.ones(len(groups[0])))
    else:
        n_spikes_per_event.append(len(groups[0]) + 1)
    for group in groups[1:]:
        if False in group:
            n_spikes_per_event.extend(np.ones(len(group)-1))  # -1 because 1st ISI belongs to the last burst AP
        else:
            n_spikes_per_event.append(len(group)+1)
    return np.array(n_spikes_per_event)


def get_ISI_idx_per_event(burst_ISI_indicator):
    ISI_idx_per_event = [0]
    counter = 1
    for i in range(1, len(burst_ISI_indicator)):
        if (burst_ISI_indicator[i-1] == 0 and burst_ISI_indicator[i] == 0) \
                or (burst_ISI_indicator[i-1] == 0 and burst_ISI_indicator[i] == 1):
            ISI_idx_per_event.append(counter)
        counter += 1
    return np.array(ISI_idx_per_event)


if __name__ == '__main__':
    # test
    burst_ISI_indicator = np.array([False, True, False, False, True, True, False, False])
    assert np.array_equal(get_ISI_idx_per_event(burst_ISI_indicator), np.array([0, 1, 3, 4, 7]))
    assert np.array_equal(get_n_spikes_per_event(burst_ISI_indicator), np.array([1, 2, 1, 3, 1]))

    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/Harris/burst_len_vs_preceding_silence'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
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

        if n_spikes_per_event[-1] == 1: # shorten by 1 to be able to index ISIs (just necessary if last spike is single)
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

        # # plot for testing
        # colors = ['b', 'm', 'r', 'orange', 'y']
        # pl.figure()
        # pl.plot(t, v, 'k')
        # for i, n_spikes in enumerate(n_spikes_variants):
        #     if len(t[AP_max_idxs[ISI_idx_per_event[n_spikes_per_event == n_spikes]]]) > 0:
        #         pl.plot([t[AP_max_idxs[ISI_idx_per_event[n_spikes_per_event == n_spikes]]],
        #              t[AP_max_idxs[ISI_idx_per_event[n_spikes_per_event == n_spikes]]]-ISIs[ISI_idx_per_event[n_spikes_per_event == n_spikes] - 1]],
        #             [-90, -90], color=colors[i], linewidth=3.0)
        # pl.show()

    # save and plot
    if cell_type == 'grid_cells':
        def plot_burst_len_vs_preceding_silence(ax, cell_idx, n_spikes_variants, med_preceding_silence_cells,
                                                std_preceding_silence_cells):
            ax.bar(n_spikes_variants, med_preceding_silence_cells[cell_idx],
                             yerr=std_preceding_silence_cells[cell_idx], color='0.5', capsize=2)
            ax.set_ylim(0, 500)
            ax.set_xticks(n_spikes_variants)
            ax.set_xticklabels(n_spikes_variants_labels)

        plot_kwargs = dict(n_spikes_variants=n_spikes_variants,
                           med_preceding_silence_cells=med_preceding_silence_cells,
                           std_preceding_silence_cells=std_preceding_silence_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_burst_len_vs_preceding_silence, plot_kwargs,
                                xlabel='Burst length', ylabel='Med. prec. \nsilence (ms)',
                                save_dir_img=os.path.join(save_dir_img, 'burst_len_vs_preceding_silence.png'))
        #pl.show()

        # plot averaged over cell_groups
        cell_groups = get_cell_groups()
        fig, ax = pl.subplots(1, len(cell_groups.keys()), figsize=(10, 5), sharey='all', sharex='all')
        for i, (cell_group_name, cell_group_ids) in enumerate(cell_groups.iteritems()):
            cell_idxs = np.array([cell_ids.index(cell_id) for cell_id in cell_group_ids])

            ax[i].set_title(cell_group_name)
            ax[i].bar(n_spikes_variants, np.nanmean(np.array(med_preceding_silence_cells)[cell_idxs], 0),
                   yerr=np.nanstd(np.array(std_preceding_silence_cells)[cell_idxs], 0), color='0.5', capsize=2)
            ax[i].set_ylim(0, 500)
            ax[i].set_xticks(n_spikes_variants)
            ax[i].set_xticklabels(n_spikes_variants_labels)
            ax[i].set_xlabel('Burst length')
        ax[0].set_ylabel('Med. prec. silence (ms)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'burst_len_vs_preceding_silence_cell_groups.png'))
        pl.show()