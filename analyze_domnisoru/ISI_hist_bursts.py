from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.burst_len_vs_preceding_silence import get_n_spikes_per_event, \
    get_ISI_idx_per_event
pl.style.use('paper')


if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    AP_thresholds = {'s73_0004': -55, 's90_0006': -45, 's82_0002': -35, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    use_AP_max_idxs_domnisoru = True
    filter_long_ISIs = False
    max_ISI = 200
    burst_ISI = 8  # ms
    if filter_long_ISIs:
        save_dir_img = os.path.join(save_dir_img, 'cut_ISIs_at_'+str(max_ISI))
    save_dir_img = os.path.join(save_dir_img, cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # parameter
    bin_width = 1.0
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    # over cells
    mean_ISI_burst = np.zeros(len(cell_ids))
    std_ISI_burst = np.zeros(len(cell_ids))
    mean_ISI_doublet = np.zeros(len(cell_ids))
    std_ISI_doublet = np.zeros(len(cell_ids))
    mean_ISI_1st_triplets_and_more = np.zeros(len(cell_ids))
    std_ISI_1st_triplets_and_more = np.zeros(len(cell_ids))
    mean_ISI_2nd_triplets_and_more = np.zeros(len(cell_ids))
    std_ISI_2nd_triplets_and_more = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
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
        burst_ISI_indicator = np.concatenate((ISIs <= burst_ISI, np.array([False])))

        n_spikes_per_event = get_n_spikes_per_event(burst_ISI_indicator)
        ISI_idx_per_event = get_ISI_idx_per_event(burst_ISI_indicator)
        ISIs_doublets = ISIs[ISI_idx_per_event[n_spikes_per_event == 2]]
        mean_ISI_doublet[cell_idx] = np.mean(ISIs_doublets)
        std_ISI_doublet[cell_idx] = np.std(ISIs_doublets)

        ISIs_1st_triplets_and_more = ISIs[ISI_idx_per_event[n_spikes_per_event >= 3]]
        mean_ISI_1st_triplets_and_more[cell_idx] = np.mean(ISIs_1st_triplets_and_more)
        std_ISI_1st_triplets_and_more[cell_idx] = np.std(ISIs_1st_triplets_and_more)

        ISIs_2nd_triplets_and_more = ISIs[ISI_idx_per_event[n_spikes_per_event >= 3] + 1]
        mean_ISI_2nd_triplets_and_more[cell_idx] = np.mean(ISIs_2nd_triplets_and_more)
        std_ISI_2nd_triplets_and_more[cell_idx] = np.std(ISIs_2nd_triplets_and_more)

        ISIs_burst = ISIs[ISIs <= burst_ISI]
        mean_ISI_burst[cell_idx] = np.mean(ISIs_burst)
        std_ISI_burst[cell_idx] = np.std(ISIs_burst)

    fig, ax = pl.subplots()
    plot_with_markers(ax, std_ISI_doublet, mean_ISI_doublet, cell_ids, cell_type_dict)
    pl.xlabel('Std. doublet ISIs')
    pl.ylabel('Mean doublet ISIs')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'mean_vs_std_doublet_ISI.png'))

    fig, ax = pl.subplots()
    plot_with_markers(ax, std_ISI_burst, mean_ISI_burst, cell_ids, cell_type_dict)
    pl.xlabel('Std. burst ISIs')
    pl.ylabel('Mean burst ISIs')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'mean_vs_std_burst_ISI.png'))

    fig, ax = pl.subplots(1, 3, sharex='all', sharey='all')
    plot_with_markers(ax[0], std_ISI_doublet, mean_ISI_doublet, cell_ids, cell_type_dict)
    plot_with_markers(ax[1], std_ISI_1st_triplets_and_more, mean_ISI_1st_triplets_and_more, cell_ids, cell_type_dict)
    plot_with_markers(ax[2], std_ISI_2nd_triplets_and_more, mean_ISI_2nd_triplets_and_more, cell_ids, cell_type_dict)
    ax[0].set_title('1st ISI \ndoublets')
    ax[1].set_title('1st ISI \n$\geq$ 3 APs in burst')
    ax[2].set_title('2nd ISI \n$\geq$ 3 APs in burst')
    for i in range(3):
        ax[i].set_xlabel('Std. ISIs')
        ax[i].set_ylabel('Mean ISIs')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'mean_vs_std_diff_ISIs.png'))
    pl.show()