from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from grid_cell_stimuli.ISI_hist import get_ISIs
from cell_fitting.util import change_color_brightness
from matplotlib.colors import to_rgb
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/latuske'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']
    save_dir_img = os.path.join(save_dir_img, cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    ISI1st_mean_cells = np.zeros(len(cell_ids))
    ISI2nd_mean_cells = np.zeros(len(cell_ids))
    ISI_ratio = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        AP_max_idxs = data['spiketimes']

        # 1st and 2nd ISI in spike triplet (3 spikes within 50 ms, preceded by at least one spike by 50-200 ms)
        ISIs = get_ISIs(AP_max_idxs, t)

        # Latuske's method
        ISI1st = []
        ISI2nd = []
        for n_AP in range(1, len(AP_max_idxs)-1-1):  # -1 for looking ahead; -1 because ISIs are 1 shorter
            if 50 <= ISIs[n_AP-1]: # TODO 50 <= <= 200:
                if ISIs[n_AP] + ISIs[n_AP+1] <= 50 and ISIs[n_AP] > 8 and ISIs[n_AP+1] > 8:
                    ISI1st.append(ISIs[n_AP])
                    ISI2nd.append(ISIs[n_AP+1])

                    # pl.figure()
                    # pl.plot(t[AP_max_idxs[n_AP]:AP_max_idxs[n_AP + 2]] - t[AP_max_idxs[n_AP]],
                    #         v[AP_max_idxs[n_AP]:AP_max_idxs[n_AP + 2]], 'k')
                    # pl.show()
        ISI1st_mean_cells[cell_idx] = np.mean(ISI1st)
        ISI2nd_mean_cells[cell_idx] = np.mean(ISI2nd)
        ISI_ratio[cell_idx] = np.mean(np.array(ISI1st) / np.array(ISI2nd))  # mean ???

        # ISI1st = []
        # ISI2nd = []
        # for n_AP in range(1, len(AP_max_idxs) - 1 - 1):  # -1 for looking ahead; -1 because ISIs are 1 shorter
        #     if 100 <= ISIs[n_AP - 1]:
        #         if 8 < ISIs[n_AP] <= 50 and 8 < ISIs[n_AP + 1] <= 50:
        #             ISI1st.append(ISIs[n_AP])
        #             ISI2nd.append(ISIs[n_AP + 1])
        #
        #             pl.figure()
        #             pl.plot(t[AP_max_idxs[n_AP]:AP_max_idxs[n_AP+2]] - t[AP_max_idxs[n_AP]],
        #                     v[AP_max_idxs[n_AP]:AP_max_idxs[n_AP+2]], 'k')
        #             pl.show()
        # ISI1st_mean_cells[cell_idx] = np.mean(ISI1st)
        # ISI2nd_mean_cells[cell_idx] = np.mean(ISI2nd)
        # ISI_ratio[cell_idx] = np.mean(np.array(ISI1st) / np.array(ISI2nd))  # mean ???


    # plot (Fig. 3E in Latuske)
    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if cell_id in cell_ids_bursty else False for cell_id in cell_ids])

    fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(4, 5))
    ax1.bar(-0.05, np.mean(ISI1st_mean_cells[burst_label]), 0.15, color=change_color_brightness(to_rgb('r'), 30, 'brighter'))
    ax1.bar(0.25, np.mean(ISI2nd_mean_cells[burst_label]), 0.15, color=change_color_brightness(to_rgb('r'), 30, 'darker'))
    ax2.bar(-0.05, np.mean(ISI1st_mean_cells[~burst_label]), 0.15, color=change_color_brightness(to_rgb('b'), 30, 'brighter'))
    ax2.bar(0.25, np.mean(ISI2nd_mean_cells[~burst_label]), 0.15, color=change_color_brightness(to_rgb('b'), 30, 'darker'))
    ax1.set_ylim(0, 22)
    ax2.set_ylim(0, 22)
    ax1.set_xlim(-0.25, 0.45)
    ax2.set_xlim(-0.25, 0.45)
    ax1.set_ylabel('ISI (ms)')
    ax2.set_ylabel('')
    ax1.set_xlabel('Bursty')
    ax2.set_xlabel('Non-bursty')
    ax1.set_xticks([-0.05, 0.25])
    ax1.set_xticklabels(['1st ISI', '2nd ISI'])
    ax2.set_xticks([-0.05, 0.25])
    ax2.set_xticklabels(['1st ISI', '2nd ISI'])
    pl.tight_layout()
    #pl.savefig(os.path.join(save_dir_img, 'ISI1_ISI2_bursty_nonbursty.png'))

    fig, ax = pl.subplots(1, 1, figsize=(3, 5))
    ax.bar(0, np.mean(ISI_ratio[burst_label]), 0.15, color='r')
    ax.bar(0.2, np.mean(ISI_ratio[~burst_label]), 0.15, color='b')
    ax.axhline(1.0, 0, 1, color='0.5', linestyle='--')
    ax.set_ylim(0, None)
    ax.set_xlim(-0.2, 0.4)
    ax.set_ylabel('1st ISI / 2nd ISI')
    ax.set_xlabel('')
    ax.set_xticks([0, 0.2])
    ax.set_xticklabels(['Bursty', 'Non-bursty'], fontsize=10)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'ISI1_ratio_bursty_nonbursty_without_bursts.png'))

    pl.show()