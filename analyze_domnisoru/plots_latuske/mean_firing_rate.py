from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
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
    len_recording = np.zeros(len(cell_ids))
    firing_rate = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # firing rate
        AP_max_idxs = data['spiketimes']
        len_recording[cell_idx] = t[-1]
        firing_rate[cell_idx] = len(AP_max_idxs) / (len_recording[cell_idx] / 1000.0)

    # save
    # np.save(os.path.join(save_dir_img, 'firing_rate.npy'), firing_rate)

    # plot (Fig. 3D in Latuske)
    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if cell_id in cell_ids_bursty else False for cell_id in cell_ids])

    pl.figure(figsize=(4, 5))
    bplot = pl.boxplot([firing_rate[burst_label], firing_rate[~burst_label]], patch_artist=True,
                       labels=['Bursty', 'Non-bursty'])
    edge_color = 'k'
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        pl.setp(bplot[element], color=edge_color)
    bplot['boxes'][0].set_facecolor('r')
    bplot['boxes'][1].set_facecolor('b')
    pl.ylim(0, None)
    pl.ylabel('Firing rate (Hz)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'firingrate_bursty_nonbursty.png'))

    pl.show()