from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_latuske import load_ISIs
#pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'

    #save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'

    ISIs_cells = load_ISIs()

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    firing_rate = np.zeros(len(ISIs_cells))

    for cell_idx in range(len(ISIs_cells)):
        # compute firing rate
        len_recording_approx = np.cumsum(ISIs_cells[cell_idx])[-1]
        firing_rate[cell_idx] = (len(ISIs_cells[cell_idx]) + 1) / (len_recording_approx / 1000.)

    # save
    np.save(os.path.join(save_dir_img, 'firing_rate.npy'), firing_rate)