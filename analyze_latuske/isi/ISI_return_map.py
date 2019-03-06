from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_latuske import load_ISIs
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'
    ISIs_cells = load_ISIs(save_dir='/home/cf/Phd/programming/data/Caro/grid_cells_withfields_vt_0.pkl')

    max_ISI = 200  # None if you want to take all ISIs
    ISI_burst = 8  # ms
    bin_width = 1  # ms
    steps = np.arange(0, max_ISI + bin_width, bin_width)

    folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
    save_dir_img = os.path.join(save_dir_img, folder)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    fraction_ISI_or_ISI_next_burst = np.zeros(len(ISIs_cells))

    for cell_idx in range(len(ISIs_cells)):
        print cell_idx

        # ISIs
        ISIs = ISIs_cells[cell_idx]
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]

        fraction_ISI_or_ISI_next_burst[cell_idx] = float(sum(np.logical_or(ISIs[:-1] < ISI_burst,
                                                                           ISIs[1:] < ISI_burst))) / len(ISIs[1:])

        # plot
        # 2d return
        pl.figure()
        pl.plot(ISIs[:-1], ISIs[1:], color='0.5', marker='o', linestyle='', markersize=3)
        pl.xlabel('ISI[n] (ms)')
        pl.ylabel('ISI[n+1] (ms)')
        pl.xlim(0, max_ISI)
        pl.ylim(0, max_ISI)
        pl.tight_layout()
        pl.show()
        pl.close('all')

    # save
    np.save(os.path.join(save_dir_img, 'fraction_ISI_or_ISI_next_burst.npy'), fraction_ISI_or_ISI_next_burst)
