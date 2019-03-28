from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_latuske import load_ISIs
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from analyze_in_vivo.analyze_domnisoru.isi import plot_ISI_return_map
#pl.style.use('paper_subplots')


if __name__ == '__main__':
    #save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'

    save_dir_img = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'

    ISIs_cells = load_ISIs()

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
        # pl.figure()
        # pl.plot(ISIs[:-1], ISIs[1:], color='0.5', marker='o', linestyle='', markersize=3)
        # pl.xlabel('ISI[n] (ms)')
        # pl.ylabel('ISI[n+1] (ms)')
        # pl.xlim(0, max_ISI)
        # pl.ylim(0, max_ISI)
        # pl.tight_layout()
        # pl.show()
        # pl.close('all')

    # save
    np.save(os.path.join(save_dir_img, 'fraction_ISI_or_ISI_next_burst.npy'), fraction_ISI_or_ISI_next_burst)

    # plot
    for n in range(int(np.ceil(len(ISIs_cells)/27.))):
        end = (n+1)*27
        if end >= len(ISIs_cells):
            end = len(ISIs_cells)
        plot_kwargs = dict(ISIs_per_cell=ISIs_cells[n*27:end], max_ISI=max_ISI)
        cell_ids = [str(i) for i in range(len(ISIs_cells))]
        cell_type_dict = {str(i): 'not known' for i in cell_ids}
        plot_for_all_grid_cells(cell_ids[n*27:end], cell_type_dict, plot_ISI_return_map, plot_kwargs,
                                    xlabel='ISI[n] (ms)', ylabel='ISI[n+1] (ms)', legend=False,
                                    save_dir_img=os.path.join(save_dir_img, 'ISI_return_map_'+str(n)+'.png'))
    pl.show()