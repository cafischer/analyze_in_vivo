from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.signal import resample
from grid_cell_stimuli import compute_fft
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
pl.style.use('paper')



if __name__ == '__main__':
    # Note: no all APs are captured as the spikes are so small and noise is high and depth of hyperpolarization
    # between successive spikes varies
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/frequency_analysis'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'fVm']
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type)

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    power_cells = np.zeros(len(cell_ids), dtype=object)
    freqs_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        theta = data['fVm']

        # downsampling
        v_r, t_r = resample(v, 2**16, t)

        # power spectrum
        fft_v, freqs_cells[cell_idx] = compute_fft(v_r, (t_r[1]-t_r[0])/1000.0)
        power_cells[cell_idx] = np.abs(fft_v)**2

        # pl.figure()
        # pl.plot(freqs_cells[cell_idx], power_cells[cell_idx], 'k')
        # pl.ylim(0, np.max(power_cells[cell_idx][freqs_cells[cell_idx]>5]))
        # pl.xlim(0, None)
        # pl.show()

    # plots
    if cell_type == 'grid_cells':
        def plot(ax, cell_idx, power_cells, freqs_cells):
            ax.plot(freqs_cells[cell_idx], power_cells[cell_idx], 'k')
            ax.set_ylim(0, 2.5e8)
            ax.set_xlim(0, 25)

        plot_kwargs = dict(power_cells=power_cells, freqs_cells=freqs_cells)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot, plot_kwargs,
                                xlabel='Freq. (Hz)', ylabel='Power spectrum',
                                save_dir_img=os.path.join(save_dir_img, 'power_spectrum.png'))
        pl.show()