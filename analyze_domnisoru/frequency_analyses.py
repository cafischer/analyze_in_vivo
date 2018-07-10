from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli.theta_envelope import compute_envelope, plot_envelope
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
    mean_amp_theta = np.zeros(len(cell_ids))
    std_amp_theta = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        theta = data['fVm']

        # Hilbert transform theta
        theta_envelope = compute_envelope(theta)

        # plot_envelope(theta, theta_envelope, t)
        # pl.show()

        # mean and std of theta envelope (= amplitude of theta oscillations)
        mean_amp_theta[cell_idx] = np.mean(theta_envelope)
        std_amp_theta[cell_idx] = np.std(theta_envelope)   #/ np.sqrt(len(theta_envelope))


    # plots
    if cell_type == 'grid_cells':
        def plot(ax, cell_idx, mean_amp_theta, sem_amp_theta):
            ax.bar(0.5, mean_amp_theta[cell_idx], 0.4, yerr=sem_amp_theta[cell_idx], capsize=2, color='0.5')
            ax.set_xlim(0, 1)
            ax.set_xticks([])

        plot_kwargs = dict(mean_amp_theta=mean_amp_theta, sem_amp_theta=std_amp_theta)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot, plot_kwargs,
                                xlabel='', ylabel='Mean theta amp.',
                                save_dir_img=os.path.join(save_dir_img, 'mean_amp_theta.png'))