from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from grid_cell_stimuli.ISI_hist import get_ISIs
from sklearn import linear_model
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/velocity'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_type_dict = get_celltype_dict(save_dir)
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']
    use_AP_max_idxs_domnisoru = True

    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # over cells
    avg_velocity_cells = np.zeros(len(cell_ids), dtype=object)
    inv_ISIs_cells = np.zeros(len(cell_ids), dtype=object)

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']

        # instantaneous firing rate
        ISIs = get_ISIs(AP_max_idxs, t)
        inv_ISIs = 1. / (ISIs / 1000.)

        # velocity per ISI
        avg_velocity = np.zeros(len(inv_ISIs))
        for i, (j1, j2) in enumerate(zip(AP_max_idxs[:-1], AP_max_idxs[1:])):
            avg_velocity[i] = np.mean(velocity[j1:j2])

        inv_ISIs_cells[cell_idx] = inv_ISIs
        avg_velocity_cells[cell_idx] = avg_velocity

    def plot_velocity_vs_firing_rate(ax, cell_idx, avg_velocity_cells, inv_ISIs_cells, regression=False):
        ax.plot(avg_velocity_cells[cell_idx], inv_ISIs_cells[cell_idx], 'o', color='0.5', alpha=0.5, markersize=2)

        if regression:
            regression = linear_model.LinearRegression()
            regression.fit(np.array([avg_velocity_cells[cell_idx]]).T, np.array([inv_ISIs_cells[cell_idx]]).T)
            ax.plot(avg_velocity_cells[cell_idx],
                    regression.coef_[0, 0] * avg_velocity_cells[cell_idx] + regression.intercept_[0], 'r')

    plot_kwargs = dict(avg_velocity_cells=avg_velocity_cells, inv_ISIs_cells=inv_ISIs_cells)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_velocity_vs_firing_rate, plot_kwargs,
                            xlabel='Velocity \n(cm/sec)', ylabel='Inst. firing rate (Hz)',
                            save_dir_img=os.path.join(save_dir_img, 'velocity_vs_firing_rate.png'))

    plot_kwargs = dict(avg_velocity_cells=avg_velocity_cells, inv_ISIs_cells=inv_ISIs_cells, regression=True)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_velocity_vs_firing_rate, plot_kwargs,
                            xlabel='Velocity \n(cm/sec)', ylabel='Inst. firing rate (Hz)',
                            save_dir_img=os.path.join(save_dir_img, 'velocity_vs_firing_rate_with_regression.png'))
    pl.show()