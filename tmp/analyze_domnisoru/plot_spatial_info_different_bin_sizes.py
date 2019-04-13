from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from scipy.optimize import curve_fit
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir_firing_rate = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field/vel_thresh_1'
    save_dir_rec_info = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/recording_info'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)
    save_dir_img = os.path.join(save_dir_img, cell_type)

    bin_sizes = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    spatial_info_skaggs = np.zeros((len(cell_ids), len(bin_sizes)))
    MI_v = np.zeros((len(cell_ids), len(bin_sizes)))
    for bins_size_idx, bin_size in enumerate(bin_sizes):
        spatial_info_skaggs[:, bins_size_idx] = np.load(os.path.join(save_dir_img, 'bin_size_'+str(bin_size), 'spatial_info.npy'))
        MI_v[:, bins_size_idx] = np.load(os.path.join(save_dir_img, 'bin_size_'+str(bin_size), 'MI_v.npy'))


    def plot_vs_bin_size(ax, cell_idx, spatial_info, bin_sizes):
        ax.plot(bin_sizes, spatial_info[cell_idx, :], 'ok', markersize=4.0)
        ax.set_xlim(0, bin_sizes[-1]+1)

        # exponential fits
        bin_sizes = np.array(bin_sizes)
        def exp_fun(x, a, b):
            return a * np.exp(-b*x)
        p_opt = curve_fit(exp_fun, bin_sizes, spatial_info[cell_idx], p0=[1.0, 0.5])[0]
        #print p_opt
        ax.plot(bin_sizes, exp_fun(bin_sizes, p_opt[0], p_opt[1]), 'r', label='tau: %.2f' % p_opt[1])
        ax.legend(fontsize=8)

    plot_kwargs = dict(spatial_info=spatial_info_skaggs, bin_sizes=bin_sizes)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_vs_bin_size, plot_kwargs,
                            xlabel='Bin size \n(cm)', ylabel='Spatial info. \n(Skaggs, 1996)',
                            save_dir_img=os.path.join(save_dir_img, 'bin_size_vs_skaggs.png'))

    plot_kwargs = dict(spatial_info=MI_v, bin_sizes=bin_sizes)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_vs_bin_size, plot_kwargs,
                            xlabel='Bin size \n(cm)', ylabel='MI (mem. pot.)',
                            save_dir_img=os.path.join(save_dir_img, 'bin_size_vs_MI_v.png'))
    pl.show()