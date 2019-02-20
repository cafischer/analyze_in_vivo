import numpy as np
import matplotlib.pyplot as pl
import os
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells_grid
from cell_fitting.util import init_nan
pl.style.use('paper_subplots')


def plot_sta_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, before_AP, after_AP,
                        bins, ISI_hist_cells, ylims=(None, None)):
    if subplot_idx == 0: # STA
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
        ax.set_ylim(*ylims)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP + 5, 10))
    elif subplot_idx == 1: # ISI
        bin_width = bins[1] - bins[0]
        ax.bar(bins[:-1], ISI_hist_cells[cell_idx] / (np.sum(ISI_hist_cells[cell_idx])*bin_width),
               bin_width, color='0.5')
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP+5, 10))


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/not_detrended/all'
    cell_type = 'grid_cells'
    save_dir_img = os.path.join(save_dir_ISI_hist, 'fit_gamma')
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    before_AP = 10
    after_AP = 25
    dt = 0.05
    max_ISI = after_AP
    bin_width = 1.0  # ms
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, 'cut_ISIs_at_'+str(max_ISI), cell_type,
                                               'ISI_hist_' + str(max_ISI) + '_' + str(bin_width) + '.npy'))
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        save_dir_cell = os.path.join(save_dir_sta, cell_type, cell_id)
        sta_mean_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
        sta_std_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
    t_AP = np.arange(-before_AP, after_AP+dt, dt)


    # plot
    plot_kwargs = dict(t_AP=t_AP,
                       sta_mean_cells=sta_mean_cells,
                       sta_std_cells=sta_std_cells,
                       before_AP=before_AP,
                       after_AP=after_AP,
                       bins=bins,
                       ISI_hist_cells=ISI_hist_cells,
                       ylims=(-75, -45)  # (-75, -50)
                       )
    plot_for_all_grid_cells_grid(cell_ids, get_celltype_dict(save_dir), plot_sta_grid_on_ax, plot_kwargs,
                                 xlabel='Time (ms)', ylabel='Mem. pot. \n(mV)', n_subplots=2,
                                 save_dir_img=os.path.join(save_dir_img, 'sta_selected_and_all_APs.png'))