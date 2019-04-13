import numpy as np
import matplotlib.pyplot as pl
import os
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells_grid
from cell_fitting.util import init_nan
pl.style.use('paper_subplots')


def plot_sta_grid_on_ax(ax, cell_idx, subplot_idx, t_AP, sta_mean_cells, sta_std_cells, before_AP, after_AP,
                        bins, ISI_hist_cells, ylims=(None, None)):
    if subplot_idx == 0: # STA
        ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
        ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
                        sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='0.6')
        ax.set_ylim(*(-75, -45))
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP + 5, 10))
        ax.set_xticklabels([])
        if cell_idx == 0 or cell_idx == 9 or cell_idx == 18:
            ax.set_ylabel('$STA_V$ \n(mV)')
    elif subplot_idx == 1: # ISI
        bin_width = bins[1] - bins[0]
        ax.bar(bins[:-1], ISI_hist_cells[cell_idx] / (np.sum(ISI_hist_cells[cell_idx])*bin_width),
               bin_width, color='0.5')
        ax.set_ylim(0, 0.24)
        ax.set_xlim(-before_AP, after_AP)
        ax.set_xticks(np.arange(-before_AP, after_AP+5, 10))
        if cell_idx == 0 or cell_idx == 9 or cell_idx == 18:
            ax.set_ylabel('ISI hist. \n(norm.)')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    #save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/not_detrended/all'
    save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP/not_detrended/all'
    cell_type = 'grid_cells'
    save_dir_img = os.path.join(save_dir_sta, cell_type)
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    before_AP = 10
    after_AP = 25
    dt = 0.05
    max_ISI = 200
    bin_width = 1.0  # ms
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, 'cut_ISIs_at_'+str(max_ISI), cell_type,
                                               'ISI_hist_' + str(max_ISI) + '_' + str(bin_width) + '.npy'))
    bins = np.arange(0, max_ISI+bin_width, bin_width)

    # sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    # sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    # for cell_idx, cell_id in enumerate(cell_ids):
    #     save_dir_cell = os.path.join(save_dir_sta, cell_type, cell_id)
    sta_mean_cells = np.load(os.path.join(save_dir_sta, cell_type, 'sta_mean_'+str(before_AP)+'_'+str(after_AP)+'.npy'))
    sta_std_cells = np.load(os.path.join(save_dir_sta, cell_type, 'sta_std_'+str(before_AP)+'_'+str(after_AP)+'.npy'))
    t_AP = np.arange(-before_AP, after_AP+dt, dt)

    cell_ids_bursty = get_cell_ids_bursty()
    burst_label = np.array([True if cell_id in cell_ids_bursty else False for cell_id in cell_ids])
    colors_marker = np.zeros(len(burst_label), dtype=str)
    colors_marker[burst_label] = 'r'
    colors_marker[~burst_label] = 'b'

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
                                 xlabel='Time (ms)', n_subplots=2, colors_marker=colors_marker,
                                 save_dir_img=os.path.join(save_dir_img, 'sta_isi.png'))
    pl.show()