import numpy as np
import matplotlib.pyplot as pl
import os
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_for_all_grid_cells
from scipy.stats import gamma
from scipy.optimize import curve_fit
from cell_fitting.util import init_nan
pl.style.use('paper_subplots')


def plot_ISI_hist_gamma_on_ax(ax, cell_idx, ISI_hist_cells, max_ISI, bin_width, popt_cells, ylims=(None, None)):
    ISI_hist_norm = ISI_hist_cells[cell_idx] / (np.sum(ISI_hist_cells[cell_idx]) * bin_width)
    x_gamma = np.arange(0, max_ISI, 0.01)
    ax.bar(bins[:-1], ISI_hist_norm, bin_width, color='0.5')
    ax.plot(x_gamma, fit_fun(x_gamma, *popt_cells[cell_idx]), color='r', linewidth=1.5)
    ax.set_ylim(*ylims)


# def fit_fun(x, a1, scale1, a2, scale2, weight):
#     return gamma.pdf(x, a1, loc=0, scale=scale1) + weight * gamma.pdf(x, a2, loc=0, scale=scale2)

def fit_fun(x, a1, scale1, loc1, a2, scale2, loc2, weight):
    return gamma.pdf(x, a1, loc=loc1, scale=scale1) + weight * gamma.pdf(x, a2, loc=loc2, scale=scale2)


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    cell_type = 'grid_cells'
    save_dir_img = os.path.join(save_dir_ISI_hist, 'fit_gamma')
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_type_dict = get_celltype_dict(save_dir)
    n_trials = 500
    n_params = 7
    max_ISI = 50
    bin_width = 1.0  # ms
    ISI_hist_cells = np.load(os.path.join(save_dir_ISI_hist, 'cut_ISIs_at_'+str(max_ISI), cell_type,
                                               'ISI_hist_' + str(max_ISI) + '_' + str(bin_width) + '.npy'))
    bins = np.arange(0, max_ISI+bin_width, bin_width)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # bounds_min = np.array([0, 0, 0, 0, -1])
    # bounds_max = np.array([10, 10, 30, 10, 1])
    bounds_min = np.array([0, 0, 0, 0, 0, 0, -1])
    bounds_max = np.array([10, 10, 10, 30, 10, 10, 1])

    popt_cells = init_nan((len(cell_ids), n_params))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id
        ISI_hist_norm = ISI_hist_cells[cell_idx] / (np.sum(ISI_hist_cells[cell_idx]) * bin_width)

        rmse = 1000000
        for i in range(n_trials):
            try:
                p0 = bounds_min + (bounds_max - bounds_min) * np.random.rand(n_params)
                popt, _ = curve_fit(fit_fun, bins[:-1]+bin_width/2., ISI_hist_norm, p0=p0)
                rmse_new = np.sqrt(np.sum((fit_fun(bins[:-1]+bin_width/2., *popt) - ISI_hist_norm)**2))
                if rmse_new <= rmse:
                    popt_cells[cell_idx] = popt
                rmse = rmse_new
            except RuntimeError:
                continue
        print 'popt: ' + str(['%.1f' % p for p in popt_cells[cell_idx]])
        print 'rmse: %.2f' % rmse

    # plot
    plot_kwargs = dict(ISI_hist_cells=ISI_hist_cells,  max_ISI=max_ISI, bin_width=bin_width, popt_cells=popt_cells,
                       ylims=(0, 0.12))  # ylims=(0, 0.3)
    plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_ISI_hist_gamma_on_ax, plot_kwargs,
                            wspace=0.18, xlabel='ISI (ms)', ylabel='Rel. frequency',
                            save_dir_img=os.path.join(save_dir_img,
                                                      'ISI_hist_gamma2loc_' + str(max_ISI) + '_' + str(bin_width) + '.png'))
    pl.show()