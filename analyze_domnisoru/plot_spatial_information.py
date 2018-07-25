from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from sklearn import linear_model, decomposition
from scipy.stats import pearsonr
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

    spatial_info_skaggs = np.load(os.path.join(save_dir_img, 'spatial_info.npy'))
    MI_v = np.load(os.path.join(save_dir_img, 'MI_v.npy'))

    # MI(v) vs Skaggs
    regression = linear_model.LinearRegression()
    regression.fit(np.array([MI_v]).T, np.array([spatial_info_skaggs]).T)

    pca = decomposition.PCA(n_components=1)
    pca.fit(np.vstack([MI_v, spatial_info_skaggs]).T)

    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = ['s79_0003', 's104_0007', 's109_0002', 's110_0002', 's119_0004']
    fig, ax = pl.subplots()
    plot_with_markers(ax, MI_v, spatial_info_skaggs, cell_ids, cell_type_dict, theta_cells, DAP_cells)
    pl.plot(MI_v, regression.coef_[0, 0] * MI_v + regression.intercept_[0], 'r')
    ax.annotate('', pca.mean_, pca.mean_ + (pca.components_[0] * np.sqrt(pca.explained_variance_[0])),
                arrowprops=dict(arrowstyle='<-', color='k'))
    pl.ylabel('Skaggs, 1996')
    pl.xlabel('MI (mem. pot.)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'mi_vs_skaggs.png'))

    print 'Pearson: ', pearsonr(MI_v, spatial_info_skaggs)[0]
    fig, ax = pl.subplots()
    plot_with_markers(ax, MI_v, spatial_info_skaggs, cell_ids, cell_type_dict, theta_cells, DAP_cells)
    pl.plot(MI_v, regression.coef_[0, 0] * MI_v + regression.intercept_[0], 'r')
    pl.ylabel('Skaggs, 1996')
    pl.xlabel('MI (mem. pot.)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'mi_vs_skaggs_with_regression.png'))
    pl.show()
