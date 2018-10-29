from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from cell_fitting.util import init_nan
from cell_characteristics import to_idx
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP'
    save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    do_detrend = False
    in_field = False
    out_field = False
    before_AP = 10
    after_AP = 25
    dt = 0.05
    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)
    DAP_deflections = init_nan(len(cell_ids))
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    folder_field = {(True, False): 'in_field', (False, True): 'out_field', (False, False): 'all'}
    save_dir_sta = os.path.join(save_dir_sta, folder_detrend[do_detrend], folder_field[(in_field, out_field)],
                                cell_type)
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], folder_field[(in_field, out_field)],
                                cell_type)

    # main
    sta_mean_cells = np.zeros((len(cell_ids), before_AP_idx + after_AP_idx + 1))
    sta_mean_good_APs_cells = np.zeros((len(cell_ids), before_AP_idx+after_AP_idx+1))
    sta_diff_good_APs_cells = np.zeros((len(cell_ids), before_AP_idx+after_AP_idx))
    for cell_idx, cell_id in enumerate(cell_ids):
        # load
        sta_mean_cells[cell_idx, :] = np.load(os.path.join(save_dir_sta, cell_id, 'sta_mean.npy'))
        sta_mean_good_APs_cells[cell_idx, :] = np.load(os.path.join(save_dir_img, cell_id, 'sta_mean.npy'))
        sta_diff_good_APs_cells[cell_idx] = np.diff(sta_mean_good_APs_cells[cell_idx])  # derivative

    # average over bursty/non-bursty
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]

    sta_bursty = np.nanmean(sta_mean_good_APs_cells[burst_label, :], 0)
    sta_nonbursty = np.nanmean(sta_mean_good_APs_cells[~burst_label, :], 0)
    std_bursty = np.nanstd(sta_mean_good_APs_cells[burst_label, :], 0)
    std_nonbursty = np.nanstd(sta_mean_good_APs_cells[~burst_label, :], 0)
    t_AP = np.arange(np.shape(sta_mean_good_APs_cells)[1]) * dt
    fig, ax = pl.subplots()
    ax.fill_between(t_AP, sta_bursty - std_bursty, sta_bursty + std_bursty, color='r', alpha=0.5)
    ax.plot(t_AP, sta_bursty, 'r', label='Bursty')
    ax.fill_between(t_AP, sta_nonbursty - std_nonbursty, sta_nonbursty + std_nonbursty, color='b', alpha=0.5)
    ax.plot(t_AP, sta_nonbursty, 'b', label='Non-bursty')
    pl.legend(handles=handles_bursty, loc='upper left')
    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')
    axins = inset_axes(ax, width='50%', height='50%', loc='upper right') # bbox_to_anchor=(0.7, 0.7, 1.0, 1.0)
    axins.fill_between(t_AP, sta_bursty - std_bursty, sta_bursty + std_bursty, color='r', alpha=0.5)
    axins.plot(t_AP, sta_bursty, 'r', label='Bursty')
    axins.fill_between(t_AP, sta_nonbursty - std_nonbursty, sta_nonbursty + std_nonbursty, color='b', alpha=0.5)
    axins.plot(t_AP, sta_nonbursty, 'b', label='Non-bursty')
    axins.set_ylim(-69, -53)
    axins.set_xlim(10, 25)
    axins.spines['top'].set_visible(True)
    axins.spines['right'].set_visible(True)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k", linewidth=0.7)
    pl.savefig(os.path.join(save_dir_img2, 'sta_bursty.png'))

    # plot
    sta_bursty = np.nanmean(sta_mean_cells[burst_label, :], 0)
    sta_nonbursty = np.nanmean(sta_mean_cells[~burst_label, :], 0)
    std_bursty = np.nanstd(sta_mean_cells[burst_label, :], 0)
    std_nonbursty = np.nanstd(sta_mean_cells[~burst_label, :], 0)
    t_AP = np.arange(np.shape(sta_mean_cells)[1]) * dt
    fig, ax = pl.subplots()
    ax.fill_between(t_AP, sta_bursty - std_bursty, sta_bursty + std_bursty, color='r', alpha=0.5)
    ax.plot(t_AP, sta_bursty, 'r', label='Bursty')
    ax.fill_between(t_AP, sta_nonbursty - std_nonbursty, sta_nonbursty + std_nonbursty, color='b', alpha=0.5)
    ax.plot(t_AP, sta_nonbursty, 'b', label='Non-bursty')
    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')
    pl.legend(handles=handles_bursty)
    pl.show()

    # # PCA
    # sta_cut = sta_mean_good_APs_cells[:, before_AP_idx + to_idx(1.0, dt):before_AP_idx + to_idx(5.0, dt)]
    # sta_not_nan_idx = ~np.isnan(sta_cut[:, 0])
    # sta_centered = sta_cut[sta_not_nan_idx, :] #\
    #               # - np.mean(sta_cut[sta_not_nan_idx, :], 0)
    # pca = PCA(n_components=2)
    # pca.fit(sta_centered)
    # transformed = pca.transform(sta_centered)
    # print('Explained variance: ', pca.explained_variance_)
    #
    # # k-means
    # n_clusters = 2
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(transformed)
    # labels = kmeans.labels_
    #
    # # plot
    # fig, ax = pl.subplots()
    # # for label in range(n_clusters):
    # #     ax.plot(transformed[labels==label, 0], transformed[labels==label, 1], 'o')
    # ax.plot(transformed[labels == 0, 0], transformed[labels == 0, 1], 'or')
    # ax.plot(transformed[labels == 1, 0], transformed[labels == 1, 1], 'ob')
    # # ax.plot(transformed[labels == 2, 0], transformed[labels == 2, 1], 'oy')
    # # ax.plot(transformed[labels == 3, 0], transformed[labels == 3, 1], 'og')
    # for i in range(sum(sta_not_nan_idx)):
    #     ax.annotate(cell_ids[sta_not_nan_idx][i], xy=(transformed[i, 0], transformed[i, 1]))
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    #
    # fig, ax = pl.subplots()
    # ax.plot(np.arange(np.shape(sta_centered)[1]) * dt, pca.components_[0, :])
    # ax.plot(np.arange(np.shape(sta_centered)[1]) * dt, pca.components_[1, :])
    #
    # fig, ax = pl.subplots()
    # for s in sta_centered:
    #     ax.plot(np.arange(np.shape(sta_centered)[1]) * dt, s)
    #
    # fig, ax = pl.subplots()
    # for s in sta_mean_good_APs_cells:
    #     ax.plot(np.arange(np.shape(sta_mean_good_APs_cells)[1]) * dt, s)
    # pl.show()