from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import os
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, get_cell_ids_DAP_cells, load_cell_ids
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers

pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_sta = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/not_detrended/all/grid_cells'
    save_dir_sta_good_APs = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP/not_detrended/all/grid_cells'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    cell_type = 'DAP_cells'
    cell_ids, _ = get_cell_ids_DAP_cells()
    cell_type_dict = get_celltype_dict(save_dir)

    # parameters
    use_AP_max_idxs_domnisoru = True
    param_list = ['Vm_ljpc', 'spiketimes']
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    sta_mean_cells = np.zeros(len(cell_ids), dtype=object)
    sta_std_cells = np.zeros(len(cell_ids), dtype=object)
    v_hist_cells = np.zeros(len(cell_ids), dtype=object)
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        save_dir_cell = os.path.join(save_dir_sta, cell_id)

        sta_mean_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
        sta_std_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
        v_hist_cells[cell_idx] = np.load(os.path.join(save_dir_cell, 'v_hist.npy'))
        t_AP = np.load(os.path.join(save_dir_cell, 't_AP.npy'))
        bins_v = np.load(os.path.join(save_dir_cell, 'bins_v.npy'))
        # sta_mean_cells[cell_idx], sta_std_cells[cell_idx], v_hist_cells[cell_idx], t_AP = get_sta_for_cell_id(cell_id,
        #                                                                                                      param_list,
        #                                                                                                      save_dir)


    # goodness of recording
    # ax = pl.subplot(outer[:, :3], projection='3d')
    # grid_cells = load_cell_ids(save_dir, 'grid_cells')
    # theta_cells = load_cell_ids(save_dir, 'giant_theta')
    # DAP_cells = get_cell_ids_DAP_cells()
    # DAP_deflection = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'DAP_deflection.npy'))
    # DAP_deflection[np.isnan(DAP_deflection)] = 0
    # AP_width = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'AP_width.npy'))
    # AP_amp = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'AP_amp.npy'))
    # plot_with_markers(ax, AP_width, AP_amp, np.array(grid_cells), cell_type_dict, DAP_deflection, 'k',
    #                   theta_cells, DAP_cells)
    # ax.set_xlabel('AP width (ms)')
    # ax.set_ylabel('AP amp. (mV)')
    # ax.set_zlabel('DAP deflection (mV)')
    # ax.view_init(azim=45, elev=20)

    grid_cells = load_cell_ids(save_dir, 'grid_cells')
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()
    AP_width = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'AP_width.npy'))
    AP_amp = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'AP_amp.npy'))
    DAP_deflection = np.load(os.path.join(save_dir_characteristics, 'grid_cells', 'DAP_deflection.npy'))
    DAP_deflection[np.isnan(DAP_deflection)] = 0

    # preparation
    # X = np.vstack((AP_width, AP_amp)).T
    # mean = np.mean(X, 0)
    # std = np.std(X, 0)
    # X = (X - np.mean(X, 0)) / np.std(X, 0)  # z-standardized
    has_DAP = DAP_deflection > 0

    # SVM
    # clf = svm.SVC(kernel='linear', C=10.0)
    # clf.fit(X, has_DAP.astype(int))
    # labels_predicted = clf.predict(X).astype(bool)
    # w0 = clf.intercept_[0]
    # w1, w2 = clf.coef_[0, :]

    # clf = svm.SVC(kernel='linear', C=1.0)
    # clf.fit(np.array([AP_width]).T, has_DAP.astype(int))
    # labels_predicted_width = clf.predict(np.array([AP_width]).T).astype(bool)
    # threshold_AP_width = -clf.intercept_[0] / clf.coef_[0, 0]
    #
    # clf = svm.SVC(kernel='linear', C=1.0)
    # clf.fit(np.array([AP_amp]).T, has_DAP.astype(int))
    # labels_predicted_amp = clf.predict(np.array([AP_amp]).T).astype(bool)
    # threshold_AP_amp = -clf.intercept_[0] / clf.coef_[0, 0]
    #
    # labels_predicted = np.logical_and(labels_predicted_amp, labels_predicted_width)

    # Perceptron
    # perceptron = Perceptron(max_iter=1000, tol=1, class_weight={1: 1, 0: 1})
    # perceptron.fit(X, has_DAP.astype(int))
    # labels_predicted = perceptron.predict(X).astype(bool)
    # w0 = perceptron.intercept_
    # w1, w2 = perceptron.coef_[0, :]
    # print perceptron._max_iter

    # conservative choice
    max_AP_width_DAP = np.max(AP_width[has_DAP])
    AP_width_next = np.min(AP_width[~has_DAP][AP_width[~has_DAP] > max_AP_width_DAP])
    threshold_AP_width = np.round(max_AP_width_DAP + (AP_width_next - max_AP_width_DAP) / 2., 2)
    min_AP_amp_DAP = np.min(AP_amp[has_DAP])
    AP_amp_next = np.max(AP_amp[~has_DAP][AP_amp[~has_DAP] < min_AP_amp_DAP])
    threshold_AP_amp = np.round(AP_amp_next + (min_AP_amp_DAP - AP_amp_next) / 2., 1)
    print 'threshold_AP_width: ', threshold_AP_width
    print 'threshold_AP_amp: ', threshold_AP_amp
    labels_predicted = np.array([w <= threshold_AP_width and a >= threshold_AP_amp for w, a in zip(AP_width, AP_amp)])


    # # plot
    # fig = pl.figure(figsize=(10, 6))
    # n_rows, n_columns = 2, 5
    # outer = gridspec.GridSpec(n_rows, n_columns)
    # ax = pl.subplot(outer[:, :3])
    # fig.add_subplot(ax)
    # plot_with_markers(ax, AP_width[labels_predicted], AP_amp[labels_predicted], np.array(grid_cells)[labels_predicted],
    #                   cell_type_dict, edgecolor='r',
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional)
    # plot_with_markers(ax, AP_width[~labels_predicted], AP_amp[~labels_predicted], np.array(grid_cells)[~labels_predicted],
    #                   cell_type_dict, edgecolor='b',
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells, DAP_cells_additional=DAP_cells_additional)
    # # plot_with_markers(ax, AP_width[has_DAP], AP_amp[has_DAP], np.array(grid_cells)[has_DAP],
    # #                   cell_type_dict, edgecolor='y',
    # #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    # ax.set_xlim(0.5, None)
    # xlim = pl.xlim()
    # ylim = pl.ylim()
    # # x = np.arange(xlim[0], xlim[1]+0.1, 0.1)
    # # x = (x - mean[0]) / std[0]
    # # y = -(w1 * x + w0) / w2
    # # x = x * std[0] + mean[0]
    # # y = y * std[1] + mean[1]
    # # pl.plot(x, y, 'k')
    # pl.axvline(threshold_AP_width)
    # pl.axhline(threshold_AP_amp)
    # ax.set_xlabel('AP width (ms)')
    # ax.set_ylabel('AP amp. (mV)')
    # ax.set_ylim(ylim)
    # ax.legend(handles=[Line2D([0], [0], color='r', label='Good rec.'), Line2D([0], [0], color='b', label='Bad rec.')],
    #           loc='lower left')
    # ax.text(-0.12, 1.0, 'A', transform=ax.transAxes, size=18, weight='bold')
    #
    # # example 1 select good APs
    # cell_id = 's73_0004'
    # save_dir_cell = os.path.join(save_dir_sta, cell_id)
    # sta_mean = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
    # sta_std = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
    # v_hist = np.load(os.path.join(save_dir_cell, 'v_hist.npy'))
    # t_AP = np.load(os.path.join(save_dir_cell, 't_AP.npy'))
    # save_dir_cell = os.path.join(save_dir_sta_good_APs, cell_id)
    # sta_mean_good_APs = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
    # sta_std_good_APs = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
    #
    # ax1 = pl.subplot(outer[0, 3])
    # fig.add_subplot(ax1)
    # ax1.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    # ax1.annotate('all APs', xy=(t_AP[0], 15), textcoords='data',
    #             horizontalalignment='left', verticalalignment='top', fontsize=9)
    # plot_sta(ax1, 0, t_AP, [sta_mean], [sta_std])
    # ax1.set_xlabel('Time (ms)')
    # ax1.set_ylabel('Mem. pot. (mV)')
    # ax1.set_ylim(-75, 15)
    #
    # ax2 = pl.subplot(outer[0, 4])
    # ax2.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    # ax2.annotate('selected APs', xy=(t_AP[0], 15), textcoords='data',
    #             horizontalalignment='left', verticalalignment='top', fontsize=9)
    # plot_sta(ax2, 0, t_AP, [sta_mean_good_APs], [sta_std_good_APs])
    # ax2.set_xlabel('Time (ms)')
    # ax2.set_ylabel('Mem. pot. (mV)')
    # ax2.set_ylim(-75, 15)
    # ax1.text(-0.6, 1.0, 'B', transform=ax1.transAxes, size=18, weight='bold')
    #
    # # example 2 select good APs
    # cell_id = 's85_0007'
    # save_dir_cell = os.path.join(save_dir_sta, cell_id)
    # sta_mean = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
    # sta_std = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
    # v_hist = np.load(os.path.join(save_dir_cell, 'v_hist.npy'))
    # t_AP = np.load(os.path.join(save_dir_cell, 't_AP.npy'))
    # save_dir_cell = os.path.join(save_dir_sta_good_APs, cell_id)
    # sta_mean_good_APs = np.load(os.path.join(save_dir_cell, 'sta_mean.npy'))
    # sta_std_good_APs = np.load(os.path.join(save_dir_cell, 'sta_std.npy'))
    #
    # ax1 = pl.subplot(outer[1, 3])
    # fig.add_subplot(ax1)
    # ax1.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    # ax1.annotate('all APs', xy=(t_AP[0], 15), textcoords='data',
    #              horizontalalignment='left', verticalalignment='top', fontsize=9)
    # plot_sta(ax1, 0, t_AP, [sta_mean], [sta_std])
    # ax1.set_xlabel('Time (ms)')
    # ax1.set_ylabel('Mem. pot. (mV)')
    # ax1.set_ylim(-75, 15)
    #
    # ax2 = pl.subplot(outer[1, 4])
    # ax2.set_title(get_cell_id_with_marker(cell_id, cell_type_dict))
    # ax2.annotate('selected APs', xy=(t_AP[0], 15), textcoords='data',
    #              horizontalalignment='left', verticalalignment='top', fontsize=9)
    # plot_sta(ax2, 0, t_AP, [sta_mean_good_APs], [sta_std_good_APs])
    # ax2.set_xlabel('Time (ms)')
    # ax2.set_ylabel('Mem. pot. (mV)')
    # ax2.set_ylim(-75, 15)
    #
    # ax1.text(-0.6, 1.0, 'C', transform=ax1.transAxes, size=18, weight='bold')
    #
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'good_recordings_old.png'))

    # new plot
    fig, ax = pl.subplots()
    fig.add_subplot(ax)
    plot_with_markers(ax, AP_width[labels_predicted], AP_amp[labels_predicted], np.array(grid_cells)[labels_predicted],
                      cell_type_dict, edgecolor='#A11E22', theta_cells=theta_cells, DAP_cells=DAP_cells,
                      DAP_cells_additional=DAP_cells_additional, legend=False)
    handles = plot_with_markers(ax, AP_width[~labels_predicted],
                                AP_amp[~labels_predicted], np.array(grid_cells)[~labels_predicted],
                                cell_type_dict, edgecolor='#EBA631', theta_cells=theta_cells, DAP_cells=DAP_cells,
                                DAP_cells_additional=DAP_cells_additional, legend=False)
    # for i, cell_id in enumerate(grid_cells):
    #     ax.annotate(cell_id, xy=(AP_width[i], AP_amp[i]))

    # plot_with_markers(ax, AP_width[has_DAP], AP_amp[has_DAP], np.array(grid_cells)[has_DAP],
    #                   cell_type_dict, edgecolor='y',
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    ax.set_xlim(0.5, None)
    pl.axvline(threshold_AP_width)
    pl.axhline(threshold_AP_amp)
    ax.set_xlabel('AP width (ms)')
    ax.set_ylabel('AP amp. (mV)')
    ax.set_ylim(15, 83)
    handles_extra = [Patch(color='#A11E22', label='Good rec.'), Patch(color='#EBA631', label='Bad rec.')]
    ax.legend(handles=handles+handles_extra,
              loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'good_recordings.png'))

    pl.show()