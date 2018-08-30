from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.linear_model import Perceptron
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristic_correlations'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    save_dir_burst = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_spat_info = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_auto_corr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_DAP_cells()
    cell_type_dict = get_celltype_dict(save_dir)

    fraction_burst = np.load(os.path.join(save_dir_burst, cell_type, 'fraction_burst.npy'))
    spatial_info = np.load(os.path.join(save_dir_spat_info, cell_type, 'spatial_info.npy'))
    peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, cell_type, 'peak_ISI_hist.npy'))
    peak_auto_corr = np.load(os.path.join(save_dir_auto_corr, cell_type, 'peak_auto_corr_50.npy'))
    DAP_deflection = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_deflection.npy'))
    DAP_width = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_width.npy'))
    DAP_time = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_time.npy'))
    AP_width = np.load(os.path.join(save_dir_characteristics, cell_type, 'AP_width.npy'))
    AP_amp = np.load(os.path.join(save_dir_characteristics, cell_type, 'AP_amp.npy'))

    DAP_deflection[np.isnan(DAP_deflection)] = 0  # for plotting

    good_cell_indicator = AP_width <= 0.75
    DAP_deflection_good_cells = DAP_deflection[good_cell_indicator]
    DAP_time__good_cells = DAP_time[good_cell_indicator]
    fraction_burst_good_cells = fraction_burst[good_cell_indicator]
    #spatial_info_good_cells = spatial_info[good_cell_indicator]
    #peak_ISI_hist_good_cells = peak_ISI_hist[good_cell_indicator]
    #peak_auto_corr_good_cells = peak_auto_corr[good_cell_indicator]
    cell_ids_good_cells = np.array(cell_ids)[good_cell_indicator]

    # plots
    # pl.figure()
    # pl.plot(np.zeros(len(cell_ids_good_cells))[DAP_deflection_good_cells == 0],
    #         fraction_burst_good_cells[DAP_deflection_good_cells == 0], 'o', color='0.5')
    # pl.plot(np.ones(len(cell_ids_good_cells))[DAP_deflection_good_cells > 0],
    #         fraction_burst_good_cells[DAP_deflection_good_cells > 0], 'o', color='0.5')
    # for cell_idx in range(len(cell_ids_good_cells)):
    #     pl.annotate(cell_ids_good_cells[cell_idx], xy=((DAP_deflection_good_cells[cell_idx] > 0).astype(int),
    #                                                    fraction_burst_good_cells[cell_idx]))
    # pl.errorbar(-0.1, np.mean(fraction_burst_good_cells[DAP_deflection_good_cells == 0]),
    #         yerr=np.std(fraction_burst_good_cells[DAP_deflection_good_cells == 0]), color='k', capsize=2, marker='o')
    # pl.errorbar(0.9, np.mean(fraction_burst_good_cells[DAP_deflection_good_cells > 0]),
    #         yerr=np.std(fraction_burst_good_cells[DAP_deflection_good_cells > 0]), color='k', capsize=2, marker='o')
    # pl.xlim(-1, 2)
    # pl.xticks([0, 1], ['no', 'yes'])
    # pl.ylabel('Fraction ISI < 8 ms')
    # pl.xlabel('DAP')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_vs_frac_burst.png'))

    # pl.figure()
    # pl.plot(np.zeros(len(cell_ids_good_cells))[DAP_deflection_good_cells == 0],
    #         spatial_info_good_cells[DAP_deflection_good_cells == 0], 'o', color='0.5')
    # pl.plot(np.ones(len(cell_ids_good_cells))[DAP_deflection_good_cells > 0],
    #         spatial_info_good_cells[DAP_deflection_good_cells > 0], 'o', color='0.5')
    # for cell_idx in range(len(cell_ids_good_cells)):
    #     pl.annotate(cell_ids_good_cells[cell_idx], xy=((DAP_deflection_good_cells[cell_idx] > 0).astype(int),
    #                                                    spatial_info_good_cells[cell_idx]))
    # pl.errorbar(-0.1, np.mean(spatial_info_good_cells[DAP_deflection_good_cells == 0]),
    #             yerr=np.std(spatial_info_good_cells[DAP_deflection_good_cells == 0]), color='k', capsize=2, marker='o')
    # pl.errorbar(0.9, np.mean(spatial_info_good_cells[DAP_deflection_good_cells > 0]),
    #             yerr=np.std(spatial_info_good_cells[DAP_deflection_good_cells > 0]), color='k', capsize=2, marker='o')
    # pl.ylabel('Spatial information')
    # pl.xlabel('DAP')
    # pl.xlim(-1, 2)
    # pl.xticks([0, 1], ['no', 'yes'])
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_vs_spatial_info.png'))

    # peak_ISI_hist_good_cells = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist_good_cells])
    # pl.figure()
    # pl.plot(np.zeros(len(cell_ids_good_cells))[DAP_deflection_good_cells == 0],
    #         peak_ISI_hist_good_cells[DAP_deflection_good_cells == 0], 'o', color='0.5')
    # pl.plot(np.ones(len(cell_ids_good_cells))[DAP_deflection_good_cells > 0],
    #         peak_ISI_hist_good_cells[DAP_deflection_good_cells > 0], 'o', color='0.5')
    # for cell_idx in range(len(cell_ids_good_cells)):
    #     pl.annotate(cell_ids_good_cells[cell_idx], xy=((DAP_deflection_good_cells[cell_idx] > 0).astype(int),
    #                                                    peak_ISI_hist_good_cells[cell_idx]))
    # pl.errorbar(-0.1, np.mean(peak_ISI_hist_good_cells[DAP_deflection_good_cells == 0]),
    #             yerr=np.std(peak_ISI_hist_good_cells[DAP_deflection_good_cells == 0]), color='k', capsize=2, marker='o')
    # pl.errorbar(0.9, np.mean(peak_ISI_hist_good_cells[DAP_deflection_good_cells > 0]),
    #             yerr=np.std(peak_ISI_hist_good_cells[DAP_deflection_good_cells > 0]), color='k', capsize=2, marker='o')
    # pl.ylabel('Peak of ISI hist. (ms)')
    # pl.xlabel('DAP')
    # pl.xlim(-1, 2)
    # pl.xticks([0, 1], ['no', 'yes'])
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_vs_peak_ISI_hist.png'))

    # peak_ISI_hist = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist])  # set middle of bin as peak
    # fig, ax = pl.subplots()
    # plot_with_markers(ax, DAP_time, peak_ISI_hist, cell_ids, cell_type_dict,
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    # pl.plot(np.arange(0, 10), np.arange(0, 10), '0.5', linestyle='--')
    # # for cell_idx in range(len(cell_ids)):
    # #     pl.annotate(cell_ids[cell_idx], xy=(DAP_time[cell_idx], peak_ISI_hist[cell_idx]))
    # pl.xlim(0, 10)
    # pl.ylim(0, 10)
    # pl.ylabel('Peak of ISI hist. (ms)')
    # pl.xlabel('DAP time (ms)')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_time_vs_ISI_peak.png'))

    # pl.figure()
    # pl.plot(DAP_time, peak_auto_corr, 'o', color='0.5')
    # for cell_idx in range(len(cell_ids)):
    #     pl.annotate(cell_ids[cell_idx], xy=(DAP_time[cell_idx], peak_auto_corr[cell_idx]))
    # pl.xlim(0, 20)
    # pl.ylim(0, 20)
    # pl.ylabel('Peak of auto-correlation (ms)')
    # pl.xlabel('DAP time (ms)')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_time_vs_auto_corr_peak.png'))

    # fig, ax = pl.subplots()
    # plot_with_markers(ax, AP_amp, DAP_deflection, cell_ids, cell_type_dict, theta_cells=theta_cells,
    #                   DAP_cells=DAP_cells)
    # pl.ylabel('DAP deflection (mV)')
    # pl.xlabel('AP amplitude (mV)')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_AP_amp.png'))
    #
    # # print 'Low AP width: '
    # # print np.array(cell_ids)[AP_width < 0.7]
    # fig, ax = pl.subplots()
    # plot_with_markers(ax, AP_width, DAP_deflection, cell_ids, cell_type_dict, theta_cells=theta_cells,
    #                   DAP_cells=DAP_cells)
    # pl.ylabel('DAP deflection (mV)')
    # pl.xlabel('AP width (ms)')
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_AP_width.png'))

    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit(np.vstack((AP_width, AP_amp)).T).labels_.astype(bool)

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_with_markers(ax, AP_width[labels], AP_amp[labels], np.array(cell_ids)[labels], cell_type_dict, DAP_deflection[labels], 'r',
                      theta_cells, DAP_cells)
    plot_with_markers(ax, AP_width[~labels], AP_amp[~labels], np.array(cell_ids)[~labels], cell_type_dict, DAP_deflection[~labels], 'b',
                      theta_cells, DAP_cells)
    # ax.plot(AP_width[labels], AP_amp[labels], DAP_deflection[labels], 'or')
    # ax.plot(AP_width[~labels], AP_amp[~labels], DAP_deflection[~labels], 'ob')
    ax.set_xlabel('AP width (ms)')
    ax.set_ylabel('AP amp. (mV)')
    ax.set_zlabel('DAP deflection (mV)')
    ax.view_init(elev=28, azim=38)

    # legend
    fig_fake, ax_fake = pl.subplots()
    handles = [ax_fake.scatter(0, 0, marker='o', s=100, linewidths=0.8,
                               edgecolor='r', facecolor='None', label='Cluster 1'),
               ax_fake.scatter(0, 0, marker='o', s=100, linewidths=0.8,
                                    edgecolor='b', facecolor='None', label='Cluster 2')]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 0.1))
    pl.close(fig_fake)

    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_AP_width_vs_AP_amp.png'))
    print np.array(cell_ids)[labels]
    print np.array(cell_ids)[~labels]

    # SVM
    pl.close('all')
    X = np.vstack((AP_width, AP_amp)).T
    X = (X - np.mean(X, 0)) / np.std(X, 0)  # standardized
    clf = svm.SVC(kernel='linear', C=1.0)
    has_dap = DAP_deflection > 0
    clf.fit(X, has_dap.astype(int))
    #clf.fit(X, labels.astype(int))

    print('accuracy:', clf.score(X, labels))
    labels_predicted = clf.predict(X).astype(bool)
    w1, w2 = clf.coef_[0, :]

    fig, ax = pl.subplots()
    plot_with_markers(ax, X[:, 0], X[:, 1], np.array(cell_ids), cell_type_dict, edgecolor='k',
                      theta_cells=theta_cells, DAP_cells=DAP_cells)
    plot_with_markers(ax, X[:, 0][labels_predicted], X[:, 1][labels_predicted], np.array(cell_ids)[labels_predicted],
                      cell_type_dict, edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells)
    # plot_with_markers(ax, X[:, 0][has_dap], X[:, 1][has_dap], np.array(cell_ids)[has_dap],
    #                   cell_type_dict, edgecolor='y', theta_cells=theta_cells, DAP_cells=DAP_cells)
    xlim = pl.xlim()
    ylim = pl.ylim()
    pl.plot(np.arange(xlim[0], xlim[1], 0.1), -(w1 * np.arange(xlim[0], xlim[1], 0.1) + clf.intercept_[0]) / w2, 'k')
    ax.set_ylim(ylim)
    ax.set_xlabel('AP width (ms)')
    ax.set_ylabel('AP amp. (mV)')

    print np.array(cell_ids)[labels_predicted]
    print np.array(cell_ids)[~labels_predicted]

    # # perceptron
    # X = np.vstack((AP_width, AP_amp / 100.0)).T
    # X = (X - np.mean(X, 0)) / np.std(X, 0)  # standardized
    # clf = Perceptron(max_iter=1000, tol=1e-8, eta0=0.001)
    # clf.fit(X, (DAP_deflection > 0).astype(int))
    # #clf.fit(X, labels.astype(int))
    #
    # print clf.n_iter_
    # print('accuracy:', clf.score(X, labels))
    # labels_predicted = clf.predict(X).astype(bool)
    # w1, w2 = clf.coef_[0, :]
    #
    # fig, ax = pl.subplots()
    # plot_with_markers(ax, X[:, 0], X[:, 1], np.array(cell_ids), cell_type_dict, edgecolor='k',
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    # plot_with_markers(ax, X[:, 0][labels_predicted], X[:, 1][labels_predicted], np.array(cell_ids)[labels_predicted],
    #                   cell_type_dict, edgecolor='r',
    #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    # # plot_with_markers(ax, AP_width[labels], AP_amp[labels], np.array(cell_ids)[labels],
    # #                   cell_type_dict, edgecolor='y',
    # #                   theta_cells=theta_cells, DAP_cells=DAP_cells)
    # xlim = pl.xlim()
    # pl.plot(np.arange(xlim[0], xlim[1], 0.1), -(w1 * np.arange(xlim[0], xlim[1], 0.1) + clf.intercept_[0]) / w2, 'k')
    # ax.set_xlabel('AP width (ms)')
    # ax.set_ylabel('AP amp. (mV)')

    pl.show()