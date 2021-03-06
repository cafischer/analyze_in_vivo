import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import get_celltype_dict, load_cell_ids, get_label_burstgroups, get_cell_ids_burstgroups
from analyze_in_vivo.analyze_domnisoru.plot_utils import get_cell_id_with_marker, plot_with_markers
from scipy.stats import ttest_ind, pearsonr, median_test
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img_paper = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/paper'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_characteristics = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
    save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_bootstrap = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bootstrap/'

    cell_type_dict = get_celltype_dict(save_dir)
    max_ISI = 200
    bin_width = 1  # ms
    sigma_smooth = 1  # ms
    #before_AP = 25
    #after_AP = 25
    #t_vref = 10
    #dt = 0.05
    #AP_criterion = {'AP_amp_and_width': (40, 1)}

    before_AP = 25
    after_AP = 25
    t_vref = 10
    dt = 0.05
    AP_criterion = {'AP_amp_and_width': (40, 1)}

    # load stuff
    grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_burstgroups()['B+D']
    DAP_label = get_label_burstgroups(save_dir)['B+D']

    folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
             + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    save_dir_DAP_times = os.path.join(save_dir_characteristics, folder)
    save_dir_ISI_hist = os.path.join(save_dir_ISI_hist)
    DAP_time = np.load(os.path.join(save_dir_DAP_times, 'DAP_time.npy'))[DAP_label]

    if sigma_smooth is not None:
        ISI_peak = np.load(os.path.join(save_dir_ISI_hist, 'sigma_smooth_' + str(sigma_smooth),
                                        'peak_ISI_hist.npy'))[DAP_label]
        ISI_peak = np.array(ISI_peak, dtype=float)
    else:
        folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
        ISI_peak = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy'))[DAP_label]
        ISI_peak = np.array([(p[0] + p[1]) / 2. for p in ISI_peak])  # set middle of bin as peak

    # test
    _, p, med, _ = median_test(DAP_time, ISI_peak)
    print 'p-value: %.5f' % p
    print 'median: %.2f' % med

    # plot correlation DAP-time and peak ISI-hist
    fig, ax = pl.subplots()
    ax.plot(np.arange(0, 10), np.arange(0, 10), '0.5', linestyle='--')
    #ax.fill_between(np.arange(0, 10), np.arange(0, 10)-1, np.arange(0, 10)+1, color='0.7')
    plot_with_markers(ax, DAP_time, ISI_peak, grid_cells[DAP_label], cell_type_dict,
                      theta_cells=theta_cells, DAP_cells=DAP_cells)
    #ax.set_xlim(0, 7)
    #ax.set_ylim(0, 7)
    ax.set_xticks(np.arange(0, 8, 2))
    ax.set_yticks(np.arange(0, 8, 2))
    ax.set_aspect('equal', adjustable='box-forced')
    ax.set_ylabel('Peak of ISI hist. (ms)')
    ax.set_xlabel('Time$_{\mathrm{AP-DAP}}$ (ms)')
    pl.tight_layout()
    #pl.savefig(os.path.join(save_dir_ISI_hist, folder, 'dap_time_vs_ISI_peak.png'))
    pl.show()


    # folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
    #               + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    # DAP_time_mean = np.load(os.path.join(save_dir_bootstrap, folder_name, 'mean_DAP_time.npy'))
    # DAP_time_se = np.load(os.path.join(save_dir_bootstrap, folder_name, 'se_DAP_time.npy'))
    #
    # if sigma_smooth is not None:
    #     folder_name = 'max_ISI_' + str(max_ISI) + '_sigma_smooth_' + str(sigma_smooth)
    # else:
    #     folder_name = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
    # ISI_peak_mean = np.load(os.path.join(save_dir_bootstrap, folder_name, 'mean_ISI_peak.npy'))
    # ISI_peak_se = np.load(os.path.join(save_dir_bootstrap, folder_name, 'se_ISI_peak.npy'))

    # # plot for paper
    # corr, p_val = pearsonr(DAP_time, ISI_peak)
    # _, p_val_t = ttest_ind(DAP_time, ISI_peak)
    # print 'corr (pearson): %.2f' % corr
    # print 'p (pearson): %.2f' % p_val
    # print 'p (t-test): %.2f' % p_val_t
    #
    # f, ax = pl.subplots()
    # max_val = max(np.max(DAP_time_mean), np.max(ISI_peak_mean))
    # ax.plot([0, max_val], [0, max_val], '0.5', linestyle='--')
    # #handles = plot_with_markers(ax, DAP_time_mean, ISI_peak_mean, grid_cells[burst_label],
    # #                            cell_type_dict, theta_cells=theta_cells, edgecolor='k', legend=False)
    # ax.errorbar(DAP_time_mean, ISI_peak, xerr=DAP_time_se, linestyle='', capsize=2, color='k', marker='o', markersize=3)  # yerr=ISI_peak_se,
    # ax.set_ylabel('Peak of ISI hist. (ms)')
    # ax.set_xlabel('Time$_{AP-DAP}$ (ms)')
    # ax.set_xlim(0, 7.5)
    # ax.set_ylim(0, 7.5)
    # #ax.legend(handles=handles, loc='lower right')
    # #for i in range(len(DAP_cells)):
    # #    ax.annotate(DAP_cells[i], xy=(DAP_time_mean[i]+0.1, ISI_peak_mean[i]+0.35), fontsize=7)
    # pl.tight_layout()
    # if sigma_smooth is not None:
    #     pl.savefig(os.path.join(save_dir_img_paper, 'dap_time_vs_ISI_peak_kde.png'))
    # else:
    #     pl.savefig(os.path.join(save_dir_img_paper, 'dap_time_vs_ISI_peak_binned.png'))
    # pl.show()
