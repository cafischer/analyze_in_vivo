from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype, get_track_len, \
    get_cell_ids_bursty, get_celltype_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from grid_cell_stimuli import get_AP_max_idxs
import matplotlib.gridspec as gridspec
from scipy.stats import median_test
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/recording_info'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes', 'Y_cm']

    # parameters
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    len_recording = np.zeros(len(cell_ids))
    n_APs = np.zeros(len(cell_ids), dtype=int)
    n_runs = np.zeros(len(cell_ids), dtype=int)
    avg_firing_rate = np.zeros(len(cell_ids))
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        position = data['Y_cm']

        # get APs
        AP_max_idxs = data['spiketimes']

        # length of recording, total number of spikes, average firing rate
        len_recording[cell_idx] = t[-1] / 1000.0 / 60.0  # min
        n_APs[cell_idx] = len(AP_max_idxs)
        n_runs[cell_idx] = np.sum(np.diff(position) < -get_track_len(cell_id) / 2.) + 1  # start at 0 + # resets
        avg_firing_rate[cell_idx] = n_APs[cell_idx] / (len_recording[cell_idx] * 60.0)

    np.save(os.path.join(save_dir_img, 'avg_firing_rate.npy'), avg_firing_rate)
    np.save(os.path.join(save_dir_img, 'n_APs.npy'), n_APs)
    np.save(os.path.join(save_dir_img, 'n_runs.npy'), n_runs)

    # plot avg. firing rate in bursty and non-bursty neurons
    cell_type_dict = get_celltype_dict(save_dir)
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])

    stat, p, m, table = median_test(avg_firing_rate[burst_label], avg_firing_rate[~burst_label])
    print 'p-val: ', p  # if p is small, reject H0 that medians are the same

    fig, ax = pl.subplots()
    plot_with_markers(ax, -1 * np.ones(np.sum(burst_label)), avg_firing_rate[burst_label],
                      np.array(cell_ids)[burst_label],
                      cell_type_dict, edgecolor='r')
    plot_with_markers(ax, 1 * np.ones(np.sum(~burst_label)), avg_firing_rate[~burst_label],
                      np.array(cell_ids)[~burst_label],
                      cell_type_dict, edgecolor='b')
    ax.boxplot(avg_firing_rate[burst_label], positions=[-0.5])
    ax.boxplot(avg_firing_rate[~burst_label], positions=[1.5])
    # ax.errorbar(-0.5, np.mean(avg_firing_rate[burst_label]), yerr=np.std(avg_firing_rate[burst_label]),
    #             marker='o', capsize=2, color='r')
    # ax.errorbar(1.5, np.mean(avg_firing_rate[~burst_label]), yerr=np.std(avg_firing_rate[~burst_label]),
    #             marker='o', capsize=2, color='b')
    ax.set_xlim(-1.5, 2)
    ax.set_xticks([])
    pl.show()

    #
    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.65, hspace=0.43)

        cell_idx = 0
        for cell_idx in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[cell_idx], wspace=4.0)
            ax1 = pl.Subplot(fig, inner[0])
            ax2 = pl.Subplot(fig, inner[1])
            ax3 = pl.Subplot(fig, inner[2])
            if cell_idx < len(cell_ids):
                if get_celltype(cell_ids[cell_idx], save_dir) == 'stellate':
                    ax2.set_title(cell_ids[cell_idx] + ' ' + u'\u2605', fontsize=12)
                elif get_celltype(cell_ids[cell_idx], save_dir) == 'pyramidal':
                    ax2.set_title(cell_ids[cell_idx] + ' ' + u'\u25B4', fontsize=12)
                else:
                    ax2.set_title(cell_ids[cell_idx], fontsize=12)

                ax1.bar(0, len_recording[cell_idx], 0.9, color='0.5')
                ax2.bar(0, n_APs[cell_idx], 0.9, color='0.5')
                ax3.bar(0, avg_firing_rate[cell_idx], 0.9, color='0.5')
                #ax3.bar(0.5, number_spikes[cell_idx] / (len_recording[cell_idx] * 60.0), 0.9, color='r')

                ax1.set_xlim(-1, 1)
                ax2.set_xlim(-1, 1)
                ax3.set_xlim(-1, 1)
                ymax1 = np.round(np.max(len_recording), -1)
                ymax2 = np.round(np.max(n_APs), -1)
                ymax3 = np.round(np.max(avg_firing_rate), 0)
                ax1.set_ylim(0, ymax1)
                ax2.set_ylim(0, ymax2)
                ax3.set_ylim(0, ymax3)
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax1.set_yticks([0, np.round(ymax1 / 2.0, 0), ymax1])
                ax1.set_yticklabels([])
                ax2.set_yticks([0, np.round(ymax2 / 2.0, 0), ymax2])
                ax2.set_yticklabels([])
                ax3.set_yticks([0, np.round(ymax3 / 2.0, 0), ymax3])
                ax3.set_yticklabels([])
                if cell_idx >= (n_rows - 1) * n_columns:
                    ax1.set_xlabel('Dur. \nrec. \n(min)', fontsize=10)
                    ax2.set_xlabel('#APs', fontsize=10)
                    ax3.set_xlabel('Avg. \nf. rate \n(Hz)', fontsize=10)
                if cell_idx % n_columns == 0:
                    ax1.set_yticks([0, np.round(ymax1/2.0, 0), ymax1])
                    ax1.set_yticklabels(['0', '', '%i' % int(ymax1)], fontsize=8)
                    ax2.set_yticks([0, np.round(ymax2/2.0, 0), ymax2])
                    ax2.set_yticklabels(['0', '', '%i' % int(ymax2)], fontsize=8)
                    ax3.set_yticks([0, np.round(ymax3/2.0, 0), ymax3])
                    ax3.set_yticklabels(['0', '', '%i' % int(ymax3)], fontsize=8)
                fig.add_subplot(ax1)
                fig.add_subplot(ax2)
                fig.add_subplot(ax3)
                cell_idx += 1
        pl.subplots_adjust(left=0.03, right=0.98, top=0.96, bottom=0.07)
        pl.savefig(os.path.join(save_dir_img, 'recording_info.png'))
        pl.show()