from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from grid_cell_stimuli import get_AP_max_idxs
import matplotlib.gridspec as gridspec
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/recording_info'
    save_dir_firing_rate = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  #'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)

    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    len_recording = np.zeros(len(cell_ids))
    number_spikes = np.zeros(len(cell_ids), dtype=int)
    avg_firing_rate = np.zeros(len(cell_ids))
    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]

        # get APs
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)

        # length of recording, total number of spikes, average firing rate
        len_recording[i] = t[-1] / 1000.0 / 60.0  # min
        number_spikes[i] = len(AP_max_idxs)
        avg_firing_rate[i] = np.load(os.path.join(save_dir_firing_rate, cell_type, cell_id, 'avg_firing_rate.npy'))


    pl.close('all')
    if cell_type == 'grid_cells':
        n_rows = 3
        n_columns = 9
        fig = pl.figure(figsize=(14, 8.5))
        outer = gridspec.GridSpec(n_rows, n_columns, wspace=0.65, hspace=0.43)

        cell_idx = 0
        for i in range(n_rows * n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[i], wspace=4.0)
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
                ax2.bar(0, number_spikes[cell_idx], 0.9, color='0.5')
                ax3.bar(0, avg_firing_rate[cell_idx], 0.9, color='0.5')
                #ax3.bar(0.5, number_spikes[cell_idx] / (len_recording[cell_idx] * 60.0), 0.9, color='r')

                ax1.set_xlim(-1, 1)
                ax2.set_xlim(-1, 1)
                ax3.set_xlim(-1, 1)
                ymax1 = np.round(np.max(len_recording), -1)
                ymax2 = np.round(np.max(number_spikes), -1)
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
                if i >= (n_rows - 1) * n_columns:
                    ax1.set_xlabel('Dur. \nrec. \n(min)', fontsize=10)
                    ax2.set_xlabel('#APs', fontsize=10)
                    ax3.set_xlabel('Avg. \nf. rate \n(Hz)', fontsize=10)
                if i % n_columns == 0:
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