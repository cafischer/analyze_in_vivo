from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import os


save_dir = '../results/schmidthieber/spike_shape'

spike_characteristics_mat = np.load(os.path.join(save_dir, 'spike_characteristics_mat.npy'))
spike_characteristics_mat_vitro = np.load(os.path.join(save_dir, 'spike_characteristics_mat_vitro.npy'))
spike_characteristics_names = np.load(os.path.join(save_dir, 'spike_characteristics_names.npy'))
AP_matrix_vitro = np.load(os.path.join(save_dir, 'AP_mat_vitro.npy'))
AP_matrix = np.load(os.path.join(save_dir, 'AP_mat.npy'))
t_vitro = np.load(os.path.join(save_dir, 't_vitro.npy'))
t = np.load(os.path.join(save_dir, 't.npy'))


# plot distributions
for i in range(np.shape(spike_characteristics_mat_vitro)[1] - 2):
    min_val = min(np.min(spike_characteristics_mat_vitro[:, i]), np.min(spike_characteristics_mat[:, i]))
    max_val = max(np.max(spike_characteristics_mat_vitro[:, i]), np.max(spike_characteristics_mat[:, i]))
    bins = np.linspace(min_val, max_val, 100)
    if spike_characteristics_names[i] == 'AP_width':
        bins = np.arange(min_val, max_val, 0.05)

    hist_v, bins = np.histogram(spike_characteristics_mat_vitro[:, i], bins=bins)
    hist, bins = np.histogram(spike_characteristics_mat[:, i], bins=bins)

    ylim = 65
    dylim = 5
    if spike_characteristics_names[i] == 'AP_width':
        ylim = int(np.ceil(np.max(hist_v))) + 5
        dylim = 10

    fig, ax1 = pl.subplots()
    ax1.bar(bins[:-1], hist_v, width=bins[1] - bins[0], color='b', alpha=0.5, label='in vitro')
    ax1.set_ylim(0, ylim)
    ax1.set_yticks(range(0, ylim, dylim))
    ax2 = ax1.twinx()
    ax2.bar(bins[:-1], hist, width=bins[1] - bins[0], color='r', alpha=0.5, label='in vivo')
    ax2.set_ylim(0, 4)
    ax2.set_yticks(range(0, 4))
    ax1.set_xlabel(spike_characteristics_names[i], fontsize=16)
    ax1.set_ylabel('Count in vitro', fontsize=16)
    ax2.set_ylabel('Count in vivo', fontsize=16)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=16)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'hist_' + spike_characteristics_names[i] + '.png'))
    pl.show()


# plot spikes
pl.figure()
cmap = matplotlib.cm.get_cmap('Blues')
for i, v in enumerate(AP_matrix_vitro):
    pl.plot(t_vitro, v, color=cmap(i/np.shape(AP_matrix_vitro)[0]),
            label='in vitro' if i==len(AP_matrix_vitro)-1 else '')
for i, v in enumerate(AP_matrix):
    pl.plot(t, v, 'r', label='in vivo'if i==len(AP_matrix)-1 else '')
pl.xlabel('Time (ms)', fontsize=16)
pl.ylabel('Membrane Potential (mV)', fontsize=16)
pl.legend(fontsize=16)
pl.savefig(os.path.join(save_dir, 'spike_shapes.png'))
pl.show()

#
# # without two noise traces
# for i in range(np.shape(spike_characteristics_mat_vitro)[1] - 2):
#     hist_v, bin_e_v = np.histogram(spike_characteristics_mat_vitro[:, i], bins=60)
#     hist, bin_e = np.histogram(spike_characteristics_mat[2:, i], bins=bin_e_v)
#
#     fig, ax1 = pl.subplots()
#     ax1.bar(bin_e_v[:-1], hist_v, width=bin_e_v[1]-bin_e_v[0], color='b', alpha=0.5, label='in vitro')
#     ax2 = ax1.twinx()
#     ax2.bar(bin_e[:-1], hist, width=bin_e[1]-bin_e[0], color='r', alpha=0.5, label='in vivo')
#     pl.legend(fontsize=16)
#     ax1.set_xlabel(spike_characteristics_names[i], fontsize=16)
#     ax1.set_ylabel('Count', fontsize=16)
#     ax2.set_ylabel('Count', fontsize=16)
#     pl.savefig(os.path.join(save_dir, 'hist_select_'+spike_characteristics_names[i]+'.png'))
#     pl.show()
#
# pl.figure()
# cmap = matplotlib.cm.get_cmap('Blues')
# for i, v in enumerate(AP_matrix_vitro):
#     pl.plot(t_vitro, v, color=cmap(i/np.shape(AP_matrix_vitro)[0]), label='in vitro' if i==len(AP_matrix_vitro)-1 else '')
# for i, v in enumerate(AP_matrix[2:, :]):
#     pl.plot(t, v, 'r', label='in vivo'if i==len(AP_matrix)-1 else '')
# pl.xlabel('Time (ms)', fontsize=16)
# pl.ylabel('Membrane Potential (mV)', fontsize=16)
# pl.legend(fontsize=16)
# pl.savefig(os.path.join(save_dir, 'spike_shapes_select.png'))
pl.show()
