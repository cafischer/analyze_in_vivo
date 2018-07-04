from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from cell_characteristics import to_idx
pl.style.use('paper')


# generate APs
dt = 0.05
tstop = 50000
len_spike_train = to_idx(tstop, dt)

# # a) being rather randomly
# spike_train1 = np.random.random(len_spike_train) < 0.005
#
# b) coming in blocks
# no_spikes = True
# old_idx = 0
# spike_train2 = np.zeros(len_spike_train)
# while old_idx < len_spike_train:
#     block_len = int(np.round(np.random.normal(to_idx(500, dt), to_idx(20, dt))))
#     if block_len <= 0:
#         block_len = 100
#     if old_idx + block_len >= len_spike_train:
#         break
#     if no_spikes:
#         spike_train2[old_idx:old_idx+block_len] = np.random.random(block_len) < 0.0001
#         no_spikes = False
#     else:
#         spike_train2[old_idx:old_idx + block_len] = np.random.random(block_len) < 0.001
#         no_spikes = True
#     old_idx = old_idx + block_len
# for i in range(len(spike_train2)):
#     if spike_train2[i] == 1:
#         spike_train2[i+1:i+100] = 0

# c) unregular oscillations
freq_smooth = np.linspace(10, 50, len_spike_train)
t = np.arange(0, tstop, dt)
prob = (np.sin(2*np.pi*freq_smooth*t/1000.) - 0.9999) * 10000
spike_train3 = np.random.random(len_spike_train) < prob

# # unregular oscillations less strong edges
# freq_smooth = np.linspace(3, 6, len_spike_train)
# # freq_smooth = freq_smooth[len_spike_train:2 * len_spike_train]
# t = np.arange(0, tstop, dt)
# prob = (np.sin(2*np.pi*freq_smooth*t/1000.) + 1) / 1000.
# spike_train3 = np.random.random(len_spike_train) < prob

# # unregular oscillations strong edges
# freq_smooth = np.linspace(2, 15, len_spike_train)
# t = np.arange(0, tstop, dt)
# prob = (np.sin(2*np.pi*freq_smooth*t/1000.) - 0.3) / 200.
# spike_train3 = np.random.random(len_spike_train) < prob

# d) increasing spike prob.
# spike_prob = np.linspace(0.00001, 0.001, len_spike_train)
# spike_train2 = np.random.random(len_spike_train) < spike_prob

# pl.figure()
# pl.plot(np.arange(0, tstop, dt), spike_train1)
pl.figure()
pl.plot(np.arange(0, tstop, dt), spike_train2)
pl.figure()
pl.plot(np.arange(0, tstop, dt), spike_train3)
pl.plot(t, prob)
#pl.show()

# ISIs
# AP_max_idxs1 = np.where(spike_train1)[0]
AP_max_idxs2 = np.where(spike_train2)[0]
AP_max_idxs3 = np.where(spike_train3)[0]
# ISIs1 = np.diff(AP_max_idxs1) * dt
ISIs2 = np.diff(AP_max_idxs2) * dt
ISIs3 = np.diff(AP_max_idxs3) * dt

# 2d return
max_ISI = 200
# pl.figure()
# pl.plot(ISIs1[:-1], ISIs1[1:], color='k', marker='o', linestyle='', markersize=6)
# pl.xlabel('ISI[n] (ms)')
# pl.ylabel('ISI[n+1] (ms)')
# pl.xlim(0, max_ISI)
# pl.ylim(0, max_ISI)
# pl.tight_layout()

pl.figure()
pl.title('2')
pl.plot(ISIs2[:-1], ISIs2[1:], color='k', marker='o', linestyle='', markersize=6)
pl.xlabel('ISI[n] (ms)')
pl.ylabel('ISI[n+1] (ms)')
pl.xlim(0, max_ISI)
pl.ylim(0, max_ISI)
pl.tight_layout()
#pl.show()

pl.figure()
pl.title('3')
pl.plot(ISIs3[:-1], ISIs3[1:], color='k', marker='o', linestyle='', markersize=6)
pl.xlabel('ISI[n] (ms)')
pl.ylabel('ISI[n+1] (ms)')
pl.xlim(0, max_ISI)
pl.ylim(0, max_ISI)
pl.tight_layout()
pl.show()