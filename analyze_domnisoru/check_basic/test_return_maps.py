from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from grid_cell_stimuli import get_AP_max_idxs
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from cell_characteristics import to_idx
pl.style.use('paper')

np.random.seed(1)


def generate_poisson_spike_train(firing_rate, dt, refractory_period=None):
    refractory_period_idx = to_idx(refractory_period, dt)
    refractory_counter = 0
    spike_train = np.zeros(len(firing_rate))
    for i in range(len(firing_rate)):
        rand = np.random.uniform(0, 1)
        if firing_rate[i] * np.exp(-refractory_counter) * dt/1000.0 > rand:
            spike_train[i] = 1
            refractory_counter = refractory_period_idx
        else:
            spike_train[i] = 0
            if refractory_counter >= 1:
                refractory_counter -= 1
    return spike_train


# generate APs
dt = 0.05
tstop = 50000
len_spike_train = to_idx(tstop, dt)

mode = 4

# 1: uniform
if mode == 1:
    firing_rate = np.ones(len_spike_train) * 10  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=3)

# 2: sine oscillation
if mode == 2:
    t = np.arange(0, tstop, dt)
    firing_rate = 2 * 40 * np.sin(8 * 2 * np.pi * t/1000.0)  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=3)

    # pl.figure()
    # pl.plot(t, firing_rate)

# 3: unregular oscillations -> strong edges
if mode == 3:
    freqs = np.linspace(2, 15, len_spike_train)
    t = np.arange(0, tstop, dt)
    firing_rate = 2 * 40 * np.sin(freqs * 2 * np.pi * t / 1000.)  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=3)

    pl.figure()
    pl.plot(t, firing_rate)

# 4: using the theta from the real data
if mode == 4:
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    #cell_id = 's119_0004'  # edge-acummulating
    #cell_id = 's120_0002'  # edge-acummulating
    #cell_id = 's76_0002'  # edge-acummulating
    #cell_id = 's81_0004'  # edge-avoiding
    cell_id = 's115_0030'  # edge-avoiding
    #cell_id = 's73_0004'  # strong theta
    theta = load_data(cell_id, ['fVm'], save_dir)['fVm']
    t = np.arange(0, tstop, dt)
    firing_rate = 40 * theta[:len_spike_train]  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=3)

    pl.figure()
    pl.plot(t, firing_rate)

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

# # c) unregular oscillations
# freq_smooth = np.linspace(10, 50, len_spike_train)
# t = np.arange(0, tstop, dt)
# prob = (np.sin(2*np.pi*freq_smooth*t/1000.) - 0.9999) * 10000
# spike_train3 = np.random.random(len_spike_train) < prob

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


# plot spike train
pl.figure()
pl.plot(np.arange(0, tstop, dt), spike_train)

# ISIs
AP_max_idxs = np.where(spike_train)[0]
ISIs = np.diff(AP_max_idxs) * dt

# 2d return
max_ISI = 200
pl.figure()
pl.title('mode %i' % mode)
pl.plot(ISIs[:-1], ISIs[1:], color='k', marker='o', linestyle='', markersize=2)
pl.xlabel('ISI[n] (ms)')
pl.ylabel('ISI[n+1] (ms)')
pl.xlim(0, max_ISI)
pl.ylim(0, max_ISI)
pl.tight_layout()
pl.show()