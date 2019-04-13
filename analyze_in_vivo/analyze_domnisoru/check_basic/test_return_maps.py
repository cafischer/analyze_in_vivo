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
tstop = 100000
len_spike_train = to_idx(tstop, dt)

mode = 5

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
    firing_rate = 80 * (np.sin(freqs * 2 * np.pi * t / 1000.) - 0.3)  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=2)

    pl.figure()
    pl.plot(t, firing_rate)

# 4: using theta or ramp from the real data
if mode == 4:
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    #cell_id = 's119_0004'  # edge-acummulating
    cell_id = 's120_0002'  # edge-acummulating
    #cell_id = 's76_0002'  # edge-acummulating
    #cell_id = 's81_0004'  # edge-avoiding
    #cell_id = 's115_0030'  # edge-avoiding
    #cell_id = 's73_0004'  # large theta
    theta = load_data(cell_id, ['fVm'], save_dir)['fVm']
    ramp = load_data(cell_id, ['dcVm_ljpc'], save_dir)['dcVm_ljpc']
    ramp += np.abs(np.min(ramp))
    t = np.arange(0, tstop, dt)
    firing_rate = 10 * theta[:len_spike_train] + ramp[:len_spike_train]  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=3)

    pl.figure()
    pl.plot(t, firing_rate, 'k')
    pl.plot(t, theta[:len_spike_train], 'b')
    pl.plot(t, ramp[:len_spike_train], 'r')

# 5: using Vm without spikes from the real data
if mode == 5:
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    #cell_id = 's119_0004'  # edge-acummulating
    #cell_id = 's120_0002'  # edge-acummulating
    cell_id = 's76_0002'  # edge-acummulating
    #cell_id = 's81_0004'  # edge-avoiding
    #cell_id = 's115_0030'  # edge-avoiding
    #cell_id = 's73_0004'  # large theta
    v_without_APs = load_data(cell_id, ['Vm_wo_spikes_ljpc'], save_dir)['Vm_wo_spikes_ljpc']
    v_without_APs += np.abs(np.min(v_without_APs))
    t = np.arange(0, tstop, dt)
    firing_rate = 3 * v_without_APs[:len_spike_train]  # Hz
    spike_train = generate_poisson_spike_train(firing_rate, dt, refractory_period=2)

    pl.figure()
    pl.plot(t, firing_rate, 'k')


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

# pl.figure()
# pl.title('mode %i' % mode)
# pl.loglog(ISIs[:-1], ISIs[1:], color='k', marker='o', linestyle='', markersize=2)
# pl.xlabel('ISI[n] (ms)')
# pl.ylabel('ISI[n+1] (ms)')
# pl.xlim(0, max_ISI)
# pl.ylim(0, max_ISI)
# pl.tight_layout()
pl.show()