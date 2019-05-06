from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_cell_ids_bursty
from cell_fitting.util import init_nan
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    before_AP = 10
    after_AP = 25
    t_vref = 5
    dt = 0.05
    AP_criterion = {'None': None}

    do_detrend = False
    in_field = False
    out_field = False
    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)
    DAP_deflections = init_nan(len(cell_ids))
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_sta = os.path.join(save_dir_sta, folder_detrend[do_detrend])

    # load
    folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
             + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    sta_mean_cells = np.load(os.path.join(save_dir_sta, folder, 'sta_mean.npy'))

    # average over bursty/non-bursty
    burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])
    handles_bursty = [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]

    sta_bursty = np.mean(sta_mean_cells[burst_label], 0)
    sta_nonbursty = np.mean(sta_mean_cells[~burst_label], 0)
    std_bursty = np.std(sta_mean_cells[burst_label], 0)
    std_nonbursty = np.std(sta_mean_cells[~burst_label], 0)
    t_AP = np.arange(len(sta_mean_cells[0])) * dt

    fig, ax = pl.subplots()
    ax.fill_between(t_AP, sta_bursty - std_bursty, sta_bursty + std_bursty, color='r', alpha=0.5)
    ax.plot(t_AP, sta_bursty, 'r', label='Bursty')
    ax.fill_between(t_AP, sta_nonbursty - std_nonbursty, sta_nonbursty + std_nonbursty, color='b', alpha=0.5)
    ax.plot(t_AP, sta_nonbursty, 'b', label='Non-bursty')
    pl.legend(handles=handles_bursty, loc='upper left')
    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(0, 35)
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
    pl.savefig(os.path.join(save_dir_sta, 'sta_avg_bursty_nonbursty.png'))

    # plot
    fig, ax = pl.subplots()
    ax.fill_between(t_AP, sta_bursty - std_bursty, sta_bursty + std_bursty, color='r', alpha=0.5)
    ax.plot(t_AP, sta_bursty, 'r', label='Bursty')
    ax.fill_between(t_AP, sta_nonbursty - std_nonbursty, sta_nonbursty + std_nonbursty, color='b', alpha=0.5)
    ax.plot(t_AP, sta_nonbursty, 'b', label='Non-bursty')
    ax.set_ylabel('Mem. pot. (mV)')
    ax.set_xlabel('Time (ms)')
    pl.legend(handles=handles_bursty)
    pl.show()