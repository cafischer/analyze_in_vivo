from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from cell_characteristics import to_idx
from analyze_in_vivo.analyze_domnisoru.check_basic.in_out_field import get_starts_ends_group_of_ones
from grid_cell_stimuli.ramp_and_theta import filter
from grid_cell_stimuli import compute_fft
from scipy.signal import resample
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA'
    save_dir_in_out_field = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)[5:]

    # parameters
    use_AP_max_idxs_domnisoru = True
    velocity_threshold = 1  # cm/sec
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms', 'fY_cm', 'fvel_100ms']
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # main
    for i, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        velocity = data['vel_100ms']

        # find areas without APs and velocity < threshold
        AP_area = to_idx(2, dt)
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        spike_indicator = np.zeros(len(v))
        for AP_max_idx in AP_max_idxs:
            spike_indicator[AP_max_idx-AP_area:AP_max_idx+AP_area] = 1
        low_velocity_indicator = velocity < velocity_threshold

        starts, ends = get_starts_ends_group_of_ones(np.logical_and(~spike_indicator.astype(bool),
                                                                    low_velocity_indicator).astype(int))
        v_chunks = [v[start:end] for start, end in zip(starts, ends)]

        # # Filter out gamma frequencies
        # cutoff_low = 100.  # Hz
        # cutoff_high = 250.  # Hz
        # transition_width = 20.0  # Hz
        # ripple_attenuation = 60.0  # db
        # for v_chunk in v_chunks:
        #     if ~len(v_chunk) > to_idx(1000, dt):
        #         continue
        #     v_filtered, t_filtered, _ = filter(v_chunk, dt, ripple_attenuation, transition_width,
        #                                        cutoff_low, cutoff_high)
        #
        #     pl.figure()
        #     pl.plot(t_filtered, v_filtered)
        #     pl.show()

        # power spectrum
        for v_chunk in v_chunks:
            if ~len(v_chunk) > to_idx(1000, dt):
                continue

            # downsampling
            v_r, t_r = resample(v_chunk, 2 ** 16, t)

            # power spectrum
            fft_v, freqs = compute_fft(v_r, (t_r[1] - t_r[0]) / 1000.0)
            power = np.abs(fft_v) ** 2

            pl.figure()
            pl.plot(freqs, power, 'k')
            pl.xlabel('Frequency')
            pl.ylabel('Power')
            pl.xlim(0, 300)
            pl.ylim(0, 1e8)
            pl.tight_layout()
            pl.show()

        # # plot
        # pl.figure()
        # pl.plot(np.arange(len(v_chunks[0]))*dt, v_chunks[0])
        # pl.show()



    # def plot_sta(ax, cell_idx, t_AP, sta_mean_cells, sta_std_cells):
    #     ax.plot(t_AP, sta_mean_cells[cell_idx], 'k')
    #     ax.fill_between(t_AP, sta_mean_cells[cell_idx] - sta_std_cells[cell_idx],
    #                               sta_mean_cells[cell_idx] + sta_std_cells[cell_idx], color='k', alpha=0.5)
    #
    # plot_kwargs = dict(t_AP=t_AP, sta_mean_cells=sta_mean_cells, sta_std_cells=sta_std_cells)
    #
    # if cell_type == 'grid_cells':
    #     plot_for_all_grid_cells(cell_ids, get_celltype_dict(save_dir), plot_sta, plot_kwargs,
    #                             xlabel='Time (ms)', ylabel='Mem. pot. (mV)',
    #                             save_dir_img=os.path.join(save_dir_img, 'sta.png'))

    pl.show()