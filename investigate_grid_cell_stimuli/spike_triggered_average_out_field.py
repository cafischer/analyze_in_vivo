from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load import load_full_runs
from analyze_in_vivo.investigate_grid_cell_stimuli import detrend
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from analyze_in_vivo.spatial_firing_rate import get_spatial_firing_rate, identify_firing_fields, \
    get_start_end_idxs_in_out_field_in_time
from cell_characteristics import to_idx
import matplotlib.pyplot as pl
from sklearn.decomposition import FastICA
pl.style.use('paper')


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/STA/out_field/after_spike'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]

    # parameters
    do_detrend = True
    before_AP = 0
    after_AP = 50
    cut_before_AP = before_AP + 1.0
    cut_after_AP = after_AP - 25

    for i, cell_id in enumerate(cell_ids):
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5
        before_AP_idx = to_idx(before_AP, dt)
        after_AP_idx = to_idx(after_AP, dt)
        cut_before_AP_idx = to_idx(cut_before_AP, dt)
        cut_after_AP_idx = to_idx(cut_after_AP, dt)

        # find all spikes
        onset_idxs = get_AP_onset_idxs(v, AP_threshold)

        # detrend
        if do_detrend:
            v_detrend = detrend(v, t, cutoff_freq=5)

        # find out fields idxs
        spatial_firing_rate, positions, loc_spikes = get_spatial_firing_rate(v, t, y_pos, pos_t, h=3,
                                                                             AP_threshold=AP_threshold, bin_size=0.5,
                                                                             track_len=np.max(y_pos))
        in_field_idxs_per_field, out_field_idxs_per_field = identify_firing_fields(spatial_firing_rate,
                                                                                   fraction_from_peak_rate=0.10)
        _, start_end_idx_out_field = get_start_end_idxs_in_out_field_in_time(t, positions, y_pos, pos_t,
                                                                             in_field_idxs_per_field,
                                                                             out_field_idxs_per_field)

        # take window around each spike
        v_APs = []
        for onset_idx in onset_idxs:
            try:
                idx_field = np.where(np.array(start_end_idx_out_field) >= onset_idx)[0][0]
            except IndexError:
                continue
            start_out, end_out = start_end_idx_out_field[idx_field]
            if start_out < onset_idx <= end_out:
                if onset_idx-before_AP_idx >= 0 and onset_idx+after_AP_idx+1 <= len(v):  # able to draw window
                    v_AP = v[onset_idx-before_AP_idx:onset_idx+after_AP_idx+1]
                    if len(get_AP_onset_idxs(v_AP, AP_threshold)) == 1:  # no bursts
                        if do_detrend:
                            v_APs.append(v_detrend[onset_idx - before_AP_idx:onset_idx + after_AP_idx + 1])
                        else:
                            v_APs.append(v_AP)
        v_APs = np.vstack(v_APs)

        # STA
        spike_triggered_avg = np.mean(v_APs, 0)
        spike_triggered_std = np.std(v_APs, 0)
        t_AP = np.arange(after_AP_idx+before_AP_idx+1)*dt
        print '#APs: ' + str(len(v_APs))

        # plot
        if do_detrend:
            save_dir_cell = os.path.join(save_dir, 'detrended', cell_id)
        else:
            save_dir_cell = os.path.join(save_dir, 'not_detrended', cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        pl.figure()
        pl.plot(t_AP, spike_triggered_avg, 'b')
        pl.fill_between(t_AP, spike_triggered_avg+spike_triggered_std, spike_triggered_avg-spike_triggered_std,
                        facecolor='blue', alpha=0.5)
        pl.xlabel('Time (ms)')
        pl.ylabel('V (mV)')
        pl.title(cell_id.split('_')[1])
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'sta.png'))
        #pl.show()

        pl.figure()
        for v_AP in v_APs:
            pl.plot(t_AP, v_AP)
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'v_APs.png'))
        #pl.show()

        # pl.figure()
        # pl.plot(t, v)
        # pl.xlabel('Time (ms)')
        # pl.ylabel('Membrane Potential (mV)')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'v.png'))
        #pl.show()

        # STC
        if len(v_APs) > 10:
            len_v_AP = len(v_APs[0])
            v_APs = v_APs[:, cut_before_AP_idx:len_v_AP - cut_after_AP_idx]   # select smaller window around APs
            t_AP = t_AP[cut_before_AP_idx:len_v_AP - cut_after_AP_idx]
            v_APs_centered = v_APs - np.mean(v_APs, 0)
            cov = np.cov(v_APs.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            eigvals = abs(eigvals)
            idx_sort = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx_sort]
            eigvecs = eigvecs[:, idx_sort]
            min_expl_var = 0.8
            n = np.where(np.cumsum(eigvals) / np.sum(eigvals) >= min_expl_var)[0][0]
            chosen_eigvecs = eigvecs[:, :n+1]
            back_transform = np.dot(v_APs_centered, np.dot(chosen_eigvecs, chosen_eigvecs.T)) + np.mean(v_APs, 0)
            expl_var = eigvals / np.sum(eigvals) * 100

            pl.figure()
            for vec in v_APs:
                pl.plot(t_AP, vec)
            pl.title('AP Traces', fontsize=18)
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_cell, 'STC_APs.png'))
            #pl.show()

            pl.figure()
            for vec in back_transform:
                pl.plot(t_AP, vec)
            pl.title('Backprojected AP Traces', fontsize=18)
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_cell, 'STC_backprojected_APs.png'))
            #pl.show()

            pl.figure()
            for i, vec in enumerate(chosen_eigvecs.T):
                pl.plot(t_AP, vec, label='expl. var.: %i %%' % int(round(expl_var[i])))
            pl.title('Eigenvectors', fontsize=18)
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.legend(loc='upper left', fontsize=10)
            pl.savefig(os.path.join(save_dir_cell, 'STC_largest_eigenvecs.png'))
            #pl.show()

            pl.figure()
            pl.plot(np.arange(len(expl_var)), np.cumsum(expl_var), 'ok')
            pl.axhline(min_expl_var*100, 0, 1, color='0.5', linestyle='--',
                       label='%i %% expl. var.' % int(round(min_expl_var*100)))
            pl.title('Cumulative Explained Variance', fontsize=18)
            pl.ylabel('Percent')
            pl.xlabel('#')
            pl.ylim(0, 105)
            #pl.legend(fontsize=16)
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_cell, 'STC_eigenvals.png'))
            #pl.show()

            # ICA
            ica = FastICA(n_components=3, whiten=True)
            ica_components = ica.fit_transform(v_APs.T)  # Reconstruct signals
            pl.figure()
            for vec in ica_components.T:
                pl.plot(t_AP, vec)
            pl.title('ICA Components', fontsize=18)
            pl.ylabel('Membrane Potential (mV)')
            pl.xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_cell, 'ICA_ica_components.png'))
            #pl.show()

        pl.close('all')