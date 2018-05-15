from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load.load_schmidt_hieber import load_full_runs
from analyze_in_vivo.investigate_grid_cell_stimuli import detrend
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_stc, choose_eigvecs, \
    project_back, plots_stc, group_by_AP_max, plot_group_by_AP_max, plot_ICA, plot_all_in_one, plot_backtransform, \
    plot_PCA_3D, plot_ICA_3D, plot_clustering_kmeans
import matplotlib.pyplot as pl
from sklearn.decomposition import FastICA
pl.style.use('paper')


def find_all_APs_in_v_trace(v, before_AP_idx, after_AP_idx, AP_threshold, do_detrend=False, v_detrend=None):
    v_APs = []
    onset_idxs = get_AP_onset_idxs(v, AP_threshold)
    if len(onset_idxs) > 0:
        onset_idxs = np.insert(onset_idxs, len(onset_idxs), len(v))
        AP_max_idxs = [get_AP_max_idx(v, onset_idx, onset_next_idx) for (onset_idx, onset_next_idx)
                       in zip(onset_idxs[:-1], onset_idxs[1:])]

        for i, AP_max_idx in enumerate(AP_max_idxs):
            if (AP_max_idx is not None  # None if no AP max found (e.g. at end of v trace)
                    and AP_max_idx - before_AP_idx >= 0 and AP_max_idx + after_AP_idx < len(v)):  # able to draw window
                v_AP = v[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1]

                if before_AP_idx > AP_max_idx - onset_idxs[i]:
                    n_APs_desired = 1  # if we start before the onset, the AP belonging to the onset should be detected
                else:
                    n_APs_desired = 0  # else no AP should be detected

                if len(get_AP_onset_idxs(v_AP,
                                         AP_threshold)) == n_APs_desired:  # only take windows where there is no other AP
                    if do_detrend:
                        if cell_id == '20120213_12213':
                            if not np.any(np.diff(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                                        > 5):  # check for measurement errors
                                v_APs.append(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                        else:
                            v_APs.append(v_detrend[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1])
                    else:
                        if cell_id == '20120213_12213':
                            if not np.any(np.diff(v_AP) > 5):  # check for measurement errors
                                v_APs.append(v_AP)
                        else:
                            v_APs.append(v_AP)
    return v_APs


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/STA/whole_trace/'
    data_dir = '../data/'
    cell_ids = ["20101031_10o31c", "20110513_11513", "20110910_11910b",
                "20111207_11d07c", "20111213_11d13b", "20120213_12213"]
    #cell_ids = ["20110910_11910b"]

    # parameters
    do_detrend = True
    before_AP_sta = 25
    after_AP_sta = 25
    before_AP_stc = 0
    after_AP_stc = 25

    for i, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        v, t, x_pos, y_pos, pos_t, speed, speed_t = load_full_runs(data_dir, cell_id)
        dt = t[1] - t[0]
        AP_threshold = np.min(v) + 2./3 * np.abs(np.min(v) - np.max(v)) - 5
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)
        before_AP_idx_stc = to_idx(before_AP_stc, dt)
        after_AP_idx_stc = to_idx(after_AP_stc, dt)

        # detrend
        if do_detrend:
            v_detrend = detrend(v, t, cutoff_freq=5)
        else:
            v_detrend = None

        # plot
        if do_detrend:
            save_dir_img = os.path.join(save_dir, 'detrended', cell_id)
        else:
            save_dir_img = os.path.join(save_dir, 'not_detrended', cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        pl.figure()
        if do_detrend:
            pl.plot(t, v_detrend, 'k')
        else:
            pl.plot(t, v, 'k')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'whole_trace.png'))

        # STA
        v_APs = find_all_APs_in_v_trace(v, before_AP_idx_sta, after_AP_idx_sta, AP_threshold,
                                        do_detrend=do_detrend, v_detrend=v_detrend)
        v_APs = np.vstack(v_APs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt
        sta, sta_std = get_sta(v_APs)

        plot_sta(v_APs, t_AP, sta, sta_std, save_dir_img)

        # STC & Group by AP_max & ICA
        v_APs = find_all_APs_in_v_trace(v, before_AP_idx_stc, after_AP_idx_stc, AP_threshold,
                                        do_detrend=do_detrend, v_detrend=v_detrend)
        v_APs = np.vstack(v_APs)
        v_APs_centered = v_APs - np.mean(v_APs, 0)
        t_AP = np.arange(after_AP_idx_stc + before_AP_idx_stc + 1) * dt

        if len(v_APs) > 10:
            # STC
            eigvals, eigvecs, expl_var = get_stc(v_APs)
            chosen_eigvecs = choose_eigvecs(eigvecs, eigvals, n_eigvecs=3)
            back_projection = project_back(v_APs, chosen_eigvecs)
            plots_stc(v_APs, t_AP, back_projection, chosen_eigvecs, expl_var, save_dir_img)

            # Group by AP_max
            mean_high, std_high, mean_low, std_low, AP_max_high_labels, AP_max = group_by_AP_max(v_APs)
            plot_group_by_AP_max(mean_high, std_high, mean_low, std_low, t_AP, save_dir_img)
            mean_high_centered = mean_high - np.mean(v_APs, 0)
            mean_low_centered = mean_low - np.mean(v_APs, 0)

            # ICA
            ica = FastICA(n_components=3, whiten=True)
            ica_source = ica.fit_transform(v_APs_centered)
            plot_ICA(v_APs, t_AP, ica.mixing_, save_dir_img)

            # plot together
            plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low,
                            chosen_eigvecs, expl_var, ica.mixing_, save_dir_img)
            plot_backtransform(v_APs_centered, t_AP, mean_high_centered, mean_low_centered, std_high, std_low,
                               chosen_eigvecs, expl_var, ica_source, ica.mixing_, save_dir_img)

            #pl.close('all')
            plot_PCA_3D(v_APs_centered, chosen_eigvecs, AP_max_high_labels, AP_max, save_dir_img=save_dir_img)
            #   pl.show()
            plot_ICA_3D(v_APs_centered, ica_source, AP_max_high_labels, save_dir_img)
            plot_clustering_kmeans(v_APs, v_APs_centered, t_AP, chosen_eigvecs, 2, save_dir_img)


            # save as .npy
            np.save(os.path.join(save_dir_img, 'v_APs.npy'), v_APs)
            np.save(os.path.join(save_dir_img, 't_AP.npy'), t_AP)

        pl.close('all')