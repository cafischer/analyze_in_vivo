from __future__ import division
import numpy as np
import os
from analyze_in_vivo.load import load_VI
from analyze_in_vivo.investigate_grid_cell_stimuli import detrend
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx
from cell_characteristics import to_idx
from cell_characteristics.sta_stc import get_sta, plot_sta, get_stc, choose_eigvecs, \
    project_back, plots_stc, group_by_AP_max, plot_group_by_AP_max, plot_ICA, plot_all_in_one, plot_backtransform, \
    plot_PCA_3D, plot_ICA_3D, plot_clustering_kmeans
import matplotlib.pyplot as pl
from sklearn.decomposition import FastICA
pl.style.use('paper')


def find_all_APs_in_v_trace(v, before_AP_idx, after_AP_idx, do_detrend=False, v_detrend=None):
    if do_detrend:
        v = v_detrend
    AP_threshold = max(np.mean(v)+10.0*np.std(v), np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5)
    #AP_threshold = max(np.mean(v) + 5.0 * np.std(v), np.min(v) + 2. / 3 * np.abs(np.min(v) - np.max(v)) - 5)
    v_APs = []
    onset_idxs = get_AP_onset_idxs(v, AP_threshold)
    if len(onset_idxs) > 0:
        onset_idxs = np.insert(onset_idxs, len(onset_idxs), len(v))
        AP_max_idxs = [get_AP_max_idx(v, onset_idx, onset_next_idx) for (onset_idx, onset_next_idx)
                       in zip(onset_idxs[:-1], onset_idxs[1:])]

        # pl.figure()
        # pl.plot(v)
        # for max_i in AP_max_idxs:
        #     if max_i is not None:
        #         pl.plot(max_i, v[max_i], 'o')
        # pl.axhline(AP_threshold)
        # pl.show()

        AP_threshold += -5

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
                    s = 'y'
                    v_APs.append(v_AP)
                else:
                    s = 'n'

                # pl.figure()
                # pl.title(s)
                # pl.plot(v_AP)
                # pl.show()
    return v_APs


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/full_traces/STA/VI/'
    data_dir = '../data/'
    cell_ids = ["10o31005", "11513000", "11910001002", "11d07006", "11d13006", "12213002"]
    #cell_ids = ["11d07006"]

    # parameters
    do_detrend = True
    step_start = 534.26
    step_end = 1532.26
    before_AP_sta = 0
    after_AP_sta = 25
    before_AP_stc = 0
    after_AP_stc = 25

    for i, cell_id in enumerate(cell_ids):
        print cell_id
        # load
        v_mat, t, i_inj_mat = load_VI(data_dir, cell_id)
        dt = t[1] - t[0]
        step_start_idx = to_idx(step_start, dt)
        step_end_idx = to_idx(step_end, dt)
        before_AP_idx_sta = to_idx(before_AP_sta, dt)
        after_AP_idx_sta = to_idx(after_AP_sta, dt)
        before_AP_idx_stc = to_idx(before_AP_stc, dt)
        after_AP_idx_stc = to_idx(after_AP_stc, dt)

        v_mat = v_mat[:, step_start_idx:step_end_idx]  # use only v trace during step
        t = t[step_start_idx:step_end_idx] - t[step_start_idx]

        # pl.figure()
        # for v in v_mat:
        #     pl.plot(t, v)
        # pl.show()

        # plot
        if do_detrend:
            save_dir_img = os.path.join(save_dir, 'detrended', cell_id)
        else:
            save_dir_img = os.path.join(save_dir, 'not_detrended', cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # STA
        v_APs = []
        for v in v_mat:
            if do_detrend:
                v_detrend = detrend(v, t, cutoff_freq=5)
            else:
                v_detrend = None

            v_APs.extend(find_all_APs_in_v_trace(v, before_AP_idx_sta, after_AP_idx_sta,
                                        do_detrend=do_detrend, v_detrend=v_detrend))
        v_APs = np.vstack(v_APs)
        t_AP = np.arange(after_AP_idx_sta + before_AP_idx_sta + 1) * dt

        sta, sta_std = get_sta(v_APs)
        plot_sta(v_APs, t_AP, sta, sta_std, save_dir_img)

        # STC & Group by AP_max & ICA
        v_APs = []
        for v in v_mat:
            if do_detrend:
                v_detrend = detrend(v, t, cutoff_freq=5)
            else:
                v_detrend = None

            v_APs.extend(find_all_APs_in_v_trace(v, before_AP_idx_stc, after_AP_idx_stc,
                                        do_detrend=do_detrend, v_detrend=v_detrend))
        v_APs = np.vstack(v_APs)
        v_APs_centered = v_APs - np.mean(v_APs, 0)
        t_AP = np.arange(after_AP_idx_stc + before_AP_idx_stc + 1) * dt

        if len(v_APs) >= 5:
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
            #pl.close('all')
            plot_backtransform(v_APs_centered, t_AP, mean_high_centered, mean_low_centered, std_high, std_low,
                               chosen_eigvecs, expl_var, ica_source, ica.mixing_, save_dir_img)
            #pl.show()
            plot_PCA_3D(v_APs_centered, chosen_eigvecs, AP_max_high_labels, AP_max, save_dir_img=save_dir_img)
            plot_ICA_3D(v_APs_centered, ica_source, AP_max_high_labels, save_dir_img)
            plot_clustering_kmeans(v_APs, v_APs_centered, t_AP, chosen_eigvecs, 2, save_dir_img)

        pl.close('all')