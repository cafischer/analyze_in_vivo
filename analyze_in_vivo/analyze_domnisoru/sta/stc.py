from __future__ import division
import numpy as np
import os
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype
from cell_characteristics.sta_stc import get_stc, choose_eigvecs, project_back, plots_stc, group_by_AP_max,\
    plot_group_by_AP_max, plot_ICA, plot_all_in_one, plot_backtransform, \
    plot_PCA_3D, plot_ICA_3D, plot_clustering_kmeans
from analyze_in_vivo.analyze_schmidt_hieber import detrend
from grid_cell_stimuli import get_AP_max_idxs, find_all_AP_traces
import matplotlib.pyplot as pl
from sklearn.decomposition import FastICA
pl.style.use('paper')



if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STC'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'  # 'pyramidal_layer2'  #
    cell_ids = load_cell_ids(save_dir, cell_type)

    # parameters
    use_AP_max_idxs_domnisoru = True
    do_detrend = False
    before_AP_stc = 0
    after_AP_stc = 25
    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38,
                     's117_0002': -60, 's119_0004': -50, 's104_0007': -55,
                     's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes']
    folder_detrend = {True: 'detrended', False: 'not_detrended'}
    save_dir_img = os.path.join(save_dir_img, folder_detrend[do_detrend], cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    for i, cell_id in enumerate(cell_ids):
        print cell_id
        
        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        dt = t[1] - t[0]
        before_AP_idx_stc = to_idx(before_AP_stc, dt)
        after_AP_idx_stc = to_idx(after_AP_stc, dt)

        # detrend
        if do_detrend:
            v = detrend(v, t, cutoff_freq=5)

        # save dir
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)
            
        # STC & Group by AP_max & ICA
        if use_AP_max_idxs_domnisoru:
            AP_max_idxs = data['spiketimes']
        else:
            AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        v_APs = find_all_AP_traces(v, before_AP_idx_stc, after_AP_idx_stc, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx_stc + before_AP_idx_stc + 1) * dt
        v_APs_centered = v_APs - np.mean(v_APs, 0)

        if len(v_APs) > 10:
            # STC
            eigvals, eigvecs, expl_var = get_stc(v_APs)
            chosen_eigvecs = choose_eigvecs(eigvecs, eigvals, n_eigvecs=3)
            back_projection = project_back(v_APs, chosen_eigvecs)
            plots_stc(v_APs, t_AP, back_projection, chosen_eigvecs, expl_var, save_dir_cell)

            # Group by AP_max
            mean_high, std_high, mean_low, std_low, AP_max_high_labels, AP_max = group_by_AP_max(v_APs)
            plot_group_by_AP_max(mean_high, std_high, mean_low, std_low, t_AP, save_dir_cell)
            mean_high_centered = mean_high - np.mean(v_APs, 0)
            mean_low_centered = mean_low - np.mean(v_APs, 0)

            # ICA
            ica = FastICA(n_components=3, whiten=True)
            ica_source = ica.fit_transform(v_APs_centered)
            plot_ICA(v_APs, t_AP, ica.mixing_, save_dir_cell)

            # plot together
            plot_all_in_one(v_APs, t_AP, back_projection, mean_high, std_high, mean_low, std_low,
                            chosen_eigvecs, expl_var, ica.mixing_, save_dir_cell)
            plot_backtransform(v_APs_centered, t_AP, mean_high_centered, mean_low_centered, std_high, std_low,
                               chosen_eigvecs, expl_var, ica_source, ica.mixing_, save_dir_cell)

            #pl.close('all')
            plot_PCA_3D(v_APs_centered, chosen_eigvecs, AP_max_high_labels, AP_max, save_dir_img=save_dir_cell)
            plot_ICA_3D(v_APs_centered, ica_source, AP_max_high_labels, save_dir_cell)
            plot_clustering_kmeans(v_APs, v_APs_centered, t_AP, chosen_eigvecs, 2, save_dir_cell)
            #pl.show()
        pl.close('all')