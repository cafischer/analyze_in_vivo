from __future__ import division
import numpy as np
import os


save_dir_fig6 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig6'
save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

save_dir_ISI_hist = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
save_dir_firing_rate = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/firing_rate'
save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'
save_dir_ISI_hist_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_hist'
save_dir_spike_events_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/spike_events'
save_dir_ISI_return_map_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/ISI/ISI_return_map'
save_dir_firing_rate_latuske = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/latuske/firing_rate'
save_dir_delta_DAP = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/delta_DAP_delta_fAHP'


if not os.path.exists(save_dir_fig6):
    os.makedirs(save_dir_fig6)

max_ISI = 200
bin_width = 1  # ms
sigma_smooth = 1  # ms
burst_ISI = 8  # ms
before_AP = 25
after_AP = 25
t_vref = 10
dt = 0.05
AP_criterion = {'AP_amp_and_width': (40, 1)}
remove_cells = True

folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width) + '_sigma_smooth_' + str(sigma_smooth)

# load domnisoru
peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, folder, 'peak_ISI_hist.npy'))
fraction_burst = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_burst.npy'))
CV_ISIs = np.load(os.path.join(save_dir_ISI_hist, folder, 'CV_ISIs.npy'))
fraction_ISIs_8_25 = np.load(os.path.join(save_dir_ISI_hist, folder, 'fraction_ISIs_8_25.npy'))
firing_rate = np.load(os.path.join(save_dir_firing_rate, 'firing_rate.npy'))
v_onset_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_onset_fAHP.npy'))
v_DAP_fAHP = np.load(os.path.join(save_dir_delta_DAP, 'avg_times', 'v_DAP_fAHP.npy'))


# save
np.save(os.path.join(save_dir_fig6, 'peak_ISI_hist.npy'), peak_ISI_hist)
np.save(os.path.join(save_dir_fig6, 'fraction_burst.npy'), fraction_burst)
np.save(os.path.join(save_dir_fig6, 'CV_ISIs.npy'), CV_ISIs)
np.save(os.path.join(save_dir_fig6, 'fraction_ISIs_8_25.npy'), fraction_ISIs_8_25)
np.save(os.path.join(save_dir_fig6, 'firing_rate.npy'), firing_rate)
np.save(os.path.join(save_dir_fig6, 'v_onset_fAHP.npy'), v_onset_fAHP)
np.save(os.path.join(save_dir_fig6, 'v_DAP_fAHP.npy'), v_DAP_fAHP)