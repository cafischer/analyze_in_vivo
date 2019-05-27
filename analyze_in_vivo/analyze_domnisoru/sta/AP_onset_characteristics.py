import numpy as np
import os
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl

save_dir_sta = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/STA/good_AP_criterion/not_detrended'

# parameter
AP_criterion = {'AP_amp_and_width': (40, 1)}
t_vref = 10
before_AP = 25
after_AP = 25
dt = 0.05
AP_thresh_derivative = 15.0
dur = 10

# load
folder = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
              + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
sta_mean_cells = np.load(os.path.join(save_dir_sta, folder, 'sta_mean.npy'))
t_sta = np.arange(-before_AP, after_AP+dt, dt)

# compute
sta_derivative_cells = np.array([np.diff(sta_mean) / dt for sta_mean in sta_mean_cells])

AP_thresh_idx = np.array([get_AP_onset_idxs(sta_derivative[:to_idx(before_AP, dt)], AP_thresh_derivative)[-1]
                          for sta_derivative in sta_derivative_cells])

v_onset = np.array([sta_mean_cells[i][AP_thresh_idx[i]] for i in range(len(sta_mean_cells))])
v_start = np.array([sta_mean_cells[i][AP_thresh_idx[i] - to_idx(dur, dt)] for i in range(len(sta_mean_cells))])
linear_slope_APonset = (v_onset - v_start) / dur

# print linear_slope_APonset[0]
# pl.figure()
# pl.plot(t_sta, sta_mean_cells[0], 'k')
# pl.plot(t_sta[AP_thresh_idx[0]], v_onset[0], 'or')
# pl.plot(t_sta[AP_thresh_idx[0] - to_idx(dur, dt)], v_start[0], 'or')
# pl.show()

np.save(os.path.join(save_dir_sta, folder, 'v_onset.npy'), v_onset)
np.save(os.path.join(save_dir_sta, folder, 'linear_slope_APonset'), linear_slope_APonset)