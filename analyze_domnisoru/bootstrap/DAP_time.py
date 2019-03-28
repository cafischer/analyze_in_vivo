import numpy as np
import os
from grid_cell_stimuli import find_all_AP_traces
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_cell_ids_DAP_cells
from cell_characteristics import to_idx
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict
from cell_characteristics.analyze_APs import get_spike_characteristics
from analyze_in_vivo.analyze_domnisoru.sta import get_AP_amps_and_widths, select_APs


def get_DAP_time(v_APs, t_AP):
    sta_mean = np.mean(v_APs, 0)

    v_ref = sta_mean[before_AP_idx - to_idx(t_vref, dt)]
    spike_characteristics_dict = get_spike_characteristics_dict()
    spike_characteristics_dict['AP_max_idx'] = before_AP_idx
    spike_characteristics_dict['AP_onset'] = before_AP_idx - to_idx(1.0, dt)
    DAP_time = get_spike_characteristics(sta_mean, t_AP, ['DAP_time'], v_ref, check=False, **spike_characteristics_dict)[0]
    return DAP_time



if __name__ == '__main__':
    #save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bootstrap/'
    #save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    save_dir_characteristics = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bootstrap/'
    save_dir = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cell_type = 'grid_cells'
    cell_ids = np.array(load_cell_ids(save_dir, cell_type))
    cell_ids_DAP = get_cell_ids_DAP_cells(new=True)
    param_list = ['Vm_ljpc', 'spiketimes', 'vel_100ms']

    # parameters
    np.random.seed(1)
    n_trials = 100000
    before_AP = 25
    after_AP = 25
    t_vref = 10
    dt = 0.05
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    folder_name = AP_criterion.keys()[0] + str(AP_criterion.values()[0]) \
                  + '_before_after_AP_' + str((before_AP, after_AP)) + '_t_vref_' + str(t_vref)
    save_dir_characteristics = os.path.join(save_dir_characteristics, folder_name)
    if not os.path.exists(save_dir_characteristics):
        os.makedirs(save_dir_characteristics)

    before_AP_idx = to_idx(before_AP, dt)
    after_AP_idx = to_idx(after_AP, dt)


    mean_DAP_time = np.zeros(len(cell_ids_DAP))
    se_DAP_time = np.zeros(len(cell_ids_DAP))

    for cell_idx, cell_id in enumerate(cell_ids_DAP):
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        AP_max_idxs = data['spiketimes']
        v_APs = find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs)
        t_AP = np.arange(after_AP_idx + before_AP_idx + 1) * dt

        # sort out v_APs according to criterion
        AP_amps, AP_widths = get_AP_amps_and_widths(v_APs, t_AP, dt, before_AP_idx, t_vref)
        v_APs_good = select_APs(AP_amps, AP_widths, AP_criterion, v_APs)

        # bootstrapping
        DAP_times = np.zeros(n_trials)
        n_trial = 0
        while n_trial < n_trials-1:
            v_APs_sample = v_APs_good[np.random.randint(0, len(v_APs_good), len(v_APs_good))]  # with replacement
            DAP_times[n_trial] = get_DAP_time(v_APs_sample, t_AP)
            if not np.isnan(DAP_times[n_trial]):
                n_trial += 1

        mean_DAP_time[cell_idx] = np.mean(DAP_times)
        se_DAP_time[cell_idx] = np.std(DAP_times, ddof=1)

        print cell_id
        print 'mean: %.2f' % mean_DAP_time[cell_idx]
        print 'se: %.2f' % se_DAP_time[cell_idx]
        print 'sample size: ', len(v_APs_good)

    np.save(os.path.join(save_dir_characteristics, 'mean_DAP_time.npy'), mean_DAP_time)
    np.save(os.path.join(save_dir_characteristics, 'se_DAP_time.npy'), se_DAP_time)