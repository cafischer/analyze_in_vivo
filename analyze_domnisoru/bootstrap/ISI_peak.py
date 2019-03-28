import numpy as np
import matplotlib.pyplot as pl
import os
from grid_cell_stimuli import find_all_AP_traces
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru import perform_kde, evaluate_kde
from analyze_in_vivo.analyze_domnisoru.isi import get_ISI_hist_peak_and_width
from grid_cell_stimuli.ISI_hist import get_ISIs


def get_ISI_peak(ISIs, t_kde, sigma_smooth, bins):
    if sigma_smooth is not None:
        kde = perform_kde(ISIs, sigma_smooth)
        ISI_kde = evaluate_kde(t_kde, kde)
        peak_ISI_hist, _ = get_ISI_hist_peak_and_width(ISI_kde, t_kde)

        # pl.figure()
        # pl.plot(t_kde, ISI_kde)
        # pl.axvline(peak_ISI_hist, color='r')
        # pl.show()
    else:
        ISI_hist = np.histogram(ISIs, bins)[0]
        peak_ISI_hist = np.mean((bins[:-1][np.argmax(ISI_hist)],
                                       bins[1:][np.argmax(ISI_hist)]))

        #pl.figure()
        #pl.bar(bins[:-1], ISI_hist, width=bin_width, align='edge')
        #pl.axvline(peak_ISI_hist, color='r')
        #pl.show()
    return peak_ISI_hist


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
    n_trials = 1000  # TODO higher number 100000
    max_ISI = 200  # ms
    bin_width = 1  # ms
    sigma_smooth = 1  # ms  None for no smoothing
    dt_kde = 0.05  # ms
    bins = np.arange(0, max_ISI + bin_width, bin_width)
    t_kde = np.arange(0, max_ISI + dt_kde, dt_kde)
    if sigma_smooth is not None:
        folder = 'max_ISI_' + str(max_ISI) + '_sigma_smooth_' + str(sigma_smooth)
    else:
        folder = 'max_ISI_' + str(max_ISI) + '_bin_width_' + str(bin_width)
    save_dir_characteristics = os.path.join(save_dir_characteristics, folder)
    if not os.path.exists(save_dir_characteristics):
        os.makedirs(save_dir_characteristics)

    mean_ISI_peak = np.zeros(len(cell_ids_DAP))
    se_ISI_peak = np.zeros(len(cell_ids_DAP))

    for cell_idx, cell_id in enumerate(cell_ids_DAP):
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        AP_max_idxs = data['spiketimes']

        ISIs = get_ISIs(AP_max_idxs, t)
        if max_ISI is not None:
            ISIs = ISIs[ISIs <= max_ISI]

        # bootstrapping
        ISI_peaks = np.zeros(n_trials)
        n_trial = 0
        while n_trial < n_trials-1:
            ISIs_sample = ISIs[np.random.randint(0, len(ISIs), len(ISIs))]  # with replacement
            ISI_peaks[n_trial] = get_ISI_peak(ISIs_sample, t_kde, sigma_smooth, bins)
            if not np.isnan(ISI_peaks[n_trial]):
                n_trial += 1

        mean_ISI_peak[cell_idx] = np.mean(ISI_peaks)
        se_ISI_peak[cell_idx] = np.std(ISI_peaks, ddof=1)

        print cell_id
        print 'median: %.2f' %  np.median(ISI_peaks)
        print 'mad: %.2f' % np.median(np.abs(ISI_peaks - np.median(ISI_peaks)))
        print 'mean: %.2f' % mean_ISI_peak[cell_idx]
        print 'se: %.2f' % se_ISI_peak[cell_idx]
        print 'sample size: ', len(ISIs)

        #pl.figure()
        #pl.hist(ISI_peaks, bins=100)
        #pl.xlabel('ISI peak')
        #pl.ylabel('Frequency')
        #pl.show()

    #np.save(os.path.join(save_dir_characteristics, 'mean_ISI_peak.npy'), mean_ISI_peak)
    #np.save(os.path.join(save_dir_characteristics, 'se_ISI_peak.npy'), se_ISI_peak)