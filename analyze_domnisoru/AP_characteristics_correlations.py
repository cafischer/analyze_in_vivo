from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristic_correlations'
    save_dir_characteristics = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/AP_characteristics/all'
    save_dir_burst = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_spat_info = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    save_dir_auto_corr = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spike_time_auto_corr'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    param_list = ['Vm_ljpc', 'spiketimes']

    # parameters
    save_dir_img = os.path.join(save_dir_img, cell_type)
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    fraction_burst = np.load(os.path.join(save_dir_burst, cell_type, 'fraction_burst.npy'))
    spatial_info = np.load(os.path.join(save_dir_spat_info, cell_type, 'spatial_info.npy'))
    peak_ISI_hist = np.load(os.path.join(save_dir_ISI_hist, cell_type, 'peak_ISI_hist.npy'))
    peak_auto_corr = np.load(os.path.join(save_dir_auto_corr, cell_type, 'peak_auto_corr_50.npy'))
    DAP_deflection = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_deflection.npy'))
    DAP_width = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_width.npy'))
    DAP_time = np.load(os.path.join(save_dir_characteristics, cell_type, 'DAP_time.npy'))
    AP_width = np.load(os.path.join(save_dir_characteristics, cell_type, 'AP_width.npy'))
    AP_amp = np.load(os.path.join(save_dir_characteristics, cell_type, 'AP_amp.npy'))

    DAP_deflection[np.isnan(DAP_deflection)] = 0  # for plotting

    good_cell_indicator = AP_width <= 0.75
    DAP_deflection_good_cells = DAP_deflection[good_cell_indicator]
    DAP_time__good_cells = DAP_time[good_cell_indicator]
    fraction_burst_good_cells = fraction_burst[good_cell_indicator]
    spatial_info_good_cells = spatial_info[good_cell_indicator]
    peak_ISI_hist_good_cells = peak_ISI_hist[good_cell_indicator]
    peak_auto_corr_good_cells = peak_auto_corr[good_cell_indicator]
    cell_ids_good_cells = np.array(cell_ids)[good_cell_indicator]

    # plots
    pl.figure()
    pl.plot(np.zeros(len(cell_ids_good_cells))[DAP_deflection_good_cells == 0],
            fraction_burst_good_cells[DAP_deflection_good_cells == 0], 'o', color='0.5')
    pl.plot(np.ones(len(cell_ids_good_cells))[DAP_deflection_good_cells > 0],
            fraction_burst_good_cells[DAP_deflection_good_cells > 0], 'o', color='0.5')
    for cell_idx in range(len(cell_ids_good_cells)):
        pl.annotate(cell_ids_good_cells[cell_idx], xy=((DAP_deflection_good_cells[cell_idx] > 0).astype(int),
                                                       fraction_burst_good_cells[cell_idx]))
    pl.errorbar(-0.1, np.mean(fraction_burst_good_cells[DAP_deflection_good_cells == 0]),
            yerr=np.std(fraction_burst_good_cells[DAP_deflection_good_cells == 0]), color='k', capsize=2, marker='o')
    pl.errorbar(0.9, np.mean(fraction_burst_good_cells[DAP_deflection_good_cells > 0]),
            yerr=np.std(fraction_burst_good_cells[DAP_deflection_good_cells > 0]), color='k', capsize=2, marker='o')
    pl.xlim(-1, 2)
    pl.xticks([0, 1], ['no', 'yes'])
    pl.ylabel('Fraction ISI < 8 ms')
    pl.xlabel('DAP')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_vs_frac_burst.png'))

    pl.figure()
    pl.plot(np.zeros(len(cell_ids_good_cells))[DAP_deflection_good_cells == 0],
            spatial_info_good_cells[DAP_deflection_good_cells == 0], 'o', color='0.5')
    pl.plot(np.ones(len(cell_ids_good_cells))[DAP_deflection_good_cells > 0],
            spatial_info_good_cells[DAP_deflection_good_cells > 0], 'o', color='0.5')
    for cell_idx in range(len(cell_ids_good_cells)):
        pl.annotate(cell_ids_good_cells[cell_idx], xy=((DAP_deflection_good_cells[cell_idx] > 0).astype(int),
                                                       spatial_info_good_cells[cell_idx]))
    pl.errorbar(-0.1, np.mean(spatial_info_good_cells[DAP_deflection_good_cells == 0]),
                yerr=np.std(spatial_info_good_cells[DAP_deflection_good_cells == 0]), color='k', capsize=2, marker='o')
    pl.errorbar(0.9, np.mean(spatial_info_good_cells[DAP_deflection_good_cells > 0]),
                yerr=np.std(spatial_info_good_cells[DAP_deflection_good_cells > 0]), color='k', capsize=2, marker='o')
    pl.ylabel('Spatial information')
    pl.xlabel('DAP')
    pl.xlim(-1, 2)
    pl.xticks([0, 1], ['no', 'yes'])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_vs_spatial_info.png'))

    peak_ISI_hist_good_cells = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist_good_cells])
    pl.figure()
    pl.plot(np.zeros(len(cell_ids_good_cells))[DAP_deflection_good_cells == 0],
            peak_ISI_hist_good_cells[DAP_deflection_good_cells == 0], 'o', color='0.5')
    pl.plot(np.ones(len(cell_ids_good_cells))[DAP_deflection_good_cells > 0],
            peak_ISI_hist_good_cells[DAP_deflection_good_cells > 0], 'o', color='0.5')
    for cell_idx in range(len(cell_ids_good_cells)):
        pl.annotate(cell_ids_good_cells[cell_idx], xy=((DAP_deflection_good_cells[cell_idx] > 0).astype(int),
                                                       peak_ISI_hist_good_cells[cell_idx]))
    pl.errorbar(-0.1, np.mean(peak_ISI_hist_good_cells[DAP_deflection_good_cells == 0]),
                yerr=np.std(peak_ISI_hist_good_cells[DAP_deflection_good_cells == 0]), color='k', capsize=2, marker='o')
    pl.errorbar(0.9, np.mean(peak_ISI_hist_good_cells[DAP_deflection_good_cells > 0]),
                yerr=np.std(peak_ISI_hist_good_cells[DAP_deflection_good_cells > 0]), color='k', capsize=2, marker='o')
    pl.ylabel('Peak of ISI hist. (ms)')
    pl.xlabel('DAP')
    pl.xlim(-1, 2)
    pl.xticks([0, 1], ['no', 'yes'])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_vs_peak_ISI_hist.png'))

    peak_ISI_hist = np.array([(p[0] + p[1]) / 2. for p in peak_ISI_hist])  # set middle of bin as peak
    pl.figure()
    pl.plot(DAP_time, peak_ISI_hist, 'o', color='0.5')
    for cell_idx in range(len(cell_ids)):
        pl.annotate(cell_ids[cell_idx], xy=(DAP_time[cell_idx], peak_ISI_hist[cell_idx]))
    pl.xlim(0, 10)
    pl.ylim(0, 10)
    pl.ylabel('Peak of ISI hist. (ms)')
    pl.xlabel('DAP time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_time_vs_ISI_peak.png'))

    pl.figure()
    pl.plot(DAP_time, peak_auto_corr, 'o', color='0.5')
    for cell_idx in range(len(cell_ids)):
        pl.annotate(cell_ids[cell_idx], xy=(DAP_time[cell_idx], peak_auto_corr[cell_idx]))
    pl.xlim(0, 20)
    pl.ylim(0, 20)
    pl.ylabel('Peak of auto-correlation (ms)')
    pl.xlabel('DAP time (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_time_vs_auto_corr_peak.png'))

    pl.figure()
    pl.plot(AP_amp, DAP_deflection, 'ok')
    pl.ylabel('DAP deflection (mV)')
    pl.xlabel('AP amplitude (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_AP_amp.png'))

    print 'Low AP width: '
    print np.array(cell_ids)[AP_width < 0.7]
    pl.figure()
    pl.plot(AP_width, DAP_deflection, 'ok')
    pl.ylabel('DAP deflection (mV)')
    pl.xlabel('AP width (ms)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'DAP_deflection_vs_AP_width.png'))

    pl.show()