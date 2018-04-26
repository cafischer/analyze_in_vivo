from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from grid_cell_stimuli.remove_APs import remove_APs
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta, plot_filter
from sklearn.metrics import mean_squared_error
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/check/theta_ramp'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_names = load_cell_ids(save_dir, 'stellate_layer2')
    param_list = ['Vm_ljpc', 'Vm_wo_spikes_ljpc', 'fVm', 'dcVm_ljpc']

    # parameter
    AP_thresholds = {'s117_0002': -60, 's119_0004': -50, 's104_0007': -55, 's79_0003': -50, 's76_0002': -50,
                     's101_0009': -45}
    t_before = 3
    t_after = 6

    cutoff_ramp = 3  # Hz
    cutoff_theta_low = 5  # Hz
    cutoff_theta_high = 11  # Hz
    transition_width = 1.5  # Hz
    ripple_attenuation = 60.0  # db

    for cell_name in cell_names:
        print cell_name

        save_dir_cell = os.path.join(save_dir_img, cell_name)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        # load
        data = load_data(cell_name, param_list, save_dir)
        v = data['Vm_ljpc'][:100000]
        t = np.arange(0, len(v)) * data['dt']

        # remove spikes and test the same as Domnisoru
        v_APs_removed = remove_APs(v, t, AP_thresholds[cell_name], t_before, t_after)

        # pl.figure()
        # pl.title('APs removed')
        # pl.plot(np.arange(len(data['Vm_wo_spikes_ljpc']))*data['dt']/1000., data['Vm_wo_spikes_ljpc'], 'r', label='domnisoru')
        # pl.plot(t/1000., v_APs_removed, 'k', label='')
        # pl.ylabel('Membrane Potential (mV)')
        # pl.xlabel('Time (s)')
        # pl.legend()
        # #pl.xlim(1, 2)
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'APs_removed.png'))
        # pl.show()

        # compute theta and ramp
        ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v_APs_removed, data['dt'],
                                                                                  ripple_attenuation,
                                                                                  transition_width, cutoff_ramp,
                                                                                  cutoff_theta_low,
                                                                                  cutoff_theta_high,
                                                                                  pad_if_to_short=True)

        print 'RMS theta: %.3f' % np.sqrt(mean_squared_error(data['fVm'][:100000], theta))
        print 'RMS ramp: %.3f' % np.sqrt(mean_squared_error(data['dcVm_ljpc'][:100000], ramp))

        # plot
        t /= 1000

        plot_filter(filter_ramp, filter_theta, data['dt'], save_dir_cell)

        pl.figure()
        pl.title('Theta')
        pl.plot(np.arange(len(data['fVm']))*data['dt']/1000., data['fVm'], 'r', label='domnisoru')
        pl.plot(t, theta, 'k', label='')
        pl.ylabel('')
        pl.xlabel('Time (s)')
        pl.xlim(0, t[-1])
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'theta.png'))

        pl.figure()
        pl.title('Ramp')
        pl.plot(np.arange(len(data['fVm']))*data['dt']/1000., data['dcVm_ljpc'], 'r', label='domnisoru')
        pl.plot(t, ramp, 'k', label='')
        pl.ylabel('')
        pl.xlabel('Time (s)')
        pl.xlim(0, t[-1])
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ramp.png'))

        pl.figure()
        pl.title('Theta')
        pl.plot(np.arange(len(data['fVm']))*data['dt']/1000., data['fVm'], 'r', label='domnisoru')
        pl.plot(t, theta, 'k', label='')
        pl.ylabel('')
        pl.xlabel('Time (s)')
        pl.xlim(0, 1)
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'theta_zoom.png'))

        pl.figure()
        pl.title('Ramp')
        pl.plot(np.arange(len(data['fVm']))*data['dt']/1000., data['dcVm_ljpc'], 'r', label='domnisoru')
        pl.plot(t, ramp, 'k', label='')
        pl.ylabel('')
        pl.xlabel('Time (s)')
        pl.xlim(0, 1)
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_cell, 'ramp_zoom.png'))
        #pl.show()