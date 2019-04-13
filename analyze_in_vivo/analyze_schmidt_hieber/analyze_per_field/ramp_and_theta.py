from __future__ import division
import numpy as np
import os
import json
from analyze_in_vivo.load.load_schmidt_hieber import load_field_crossings
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta, plot_filter, plot_spectrum, plot_v_ramp_theta


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = '../results/'+folder+'/ramp_and_theta'
    save_dir_data = '../results/'+folder+'/downsampled'

    # parameters
    cutoff_ramp = 3  # Hz
    cutoff_theta_low = 5  # Hz
    cutoff_theta_high = 11  # Hz
    transition_width = 1  # Hz
    ripple_attenuation = 60.0  # db
    params = {'cutoff_ramp': cutoff_ramp, 'cutoff_theta_low': cutoff_theta_low,
              'cut_off_theta_high': cutoff_theta_high, 'transition_width': transition_width,
              'ripple_attenuation': ripple_attenuation}

    # over all field crossings
    max_ramp = []
    min_ramp = []
    max_theta = []
    min_theta = []
    for file_name in os.listdir(save_dir_data):

        # load
        v = np.load(os.path.join(save_dir_data, file_name, 'v_AP_removed.npy'))
        t = np.load(os.path.join(save_dir_data, file_name, 't.npy'))
        dt = t[1] - t[0]

        # get ramp and theta
        ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v, dt, ripple_attenuation,
                                                                                  transition_width, cutoff_ramp,
                                                                                  cutoff_theta_low,
                                                                                  cutoff_theta_high,
                                                                                  pad_if_to_short=True)

        # save and plot
        save_dir_cell_field_crossing = os.path.join(save_dir, file_name)
        if not os.path.exists(save_dir_cell_field_crossing):
            os.makedirs(save_dir_cell_field_crossing)

        max_ramp.append(np.max(ramp))
        min_ramp.append(np.min(ramp))
        max_theta.append(np.max(theta))
        min_theta.append(np.min(theta))

        np.save(os.path.join(save_dir_cell_field_crossing, 'ramp.npy'), ramp)
        np.save(os.path.join(save_dir_cell_field_crossing, 'theta.npy'), theta)
        np.save(os.path.join(save_dir_cell_field_crossing, 't.npy'), t_ramp_theta)

        with open(os.path.join(save_dir_cell_field_crossing, 'params'), 'w') as f:
            json.dump(params, f)

        plot_filter(filter_ramp, filter_theta, dt, save_dir_cell_field_crossing)
        plot_spectrum(v, ramp, theta, dt, save_dir_cell_field_crossing)
        plot_v_ramp_theta(v, t, ramp, theta, t_ramp_theta, save_dir_cell_field_crossing)

    print 'Mean Ramp_{max-min}: %.2f' % np.mean(np.array(max_ramp) - np.array(min_ramp))
    print 'Std Ramp_{max-min}: %.2f' % np.std(np.array(max_ramp) - np.array(min_ramp))
    print 'Mean Theta_{max-min}: %.2f' % np.mean(np.array(max_theta) - np.array(min_theta))
    print 'Std Theta_{max-min}: %.2f' % np.std(np.array(max_theta) - np.array(min_theta))