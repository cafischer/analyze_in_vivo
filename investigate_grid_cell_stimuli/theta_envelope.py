import numpy as np
import os
import json
from grid_cell_stimuli.theta_envelope import compute_envelope, plot_envelope


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = './results/' + folder + '/ramp_and_theta'
    save_dir_data = './results/' + folder + '/ramp_and_theta'

    # over all field crossings
    for file_name in os.listdir(save_dir_data):

        # load
        theta = np.load(os.path.join(save_dir_data, file_name, 'theta.npy'))
        t = np.load(os.path.join(save_dir_data, file_name, 't.npy'))
        dt = t[1] - t[0]

        # compute the envelope
        theta_envelope = compute_envelope(theta)

        # save and plot
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'theta_envelope.npy'), theta_envelope)

        plot_envelope(theta, theta_envelope, t, save_dir)