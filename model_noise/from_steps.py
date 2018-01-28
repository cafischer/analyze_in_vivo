from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from load import load_VI
from scipy.optimize import curve_fit
from grid_cell_stimuli.remove_APs import remove_APs
from grid_cell_stimuli.downsample import antialias_and_downsample
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta, plot_spectrum


def remove_theta(v_mat, t):
    v_mat_new = np.zeros(np.shape(v_mat))

    for i in range(np.shape(v_mat)[0]):
        # remove APs and downsampling
        cutoff_freq = 2000  # Hz
        dt_new_max = 1. / cutoff_freq * 1000  # ms
        transition_width = 5.0  # Hz
        ripple_attenuation = 60.0  # db
        AP_threshold = max(-50, np.min(v_mat[i]) + np.abs(np.max(v_mat[i]) - np.min(v_mat[i])) * (2. / 3))
        print 'AP_threshold %.2f' % AP_threshold
        v_downsampled, t_downsampled, filter = antialias_and_downsample(
            remove_APs(v_mat[i], t, AP_threshold, t_before, t_after),
            dt, ripple_attenuation, transition_width,
            cutoff_freq, dt_new_max)

        # get theta
        cutoff_ramp = 3  # Hz
        cutoff_theta_low = 4  # Hz TODO: 5
        cutoff_theta_high = 11  # Hz
        transition_width = 1  # Hz
        ripple_attenuation = 60.0  # db
        dt_d = t_downsampled[1] - t_downsampled[0]
        ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v_downsampled, dt_d,
                                                                                  ripple_attenuation,
                                                                                  transition_width, cutoff_ramp,
                                                                                  cutoff_theta_low,
                                                                                  cutoff_theta_high,
                                                                                  pad_if_to_short=True)
        # remove theta
        v_mat_new[i] = v_mat[i] - np.interp(t, t_ramp_theta, theta)

        plot_spectrum(v_mat[i], np.interp(t, t_ramp_theta, ramp), np.interp(t, t_ramp_theta, theta), dt, save_dir, show=False)

        pl.figure()
        pl.plot(t, v_mat[i], 'k')
        pl.plot(t, np.interp(t, t_ramp_theta, theta), 'g')
        pl.plot(t_downsampled, theta, 'b')
        pl.plot(t, v_mat_new[i], 'r')
        pl.show()
    return v_mat_new


def gauss(x, mu, sig):
    return np.exp(-(x-mu)**2 / (2*sig**2))


def fit_gauss_to_hist(v, n_bins):
    hist, bins = np.histogram(v, bins=n_bins)
    norm_fac = np.max(hist)
    hist = hist / norm_fac
    bin_midpoints = np.array([(e_s + e_e) / 2 for (e_s, e_e) in zip(bins[:-1], bins[1:])])
    p_opt, _ = curve_fit(gauss, bin_midpoints, hist, p0=(np.mean(v_mat_new[i]), np.std(v_mat_new[i])))
    return p_opt, bin_midpoints, norm_fac


if __name__ == '__main__':

    save_dir = '../results/schmidthieber/spike_shape/traces'
    data_dir = '../data/'
    cell_ids = ["10o31005", "11513000", "11910001002", "11d07006", "11d13006", "12213002"]
    n_bins = 25
    t_before = 3
    t_after = 6

    Ei = 0 # TODO
    Ee = -60 # TODO
    g_pas = 0 # TODO
    E_pas = -70 # TODO
    a = 1 # TODO
    tau_e = 1 # TODO
    tau_i = 1 # TODO
    c_m = 1 # TODO

    for cell_id in cell_ids:
        v_mat, t, i_inj_mat = load_VI(data_dir, cell_id)
        dt = t[1] - t[0]
        start_step = np.where(np.diff(np.abs(i_inj_mat[0, :])) > 0.05)[0][0] + 1
        end_step = np.where(-np.diff(np.abs(i_inj_mat[0, :])) > 0.05)[0][0]
        start_step += 5000  # TODO
        end_step -= 5000  # TODO

        pl.figure()
        for v in v_mat:
            pl.plot(np.arange(len(v)) * dt, v)
        #pl.show()

        save_dir_fig = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        v_mat_during_step = np.zeros((len(v_mat), end_step-start_step))
        for i in range(np.shape(v_mat)[0]):
            v_mat_during_step[i] = v_mat[i, start_step:end_step]

        #v_mat_new = remove_theta(v_mat_during_step, t[start_step:end_step]-t[start_step])
        v_mat_new = v_mat_during_step

        for i in range(np.shape(v_mat_new)[0]):
            i_amp = np.round(np.mean(i_inj_mat[i, start_step:end_step]), 2)

            if i_amp > -0.05 and i_amp < 0.05:
                # fit gaussian
                p_opt, bin_midpoints, norm_fac = fit_gauss_to_hist(v_mat_new[i], n_bins)

                pl.figure()
                pl.plot(t[start_step:end_step], v_mat_new[i], 'k')
                pl.ylabel('Membrane potential (mV)', fontsize=16)
                pl.xlabel('Time (ms)', fontsize=16)

                pl.figure()
                pl.hist(v_mat_new[i], bins=n_bins, weights=np.ones(len(v_mat_new[i]))/norm_fac)
                pl.plot(bin_midpoints, gauss(bin_midpoints, p_opt[0], p_opt[1]), 'r')
                pl.plot(bin_midpoints, gauss(bin_midpoints, np.mean(v_mat_new[i]), np.std(v_mat_new[i])), 'g')
                pl.xlabel('Membrane potential (mV)', fontsize=16)
                pl.ylabel('Count', fontsize=16)
                #pl.show()

                i_amp1 = i_amp
                mu1, sig1 = p_opt

            if i_amp > 0.05 and i_amp < 0.15:
                p_opt, bin_midpoints, norm_fac = fit_gauss_to_hist(v_mat_new[i], n_bins)

                pl.figure()
                pl.plot(t[start_step:end_step], v_mat_new[i], 'k')
                pl.ylabel('Membrane potential (mV)', fontsize=16)
                pl.xlabel('Time (ms)', fontsize=16)

                pl.figure()
                pl.hist(v_mat_new[i], bins=n_bins, weights=np.ones(len(v_mat_new[i]))/norm_fac)
                pl.plot(bin_midpoints, gauss(bin_midpoints, p_opt[0], p_opt[1]), 'r')
                pl.plot(bin_midpoints, gauss(bin_midpoints, np.mean(v_mat_new[i]), np.std(v_mat_new[i])), 'g')
                pl.xlabel('Membrane potential (mV)', fontsize=16)
                pl.ylabel('Count', fontsize=16)
                #pl.show()

                i_amp2 = i_amp
                mu2, sig2 = p_opt

        ge0 = ((i_amp1 - i_amp2) * (sig2**2 * (Ei - mu1)**2 - sig1**2 * (Ei - mu2)**2)) / \
              (((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ee - Ei) * (mu1 - mu2)**2) \
              - ((i_amp1 - i_amp2) * (Ei -mu2) + (i_amp2 - g_pas * a * (Ei - E_pas))*(mu1 - mu2)) / \
              ((Ee - Ei) * (mu1 - mu2))
        gi0 = ((i_amp1 - i_amp2) * (sig2**2 * (Ee - mu1)**2 - sig1**2 * (Ee - mu2)**2)) / \
              (((Ee - mu1) * (Ei - mu2) + (Ee - mu2) * (Ei - mu1)) * (Ei - Ee) * (mu1 - mu2)**2) \
              - ((i_amp1 - i_amp2) * (Ee -mu2) + (i_amp2 - g_pas * a * (Ee - E_pas))*(mu1 - mu2)) / \
              ((Ei - Ee) * (mu1 - mu2))
        var_e = (2 * a * c_m * (i_amp1 - i_amp2) * (sig1**2 * (Ei - mu2)**2 - sig2**2 * (Ei - mu1)**2)) \
                / (tau_e * ((Ee - mu1)*(Ei - mu2) + (Ee - mu2)*(Ei - mu1)) * (Ee - Ei) * (mu1 - mu2)**2)
        var_i = (2 * a * c_m * (i_amp1 - i_amp2) * (sig1**2 * (Ee - mu2)**2 - sig2**2 * (Ee - mu1)**2)) \
                / (tau_i * ((Ee - mu1)*(Ei - mu2) + (Ee - mu2)*(Ei - mu1)) * (Ei - Ee) * (mu1 - mu2)**2)