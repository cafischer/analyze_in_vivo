from scipy.optimize import curve_fit
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as pl
from load import load_VI
from cell_characteristics import to_idx


def estimate_passive_parameters(v, t, i_inj):
    """

    :param v: Trace where hyper-polarized a little.
    :param t:
    :param i_inj:
    :return:
    """

    pl.figure()
    pl.plot(t, v)

    start_step = np.where(np.diff(np.abs(i_inj)) > 0.05)[0][0] + 1
    end_step = np.where(-np.diff(np.abs(i_inj)) > 0.05)[0][0]

    # fit tau
    peak_hyperpolarization = argrelmin(v[start_step:end_step], order=to_idx(100, t[1]-t[0]))[0][0] + start_step
    v_expdecay = v[start_step:peak_hyperpolarization] - v[start_step]
    t_expdecay = t[start_step:peak_hyperpolarization] - t[start_step]
    v_diff = np.abs(v_expdecay[-1] - v_expdecay[0])

    def exp_decay(t, tau):
        return v_diff * np.exp(-t / tau) - v_diff

    tau_m, _ = curve_fit(exp_decay, t_expdecay, v_expdecay)  # ms
    tau_m = tau_m[0]

    pl.figure()
    pl.plot(t_expdecay, v_expdecay, 'k')
    pl.plot(t_expdecay, exp_decay(t_expdecay, tau_m), 'r')
    pl.show()

    # compute Rin
    last_fourth_i_inj = 3/4 * (end_step - start_step) + start_step

    v_rest = np.mean(v[0:start_step - 1])
    i_rest = np.mean(i_inj[0:start_step - 1])
    v_in = np.mean(v[last_fourth_i_inj:end_step]) - v_rest
    i_in = np.mean(i_inj[last_fourth_i_inj:end_step]) - i_rest

    r_in = v_in / i_in  # mV / nA = MOhm

    # compute capacitance
    c_m = tau_m / r_in * 1000  # ms / MOhm to pF

    print 'tau: ' + str(tau_m) + ' ms'
    print 'Rin: ' + str(r_in) + ' MOhm'
    print 'c_m: ' + str(c_m) + ' pF'

    # TODO:
    # estimate cell size
    # c_m_ind = 1.0 * 1e6  # pF/cm2
    # cell_area = 1.0 / (c_m_ind / c_m)  # cm2
    # diam = np.sqrt(cell_area / np.pi) * 1e4  # um
    # print 'Estimated diam: ' + str(diam)
    # #
    # L = 100  # um
    # diam = 100  # um
    # print 'If L={0} and diam={1}: '.format(L, diam)
    # L = L * 1e-4  # cm
    # diam = diam * 1e-4  # cm
    # cell_area = get_cellarea(L, diam)  # cm2
    #
    # g_pas = 1 / convert_unit_prefix('M', r_in) / cell_area  # S/cm2
    # print 'g_pas: ' + str(g_pas) + ' S/cm2'
    # c_m_per_area = c_m * 1e-6 / cell_area  # uF/cm2
    # print 'c_m: ' + str(c_m_per_area) + ' uF/cm2'

    return c_m, r_in, tau_m


if __name__ == '__main__':
    data_dir = '../data/'
    cell_id = "10o31005"

    v_mat, t, i_inj_mat = load_VI(data_dir, cell_id)
    v = v_mat[0, :]
    i_inj = i_inj_mat[0, :]

    c_m, r_in, tau_m = estimate_passive_parameters(v, t, i_inj)

