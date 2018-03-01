from scipy.optimize import curve_fit
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as pl
from load import load_VI
from cell_characteristics import to_idx
from cell_fitting.util import convert_from_unit


def get_cellarea(L, diam):
    """
    Takes length and diameter of some cell segment and returns the area of that segment (assuming it to be the surface
    of a cylinder without the circle surfaces as in Neuron).
    :param L: Length (um).
    :type L: float
    :param diam: Diameter (um).
    :type diam: float
    :return: Cell area (um).
    :rtype: float
    """
    return L * diam * np.pi


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
    peak_hyperpolarization = argrelmin(v[start_step:end_step], order=to_idx(20, t[1]-t[0]))[0][0] + start_step
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

    # estimate cell size
    c_m_ind = 1.0 * 1e6  # pF/cm2  # from experiments
    cell_area = 1.0 / (c_m_ind / c_m)  # cm2
    diam = np.sqrt(cell_area / np.pi) * 1e4  # um

    # estimate g_pas
    g_pas = 1 / convert_from_unit('M', r_in) / cell_area  # S/cm2

    # normalize cm by cell area
    c_m_per_area = c_m * 1e-6 / cell_area  # uF/cm2  # should be 1 as assumed from experiments

    print 'tau: ' + str(tau_m) + ' ms'
    print 'Rin: ' + str(r_in) + ' MOhm'
    print 'c_m: ' + str(c_m) + ' pF'
    print 'diam: ' + str(diam) + ' um'
    print 'g_pas: ' + str(g_pas) + ' S/cm2'
    # print 'c_m: ' + str(c_m_per_area) + ' uF/cm2'

    return cell_area, g_pas


if __name__ == '__main__':
    data_dir = '../data/'
    for cell_id in ["11513000", "11910001002", "11d07006", "11d13006", "12213002"]:
        v_mat, t, i_inj_mat = load_VI(data_dir, cell_id)
        v = v_mat[0, :]
        i_inj = i_inj_mat[0, :]

        cell_area, g_pas = estimate_passive_parameters(v, t, i_inj)



