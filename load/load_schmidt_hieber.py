import os
import pickle
import numpy as np
from scipy.io import loadmat


def get_stellate_info(data_dir):
    with open(os.path.join(data_dir, 'stelldb.pck'), 'rb') as stelldb:
        stellate_info = pickle.load(stelldb)
    return stellate_info


def timearray(array, dt):
    return np.arange(array.shape[0]) * dt


def load_field_crossings(data_dir, cell_id, n_field, n_crossing):
    datfile_time_mat = os.path.join(data_dir, "dat", "{0}_field{1:02d}_crossing_extended{2:02d}_time.mat"
                                    .format(cell_id, n_field, n_crossing))
    dat_time_mat = loadmat(datfile_time_mat)
    v_dt = dat_time_mat['V_data_dt'][0][0]
    v = dat_time_mat['V_data_time'].squeeze()
    pos_times = dat_time_mat['postimes'].squeeze()
    pos_times -= pos_times[0]
    x_pos = dat_time_mat['xpos'].squeeze()
    y_pos = dat_time_mat['ypos'].squeeze()
    speed_dt = dat_time_mat['speed_dt'][0][0]
    speed = dat_time_mat['speed_time'].squeeze()
    # V_sub_dt = dat_time_mat['V_clipped_red_dt'][0][0]
    # V_sub = dat_time_mat['V_clipped_red_time'].squeeze()
    # V_mpo_dt = dat_time_mat['V_envelope_dt'][0][0]
    # V_mpo = dat_time_mat['V_envelope_time'].squeeze()
    return v, timearray(v, v_dt), x_pos, y_pos, pos_times, speed, timearray(speed, speed_dt)


def load_full_runs(data_dir, cell_id):
    datfile_time_mat = os.path.join(data_dir, "dat", "{0}raw.mat".format(cell_id))
    dat_time_mat = loadmat(datfile_time_mat)
    v_dt = dat_time_mat['V_data_dt'][0][0]
    v = dat_time_mat['V_data_time'].squeeze()
    x_pos = dat_time_mat['xpos'].squeeze()
    y_pos = dat_time_mat['ypos'].squeeze()
    pos_times = timearray(x_pos, dat_time_mat['pos_dt'].squeeze())
    speed_dt = dat_time_mat['speed_dt'][0][0]
    speed = dat_time_mat['speed_time'].squeeze()
    # V_sub_dt = dat_time_mat['V_sub_dt'][0][0]
    # V_sub = dat_time_mat['V_sub_time'].squeeze()
    return v, timearray(v, v_dt), x_pos, y_pos, pos_times, speed, timearray(speed, speed_dt)


def load_VI(data_dir, cell_id):
    """
    Cells available:
    cells_S2 = [
    "10217003",
    "10n10000",
    "11303000",
    "11910002",
    "11n30004",
    "11d13006"]

    cells_S7 = [
    "10o31005",
    "11513000",
    "11910001002",
    "11d07006",
    "11d13006",
    "12213002"]
    """
    vi_data = loadmat(os.path.join(data_dir, "dat", cell_id + '.mat'))
    assert vi_data['ch1units'][0] == 'mV'
    assert vi_data['ch2units'][0] == 'pA'

    dt = vi_data['dt'][0][0]
    v = vi_data['data'][0, :]  # mV
    i_inj = vi_data['data'][1, :] / 1000  # nA
    return v, timearray(v[0], dt), i_inj


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    data_dir = '../data/'

    # stellate_info = get_stellate_info(data_dir)
    # print stellate_info
    #
    # cell_ids = stellate_info.keys()
    # cell_id = cell_ids[2]
    # n_field = 0
    # n_crossing = 0
    #
    # v, t, x_pos, y_pos, pos_t, speed, speed_t = load_field_crossings(data_dir, cell_id, n_field, n_crossing)
    #
    # pl.figure()
    # pl.plot(t, v)
    # pl.show()
    #
    # pl.figure()
    # pl.plot(pos_t, y_pos)
    # pl.show()
    #
    # pl.figure()
    # pl.plot(y_pos, x_pos)
    # pl.axis('equal')
    # pl.show()
    #
    # pl.figure()
    # pl.plot(speed_t, speed)
    # pl.show()

    cell_id = "10n10000"

    v, t, i_inj = load_VI(data_dir, cell_id)

    step = 10
    pl.figure()
    pl.plot(t, v[step, :])
    pl.show()