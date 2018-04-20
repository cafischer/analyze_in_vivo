from scipy.io import loadmat
import os
import matplotlib.pyplot as pl
import numpy as np
pl.style.use('paper')


def load_data(cell_name, param_list, save_dir):
    data_set = {}
    for param in param_list:
        file_name = cell_name+'_'+param+'.mat'
        file = loadmat(os.path.join(save_dir, file_name))
        data = file[param]
        if 'Y' in param:
            data += 200  # to bring position between 0 and 400
        data_set[param] = data[:, 0]

    # add time
    dt = 0.05
    t = np.arange(0, len(data)) * dt
    data_set['t'] = t
    return data_set


def load_grid_cell_names(save_dir):
    grid_cell_names_ = loadmat(os.path.join(save_dir, 'cl_ah.mat'))
    grid_cell_names = [str(x[0]) for x in grid_cell_names_['cl_ah'][0][0][0][0]]
    return grid_cell_names


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    grid_cell_names = load_grid_cell_names(save_dir)

    grid_cell_name = grid_cell_names[0]
    param_list = ['Vm_ljpc', 'Y_cm']
    data = load_data(grid_cell_name, param_list, save_dir)

    fig, axes = pl.subplots(len(data), 1, sharex='all')
    for i, (param, trace) in enumerate(data.iteritems()):
        axes[i].plot(data['t'] / 1000., trace, 'k')
        axes[i].set_ylabel(param)
    axes[-1].set_xlabel('Time (s)')
    pl.tight_layout()
    pl.show()