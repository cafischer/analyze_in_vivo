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
        shape = np.shape(file[param])
        if shape[0] == 1:
            data_set[param] = data[0, :]
        elif shape[1] == 1:
            data_set[param] = data[:, 0]
        else:
            data_set[param] = data

    # add time
    data_set['dt'] = 0.05
    return data_set


def load_field_indices(cell_name, save_dir):
    file_name = cell_name+'_all_fields_fY_ah.mat'
    file = loadmat(os.path.join(save_dir, file_name))
    in_field_idxs = file['fininds'][:, 0] - 1  # -1 for matlab indices
    out_field_idxs = file['foutinds'][:, 0] - 1
    return in_field_idxs, out_field_idxs


def load_cell_ids(save_dir, cell_type='grid_cells'):
    file = loadmat(os.path.join(save_dir, 'cl_ah.mat'))['cl_ah']
    grid_cells_tmp = file['gridlist']
    grid_cells = [str(x[0]) for x in grid_cells_tmp[0][0][0]]
    pyramidal_cells_tmp = file['pyramidal_grid']
    pyramidal_cells = [str(x[0][0]) for x in pyramidal_cells_tmp[0][0]]
    stellate_cells_tmp = file['stellate_grid']
    stellate_cells = [str(x[0][0]) for x in stellate_cells_tmp[0][0]]
    layer2_cells_tmp = file['l2_grid']
    layer2_cells = [str(x[0]) for x in layer2_cells_tmp[0, 0][0]]
    layer3_cells_tmp = file['l3_grid']
    layer3_cells = [str(x[0]) for x in layer3_cells_tmp[0, 0][0]]
    giant_theta_cells_tmp = file['giant_theta_grid']
    giant_theta_cells = [str(x[0][0]) for x in giant_theta_cells_tmp[0, 0]]
    small_theta_cells_tmp = file['st_grid']  # small theta
    small_theta_cells = [str(x[0][0]) for x in small_theta_cells_tmp[0, 0]]

    if cell_type == 'grid_cells':
        cells = grid_cells
    elif cell_type == 'stellate_layer2':
        cells = list(set(layer2_cells).intersection(stellate_cells))
    elif cell_type == 'pyramidal_layer2':
        cells = list(set(layer2_cells).intersection(pyramidal_cells))
    elif cell_type == 'pyramidal_layer3':
        cells = list(set(layer3_cells).intersection(pyramidal_cells))
    elif cell_type == 'giant_theta':
        cells = giant_theta_cells
    elif cell_type == 'small_theta':
        cells = small_theta_cells
    else:
        raise ValueError('Cell type not available!')

    cells = sorted(cells, key=lambda x: int(x[1:].split('_')[0]))
    return cells


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    grid_cell_names = load_cell_ids(save_dir, 'stellate_layer2')

    grid_cell_name = grid_cell_names[3]
    print grid_cell_name
    param_list = ['Vm_ljpc', 'Y_cm']
    data = load_data(grid_cell_name, param_list, save_dir)

    fig, axes = pl.subplots(1, 1, sharex='all')
    axes.plot(np.arange(len(data['Vm_ljpc'])) * data['dt'] / 1000., data['Vm_ljpc'], 'k')
    axes.set_ylabel('Membrane Potential')
    axes.set_xlabel('Time (s)')
    pl.tight_layout()
    pl.show()