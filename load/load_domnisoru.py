from scipy.io import loadmat
import os
import matplotlib.pyplot as pl
import numpy as np
import json
pl.style.use('paper')


def load_data(cell_id, param_list, save_dir):
    data_set = {}
    for param in param_list:
        file_name = cell_id + '_' + param + '.mat'
        file = loadmat(os.path.join(save_dir, file_name))
        data = file[param]
        if 'Y' in param:
            data += 196.8929860286835  # to make position start at 0
            assert np.all(data >= 0)
            assert np.all(data < get_track_len(cell_id))
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
        cell_ids = grid_cells
    elif cell_type == 'stellate_cells':
        cell_ids = stellate_cells
    elif cell_type == 'pyramidal_cells':
        cell_ids = pyramidal_cells
    elif cell_type == 'stellate_layer2':
        cell_ids = list(set(layer2_cells).intersection(stellate_cells))
    elif cell_type == 'pyramidal_layer2':
        cell_ids = list(set(layer2_cells).intersection(pyramidal_cells))
    elif cell_type == 'pyramidal_layer3':
        cell_ids = list(set(layer3_cells).intersection(pyramidal_cells))
    elif cell_type == 'giant_theta':
        cell_ids = giant_theta_cells
    elif cell_type == 'small_theta':
        cell_ids = small_theta_cells
    else:
        raise ValueError('Cell type not available!')
    if 's66_0003' in cell_ids:  # remove because data is not reliable
        cell_ids.remove('s66_0003')
    cell_ids = sorted(cell_ids, key=lambda x: int(x[1:].split('_')[0]))
    return cell_ids


def get_cell_ids_DAP_cells(new=False):
    if new:
        return ['s67_0000', 's73_0004', 's79_0003', 's104_0007', 's109_0002', 's110_0002', 's119_0004']
    else:
        return ['s79_0003', 's104_0007', 's109_0002', 's110_0002', 's119_0004'], ['s73_0004', 's85_0007']


def get_cell_ids_bursty():
    return np.array(['s43_0003', 's67_0000', 's73_0004', 's76_0002', 's79_0003',
                     's82_0002', 's95_0006', 's101_0009', 's104_0007', 's109_0002',
                     's110_0002', 's117_0002', 's118_0002', 's119_0004', 's120_0002'], dtype=str)


def get_cell_ids_good_recording():
    return ['s74_0006', 's79_0003', 's82_0002', 's84_0002', 's85_0007', 's104_0007', 's109_0002', 's110_0002',
            's119_0004']


def get_celltype_dict(save_dir):
    with open(os.path.join(save_dir, 'cell_types.json'), 'r') as f:
        cell_type_dict = json.load(f)
    return cell_type_dict


def get_celltype(cell_id, save_dir):
    file = loadmat(os.path.join(save_dir, 'cl_ah.mat'))['cl_ah']
    pyramidal_cells_tmp = file['pyramidal_grid']
    pyramidal_cells = [str(x[0][0]) for x in pyramidal_cells_tmp[0][0]]
    stellate_cells_tmp = file['stellate_grid']
    stellate_cells = [str(x[0][0]) for x in stellate_cells_tmp[0][0]]

    if cell_id in stellate_cells:
        return 'stellate'
    elif cell_id in pyramidal_cells:
        return 'pyramidal'
    else:
        return 'not known'


def get_track_len(cell_id):
    if cell_id == 's82_0002' or cell_id == 's84_0002':
        return 620.0  # cm
    else:
        return 410.0  # cm


def get_last_bin_edge(cell_id):
    if cell_id == 's82_0002' or cell_id == 's84_0002':
        return 616.58592993193  # cm
    else:
        return 405.70317910647157  # cm

    # bin edges in Domnisoru
    # units VR: -1140 : 28.94973617378945 : 1209 (2430 for 6 m tracks)
    # units cm: -196.8929860286835 : 5.0 : 208.81019307778803 (419.69294390324643 for 6 m tracks)



def get_cell_groups():
    """
    Cell groups based on ISI return maps.
    """
    # TODO: proper criterion for recognition of groups
    edge_accumulating = ['s66_0003', 's76_0002', 's79_0003', 's101_0009', 's117_0002', 's119_0004', 's120_0002']
    edge_avoiding = ['s81_0004', 's84_0002', 's85_0007', 's90_0006', 's96_0009', 's100_0006', 's115_0018', 's115_0024',
                's115_0030', 's120_0023']
    theta = ['s67_0000', 's73_0004', 's109_0002', 's118_0002']
    dap = ['s79_0003', 's104_0007', 's109_0002', 's110_0002', 's119_0004']
    all = ['s43_0003', 's66_0003', 's67_0000', 's73_0004', 's74_0006', 's76_0002', 's79_0003', 's81_0004', 's82_0002',
           's84_0002', 's85_0007', 's90_0006', 's95_0006', 's96_0009', 's100_0006', 's101_0009', 's104_0007',
           's109_0002',  's110_0002', 's115_0018', 's115_0024', 's115_0030', 's117_0002', 's118_0002', 's119_0004',
           's120_0002', 's120_0023']
    return {'edge-accumulating': edge_accumulating, 'edge-avoiding': edge_avoiding, 'theta': theta, 'DAP': dap,
            'all': all}


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_ids = load_cell_ids(save_dir, 'grid_cells')

    # grid_cell_name = grid_cell_names[3]
    # print grid_cell_name
    # param_list = ['Vm_ljpc', 'Y_cm']
    # data = load_ISIs(grid_cell_name, param_list, save_dir)
    #
    # fig, axes = pl.subplots(1, 1, sharex='all')
    # axes.plot(np.arange(len(data['Vm_ljpc'])) * data['dt'] / 1000., data['Vm_ljpc'], 'k')
    # axes.set_ylabel('Membrane Potential')
    # axes.set_xlabel('Time (s)')
    # pl.tight_layout()
    # pl.show()

    # import json
    # cell_type_dict = {cell_id: get_celltype(cell_id, save_dir) for cell_id in cell_ids}
    # with open(os.path.join(save_dir, 'cell_types.json'), 'w') as f:
    #     json.dump(cell_type_dict, f, indent=4)