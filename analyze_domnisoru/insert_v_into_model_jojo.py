from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from nrn_wrapper import Cell
from cell_fitting.optimization.simulate import get_standard_simulation_params, iclamp_handling_onset
pl.style.use('paper_subplots')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/v_as_input'
    model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2/cell_rounded.json'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'
    cell_type = 'giant_theta'
    cell_ids = load_cell_ids(save_dir, cell_type)

    # parameters
    param_list = ['Vm_ljpc', 'Vm_wo_spikes_ljpc']
    resistance = 11.  # s73_0004: 10.  s104_0007: 12.
    add_factor = 0.75  # s73_0004: 0.78  s104_0007: 0.74

    # create cell model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    simulation_params = get_standard_simulation_params()

    # main
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc'][:1000000]
        v_sub = data['Vm_wo_spikes_ljpc'][:1000000]
        t = np.arange(0, len(v_sub)) * data['dt']
        dt = t[1] - t[0]

        # simulate model cell
        i_inj = v_sub / resistance
        i_inj -= np.min(i_inj) * add_factor
        simulation_params['i_inj'] = i_inj
        simulation_params['tstop'] = t[-1]
        simulation_params['dt'] = dt
        v_model, t_model, i_inj_model = iclamp_handling_onset(cell, **simulation_params)

        if not os.path.exists(os.path.join(save_dir_img, cell_id)):
            os.makedirs(os.path.join(save_dir_img, cell_id))

        # plot
        pl.figure()
        pl.title('Current')
        pl.plot(t_model, i_inj)
        pl.xlim(20000, 20800)
        #pl.xlim(23000, 23800)
        pl.savefig(os.path.join(save_dir_img, cell_id, 'current.png'))

        pl.figure()
        pl.title('Model')
        pl.plot(t_model, v_model)
        pl.xlim(20000, 20800)
        #pl.xlim(23000, 23800)
        pl.savefig(os.path.join(save_dir_img, cell_id, 'model.png'))

        pl.figure()
        pl.title(cell_id)
        pl.plot(t, v)
        #pl.plot(t, v_sub)
        pl.xlim(20000, 20800)
        #pl.xlim(23000, 23800)
        pl.savefig(os.path.join(save_dir_img, cell_id, 'data.png'))
       # pl.show()