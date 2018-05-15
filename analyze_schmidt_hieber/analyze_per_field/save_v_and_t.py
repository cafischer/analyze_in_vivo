from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as pl
from load import load_field_crossings, get_stellate_info


if __name__ == '__main__':

    folder = 'schmidthieber'
    save_dir = '../results/' + folder + '/data'
    save_dir_data = '../data/'

    # load and save v and t
    stellate_info = get_stellate_info(save_dir_data)

    for cell_id in stellate_info.keys():
        for field in stellate_info[cell_id].keys():
            for crossing in range(stellate_info[cell_id][field]):
                print cell_id

                # load
                v, t, x_pos, y_pos, pos_t, speed, speed_t = load_field_crossings(save_dir_data, cell_id, field,
                                                                                 crossing)

                # save and plot
                save_dir_cell_field_crossing = os.path.join(save_dir, cell_id + '_' + str(field) + '_' + str(crossing))
                if not os.path.exists(save_dir_cell_field_crossing):
                    os.makedirs(save_dir_cell_field_crossing)

                np.save(os.path.join(save_dir_cell_field_crossing, 'v.npy'), v)
                np.save(os.path.join(save_dir_cell_field_crossing, 't.npy'), t)
                np.save(os.path.join(save_dir_cell_field_crossing, 'position.npy'), y_pos)
                np.save(os.path.join(save_dir_cell_field_crossing, 'pos_t.npy'), pos_t)

                pl.figure()
                pl.plot(t, v, 'k')
                pl.ylabel('Membrane potential (mV)', fontsize=16)
                pl.xlabel('Time (ms)', fontsize=16)
                pl.savefig(os.path.join(save_dir_cell_field_crossing, 'v.svg'))
                pl.show()

                # pl.figure()
                # pl.title(cell_id + '_' + str(field) + '_' + str(crossing))
                # pl.plot(pos_t, y_pos, 'k')
                # pl.ylabel('Position (cm)', fontsize=16)
                # pl.xlabel('Time (ms)', fontsize=16)
                # pl.savefig(os.path.join(save_dir_cell_field_crossing, 'y_pos.svg'))
                # pl.show()