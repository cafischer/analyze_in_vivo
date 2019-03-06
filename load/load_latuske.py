import pickle
import numpy as np
import sys
from analyze_in_vivo.load.mec_classes import mec_classes as mec_classes



def load_ISIs(save_dir='/home/cf/Phd/programming/data/Caro/grid_cells_withfields_vt_0.pkl'):
    sys.modules['mec_classes'] = mec_classes  # pickle needs the same module structure see:
    # https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory

    pkl_file = open(save_dir, 'rb')
    session_cells = pickle.load(pkl_file)
    grid_cells = [cell for cell in session_cells.cells]

    ISIs_cells = np.zeros(len(grid_cells), dtype=object)
    for i in range(len(grid_cells)):
        gc = grid_cells[i]
        gc.st = gc.st * 1000  # spike times, change unit to ms
        ISIs_cells[i] = np.diff(gc.st)

    return ISIs_cells