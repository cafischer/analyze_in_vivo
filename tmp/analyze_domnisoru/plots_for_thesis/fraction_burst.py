import matplotlib.pyplot as pl
import numpy as np
import os
from analyze_in_vivo.load.load_domnisoru import get_cell_ids_DAP_cells, get_celltype_dict, load_cell_ids
pl.style.use('paper')


if __name__ == '__main__':
    save_dir_img = '/home/cf/Dropbox/thesis/figures_results'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist'
    cell_type_dict = get_celltype_dict(save_dir)
    grid_cells = np.array(load_cell_ids(save_dir, 'grid_cells'))
    stellate_cells = load_cell_ids(save_dir, 'stellate_cells')
    pyramidal_cells = load_cell_ids(save_dir, 'pyramidal_cells')
    stellate_idxs = np.array([np.where(cell_id == grid_cells)[0][0] for cell_id in stellate_cells])
    pyramidal_idxs = np.array([np.where(cell_id == grid_cells)[0][0] for cell_id in pyramidal_cells])

    # load
    fraction_burst = np.load(os.path.join(save_dir_ISI_hist,  'cut_ISIs_at_200', 'grid_cells', 'fraction_burst.npy'))

    # plot
    fig, ax = pl.subplots(figsize=(4, 6))
    ax.plot(np.zeros(len(stellate_cells)), fraction_burst[stellate_idxs], 'ok')
    ax.errorbar(0.2, np.mean(fraction_burst[stellate_idxs]), yerr=np.std(fraction_burst[stellate_idxs]),
                marker='o', color='k', capsize=3)
    ax.plot(np.ones(len(pyramidal_cells))*0.6, fraction_burst[pyramidal_idxs], 'ok')
    ax.errorbar(0.8, np.mean(fraction_burst[pyramidal_idxs]), yerr=np.std(fraction_burst[pyramidal_idxs]),
                marker='o', color='k', capsize=3)
    ax.set_xlim(-0.2, 1.0)
    ax.set_xticks([0, 0.6])
    ax.set_xticklabels(['Stellate', 'Pyramidal'])
    ax.set_ylabel('Fraction burst')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'fraction_burst.png'))
    pl.show()
