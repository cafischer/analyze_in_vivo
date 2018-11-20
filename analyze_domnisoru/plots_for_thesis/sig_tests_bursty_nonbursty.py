from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.stats import ttest_ind
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_celltype_dict, get_cell_ids_bursty, get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from analyze_in_vivo.analyze_domnisoru.plot_utils import horizontal_square_bracket, get_star_from_p_val
pl.style.use('paper_subplots')


save_dir_img2 = '/home/cf/Dropbox/thesis/figures_results'
save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
save_dir_ISI_hist = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_hist/cut_ISIs_at_200/grid_cells'
save_dir_n_spikes_in_burst = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/bursting/grid_cells'
save_dir_ISI_return_map = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/ISI_return_map/cut_ISIs_at_200/grid_cells'
cell_type_dict = get_celltype_dict(save_dir)
theta_cells = load_cell_ids(save_dir, 'giant_theta')
DAP_cells = get_cell_ids_DAP_cells()

fraction_burst = np.load(os.path.join(save_dir_ISI_hist, 'fraction_burst.npy'))
fraction_single = np.load(os.path.join(save_dir_n_spikes_in_burst, 'fraction_single.npy'))
fraction_ISI_or_ISI_next_burst = np.load(os.path.join(save_dir_ISI_return_map, 'fraction_ISI_or_ISI_next_burst.npy'))

cell_ids = np.array(load_cell_ids(save_dir, 'grid_cells'))
burst_label = np.array([True if cell_id in get_cell_ids_bursty() else False for cell_id in cell_ids])

# significance tests
_, p_val_fraction_burst = ttest_ind(fraction_burst[burst_label], fraction_burst[~burst_label])
_, p_val_fraction_single = ttest_ind(fraction_single[burst_label], fraction_single[~burst_label])
_, p_val_fraction_ISI_or_ISI_next_burst = ttest_ind(fraction_ISI_or_ISI_next_burst[burst_label],
                                                    fraction_ISI_or_ISI_next_burst[~burst_label])

# plot
n_bursty = sum(burst_label)
n_nonbursty = sum(~burst_label)

fig, axes = pl.subplots(1, 3)

plot_with_markers(axes[0], np.zeros(n_bursty), fraction_burst[burst_label], cell_ids[burst_label], cell_type_dict,
                  edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
plot_with_markers(axes[0], np.ones(n_nonbursty), fraction_burst[~burst_label], cell_ids[~burst_label], cell_type_dict,
                  edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
horizontal_square_bracket(axes[0], get_star_from_p_val(p_val_fraction_burst), 0, 1, 1.03, 1.04, 0.0)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['bursty', 'non-bursty'])
axes[0].set_xlim([-0.5, 1.5])
axes[0].set_ylim([0, 1.1])
axes[0].set_ylabel('Fraction ISIs $\leq$ 8ms')

plot_with_markers(axes[1], np.zeros(n_bursty), fraction_single[burst_label], cell_ids[burst_label], cell_type_dict,
                  edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
plot_with_markers(axes[1], np.ones(n_nonbursty), fraction_single[~burst_label], cell_ids[~burst_label], cell_type_dict,
                  edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
horizontal_square_bracket(axes[1], get_star_from_p_val(p_val_fraction_single), 0, 1, 1.03, 1.04, 0.0)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['bursty', 'non-bursty'])
axes[1].set_xlim([-0.5, 1.5])
axes[1].set_ylim([0, 1.1])
axes[1].set_ylabel('Fraction single spikes')

plot_with_markers(axes[2], np.zeros(n_bursty), fraction_ISI_or_ISI_next_burst[burst_label],
                  cell_ids[burst_label], cell_type_dict,
                  edgecolor='r', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
plot_with_markers(axes[2], np.ones(n_nonbursty), fraction_ISI_or_ISI_next_burst[~burst_label],
                  cell_ids[~burst_label], cell_type_dict,
                  edgecolor='b', theta_cells=theta_cells, DAP_cells=DAP_cells, legend=False)
horizontal_square_bracket(axes[2], get_star_from_p_val(p_val_fraction_ISI_or_ISI_next_burst), 0, 1, 1.03, 1.04, 0.0)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['bursty', 'non-bursty'])
axes[2].set_xlim([-0.5, 1.5])
axes[2].set_ylim([0, 1.1])
axes[2].set_ylabel('Fraction ISI[n] or ISI[n+1] $\leq$ 8ms')

pl.tight_layout()
pl.savefig(os.path.join(save_dir_img2, 'sig_tests_bursty_nonbursty'))
pl.show()