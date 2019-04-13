import numpy as np
from analyze_in_vivo.load.load_domnisoru import load_data, load_field_indices, get_last_bin_edge

save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
cell_id = 's43_0003'
#cell_id = 's82_0002'
bins = np.arange(0, get_last_bin_edge(cell_id), 5)

# load
in_field_idx, _ = load_field_indices(cell_id, save_dir)
position_thresholded = load_data(cell_id, ['fY_cm'], save_dir)['fY_cm']

in_field_bool_idx = np.zeros(len(position_thresholded), dtype=bool)
in_field_bool_idx[in_field_idx] = True

# get labels for each bin
bins_in_field = np.unique(np.digitize(position_thresholded, bins)[in_field_bool_idx])
bins_not_in_field = np.unique(np.digitize(position_thresholded, bins)[~in_field_bool_idx])

# test that no bin defined as in-field and not in-field
overlapping_bins = bins_in_field[np.array([bin in bins_not_in_field for bin in bins_in_field])]
print 'overlapping bins: ', overlapping_bins

# check positions for an example overlapping bin
overlapping_bin = overlapping_bins[0]
position_in_field = position_thresholded[np.logical_and(in_field_bool_idx,
                                                        np.digitize(position_thresholded, bins) == overlapping_bin)]
position_not_in_field = position_thresholded[np.logical_and(~in_field_bool_idx,
                                                            np.digitize(position_thresholded, bins) == overlapping_bin)]

print 'in-field (bin %i)' % overlapping_bin
print 'min: %.2f, max: %.2f' % (np.min(position_in_field), np.max(position_in_field))
print 'not in-field (bin %i)' % overlapping_bin
print 'min: %.2f, max: %.2f' % (np.min(position_not_in_field), np.max(position_not_in_field))