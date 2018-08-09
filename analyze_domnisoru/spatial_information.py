from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data, get_celltype_dict, get_last_bin_edge, \
    get_cell_ids_DAP_cells
from analyze_in_vivo.analyze_domnisoru.position_vs_firing_rate import get_spike_train
from grid_cell_stimuli import get_AP_max_idxs
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers, plot_for_all_grid_cells
from cell_fitting.util import init_nan
from sklearn import linear_model, decomposition
pl.style.use('paper')


def get_MI(x, y, bins_x, bins_y, n_bins_x, n_bins_y, save_dir_img=None, file_name='MI.png'):
    p_x = np.histogram(x, bins_x)[0] / float(len(x))
    p_y = np.histogram(y, bins_y)[0] / float(len(y))
    prob_x_and_y = np.histogram2d(x, y, (bins_x, bins_y))[0] / float(len(x))
    prob_x_times_y = np.tile(np.array([p_x]).T, n_bins_y) * np.tile(p_y, (n_bins_x, 1))

    prob_x_and_y_flat = prob_x_and_y.flatten()
    prob_x_times_y_flat = prob_x_times_y.flatten()
    summands = np.zeros((n_bins_x * n_bins_y))
    not_zero = np.logical_and(prob_x_and_y_flat != 0, prob_x_times_y_flat != 0)
    summands[not_zero] = prob_x_and_y_flat[not_zero] \
                         * (np.log(prob_x_and_y_flat[not_zero] / prob_x_times_y_flat[not_zero]) / np.log(2))
    MI = np.sum(summands)

    # plots for testing
    pl.set_cmap('plasma')
    x, y = np.meshgrid(bins_y, bins_x)
    # pl.figure()
    # pl.pcolor(x, y, prob_x_and_y)
    # pl.colorbar()
    # pl.clim(0, 0.01)
    #
    # pl.figure()
    # pl.pcolor(x, y, prob_x_times_y)
    # pl.colorbar()
    # pl.clim(0, 0.01)
    #
    pl.figure()
    pl.title('MI %.2f' % MI)
    MI_in_shape = init_nan(np.shape(prob_x_and_y))
    non_zero = np.logical_and(prob_x_and_y != 0, prob_x_times_y != 0)
    MI_in_shape[non_zero] = prob_x_and_y[non_zero] * (np.log(prob_x_and_y[non_zero] / prob_x_times_y[non_zero]) / np.log(2))
    pl.pcolor(x, y, MI_in_shape)
    pl.colorbar()
    min_MI_v = np.min(np.min(MI_in_shape))
    max_MI_v = np.max(np.max(MI_in_shape))
    # if min_MI_v < -0.02:
    #     print 'min_MI_v', min_MI_v
    # if max_MI_v > 0.35:
    #     print 'max_MI_v', max_MI_v
    pl.clim(-0.02, 0.03)
    pl.xlabel('Position (cm)')
    pl.ylabel('Mem. Pot. (mV)')
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, file_name))
    #pl.show()
    return MI


if __name__ == '__main__':
    save_dir_img = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/spatial_info'
    save_dir_firing_rate = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/position_vs_firing_rate/vel_thresh_1'
    save_dir_fields = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/in_out_field/vel_thresh_1'
    save_dir_rec_info = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/recording_info'
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cell_type = 'grid_cells'
    cell_ids = load_cell_ids(save_dir, cell_type)
    cell_type_dict = get_celltype_dict(save_dir)

    AP_thresholds = {'s73_0004': -50, 's90_0006': -45, 's82_0002': -38, 's117_0002': -60, 's119_0004': -50,
                     's104_0007': -55, 's79_0003': -50, 's76_0002': -50, 's101_0009': -45}
    param_list = ['Vm_ljpc', 'spiketimes', 'Y_cm']

    # parameters
    bin_size = 10.0  # cm  # done: 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    bins_v = np.arange(-90, 20, 5)  # mV
    use_AP_max_idxs_domnisoru = True
    save_dir_img = os.path.join(save_dir_img, cell_type, 'bin_size_'+str(bin_size))
    save_dir_firing_rate = os.path.join(save_dir_firing_rate, cell_type, 'bin_size_'+str(bin_size))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    # load
    avg_firing_rate = np.load(os.path.join(save_dir_rec_info, cell_type, 'avg_firing_rate.npy'))
    n_runs = np.load(os.path.join(save_dir_rec_info, cell_type, 'n_runs.npy'))
    n_fields = np.load(os.path.join(save_dir_fields, cell_type, 'n_fields.npy'))

    inv_entropy = np.zeros(len(cell_ids))
    spatial_info_skaggs = np.zeros(len(cell_ids))
    MI_v = np.zeros(len(cell_ids))
    # MI_spiketrain = np.zeros(len(cell_ids))
    position_cells = []
    firing_rate_cells = []
    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        # load
        data = load_data(cell_id, param_list, save_dir)
        v = data['Vm_ljpc']
        t = np.arange(0, len(v)) * data['dt']
        pos = data['Y_cm']
        dt = t[1] - t[0]
        firing_rate = np.load(os.path.join(save_dir_firing_rate, cell_id, 'firing_rate.npy'))
        position = np.load(os.path.join(save_dir_firing_rate, cell_id, 'position.npy'))
        occupancy_prob = np.load(os.path.join(save_dir_firing_rate, cell_id, 'occupancy_prob.npy'))
        firing_rate_cells.append(firing_rate)
        position_cells.append(position)
        firing_rate_not_nan = firing_rate[~np.isnan(firing_rate)]
        bin_size_position = position[1] - position[0]

        # entropy
        prob_position = firing_rate_not_nan / (np.sum(firing_rate_not_nan) * bin_size_position)
        summands = np.zeros(len(firing_rate_not_nan))
        summands[prob_position != 0] = prob_position[prob_position != 0] * (np.log(prob_position[prob_position != 0])
                                                                             / np.log(2))  # by definition: if prob=0,
                                                                                           # then summand = 0
        entropy_cell = -np.sum(summands)

        prob_position_uniform = np.ones(len(position)) / (len(position) * bin_size_position)
        summands = prob_position_uniform * (np.log(prob_position_uniform) / np.log(2))
        entropy_uniform = -np.sum(summands)
        inv_entropy[cell_idx] = 1 - entropy_cell / entropy_uniform

        # Skaggs, 1996 spatial information
        scaled_firing_rate = firing_rate_not_nan / np.mean(firing_rate_not_nan)
        occupancy_prob_not_nan = occupancy_prob[~np.isnan(firing_rate)]
        summands = np.zeros(len(firing_rate_not_nan))
        summands[scaled_firing_rate != 0] = occupancy_prob_not_nan[scaled_firing_rate != 0] \
                                           * scaled_firing_rate[scaled_firing_rate != 0] \
                                           * (np.log(scaled_firing_rate[scaled_firing_rate != 0]) / np.log(2))
        spatial_info_skaggs[cell_idx] = np.sum(summands)

        # Mutual information: voltage and position
        n_bins_v = len(bins_v) - 1
        v_binned = np.digitize(v, bins_v) - 1
        bins_pos = np.arange(0, get_last_bin_edge(cell_id) + bin_size, bin_size)
        n_bins_pos = len(bins_pos) - 1
        pos_binned = np.digitize(pos, bins_pos) - 1

        # plot
        save_dir_cell = os.path.join(save_dir_img, cell_id)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)
        # pl.close('all')
        # pl.figure()
        # pl.plot(pos, v, 'k')
        # pl.xlabel('Position (cm)')
        # pl.ylabel('Mem. Pot. (mV)')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_cell, 'pos_vs_v.png'))

        MI_v[cell_idx] = get_MI(v_binned, pos_binned, bins_v, bins_pos, n_bins_v, n_bins_pos)

        # # Mutual information: spiketrain and position
        # if use_AP_max_idxs_domnisoru:
        #     AP_max_idxs = data['spiketimes']
        # else:
        #     AP_max_idxs = get_AP_max_idxs(v, AP_thresholds[cell_id], dt)
        # spike_train = get_spike_train(AP_max_idxs, len(v))
        # MI_spiketrain[cell_idx] = get_MI(spike_train, pos_binned, np.array([0, 1, 2]), bins_pos,
        #                                  2, n_bins_pos)

    np.save(os.path.join(save_dir_img, 'spatial_info.npy'), spatial_info_skaggs)
    np.save(os.path.join(save_dir_img, 'MI_v.npy'), MI_v)

    pl.close('all')
    if cell_type == 'grid_cells':
        def plot_firing_rate_and_spatial_info(ax, cell_idx, position_cells, firing_rate_cells, spatial_info_skaggs):
            ax.plot(position_cells[cell_idx], firing_rate_cells[cell_idx], 'k')
            ax.annotate('Skaggs, 1996: %.2f' % spatial_info_skaggs[cell_idx],
                         xy=(0.1, 0.9), xycoords='axes fraction', fontsize=8, ha='left', va='top',
                         bbox=dict(boxstyle='round', fc='w', edgecolor='0.8', alpha=0.8))

        plot_kwargs = dict(position_cells=position_cells, firing_rate_cells=firing_rate_cells,
                           spatial_info_skaggs=spatial_info_skaggs)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_firing_rate_and_spatial_info, plot_kwargs,
                        xlabel='Position (cm)', ylabel='Firing rate (Hz)',
                        save_dir_img=os.path.join(save_dir_img, 'firing_rate_spatial_info.png'))

        def plot_spatial_info(ax, cell_idx, spatial_info):
            ax.bar(1, (spatial_info[cell_idx]), width=0.8, color='0.5')

            ax.set_xlim(0, 2)
            ax.set_ylim(0, np.max(spatial_info))
            ax.set_xticks([])

        plot_kwargs = dict(spatial_info=spatial_info_skaggs)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_spatial_info, plot_kwargs,
                                xlabel='', ylabel='Spatial information',
                                save_dir_img=os.path.join(save_dir_img, 'spatial_info_skaggs.png'))

        plot_kwargs = dict(spatial_info=MI_v)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_spatial_info, plot_kwargs,
                                xlabel='', ylabel='MI (mem. pot.)',
                                save_dir_img=os.path.join(save_dir_img, 'MI_v.png'))

        plot_kwargs = dict(spatial_info=inv_entropy)
        plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_spatial_info, plot_kwargs,
                                xlabel='', ylabel='Inv. entropy',
                                save_dir_img=os.path.join(save_dir_img, 'inv_entropy.png'))

        # pl.figure()
        # pl.plot(inv_entropy, spatial_info_skaggs, 'ok')
        # pl.ylabel('Skaggs, 1996')
        # pl.xlabel('Inv. entropy')
        # pl.tight_layout()
        # pl.savefig(os.path.join(save_dir_img, 'entropy_vs_skaggs.png'))

        # MI(v) vs Skaggs
        regression = linear_model.LinearRegression()
        regression.fit(np.array([MI_v]).T, np.array([spatial_info_skaggs]).T)

        pca = decomposition.PCA(n_components=1)
        pca.fit(np.vstack([MI_v, spatial_info_skaggs]).T)

        theta_cells = load_cell_ids(save_dir, 'giant_theta')
        DAP_cells = get_cell_ids_DAP_cells()
        fig, ax = pl.subplots()
        plot_with_markers(ax, MI_v, spatial_info_skaggs, cell_ids, cell_type_dict, theta_cells, DAP_cells)
        pl.plot(MI_v, regression.coef_[0, 0] * MI_v + regression.intercept_[0], 'r')
        ax.annotate('', pca.mean_, pca.mean_ + (pca.components_[0] * np.sqrt(pca.explained_variance_[0])),
                    arrowprops=dict(arrowstyle='<-', color='k'))
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('MI (mem. pot.)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'mi_vs_skaggs.png'))

        fig, ax = pl.subplots()
        plot_with_markers(ax, MI_v, spatial_info_skaggs, cell_ids, cell_type_dict, theta_cells, DAP_cells)
        pl.plot(MI_v, regression.coef_[0, 0] * MI_v + regression.intercept_[0], 'r')
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('MI (mem. pot.)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'mi_vs_skaggs_with_regression.png'))
        pl.show()

        # plot dependence of Skaggs, 1996 measure on firing characteristics
        fig, ax = pl.subplots()
        plot_with_markers(ax, avg_firing_rate, spatial_info_skaggs, cell_ids, cell_type_dict)
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('Avg. firing rate (Hz)')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'firing_rate_vs_skaggs.png'))

        fig, ax = pl.subplots()
        plot_with_markers(ax, n_runs, spatial_info_skaggs, cell_ids, cell_type_dict)
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('# Runs')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'n_runs_vs_skaggs.png'))

        fig, ax = pl.subplots()
        pl.plot(n_fields, spatial_info_skaggs, 'ok')
        plot_with_markers(ax, n_fields, spatial_info_skaggs, cell_ids, cell_type_dict)
        pl.ylabel('Skaggs, 1996')
        pl.xlabel('# Fields')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'n_fields_vs_skaggs.png'))

        pl.show()