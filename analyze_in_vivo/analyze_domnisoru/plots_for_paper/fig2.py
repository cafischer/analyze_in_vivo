from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, get_cell_ids_DAP_cells, get_celltype_dict, \
    get_label_burstgroups, get_colors_burstgroups, get_cell_layer_dict
from analyze_in_vivo.analyze_domnisoru.plot_utils import plot_with_markers
from sklearn.linear_model import LinearRegression
pl.style.use('paper')


def center_STA_to_AP_onset(sta_mean_cells, v_onset_cells):
    sta_mean_cells_centered = np.zeros((len(sta_mean_cells), len(sta_mean_cells[0])))
    for cell_idx in range(len(sta_mean_cells)):
        sta_mean_cells_centered[cell_idx] = sta_mean_cells[cell_idx] - v_onset_cells[cell_idx]
    return sta_mean_cells_centered


if __name__ == '__main__':
    save_dir_img = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper'
    save_dir = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/extra'
    save_dir_fig2 = '/home/cfischer/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/paper/fig2'
    
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    cell_type = 'grid_cells'
    grid_cells = np.array(load_cell_ids(save_dir, cell_type))
    cell_layer_dict = get_cell_layer_dict(save_dir)
    labels_burstgroups = get_label_burstgroups()
    colors_burstgroups = get_colors_burstgroups()

    # parameters
    with_selection = True
    use_avg_times = False
    thresh = '1der'
    AP_thresh_derivative = 15
    do_detrend = False
    dt = 0.05
    before_AP = 25  # ms
    after_AP = 25  # ms
    t_vref = 10  # ms
    AP_criterion = {'AP_amp_and_width': (40, 1)}
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells = get_cell_ids_DAP_cells(new=True)
    cell_type_dict = get_celltype_dict(save_dir)

    # load
    sta_mean_cells = np.load(os.path.join(save_dir_fig2, 'sta_mean_cells.npy'))
    t_sta = np.arange(-before_AP, after_AP+dt, dt)
    v_onset_fAHP = np.load(os.path.join(save_dir_fig2, 'v_onset_fAHP.npy'))
    v_DAP_fAHP = np.load(os.path.join(save_dir_fig2, 'v_DAP_fAHP.npy'))
    AP_onset_idx_cells = np.load(os.path.join(save_dir_fig2, 'AP_onset_idx_cells.npy'))
    DAP_max_idx_cells = np.load(os.path.join(save_dir_fig2, 'DAP_max_idx_cells.npy'))
    fAHP_min_idx_cells = np.load(os.path.join(save_dir_fig2, 'fAHP_min_idx_cells.npy'))
    AP_max_idx_cells = np.load(os.path.join(save_dir_fig2, 'AP_max_idx_cells.npy'))
    v_onset = np.load(os.path.join(save_dir_fig2, 'v_onset.npy'))
    v_fAHP = np.load(os.path.join(save_dir_fig2, 'v_fAHP.npy'))
    v_DAP = np.load(os.path.join(save_dir_fig2, 'v_DAP.npy'))

    # average v_onset values
    print 'mean v_onset B+D: %.2f' % np.mean(v_onset[labels_burstgroups['B+D']])
    print 'mean v_onset B: %.2f' % np.mean(v_onset[labels_burstgroups['B']])
    print 'mean v_onset NB: %.2f' % np.mean(v_onset[labels_burstgroups['NB']])
    print 'std v_onset B+D: %.2f' % np.std(v_onset[labels_burstgroups['B+D']])
    print 'std v_onset B: %.2f' % np.std(v_onset[labels_burstgroups['B']])
    print 'std v_onset NB: %.2f' % np.std(v_onset[labels_burstgroups['NB']])

    # linear regression
    v_onset_fAHP_bursty = v_onset_fAHP[np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])]
    v_DAP_fAHP_bursty = v_DAP_fAHP[np.logical_or(labels_burstgroups['B'], labels_burstgroups['B+D'])]
    slope = LinearRegression().fit(np.array([v_onset_fAHP_bursty]).T, v_DAP_fAHP_bursty).coef_[0]
    print 'Slope: %.3f' % slope

    # bootstraping
    n_bootstraps = 10000
    slopes = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        idxs = np.random.randint(0, len(v_onset_fAHP_bursty), len(v_onset_fAHP_bursty))
        v_onset_fAHP_sample = v_onset_fAHP_bursty[idxs]
        v_DAP_fAHP_sample = v_DAP_fAHP_bursty[idxs]
        slopes[i] = LinearRegression().fit(np.array([v_onset_fAHP_sample]).T, v_DAP_fAHP_sample).coef_
    print 'Mean slope: %.3f' % np.mean(slopes)
    print 'SE slope: %.3f' % np.std(slopes)
    print 'Bias: %.3f' % (np.mean(slopes) - slope)

    # plot
    fig = pl.figure(figsize=(6, 7.5))
    outer = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])
    cell_examples = ['s84_0002', 's109_0002', 's118_0002']
    colors_examples = [colors_burstgroups['NB'], colors_burstgroups['B+D'], colors_burstgroups['B']]
    letters_examples = ['a', 'b', 'c']

    # A
    inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, :])
    ax = pl.Subplot(fig, inner[0])
    fig.add_subplot(ax)
    plot_with_markers(ax, v_onset_fAHP[labels_burstgroups['B']], v_DAP_fAHP[labels_burstgroups['B']],
                      grid_cells[labels_burstgroups['B']], cell_type_dict, theta_cells=theta_cells,
                      edgecolor=colors_burstgroups['B'], legend=False)
    plot_with_markers(ax, v_onset_fAHP[labels_burstgroups['B+D']], v_DAP_fAHP[labels_burstgroups['B+D']],
                      grid_cells[labels_burstgroups['B+D']], cell_type_dict, theta_cells=theta_cells,
                      edgecolor=colors_burstgroups['B+D'], legend=False)
    handles = plot_with_markers(ax, v_onset_fAHP[labels_burstgroups['NB']], v_DAP_fAHP[labels_burstgroups['NB']],
                                grid_cells[labels_burstgroups['NB']], cell_type_dict, theta_cells=theta_cells,
                                edgecolor=colors_burstgroups['NB'], legend=False)
    ax.set_ylabel(r'$\mathrm{\Delta V_{DAP}}$', horizontalalignment='left', y=0.0)
    ax.set_xlabel(r'$\mathrm{\Delta V_{fAHP}}$', horizontalalignment='right', x=1.0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    handles += [Patch(color=colors_burstgroups['B+D'], label='Bursty+DAP'),
                Patch(color=colors_burstgroups['B'], label='Bursty-DAP'),
                Patch(color=colors_burstgroups['NB'], label='Non-bursty')]
    # TODO add legend for layers (or maybe not?) handles += []
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    for cell_idx, cell_id in enumerate(grid_cells):
        if not cell_id == 's81_0004' and not cell_id == 's115_0030':
            ax.annotate(cell_layer_dict[cell_id], xy=(v_onset_fAHP[cell_idx]+0.2, v_DAP_fAHP[cell_idx]-0.3), fontsize=8)

    for i, cell_id in enumerate(cell_examples):
        cell_idx = np.where(grid_cells == cell_id)[0][0]
        ax.annotate('('+letters_examples[i]+')', xy=(v_onset_fAHP[cell_idx]-0.15, v_DAP_fAHP[cell_idx]),
                    va='bottom', ha='right', fontsize=10)

    # arrows for error propagation
    ax.annotate('', xytext=(-3., 2.5),
                  xy=(-1.5, 2.5),
                  arrowprops=dict(arrowstyle="<|-|>", shrinkA=5, shrinkB=1, color='g'))
    ax.annotate(r'$\mathrm{AP_{thresh}}$',
                  xy=(-2.25, 2.75),
                  ha='center', va='center')
    ax.annotate('', xytext=(-1.5, 2.5),
                  xy=(-1.5, 1.0),
                  arrowprops=dict(arrowstyle="<|-|>", shrinkA=1, shrinkB=3, color='g'))
    ax.annotate(r'$\mathrm{DAP_{max}}$',
                  xy=(-1.5, 1.7),
                  ha='left', va='center')
    ax.annotate('', xytext=(-3., 2.5),
                 xy=(-1.5, 1.0),
                 arrowprops=dict(arrowstyle="<|-|>", shrinkA=5, shrinkB=2, color='g'))
    ax.annotate(r'$\mathrm{fAHP_{min}}$',
                xy=(-2.05, 1.8),
                ha='right', va='top')

    # inset
    axin = inset_axes(ax, width='40%', height='45%', loc='lower left')
    cell_id = 's104_0007'
    cell_idx = np.where(cell_id == grid_cells)[0]
    idx1 = int(round(12 / dt))
    idx2 = int(round(5 / dt))
    axin.plot(t_sta[idx1:-idx2], sta_mean_cells[cell_idx][0][idx1:-idx2], color='k')
    axin.set_ylim(-69.5, -55)
    axin.set_xlim(-15, t_sta[-1])
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_xlabel('Time', horizontalalignment='right', x=1.0, fontsize=10)
    axin.set_ylabel('Voltage', horizontalalignment='right', y=1.0, fontsize=10)

    axin.plot([t_sta[0], t_sta[AP_onset_idx_cells[cell_idx]]], [v_onset[cell_idx], v_onset[cell_idx]], color='g')  # AP onset line
    axin.plot([t_sta[0], t_sta[-1]+5], [v_fAHP[cell_idx], v_fAHP[cell_idx]], color='g')  # fAHP min. line
    axin.plot([t_sta[int(DAP_max_idx_cells[cell_idx])], t_sta[-1]+5], [v_DAP[cell_idx], v_DAP[cell_idx]], color='g')  # DAP max. line
    # annotate delta fAHP
    # axin.annotate('', xytext=(t_sta[to_idx(6.5, dt)], v_fAHP[cell_idx]),
    #               xy=(t_sta[to_idx(6.5, dt)], v_onset[cell_idx]),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta V_{fAHP}}$',
                  xy=(t_sta[int(round(10.2 / dt))], (v_fAHP[cell_idx] - v_onset[cell_idx])/2. + v_onset[cell_idx]),
                  ha='left', va='center')
    # annotate delta DAP
    # axin.annotate('', xytext=(t_sta[-to_idx(6.5, dt)], v_fAHP[cell_idx]),
    #               xy=(t_sta[-to_idx(6.5, dt)], v_DAP[cell_idx]),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta V_{DAP}}$',
                  xy=(t_sta[-int(round(8.5 / dt))], (v_DAP[cell_idx] - v_fAHP[cell_idx])/2. + v_fAHP[cell_idx]-0.3),
                  ha='left', va='center')

    axin.plot([t_sta[int(DAP_max_idx_cells[cell_idx])], t_sta[int(DAP_max_idx_cells[cell_idx])]],
              [axin.get_ylim()[0], v_DAP[cell_idx]], color='g', linestyle='--')  # t_DAP line
    axin.plot([t_sta[int(AP_max_idx_cells[cell_idx])], t_sta[int(AP_max_idx_cells[cell_idx])]],
              [-62, axin.get_ylim()[0]], color='g', linestyle='--')  # t_AP line
    axin.plot([t_sta[int(fAHP_min_idx_cells[cell_idx])], t_sta[int(fAHP_min_idx_cells[cell_idx])]],
              [-62, v_fAHP[cell_idx]], color='g', linestyle='--')  # t_fAHP line

    # annotate t_DAP
    # axin.annotate('', xytext=(t_sta[int(AP_max_idx_cells[cell_idx])], axin.get_ylim()[0]),
    #               xy=(t_sta[int(DAP_max_idx_cells[cell_idx])], axin.get_ylim()[0]),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta t_{DAP}}$',
                  xy=(t_sta[int(AP_max_idx_cells[cell_idx])]
                      + (t_sta[int(DAP_max_idx_cells[cell_idx])] - t_sta[int(AP_max_idx_cells[cell_idx])])/2. + 0.9,
                      axin.get_ylim()[0]-0.065),
                  ha='center', va='bottom')
    # annotate t_fAHP
    # axin.annotate('', xytext=(t_sta[int(AP_max_idx_cells[cell_idx])], -62),
    #               xy=(t_sta[int(fAHP_min_idx_cells[cell_idx])], -62),
    #               arrowprops=dict(arrowstyle="<->", shrinkA=0.0, shrinkB=0.0))
    axin.annotate(r'$\mathrm{\Delta t_{fAHP}}$',
                  xy=(t_sta[int(AP_max_idx_cells[cell_idx])]
                      + (t_sta[int(fAHP_min_idx_cells[cell_idx])] - t_sta[int(AP_max_idx_cells[cell_idx])])/2.,
                      -62),
                  ha='left', va='bottom')

    ax.text(-0.164, 1.0, 'A', transform=ax.transAxes, size=18)

    # B
    for i, cell_id in enumerate(cell_examples):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1, i])
        ax = pl.Subplot(fig, inner[0])
        fig.add_subplot(ax)
        cell_idx = np.where(cell_id == grid_cells)[0]
        ax.plot(t_sta, sta_mean_cells[cell_idx][0], color=colors_examples[i])
        ax.set_ylim(-75, 0)
        #ax.plot(t_sta, sta_mean_cells[cell_idx][0] - v_onset[cell_idx], color=colors_examples[i])
        #ax.set_ylim(-10, 65)
        ax.set_xlabel('Time (ms)')
        ax.set_title('(' + letters_examples[i] + ')' + '     ' + cell_id, loc='left', fontsize=10)
        if i == 0:
            ax.set_ylabel(r'$\mathrm{V_{STA}}$ (mV)')
            ax.text(-0.6, 1.05, 'B', transform=ax.transAxes, size=18)


    # C
    sta_mean_cells_centered = center_STA_to_AP_onset(sta_mean_cells, v_onset)

    groups = ['NB', 'B+D', 'B']
    group_names = ['Non-bursty', 'Bursty with DAP', 'Bursty without DAP']
    for i, group in enumerate(groups):
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2, i])
        ax = pl.Subplot(fig, inner[0])
        fig.add_subplot(ax)
        ax.plot(t_sta, np.mean(sta_mean_cells_centered[labels_burstgroups[group]], 0),
                        color=colors_burstgroups[group])
        ax.fill_between(t_sta,
                                np.mean(sta_mean_cells_centered[labels_burstgroups[group]], 0) -
                                np.std(sta_mean_cells_centered[labels_burstgroups[group]], 0),
                                np.mean(sta_mean_cells_centered[labels_burstgroups[group]], 0) +
                                np.std(sta_mean_cells_centered[labels_burstgroups[group]], 0),
                                color=colors_burstgroups[group], alpha=0.5)
        ax.set_xlim(-5, 25)
        ax.set_ylim(-10, 65)
        ax.set_xlabel('Time (ms)')
        if i == 0:
            ax.set_ylabel(r'$\mathrm{V_{STA} - V_{thresh}}$ (mV)')
            ax.text(-0.6, 1.05, 'C', transform=ax.transAxes, size=18)

        # inset
        axin = inset_axes(ax, width='45%', height='45%')
        axin.plot(t_sta, np.mean(sta_mean_cells[labels_burstgroups[group]], 0),
                        color=colors_burstgroups[group])
        axin.fill_between(t_sta,
                        np.mean(sta_mean_cells[labels_burstgroups[group]], 0) -
                        np.std(sta_mean_cells[labels_burstgroups[group]], 0),
                        np.mean(sta_mean_cells[labels_burstgroups[group]], 0) +
                        np.std(sta_mean_cells[labels_burstgroups[group]], 0),
                        color=colors_burstgroups[group], alpha=0.5)
        axin.set_xlim(-25, 25)
        axin.set_ylim(-75, -52)
        axin.set_xticks([-25, 0, 25])
        axin.set_xticklabels([-25, 0, 25], fontsize=8)
        axin.set_yticks([-70, -60])
        axin.set_yticklabels([-70, -60], fontsize=8)

        ax.annotate(group_names[i], xy=(0.5, -0.6), xycoords='axes fraction', fontsize=12, ha='center')
        if i == 0:
            axin.set_ylabel(r'$\mathrm{V_{STA}}$ (mV)', fontsize=10, labelpad=-2)

    pl.tight_layout()
    pl.subplots_adjust(wspace=0.3, hspace=0.35, left=0.14, right=0.98, top=0.96, bottom=0.12)
    pl.savefig(os.path.join(save_dir_img, 'fig2.png'))
    pl.show()