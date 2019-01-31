import matplotlib.pyplot as pl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from analyze_in_vivo.load.load_domnisoru import get_cell_ids_DAP_cells, load_cell_ids


def plot_with_markers(ax, x, y, cell_ids, cell_type_dict, z=None, edgecolor='k', theta_cells=None, DAP_cells=None,
                      DAP_cells_additional=None, legend=True):
    for cell_idx in range(len(cell_ids)):
        hatch = ''
        facecolor = 'None'
        if theta_cells is not None:
            if cell_ids[cell_idx] in theta_cells:
                hatch = '|||||'
        if DAP_cells is not None:
            if cell_ids[cell_idx] in DAP_cells:
                hatch = '-----'
        if DAP_cells_additional is not None:
            if cell_ids[cell_idx] in DAP_cells_additional:
                hatch = '/////'
        if theta_cells is not None and DAP_cells is not None:
            if cell_ids[cell_idx] in theta_cells and cell_ids[cell_idx] in DAP_cells:
                hatch = '+++++'
        if theta_cells is not None and DAP_cells_additional is not None:
            if cell_ids[cell_idx] in theta_cells and cell_ids[cell_idx] in DAP_cells_additional:
                hatch = '|||||/////'

        if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
            if z is not None:
                ax.scatter(x[cell_idx], y[cell_idx], z[cell_idx], marker='*', hatch=hatch, s=150, linewidths=0.8,
                           edgecolor=edgecolor, facecolor=facecolor)
            else:
                ax.scatter(x[cell_idx], y[cell_idx], marker='*', hatch=hatch, s=150, linewidths=0.8,
                           edgecolor=edgecolor, facecolor=facecolor)
        elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
            if z is not None:
                ax.scatter(x[cell_idx], y[cell_idx], z[cell_idx], marker='^', hatch=hatch, s=100, linewidths=0.8,
                           edgecolor=edgecolor, facecolor=facecolor)
            else:
                ax.scatter(x[cell_idx], y[cell_idx], marker='^', hatch=hatch, s=100, linewidths=0.8,
                           edgecolor=edgecolor, facecolor=facecolor)
        else:
            if z is not None:
                ax.scatter(x[cell_idx], y[cell_idx], z[cell_idx], marker='o', hatch=hatch, s=100, linewidths=0.8,
                           edgecolor=edgecolor, facecolor=facecolor)
            else:
                ax.scatter(x[cell_idx], y[cell_idx], marker='o', hatch=hatch, s=100, linewidths=0.8,
                           edgecolor=edgecolor, facecolor=facecolor)

    # legend
    fig_fake, ax_fake = pl.subplots()
    handles = [ax_fake.scatter(0, 0, marker='*', s=150, linewidths=0.8,
                               edgecolor='k', facecolor='None', label='Stellate'),
               ax_fake.scatter(0, 0, marker='^', s=100, linewidths=0.8,
                               edgecolor='k', facecolor='None', label='Pyramidal'),
               ax_fake.scatter(0, 0, marker='o', s=100, linewidths=0.8,
                               edgecolor='k', facecolor='None', label='Non-identified')]
    if theta_cells is not None:
        handles += [ax_fake.scatter(0, 0, marker='s', hatch='|||||', s=100, linewidths=0.8,
                    edgecolor='w', facecolor='None', label='Large-theta')]
    if DAP_cells is not None:
        handles += [ax_fake.scatter(0, 0, marker='s', hatch='-----', s=100, linewidths=0.8,
                    edgecolor='w', facecolor='None', label='DAP')]
    if DAP_cells_additional is not None:
        handles += [ax_fake.scatter(0, 0, marker='s', hatch='/////', s=100, linewidths=0.8,
                    edgecolor='w', facecolor='None', label='DAP (add.)')]
    if legend:
        legend = ax.legend(handles=handles, loc='best')
        ax.add_artist(legend)
    pl.close(fig_fake)
    return handles


def get_handles_all_markers():
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    fig_fake, ax_fake = pl.subplots()
    handles = [ax_fake.scatter(0, 0, marker='*', s=150, linewidths=0.8,
                               edgecolor='k', facecolor='None', label='Stellate'),
               ax_fake.scatter(0, 0, marker='^', s=100, linewidths=0.8,
                               edgecolor='k', facecolor='None', label='Pyramidal'),
               ax_fake.scatter(0, 0, marker='o', s=100, linewidths=0.8,
                               edgecolor='k', facecolor='None', label='Non-identified')]

    handles += [ax_fake.scatter(0, 0, marker='s', hatch='|||||', s=100, linewidths=0.8,
                                    edgecolor='w', facecolor='None', label='Large theta')]
    handles += [ax_fake.scatter(0, 0, marker='s', hatch='-----', s=100, linewidths=0.8,
                                    edgecolor='w', facecolor='None', label='DAP')]
    handles += [ax_fake.scatter(0, 0, marker='s', hatch='/////', s=100, linewidths=0.8,
                    edgecolor='w', facecolor='None', label='DAP (add.)')]
    pl.close(fig_fake)
    return handles


def get_handles_for_cell_id(cell_id, cell_type_dict, color='k'):
    save_dir = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    theta_cells = load_cell_ids(save_dir, 'giant_theta')
    DAP_cells, DAP_cells_additional = get_cell_ids_DAP_cells()

    hatch = None
    if theta_cells is not None:
        if cell_id in theta_cells:
            hatch = '|||||'
    if DAP_cells is not None:
        if cell_id in DAP_cells:
            hatch = '-----'
    if DAP_cells_additional is not None:
        if cell_id in DAP_cells_additional:
            hatch = '/////'
    if theta_cells is not None and DAP_cells is not None:
        if cell_id in theta_cells and cell_id in DAP_cells:
            hatch = '+++++'
    if theta_cells is not None and DAP_cells_additional is not None:
        if cell_id in theta_cells and cell_id in DAP_cells_additional:
            hatch = '|||||/////'

    fig_fake, ax_fake = pl.subplots()
    if cell_type_dict[cell_id] == 'stellate':
        handle = ax_fake.scatter(0, 0, marker='*', s=150, linewidths=0.8, hatch=hatch,
                                 edgecolor=color, facecolor='None', label=cell_id)
    elif cell_type_dict[cell_id] == 'pyramidal':
        handle = ax_fake.scatter(0, 0, marker='^', s=100, linewidths=0.8, hatch=hatch,
                                 edgecolor=color, facecolor='None', label=cell_id)
    else:
        handle = ax_fake.scatter(0, 0, marker='o', s=100, linewidths=0.8, hatch=hatch,
                                 edgecolor=color, facecolor='None', label=cell_id)
    pl.close(fig_fake)
    return handle


def plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, fig_title=None,
                            sharey='all', sharex='all', colors_marker=None, wspace=None, save_dir_img=None):
    plot_for_cell_group(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, (15, 8.5), (3, 9),
                        fig_title=fig_title, sharey=sharey, sharex=sharex, colors_marker=colors_marker,
                        wspace=wspace, save_dir_img=save_dir_img)


def plot_for_cell_group(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, figsize, n_rows_n_columns=None,
                        fig_title=None, sharey='all', sharex='all', colors_marker=None, wspace=None, save_dir_img=None):
    if n_rows_n_columns is not None:
        n_rows, n_columns = n_rows_n_columns
    else:
        n_rows, n_columns = find_most_equal_divisors(len(cell_ids))
    if figsize is None:
        figsize = (4.5 * n_rows, 2.0 * n_columns)
    if colors_marker is None:
        colors_marker = np.array(['k'] * len(cell_ids))

    fig, axes = pl.subplots(n_rows, n_columns, sharey=sharey, sharex=sharex, squeeze=False,
                            figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=14)
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(n_columns):
            if cell_idx < len(cell_ids):
                plot_fun(axes[i1, i2], cell_idx, **plot_kwargs)

                # title (given as legend)
                #axes[i1, i2].set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict))
                handle = get_handles_for_cell_id(cell_ids[cell_idx], cell_type_dict, colors_marker[cell_idx])
                leg = axes[i1, i2].legend(handles=[handle], bbox_to_anchor=(0, 1.01, 1, 0.1), loc="lower left",
                                          frameon=False, handletextpad=0.1, mode='expand')
                # if colors_marker[0] == 'k':
                #     leg = axes[i1, i2].legend(handles=[handle], bbox_to_anchor=(0, 1.01, 1, 0.1), loc="lower left",
                #                               frameon=False, handletextpad=0.1, mode='expand')
                # else:
                #     leg = axes[i1, i2].legend(handles=[handle], bbox_to_anchor=(-0.27, 1.01, 1, 0.1), loc="lower left",
                #                               frameon=False, handletextpad=0.1, mode='expand')
                leg.legendHandles[0].set_edgecolor(
                    colors_marker[cell_idx])  # fixes bug that hatch is not in same edgecolor

                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel(xlabel)
                if i2 == 0:
                    axes[i1, i2].set_ylabel(ylabel)
                cell_idx += 1
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    handles = get_handles_all_markers()
    if colors_marker[0] != 'k':
        handles += [Patch(color='r', label='Bursty'), Patch(color='b', label='Non-bursty')]
    axes[-1, -1].legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.025, -0.025))
    #axes[-1, -1].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 0.25))  # for ISI return map
    pl.tight_layout()
    if fig_title is not None:
        pl.subplots_adjust(top=0.92)
    if wspace is not None:
        pl.subplots_adjust(wspace=wspace)
    pl.savefig(save_dir_img)


def plot_for_all_grid_cells_grid(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, n_subplots,
                                 wspace=None, fig_title=None, colors_marker=None, save_dir_img=None):
    plot_for_cell_group_grid(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, n_subplots, (15, 8.5),
                             (3, 9), wspace=wspace, fig_title=fig_title, colors_marker=colors_marker,
                             save_dir_img=save_dir_img)


def plot_for_cell_group_grid(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, n_subplots, figsize=None,
                             n_rows_n_columns=None, fig_title=None, wspace=None, hspace=0.1, colors_marker=None,
                             save_dir_img=None):
    if n_rows_n_columns is not None:
        n_rows, n_columns = n_rows_n_columns
    else:
        n_rows, n_columns = find_most_equal_divisors(len(cell_ids))
    if figsize is None:
        figsize = (4.5 * n_rows, 2.0 * n_columns)
    if colors_marker is None:
        colors_marker = np.array(['k'] * len(cell_ids))

    fig = pl.figure(figsize=figsize)
    outer = gridspec.GridSpec(n_rows, n_columns, wspace=wspace)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)

    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(n_columns):
            inner = gridspec.GridSpecFromSubplotSpec(n_subplots, 1, subplot_spec=outer[cell_idx], hspace=hspace)

            if cell_idx < len(cell_ids):
                for subplot_idx in range(n_subplots):
                    ax = pl.Subplot(fig, inner[subplot_idx])
                    fig.add_subplot(ax)

                    if subplot_idx == 0:
                        #ax.set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict))
                        handle = get_handles_for_cell_id(cell_ids[cell_idx], cell_type_dict, colors_marker[cell_idx])
                        leg = ax.legend(handles=[handle], bbox_to_anchor=(-0.2, 1.01, 1, 0.1), loc="lower left",
                                  frameon=False, handletextpad=0.1, mode='expand')
                        leg.legendHandles[0].set_edgecolor(
                            colors_marker[cell_idx])  # fixes bug that hatch is not in same edgecolor

                    if i1 == (n_rows - 1):
                        ax.set_xlabel(xlabel)
                    if i2 == 0:
                        ax.set_ylabel(ylabel)

                    plot_fun(ax, cell_idx, subplot_idx, **plot_kwargs)
                cell_idx += 1
    ax = pl.Subplot(fig, outer[n_rows-1, n_columns-1])
    fig.add_subplot(ax)
    handles = get_handles_all_markers()
    if colors_marker[0] != 'k':
        handles += [Patch(color='b', label='Non-bursty'), Patch(color='r', label='Bursty')]
    ax.legend(handles=handles, loc="lower right")
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    pl.tight_layout()
    if fig_title is not None:
        pl.subplots_adjust(top=0.92)
    #pl.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.95)
    if save_dir_img is not None:
        pl.savefig(save_dir_img)


def get_cell_id_with_marker(cell_id, cell_type_dict):
    if cell_type_dict[cell_id] == 'stellate':
        return cell_id + ' ' + u'\u2605'
    elif cell_type_dict[cell_id] == 'pyramidal':
        return cell_id + ' ' + u'\u25B4'
    else:
        return cell_id


def get_divisors(x):
    divisors = []
    for divisor in range(2, x+1):
        if x % divisor == 0:
            divisors.append((divisor, x / divisor))
    return divisors


def find_most_equal_divisors(x):
    divisors = get_divisors(x)
    diff = [np.abs(divisor_pair[1] - divisor_pair[0]) for divisor_pair in divisors]
    return divisors[np.argmin(diff)]


def horizontal_square_bracket(ax, star, x1, x2, y1, y2, dtext):
    ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], lw=1.5, c='k')
    if star == 'n.s.':
        fontsize = 10
    else:
        fontsize = 12
    ax.text((x1 + x2) * 0.5, y2 + dtext, star, ha='center', color='k', fontsize=fontsize)


def get_star_from_p_val(p):
    star_idx = np.where([p < 0.01, p < 0.001, p < 0.0001])[0]
    if len(star_idx) == 0:
        star_idx = 0
    else:
        star_idx = star_idx[-1] + 1
    stars = ['n.s.', '*', '**', '***']
    star = stars[star_idx]
    return star


if __name__ == '__main__':
    print find_most_equal_divisors(5)