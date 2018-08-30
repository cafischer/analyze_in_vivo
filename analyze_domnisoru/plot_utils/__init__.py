import matplotlib.pyplot as pl
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec


def plot_with_markers(ax, x, y, cell_ids, cell_type_dict, z=None, edgecolor='k', theta_cells=None, DAP_cells=None,
                      legend=True):
    for cell_idx in range(len(cell_ids)):
        hatch = ''
        facecolor = 'None'
        if theta_cells is not None:
            if cell_ids[cell_idx] in theta_cells:
                hatch = '|||||'
        if DAP_cells is not None:
            if cell_ids[cell_idx] in DAP_cells:
                hatch = '-----'
        if theta_cells is not None and DAP_cells is not None:
            if cell_ids[cell_idx] in theta_cells and cell_ids[cell_idx] in DAP_cells:
                hatch = '+++++'

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
    if legend:
        fig_fake, ax_fake = pl.subplots()
        handles = [ax_fake.scatter(0, 0, marker='*', s=150, linewidths=0.8,
                             edgecolor='k', facecolor='None', label='Stellate'),
                        ax_fake.scatter(0, 0, marker='^', s=100, linewidths=0.8,
                                        edgecolor='k', facecolor='None', label='Pyramidal'),
                        ax_fake.scatter(0, 0, marker='o', s=100, linewidths=0.8,
                                        edgecolor='k', facecolor='None', label='Non-identified')]
        if theta_cells is not None:
            handles += [ax_fake.scatter(0, 0, marker='o', hatch='|||||', s=100, linewidths=0.8,
                             edgecolor='k', facecolor='None', label='Large theta')]
        if DAP_cells is not None:
            handles += [ax_fake.scatter(0, 0, marker='o', hatch='-----', s=100, linewidths=0.8,
                             edgecolor='k', facecolor='None', label='DAP')]
        legend = ax.legend(handles=handles, loc='best')
        ax.add_artist(legend)
        pl.close(fig_fake)


def plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, fig_title=None,
                            sharey='all', sharex='all', save_dir_img=None):
    plot_for_cell_group(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, (14, 8.5), (3, 9),
                        fig_title=fig_title, sharey=sharey, sharex=sharex, save_dir_img=save_dir_img)


def plot_for_cell_group(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, figsize, n_rows_n_columns=None,
                        fig_title=None, sharey='all', sharex='all', save_dir_img=None):
    if n_rows_n_columns is not None:
        n_rows, n_columns = n_rows_n_columns
    else:
        n_rows, n_columns = find_most_equal_divisors(len(cell_ids))
    if figsize is None:
        figsize = (4.5 * n_rows, 2.0 * n_columns)
    fig, axes = pl.subplots(n_rows, n_columns, sharey=sharey, sharex=sharex, squeeze=False,
                            figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=14)
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(n_columns):
            if cell_idx < len(cell_ids):
                axes[i1, i2].set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict))

                plot_fun(axes[i1, i2], cell_idx, **plot_kwargs)

                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel(xlabel)
                if i2 == 0:
                    axes[i1, i2].set_ylabel(ylabel)
                cell_idx += 1
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    pl.tight_layout()
    if fig_title is not None:
        pl.subplots_adjust(top=0.92)
    pl.savefig(save_dir_img)


def plot_for_all_grid_cells_grid(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, n_subplots,
                                 fig_title=None, save_dir_img=None):
    plot_for_cell_group_grid(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, n_subplots, (14, 8.5),
                             (3, 9), fig_title=fig_title, save_dir_img=save_dir_img)


def plot_for_cell_group_grid(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, n_subplots, figsize=None,
                             n_rows_n_columns=None, fig_title=None, hspace=0.1, save_dir_img=None):
    if n_rows_n_columns is not None:
        n_rows, n_columns = n_rows_n_columns
    else:
        n_rows, n_columns = find_most_equal_divisors(len(cell_ids))
    if figsize is None:
        figsize = (4.5 * n_rows, 2.0 * n_columns)

    fig = pl.figure(figsize=figsize)
    outer = gridspec.GridSpec(n_rows, n_columns)

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
                        ax.set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict))

                    if i1 == (n_rows - 1):
                        ax.set_xlabel(xlabel)
                    if i2 == 0:
                        ax.set_ylabel(ylabel)

                    plot_fun(ax, cell_idx, subplot_idx, **plot_kwargs)
                cell_idx += 1
    pl.tight_layout()
    if fig_title is not None:
        pl.subplots_adjust(top=0.92)
    #pl.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.95)
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


if __name__ == '__main__':
    print find_most_equal_divisors(5)