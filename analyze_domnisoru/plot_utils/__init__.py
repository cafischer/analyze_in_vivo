import matplotlib.pyplot as pl
import numpy as np


def plot_with_markers(ax, x, y, cell_ids, cell_type_dict):
    for cell_idx in range(len(cell_ids)):
        if cell_type_dict[cell_ids[cell_idx]] == 'stellate':
            ax.plot(x[cell_idx], y[cell_idx], 'k', marker='*', markersize=7)
        elif cell_ids[cell_idx] == 's82_0002':
            ax.plot(x[cell_idx], y[cell_idx], marker='^', markerfacecolor='None', markeredgecolor='k')
        elif cell_ids[cell_idx] == 's84_0002':
            ax.plot(x[cell_idx], y[cell_idx], marker='o', markerfacecolor='None', markeredgecolor='k')
        elif cell_type_dict[cell_ids[cell_idx]] == 'pyramidal':
            ax.plot(x[cell_idx], y[cell_idx], 'k', marker='^')
        else:
            ax.plot(x[cell_idx], y[cell_idx], 'k', marker='o')


def plot_for_all_grid_cells(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, fig_title=None,
                            sharey='all', save_dir_img=None):
    plot_for_cell_group(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, (14, 8.5), (3, 9),
                        fig_title=fig_title, sharey=sharey, save_dir_img=save_dir_img)


def plot_for_cell_group(cell_ids, cell_type_dict, plot_fun, plot_kwargs, xlabel, ylabel, figsize, n_rows_n_columns=None,
                        fig_title=None, sharey='all', save_dir_img=None):
    if n_rows_n_columns is not None:
        n_rows, n_columns = n_rows_n_columns
    else:
        n_rows, n_columns = find_most_equal_divisors(len(cell_ids))
    if figsize is None:
        figsize = (4.5 * n_rows, 2.0 * n_columns)
    fig, axes = pl.subplots(n_rows, n_columns, sharex='all', sharey=sharey, squeeze=False,
                            figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=16)
    cell_idx = 0
    for i1 in range(n_rows):
        for i2 in range(n_columns):
            if cell_idx < len(cell_ids):
                axes[i1, i2].set_title(get_cell_id_with_marker(cell_ids[cell_idx], cell_type_dict), fontsize=12)

                plot_fun(axes[i1, i2], cell_idx, **plot_kwargs)

                if i1 == (n_rows - 1):
                    axes[i1, i2].set_xlabel(xlabel)
                if i2 == 0:
                    axes[i1, i2].set_ylabel(ylabel)
                cell_idx += 1
            else:
                axes[i1, i2].spines['left'].set_visible(False)
                axes[i1, i2].spines['bottom'].set_visible(False)
                axes[i1, i2].set_xticks([])
                axes[i1, i2].set_yticks([])
    pl.tight_layout()
    if fig_title is not None:
        pl.subplots_adjust(top=0.92)
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
    print find_most_equal_divisors(2)