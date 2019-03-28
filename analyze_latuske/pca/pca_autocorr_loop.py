import numpy as np
import matplotlib.pyplot as pl
import os
from cell_characteristics import to_idx
from analyze_in_vivo.load.load_domnisoru import load_cell_ids, load_data
from analyze_in_vivo.load.load_latuske import load_ISIs
from analyze_in_vivo.analyze_domnisoru.pca import perform_PCA
from grid_cell_stimuli.ISI_hist import get_ISIs
from analyze_in_vivo.analyze_domnisoru.autocorr import *
#pl.style.use('paper_subplots')


if __name__ == '__main__':
    #save_dir_domnisoru = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    #save_dir_autocorr_domnisoru = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'
    save_dir_domnisoru =  '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    save_dir_autocorr_domnisoru = '/home/cfischer/PycharmProjects/analyze_in_vivo/analyze_in_vivo/results/domnisoru/whole_trace/autocorr'

    max_lags = np.arange(2, 200+1, 1)  # ms
    bin_width = 1  # ms
    sigma_smooth = None
    dt_kde = 0.05
    n_components = 5
    remove_cells = True  # take out autocorrelation for cell s104_0007 and s110_0002
    use_latuske = True
    use_domnisoru = False
    normalization = 'sum'

    explained_vars = np.zeros((len(max_lags), n_components))

    for idx, max_lag in enumerate(max_lags):
        # compute autocorr latuske
        if use_latuske:
            ISIs_cells = load_ISIs()
            cell_ids_latuske = np.array([str(i) for i in range(len(ISIs_cells))])
            autocorr_cells_latuske = np.zeros((len(cell_ids_latuske), int(2*to_idx(max_lag, bin_width))+1))
            for cell_idx, cell_id in enumerate(cell_ids_latuske):
                autocorr_cells_latuske[cell_idx, :], t_autocorr, bins = get_autocorrelation_by_ISIs(ISIs_cells[cell_idx],
                                                                                                      max_lag=max_lag,
                                                                                                      bin_width=bin_width,
                                                                                                      normalization=normalization)
        # compute autocorr domnisoru
        if use_domnisoru:
            cell_ids_domnisoru = np.array(load_cell_ids(save_dir_domnisoru, 'grid_cells'))
            autocorr_cells_domnisoru = np.zeros((len(cell_ids_domnisoru), int(2*to_idx(max_lag, bin_width))+1))
            for cell_idx, cell_id in enumerate(cell_ids_domnisoru):
                data = load_data(cell_id, ['Vm_ljpc', 'spiketimes'], save_dir_domnisoru)
                t = np.arange(len(data['Vm_ljpc'])) * data['dt']
                ISIs = get_ISIs(data['spiketimes'], t)
                autocorr_cells_domnisoru[cell_idx, :], t_autocorr, bins = get_autocorrelation_by_ISIs(ISIs, max_lag=max_lag,
                                                                                                bin_width=bin_width,
                                                                                                normalization=normalization)
            autocorr_cells_domnisoru_for_pca = autocorr_cells_domnisoru

        if sigma_smooth is not None:
            t_autocorr = np.arange(-max_lag, max_lag + dt_kde, dt_kde)
        else:
            t_autocorr = np.arange(-max_lag, max_lag + bin_width, bin_width)

        # PCA
        if use_domnisoru and remove_cells:
            idx_s104_0007 = np.where(np.array(cell_ids_domnisoru) == 's104_0007')[0][0]
            idx_s110_0002 = np.where(np.array(cell_ids_domnisoru) == 's110_0002')[0][0]
            idxs = range(len(cell_ids_domnisoru))
            idxs.remove(idx_s104_0007)
            idxs.remove(idx_s110_0002)
            autocorr_cells_domnisoru_for_pca = autocorr_cells_domnisoru[np.array(idxs)]

        if use_domnisoru and use_latuske:
            autocorr_cells_for_pca = np.vstack((autocorr_cells_latuske, autocorr_cells_domnisoru_for_pca))
        elif use_domnisoru and not use_latuske:
            autocorr_cells_for_pca = autocorr_cells_domnisoru_for_pca
        elif use_latuske and not use_domnisoru:
            autocorr_cells_for_pca = autocorr_cells_latuske
        else:
            raise ValueError('Either Domnisoru or Latuske must be selected!')

        projected, components, explained_var = perform_PCA(autocorr_cells_for_pca, n_components)

        explained_vars[idx, :] = explained_var


    # plot
    pl.figure()
    for i in range(n_components):
        pl.plot(max_lags, explained_vars[:, i], label='PC'+str(i+1))
    pl.plot(max_lags, explained_vars[:, 0] + explained_vars[:, 1], label='PC1+PC2', linestyle='--')
    pl.plot(max_lags, explained_vars[:, 0] + explained_vars[:, 1] + explained_vars[:, 2], label='PC1+PC2+PC3', linestyle='--')
    pl.ylabel('Explained variance')
    pl.xlabel('Max lag')
    pl.ylim(0, 1)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_autocorr_domnisoru, 'exp_var_latuske.png'))
    pl.show()