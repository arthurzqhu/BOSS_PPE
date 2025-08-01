import numpy as np
import pandas as pd
import tensorflow as tf
import netCDF4 as nc
# from numba import cuda
import gc
from sklearn import preprocessing
import sklearn.model_selection as mod_sec
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

def get_param_interest_idx(dataset):
    n_param_nevp = dataset.getncattr('n_param_nevp')
    n_param_condevp = dataset.getncattr('n_param_condevp')
    n_param_coal = dataset.getncattr('n_param_coal')
    n_param_sed = dataset.getncattr('n_param_sed')
    is_perturbed_nevp = bool(dataset.getncattr('is_perturbed_nevp'))
    is_perturbed_condevp = bool(dataset.getncattr('is_perturbed_condevp'))
    is_perturbed_coal = bool(dataset.getncattr('is_perturbed_coal'))
    is_perturbed_sed = bool(dataset.getncattr('is_perturbed_sed'))

    param_interest_idx = []
    if is_perturbed_nevp:
        param_interest_idx.append(np.arange(n_param_nevp))
    if is_perturbed_condevp:
        param_interest_idx.append(np.arange(n_param_nevp, 
                                            n_param_nevp + n_param_condevp))
    if is_perturbed_coal:
        param_interest_idx.append(np.arange(n_param_nevp + n_param_condevp, 
                                            n_param_nevp + n_param_condevp + n_param_coal))
    if is_perturbed_sed:
        param_interest_idx.append(np.arange(n_param_nevp + n_param_condevp + n_param_coal, 
                                            n_param_nevp + n_param_condevp + n_param_coal + n_param_sed))
    param_interest_idx = np.concatenate(param_interest_idx)
    return param_interest_idx

def get_params(basepath, filename):
    dataset = nc.Dataset(basepath + filename, mode='r')
    vars_vn = dataset.getncattr('init_var')
    
    param_interest_idx = get_param_interest_idx(dataset)

    if isinstance(vars_vn, str):
        vars_vn = [vars_vn]
    PPE_vns = [istr + "_PPE" for istr in vars_vn]
    ppe_initcon_matrix = []
    
    for PPE_vn in PPE_vns:
        ppe_initcon_matrix.append(np.expand_dims(dataset.variables[PPE_vn][:], axis=1))
    
    params_PPE = dataset.variables['params_PPE'][:][:, param_interest_idx]
    ppe_initcon_matrix.append(params_PPE)
    params_train = np.concatenate(ppe_initcon_matrix, axis=1)

    return {'pnames': dataset.variables['param_names'][:],
            'vals': params_train,
            'param_interest_idx': param_interest_idx}

def get_train_val_tgt_data(basepath, filename, param_train, transform_method, 
                           l_multi_output=False, test_size=0.2, random_state=1,
                           set_nan_to_neg1001=True):
    scalers = {}

    # always use minmaxscale for parameters to avoid extrapolation
    minmaxscale = preprocessing.MinMaxScaler().fit(param_train['vals'])
    x_all = minmaxscale.transform(param_train['vals'])
    scalers['x'] = minmaxscale
    scalers['y'] = []

    ppe_info = {}
    dataset = nc.Dataset(basepath + filename, mode='r')
    ppe_info['nppe'], ppe_info['nparam_init'] = param_train['vals'].shape
    ppe_info['n_init'] = getattr(dataset, 'n_init')
    init_vars = getattr(dataset, 'init_var')
    if isinstance(init_vars, str):
        init_vars = [init_vars]

    tgt_initvar_matrix = []

    for init_var in init_vars:
        tgt_initvar_matrix.append(np.expand_dims(dataset.variables['case_' + init_var][:], axis=1))

    eff0s = getattr(dataset, 'thresholds_eff0')
    var_constraints = getattr(dataset, 'var_constraints')
    ppe_var_names = ['ppe_' + i for i in var_constraints]
    ppe_raw_vals = [dataset.variables[i][:] for i in ppe_var_names]
    tgt_var_names = ['tgt_' + i for i in var_constraints]
    tgt_raw_vals = [dataset.variables[i][:] for i in tgt_var_names]

    y_train = {}
    y_val = {}

    ppe_info['nobs'] = [np.prod(i.shape[1:]) for i in ppe_raw_vals]
    ppe_info['ncases'] = tgt_raw_vals[0].shape[0]
    ppe_info['nvar'] = len(var_constraints)
    ppe_info['npar'] = ppe_info['nparam_init'] - ppe_info['n_init']
    ppe_info['eff0s'] = eff0s
    ppe_info['var_constraints'] = var_constraints

    ppe_var_presence = []
    tgt_var_presence = []
    ppe_norm = []
    tgt_norm = []
    ppe_data = []
    tgt_data = []

    if 'V_M' in var_constraints:
        transform_method = 'minmaxscale'
        dmin = 0.
        dmax = 9.2
        drange = dmax - dmin
    
    for ivar, eff0 in enumerate(tqdm(eff0s, desc='Transforming data...')):
        cloud_filter = None

        # # WARNING: only use this filter when advection is off
        # cloud_filter = ppe_raw_vals[ivar][0]>0.
        n_cloud = np.sum(cloud_filter)
            
        if cloud_filter is not None:
            ppe_raw_val_reshaped = ppe_raw_vals[ivar][:, cloud_filter].reshape(ppe_info['nppe'], n_cloud)
            tgt_raw_val_reshaped = tgt_raw_vals[ivar][:, cloud_filter].reshape(ppe_info['ncases'], n_cloud)
            ppe_raw_val_reshaped[ppe_raw_val_reshaped<0] = 0.
            tgt_raw_val_reshaped[tgt_raw_val_reshaped<0] = 0.
        else:
            if ppe_raw_vals[ivar].ndim >= 2:
                ppe_raw_val_reshaped = np.reshape(ppe_raw_vals[ivar], (ppe_info['nppe'], np.prod(ppe_raw_vals[ivar].shape[1:])))
                tgt_raw_val_reshaped = np.reshape(tgt_raw_vals[ivar], (ppe_info['ncases'], np.prod(tgt_raw_vals[ivar].shape[1:])))
                # WARNING: temporary limiter
                ppe_raw_val_reshaped[ppe_raw_val_reshaped<0] = 0.
                tgt_raw_val_reshaped[tgt_raw_val_reshaped<0] = 0.
            else:
                ppe_raw_val_reshaped = ppe_raw_vals[ivar].reshape(-1, 1)
                tgt_raw_val_reshaped = tgt_raw_vals[ivar].reshape(-1, 1)

        ppe_var_presence.append(ppe_raw_val_reshaped > eff0/100)
        tgt_var_presence.append(tgt_raw_val_reshaped > eff0/100)

        if 'V_M' in var_constraints[ivar]:
            mmscale = preprocessing.MinMaxScaler().fit(ppe_raw_val_reshaped)
            # manually set all minmaxscale to the same range
            mmscale.data_min_[:] = dmin
            mmscale.data_max_[:] = dmax
            mmscale.data_range_[:] = drange
            mmscale.scale_[:] = 1/drange
            scalers['y'].append(mmscale)
        else:
            if transform_method == 'standard_scaler':
                ppe_norm.append(ppe_raw_val_reshaped)
                tgt_norm.append(tgt_raw_val_reshaped)
                standscale = preprocessing.StandardScaler().fit(ppe_raw_val_reshaped)
                scalers['y'].append(standscale)
            elif transform_method == 'standard_scaler_asinh':
                ppe_norm.append(smooth_linlog(ppe_raw_val_reshaped, eff0))
                tgt_norm.append(smooth_linlog(tgt_raw_val_reshaped, eff0))
                standscale = preprocessing.StandardScaler().fit(ppe_norm[-1])
                scalers['y'].append(standscale)
            elif transform_method == 'standard_scaler_log':
                ppe_norm.append(np.log10(ppe_raw_val_reshaped))
                tgt_norm.append(np.log10(tgt_raw_val_reshaped))
                standscale = preprocessing.StandardScaler().fit(ppe_norm[-1])
                scalers['y'].append(standscale)
            elif transform_method == 'minmaxscale_asinh':
                ppe_norm.append(smooth_linlog(ppe_raw_val_reshaped, eff0))
                tgt_norm.append(smooth_linlog(tgt_raw_val_reshaped, eff0))
                mmscale = preprocessing.MinMaxScaler().fit(ppe_norm[-1])
                mmscale.data_min_[:] = min(mmscale.data_min_)
                mmscale.data_max_[:] = max(mmscale.data_max_)
                mmscale.data_range_[:] = max(mmscale.data_max_) - min(mmscale.data_min_)
                mmscale.scale_ = 1/mmscale.data_range_
                scalers['y'].append(mmscale)

    mom_consistency_mask = np.min(np.array(ppe_var_presence), axis=0)
    scale_mask = np.max(mom_consistency_mask, axis=0)

    if len(ppe_norm) > 0:
        for i, iscale in enumerate(scalers['y']):
            iscale.scale_ = iscale.scale_ * scale_mask
            dat = iscale.transform(ppe_norm[i])
            # dat[ppe_norm[i].mask] = np.nan
            ppe_data.append(dat)
            dat[np.isinf(dat)] = np.nan
            dat = iscale.transform(tgt_norm[i])
            dat[tgt_norm[i].mask] = np.nan
            tgt_data.append(dat)
            dat[np.isinf(dat)] = np.nan

    for ivar, (ppe_varp_tmp, ppe_varr_tmp) in enumerate(zip(ppe_var_presence, ppe_data)):
        x_train, x_val, y_train_wpresence_single, y_val_wpresence_single = \
            mod_sec.train_test_split(x_all, ppe_varp_tmp, test_size=test_size, random_state=random_state)
        _, _, y_train_rawv_single_tmp, y_val_rawv_single_tmp =\
            mod_sec.train_test_split(x_all, ppe_varr_tmp, test_size=test_size, random_state=random_state)
        if set_nan_to_neg1001:
            y_train_rawv_single_tmp = np.nan_to_num(y_train_rawv_single_tmp, nan=-1001, neginf=-1001, posinf=-1001)
            y_val_rawv_single_tmp = np.nan_to_num(y_val_rawv_single_tmp, nan=-1001, neginf=-1001, posinf=-1001)
        y_train[f'water_{ivar}'] = y_train_rawv_single_tmp
        y_val[f'water_{ivar}'] = y_val_rawv_single_tmp

        if l_multi_output:
            y_train[f'presence_{ivar}'] = y_train_wpresence_single
            y_val[f'presence_{ivar}'] = y_val_wpresence_single
    
    return x_train, x_val, y_train, y_val, tgt_data, tgt_initvar_matrix, ppe_info, scalers

def plot_emulator_results(x_val, y_val, model, ppe_info, transform_method, scalers,
                          plot_uncertainty=False):

    y_mdl_inv, y_tgt_inv, y_mdl, y_tgt, y_mdl_unc = apply_model(model, x_val, y_val, ppe_info, transform_method, scalers)
    
    plot_2dhist(y_tgt, y_mdl, ppe_info, 'Normalized')
    plot_2dhist(y_tgt_inv, y_mdl_inv, ppe_info, 'Raw values')
    if plot_uncertainty:
        plot_2dhist_unc(y_tgt, y_mdl, y_mdl_unc, ppe_info)

def plot_2dhist_unc(y_tgt, y_mdl, y_mdl_unc, ppe_info):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']

    fig = plt.figure(figsize=(15,15/nvar))
    gs = gridspec.GridSpec(1,nvar)

    for i, (yt_tmp, yp_tmp, yunc_tmp) in enumerate(zip(y_tgt, y_mdl, y_mdl_unc)):
        ax = fig.add_subplot(gs[i])
        ax.set_aspect('equal')

        vpoint = np.logical_and(np.isfinite(yt_tmp), np.isfinite(yp_tmp))
        x_tmp = yt_tmp[vpoint]
        y_tmp = yp_tmp[vpoint]
        unc_tmp = yunc_tmp[vpoint]
        
        # Define bin edges
        xbins = np.linspace(np.min(x_tmp), np.max(x_tmp), 51)
        ybins = np.linspace(np.min(y_tmp), np.max(y_tmp), 61)

        # Digitize to get bin indices for each point
        xidx_tmp = np.digitize(x_tmp, xbins) - 1
        yidx_tmp = np.digitize(y_tmp, ybins) - 1

        # Prepare 2D array for mean uncertainty
        mean_unc = np.full((len(xbins)-1, len(ybins)-1), np.nan)
        counts = np.zeros_like(mean_unc)

        # Use numpy's bincount to vectorize the mean calculation for each bin
        # Flatten 2D bin indices to 1D for bincount
        flat_idx_tmp = xidx_tmp * mean_unc.shape[1] + yidx_tmp
        # Only consider valid indices (within bounds)
        valid = (xidx_tmp >= 0) & (xidx_tmp < mean_unc.shape[0]) & (yidx_tmp >= 0) & (yidx_tmp < mean_unc.shape[1])
        flat_idx_tmp = flat_idx_tmp[valid]
        unc_valid_tmp = unc_tmp[valid]

        # Compute sum and count for each bin
        sum_unc = np.bincount(flat_idx_tmp, weights=unc_valid_tmp, minlength=mean_unc.size)
        count_unc = np.bincount(flat_idx_tmp, minlength=mean_unc.size)

        # Avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_vals = sum_unc / count_unc
        # Reshape to 2D
        mean_unc[:,:] = mean_vals.reshape(mean_unc.shape)
        counts[:,:] = count_unc.reshape(mean_unc.shape)

        # Plot with pcolormesh (transpose so axes match)
        pcm = ax.pcolormesh(
            xbins, ybins, mean_unc.T, cmap='magma', shading='auto'
        )
        plt.colorbar(pcm, ax=ax, label='Mean ln(uncertainty)')
        ax.set_aspect('equal')
        ax_min = max([ax.get_ylim()[0]] + [ax.get_xlim()[0]])
        ax_max = min([ax.get_ylim()[1]] + [ax.get_xlim()[1]])
        ax.plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange')
        ax.set_title(var_constraints[i])
        ax.set_xlabel('normalized BOSS output')
        ax.set_ylabel('normalized emulator output')

    fig.tight_layout()
        

def plot_2dhist(y_tgt, y_mdl, ppe_info, title):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']

    fig = plt.figure(figsize=(15,15/nvar))
    gs = gridspec.GridSpec(1,nvar)

    for i, (yt_tmp, yp_tmp) in enumerate(zip(y_tgt, y_mdl)):
        ax = fig.add_subplot(gs[i])
        ax.set_aspect('equal')

        if title == 'Normalized':
            vpoint = np.logical_and(np.isfinite(yt_tmp), np.isfinite(yp_tmp))
            hist, xedges, yedges = np.histogram2d(yt_tmp[vpoint], yp_tmp[vpoint], bins=[50, 60])
        elif title == 'Raw values':
            vpoint = np.logical_and(yt_tmp>0, yp_tmp>0)
            hist, xedges, yedges = np.histogram2d(np.log10(yt_tmp[vpoint]), np.log10(yp_tmp[vpoint]), bins=[50, 60])

        hist_min = 1e-6
        hist = hist/hist.sum()
        hist = np.log10(np.maximum(hist, hist_min)).T
        plt.pcolor(xedges, yedges, hist, cmap='viridis', shading='auto')
        plt.colorbar()

        ax = plt.gca()
        ax.set_aspect('equal')
        ax_min = max([ax.get_ylim()[0]] + [ax.get_xlim()[0]])
        ax_max = min([ax.get_ylim()[1]] + [ax.get_xlim()[1]])
        plt.plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange')
        plt.title(var_constraints[i])
    
    fig.suptitle(title)
    fig.tight_layout()

def apply_model(model, x_val, y_val, ppe_info, transform_method, scalers):
    eff0s = ppe_info['eff0s']
    nvar = ppe_info['nvar']
    nobs = ppe_info['nobs']

    y_mdl_inv = []
    y_tgt_inv = []
    y_mdl = []
    y_tgt = []
    y_mdl_unc = []

    for i in range(nvar):
        if type(model(x_val)) is dict: # multi-output model (explicit, with uncertainty)
            presence = model(x_val)[f'presence_{i}'].numpy()
            y_model_raw = model(x_val)[f'water_{i}'][:,:nobs[i]].numpy().astype('float64')
            y_model_unc = model(x_val)[f'water_{i}'][:,nobs[i]:].numpy().astype('float64')
            y_val_raw = y_val[f'water_{i}'].astype('float64')
            y_mdl_inv.append(presence * inverse_transform_data(y_model_raw, eff0s[i], transform_method, scalers['y'][i]))
            y_tgt_inv.append(inverse_transform_data(y_val_raw, eff0s[i], transform_method, scalers['y'][i]))
            y_mdl.append(y_model_raw)
            y_tgt.append(y_val_raw)
            y_mdl_unc.append(y_model_unc)
        else: # single-output model (implicit, no uncertainty calculation implemented)
            y_model = model(x_val)[i].numpy().astype('float64')
            y_val_raw = y_val[i].numpy().astype('float64')
            y_mdl_inv.append(inverse_transform_data(y_model, eff0s[i], transform_method, scalers['y'][i]))
            y_tgt_inv.append(inverse_transform_data(y_val_raw, eff0s[i], transform_method, scalers['y'][i]))
            y_mdl.append(y_model)
            y_tgt.append(y_val[i].numpy().astype('float64'))
    
    for var_tmp in y_tgt:
        var_tmp[var_tmp<-999] = np.nan
    
    return y_mdl_inv, y_tgt_inv, y_mdl, y_tgt, y_mdl_unc

def inverse_transform_data(y, eff0, transform_method, scaler):
    # print(np.max(y), np.min(y), eff0)
    if 'asinh' in transform_method:
        return inv_smooth_linlog(scaler.inverse_transform(y), eff0)
    elif 'log' in transform_method:
        return scaler.inverse_transform(10**y)

smooth_linlog = lambda y, eff0: eff0*np.arcsinh(y/eff0)
inv_smooth_linlog = lambda y, eff0: eff0*np.sinh(y/eff0)
boxcox = lambda y, lam: (y**lam - 1)/lam if lam != 0 else np.log(y)
inv_boxcox = lambda y, lam: (y*lam+1)**(1/lam)