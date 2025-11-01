import numpy as np
import pandas as pd
import netCDF4 as nc
# from numba import cuda
import gc
from sklearn import preprocessing
import sklearn.model_selection as mod_sec
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random

ncol_max = 4

def get_param_interest_idx(dataset, return_perturbed_groupname=False):
    """
    Get the indices of the parameters of interest
    netCDF file should have the following global attributes:
        'n_param_*': number of parameters for each parameter group [int]
        'is_perturbed_*': whether the parameter group is perturbed by PPE [int]
    """

    n_param_attrs = [attr for attr in dataset.ncattrs() if "n_param" in attr]
    param_group_names = [attr.replace("n_param_", "") for attr in n_param_attrs]
    param_interest_idx = []
    perturbed_pgroup = []
    idx_start = 0

    for p_groupname in param_group_names:
        if dataset.getncattr(f"is_perturbed_{p_groupname}"):
            param_interest_idx.append(np.arange(idx_start, idx_start + dataset.getncattr(f"n_param_{p_groupname}")))
            perturbed_pgroup.append(p_groupname)
        idx_start += dataset.getncattr(f"n_param_{p_groupname}")
    # param_interest_idx = np.concatenate(param_interest_idx)

    if return_perturbed_groupname:
        return param_interest_idx, perturbed_pgroup
    else:
        return param_interest_idx
    

def get_params(basepath, filename):
    """
    Get the parameters of interest
    """
    dataset = nc.Dataset(basepath + filename, mode='r')
    vars_vn = dataset.getncattr('init_var')
    
    param_interest_idx = np.concatenate(get_param_interest_idx(dataset))

    if isinstance(vars_vn, str):
        vars_vn = [vars_vn]
    PPE_vns = [istr + "_PPE" for istr in vars_vn]
    ppe_initcon_matrix = []
    
    for PPE_vn in PPE_vns:
        if PPE_vn in dataset.variables:
            ppe_initcon_matrix.append(np.expand_dims(dataset.variables[PPE_vn][:], axis=1))
    
    params_PPE = dataset.variables['params_PPE'][:][:, param_interest_idx]
    ppe_initcon_matrix.append(params_PPE)
    params_train = np.concatenate(ppe_initcon_matrix, axis=1)

    pnames = dataset.variables['param_names'][:]
    dataset.close()

    return {'pnames': pnames,
            'vals': params_train,
            'param_interest_idx': param_interest_idx}

def get_train_val_tgt_data(basepath, filename, param_train, transform_methods, 
                           l_multi_output=False, test_size=0.2, random_state=1,
                           set_nan_to_neg1001=True, var_select=None, var_limit_zero=None,
                           throw_away_ratio=0.):
    """
    Get the training, validation, and target data.
    Args:
        basepath: path to the base directory
        filename: name of the netCDF file
            should have the following global attributes:
                'n_init': number of initial condition variables perturbed by PPE [int]
                'init_var': names of (varying) initial condition variables [string]
                'thresholds_eff0': thresholds for the constraint variables [float]
                'var_constraints': names of constraint variables [string]
                'n_param_*': number of parameters for each parameter group [int]
                'is_perturbed_*': whether the parameter group is perturbed by PPE [int]
            should have the following dimensions:
                ncases: number of cases run by target model
                nppe: number of PPE members
                nparams: number of parameters used by the model being trained
            should have the following variables:
                param_names: names of parameters [string] (optional but recommended)
                [init_var]_PPE: initial condition used by PPE members [float]
                ppe_[var_constraints]: constraint variables from each PPE member [float]
                params_PPE: parameters used by the model being trained [float]
                tgt_[var_constraints]: constraint variables from the target model [float]
                case_[init_var]: initial condition used by the target model [float]
        param_train: parameters of interest, output of get_params
        transform_methods: a string or a list of shape [nvar] of strings describing methods to transform the data, one of:
            'standard_scaler': standard scaler
            'standard_scaler_asinh': standard scaler with asinh transformation
            'standard_scaler_log': standard scaler with log transformation
            'minmaxscale_asinh': minmax scaler with asinh transformation
            'minmaxscale': minmax scaler
        l_multi_output: whether to use multi-output model: presence of water (boolean) and amount of water (float)
            set to True for multi-output model, False for CRPS (default: False)
        test_size: size of the test set (default: 0.2). If 0, use all data for training.
        random_state: random seed (default: 1)
        set_nan_to_neg1001: whether to set nan to -1001 (default: True)
        var_select: select specific variables to use as constraint variables. If None, use all variables. (default: None)
        var_limit_zero: choose one variable to limit the data to be greater than zero. (default: None)
        throw_away_ratio: ratio of data to throw away. If 0, use all data. (default: 0)
    Returns:
        x_train: training parameters 
            minmaxscaled
            shape: [ntrain, nparam_init]
        x_val: similar to x_train but for validation parameters
        y_train: training data 
            user specified transform method: 'standard_scalar_asinh' is preferred for moment values, 'minmaxscale_asinh' is preferred for fall speed
            has keys: VAR (constraint variable), 'presence_VAR' (presence of constraint variable, if l_multi_output is True)
            VAR refers to all the constraints, which are elements of `var_constraints`
            shape: [ntrain, nobs]
        y_val: similar to y_train but for validation data
        tgt_data: target data
            a list of length `nvar` of target data, each element is a 2D array of shape [ncases, nobs]
        tgt_initvar_matrix
            the list of initial conditions for the target data
            shape: [ncases]
        ppe_info: 
            information about the PPE, has keys:
            'nppe': number of PPE members
            'nvar': number of constraint variables
            'npar': number of parameters
            'n_init': number of initial condition variables perturbed by PPE
            'nparam_init': number of parameters + initial conditions (total perturbed parameters) = nppe + n_init
            'nobs': number of observations for each constraint variable
            'ncases': number of cases run by target model (TAU by default)
            'eff0s': used for asinh transformation, thresholds for the constraint variables, below which the transformed data loglike, above which it's linear like
            'var_constraints': names of constraint variables
        scalers: scalers for the data
            has keys: 'x' and 'y', scalers['y'] is a list of length `nvar` of scalers, each element is a scaler type object
    """

    scalers = {}

    ppe_info = {}
    dataset = nc.Dataset(basepath + filename, mode='r')
    ppe_info['n_init'] = getattr(dataset, 'n_init')
    init_vars = getattr(dataset, 'init_var')
    if isinstance(init_vars, str):
        init_vars = [init_vars]

    tgt_initvar_matrix = []

    for init_var in init_vars:
        if 'case_' + init_var in dataset.variables:
            tgt_initvar_matrix.append(np.expand_dims(dataset.variables['case_' + init_var][:], axis=1))

    if 'thresholds_eff0' in dataset.ncattrs():
        eff0s = getattr(dataset, 'thresholds_eff0')
    else:
        eff0s = []
    eff0s[4] = 1e-8
    eff0s[5] = 1e-8
    var_constraints = getattr(dataset, 'var_constraints')
    if var_select is not None:
        # check if all variables in var_select are in var_constraints
        var_missing = [v for v in var_select if v not in var_constraints]
        if var_missing:
            raise KeyError(f"Missing keys in var_constraints: {var_missing}")
        else:
            # get the indices of var_constraints for each element of var_select
            var_indices = [list(var_constraints).index(v) for v in var_select]
            var_constraints = [var_constraints[i] for i in var_indices]
            eff0s = [eff0s[i] for i in var_indices]
    nvar = len(var_constraints)
    ppe_var_names = ['ppe_' + i for i in var_constraints]
    ppe_raw_vals = [dataset.variables[i][:] for i in ppe_var_names]
    tgt_var_names = ['tgt_' + i for i in var_constraints]
    tgt_raw_vals = [dataset.variables[i][:] for i in tgt_var_names]

    if var_limit_zero is not None:
        if not isinstance(var_limit_zero, str):
            raise TypeError('optional `var_limit_zero` needs to be a string')

        idx = filter_zeros_and_get_indices(ppe_raw_vals[var_constraints.index(var_limit_zero)], throw_away_ratio, seed=random_state)
        ppe_vals_temp = ppe_raw_vals.copy()
        ppe_raw_vals = [i[idx] for i in ppe_vals_temp]
        param_train['vals'] = param_train['vals'][idx,:]

    # always use minmaxscale for parameters to avoid extrapolation
    minmaxscale = preprocessing.MinMaxScaler().fit(param_train['vals'])
    x_all = minmaxscale.transform(param_train['vals'])
    scalers['x'] = minmaxscale
    scalers['y'] = []

    y_train = {}
    y_val = {}


    ppe_info['nppe'], ppe_info['nparam_init'] = param_train['vals'].shape
    ppe_info['nobs'] = [int(np.prod(i.shape[1:])) for i in ppe_raw_vals]
    ppe_info['ncases'] = tgt_raw_vals[0].shape[0]
    ppe_info['nvar'] = nvar
    ppe_info['npar'] = ppe_info['nparam_init'] - ppe_info['n_init']
    ppe_info['eff0s'] = eff0s
    ppe_info['var_constraints'] = var_constraints

    ppe_var_presence = []
    tgt_var_presence = []
    ppe_norm = []
    tgt_norm = []
    ppe_data = []
    tgt_data = []
    tgt_unc = []

    for ivar, (varcon, eff0) in enumerate(tqdm(zip(var_constraints, eff0s), desc='Transforming data...')):
        # get obs uncertainty
        # print(dataset.variables[f'tgt_unc_{varcon}'][:])
        if f'tgt_unc_{varcon}' in dataset.variables:
            tgt_unc.append(dataset.variables[f'tgt_unc_{varcon}'][:])

        # get transform method
        if isinstance(transform_methods, str):
            transform_method = transform_methods
        else:
            if len(transform_methods) != len(var_constraints):
                raise ValueError(f"transform_methods ({len(transform_methods)}) must be a string \
                    or a list of length same as var_constraints ({len(var_constraints)})")
            transform_method = transform_methods[ivar]

        # reshape data
        if ppe_raw_vals[ivar].ndim >= 2:
            ppe_raw_val_reshaped = np.reshape(ppe_raw_vals[ivar], (ppe_info['nppe'], np.prod(ppe_raw_vals[ivar].shape[1:])))
            tgt_raw_val_reshaped = np.reshape(tgt_raw_vals[ivar], (ppe_info['ncases'], np.prod(tgt_raw_vals[ivar].shape[1:])))
        else:
            ppe_raw_val_reshaped = ppe_raw_vals[ivar].reshape(-1, 1)
            tgt_raw_val_reshaped = tgt_raw_vals[ivar].reshape(-1, 1)

        ppe_var_presence.append(ppe_raw_val_reshaped > eff0/100)
        tgt_var_presence.append(tgt_raw_val_reshaped > eff0/100)

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
        elif transform_method == 'minmaxscale':
            ppe_norm.append(ppe_raw_val_reshaped)
            tgt_norm.append(tgt_raw_val_reshaped)
            mmscale = preprocessing.MinMaxScaler().fit(ppe_raw_val_reshaped)
            scalers['y'].append(mmscale)
        elif transform_method == 'minmaxscale_asinh':
            ppe_norm.append(smooth_linlog(ppe_raw_val_reshaped, eff0))
            tgt_norm.append(smooth_linlog(tgt_raw_val_reshaped, eff0))
            mmscale = preprocessing.MinMaxScaler().fit(ppe_norm[-1])
            mmscale.data_min_[:] = min(mmscale.data_min_)
            mmscale.data_max_[:] = max(mmscale.data_max_)
            mmscale.data_range_[:] = max(mmscale.data_max_) - min(mmscale.data_min_)
            mmscale.scale_ = 1/mmscale.data_range_
            scalers['y'].append(mmscale)

    # print([i.shape for i in ppe_var_presence])
    # mom_consistency_mask = np.min(np.array(ppe_var_presence), axis=0)
    # scale_mask = np.max(mom_consistency_mask, axis=0)

    if len(ppe_norm) > 0:
        for i, iscale in enumerate(scalers['y']):
            # iscale.scale_ = iscale.scale_ * scale_mask
            dat = iscale.transform(ppe_norm[i])
            # dat[ppe_norm[i].mask] = np.nan
            ppe_data.append(dat)
            dat[np.isinf(dat)] = np.nan
            dat = iscale.transform(tgt_norm[i])
            dat[tgt_norm[i].mask] = np.nan
            tgt_data.append(dat)
            dat[np.isinf(dat)] = np.nan

    for ivar, (ppe_varr_tmp, ppe_varp_tmp) in enumerate(zip(ppe_data, ppe_var_presence)):
        varcon = var_constraints[ivar]
        if test_size > 0:
            x_train, x_val, y_train_rawv_single_tmp, y_val_rawv_single_tmp =\
                mod_sec.train_test_split(x_all, ppe_varr_tmp, test_size=test_size, random_state=random_state)
            # print(np.where(np.isin(x_train, x_all))[0])
        else:
            x_train = x_all
            x_val = None
            y_train_rawv_single_tmp = ppe_varr_tmp
            y_val_rawv_single_tmp = None
        if set_nan_to_neg1001:
            y_train_rawv_single_tmp = np.nan_to_num(y_train_rawv_single_tmp, nan=-1001, neginf=-1001, posinf=-1001)
            y_val_rawv_single_tmp = np.nan_to_num(y_val_rawv_single_tmp, nan=-1001, neginf=-1001, posinf=-1001)
        y_train[varcon] = y_train_rawv_single_tmp
        y_val[varcon] = y_val_rawv_single_tmp

        if l_multi_output:
            if test_size > 0:
                x_train, x_val, y_train_wpresence_single, y_val_wpresence_single = \
                    mod_sec.train_test_split(x_all, ppe_varp_tmp, test_size=test_size, random_state=random_state)
            else:
                x_train = x_all
                x_val = None
                y_train_wpresence_single = ppe_varp_tmp
                y_val_wpresence_single = None
            y_train[f'presence_{varcon}'] = y_train_wpresence_single
            y_val[f'presence_{varcon}'] = y_val_wpresence_single
    
    dataset.close()
    return x_train, x_val, y_train, y_val, tgt_data, tgt_unc, tgt_initvar_matrix, ppe_info, scalers

def plot_emulator_results(x_val, y_val, model, ppe_info, transform_methods, scalers,
                          l_plot_uncertainty=False, l_plot_log=True, l_plot_scatter=False):

    y_mdl_inv, y_tgt_inv, y_mdl, y_tgt, y_mdl_unc = apply_model(model, x_val, y_val, ppe_info, transform_methods, scalers)

    if l_plot_scatter:
        plot_scatter(y_tgt, y_mdl, ppe_info, 'Normalized', l_plot_log)
        plot_scatter(y_tgt_inv, y_mdl_inv, ppe_info, 'Raw values', l_plot_log)
    else:
        plot_2dhist(y_tgt, y_mdl, ppe_info, 'Normalized', l_plot_log)
        plot_2dhist(y_tgt_inv, y_mdl_inv, ppe_info, 'Raw values', l_plot_log)
    if l_plot_uncertainty:
        plot_2dhist_unc(y_tgt, y_mdl, y_mdl_unc, ppe_info)

def plot_scatter(y_tgt, y_mdl, ppe_info, title, l_plot_log):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']
    
    ncol = min(ncol_max, nvar)
    nrow = int(np.ceil(nvar/ncol_max))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, nrow*2))
    axs = axs.flatten()
    
    for i, (yt_tmp, yp_tmp) in enumerate(zip(y_tgt, y_mdl)):
        axs[i].set_aspect('equal')
        axs[i].scatter(yt_tmp, yp_tmp, alpha=0.1)
        axs[i].set_title(var_constraints[i])
        if title == 'Raw values' and l_plot_log:
            axs[i].set_xlabel('log10 target output')
            axs[i].set_ylabel('log10 emulator output')
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
        else:
            axs[i].set_xlabel('target output')
            axs[i].set_ylabel('emulator output')
        ax_min = max([axs[i].get_ylim()[0]] + [axs[i].get_xlim()[0]])
        ax_max = min([axs[i].get_ylim()[1]] + [axs[i].get_xlim()[1]])
        axs[i].plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange')
    
    fig.suptitle(title)
    fig.tight_layout() 
    

def plot_2dhist_unc(y_tgt, y_mdl, y_mdl_unc, ppe_info):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']

    ncol = min(ncol_max, nvar)
    nrow = int(np.ceil(nvar/ncol_max))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, nrow*2), sharex=True)
    axs = axs.flatten()

    for i, (yt_tmp, yp_tmp, yunc_tmp) in enumerate(zip(y_tgt, y_mdl, y_mdl_unc)):
        axs[i].set_aspect('equal')

        vpoint = np.logical_and(np.isfinite(yt_tmp), np.isfinite(yp_tmp))
        # vpoint = np.logical_and(yt_tmp>0, yp_tmp>0)
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
        pcm = axs[i].pcolor(xbins, ybins, mean_unc.T, cmap='magma', shading='auto')
        # plt.colorbar(pcm, ax=axs[i], label='Mean ln(uncertainty)')

        ax_min = max([axs[i].get_ylim()[0]] + [axs[i].get_xlim()[0]])
        ax_max = min([axs[i].get_ylim()[1]] + [axs[i].get_xlim()[1]])
        axs[i].plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange')
        axs[i].set_title(var_constraints[i])
        axs[i].set_xlabel('log10 BOSS output')
        axs[i].set_ylabel('log10 emulator output')

    fig.tight_layout()
        

def plot_2dhist(y_tgt, y_mdl, ppe_info, title, l_plot_log):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']

    ncol = min(ncol_max, nvar)
    nrow = int(np.ceil(nvar/ncol_max))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, nrow*2))
    axs = axs.flatten()
    
    for i, (yt_tmp, yp_tmp) in enumerate(zip(y_tgt, y_mdl)):
        # ax = fig.add_subplot(gs[i])
        axs[i].set_aspect('equal')

        if title == 'Normalized':
            vpoint = np.logical_and(np.isfinite(yt_tmp), np.isfinite(yp_tmp))
            hist, xedges, yedges = np.histogram2d(yt_tmp[vpoint], yp_tmp[vpoint], bins=[50, 60])
        elif title == 'Raw values':
            vpoint = np.logical_and(yt_tmp>0, yp_tmp>0)
            # vpoint = np.logical_and(np.isfinite(yt_tmp), np.isfinite(yp_tmp))
            if l_plot_log:
                hist, xedges, yedges = np.histogram2d(np.log10(yt_tmp[vpoint]), np.log10(yp_tmp[vpoint]), bins=[50, 60])
            else:
                hist, xedges, yedges = np.histogram2d(yt_tmp[vpoint], yp_tmp[vpoint], bins=[50, 60])

        hist_min = 1e-6
        hist = hist/hist.sum()
        if l_plot_log:
            hist = np.log10(np.maximum(hist, hist_min)).T
        else:
            hist = np.maximum(hist, hist_min).T
        pclr_hist = axs[i].pcolor(xedges, yedges, hist, cmap='viridis', shading='auto')
        # plt.colorbar(pclr_hist, ax=axs[i])

        ax_min = max([axs[i].get_ylim()[0]] + [axs[i].get_xlim()[0]])
        ax_max = min([axs[i].get_ylim()[1]] + [axs[i].get_xlim()[1]])
        axs[i].plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange')
        axs[i].set_title(var_constraints[i])
        if l_plot_log:
            axs[i].set_xlabel('log10 target output')
            axs[i].set_ylabel('log10 emulator output')
        else:
            axs[i].set_xlabel('target output')
            axs[i].set_ylabel('emulator output')
    
    fig.suptitle(title)
    fig.tight_layout()

def apply_model(model, x_val, y_val, ppe_info, transform_methods, scalers):
    eff0s = ppe_info['eff0s']
    nvar = ppe_info['nvar']
    nobs = ppe_info['nobs']
    varcons = ppe_info['var_constraints']

    y_mdl_inv = []
    y_tgt_inv = []
    y_mdl = []
    y_tgt = []
    y_mdl_unc = []

    for i, varcon in enumerate(varcons):
        
        if isinstance(transform_methods, str):
            transform_method = transform_methods
        else:
            if len(transform_methods) != nvar:
                raise ValueError(f"transform_methods ({len(transform_methods)}) must be a string \
                    or a list of length same as var_constraints ({nvar})")
            transform_method = transform_methods[i]

        if type(model(x_val)) is dict:
            presence = 1.
            if f'presence_{varcon}' in model(x_val).keys():
                presence = model(x_val)[f'presence_{varcon}'].numpy()
                presence[presence<0.5] = 0.
                presence[presence>=0.5] = 1.
                
            y_model_raw = model(x_val)[varcon][:,:nobs[i]]
            y_model_unc = softplus(model(x_val)[varcon][:,nobs[i]:])
            if type(y_val) is dict:
                y_val_raw = y_val[varcon][:,:nobs[i]]
            else:
                y_val_raw = y_val[i]
            
            y_mdl_inv.append(presence * inverse_transform_data(y_model_raw, transform_method, scalers['y'][i], eff0s[i]))
            y_tgt_inv.append(inverse_transform_data(y_val_raw, transform_method, scalers['y'][i], eff0s[i]))
            y_mdl.append(y_model_raw)
            y_tgt.append(y_val_raw)
            y_mdl_unc.append(y_model_unc)
        else:
            raise ValueError('model output is not a dictionary. Not yet implemented.')
            # y_model = model(x_val)[i].numpy().astype('float64')
            # y_val_raw = y_val[i].numpy().astype('float64')
            # y_mdl_inv.append(inverse_transform_data(y_model, transform_method, scalers['y'][i], eff0s[i]))
            # y_tgt_inv.append(inverse_transform_data(y_val_raw, transform_method, scalers['y'][i], eff0s[i]))
            # y_mdl.append(y_model)
            # y_tgt.append(y_val[i].numpy().astype('float64'))
    
    for var_tmp in y_tgt:
        var_tmp[var_tmp<-999] = np.nan
    
    return y_mdl_inv, y_tgt_inv, y_mdl, y_tgt, y_mdl_unc

def filter_zeros_and_get_indices(input_list, throw_away_ratio, seed=None):
    if seed:
        random.seed(seed)

    nppe = len(input_list)
    max_zeros_to_remove = int(nppe * throw_away_ratio)

    # Get the indices of non-zeros and zeros
    non_zero_indices = [i for i, x in enumerate(input_list) if x != 0]
    zero_indices = [i for i, x in enumerate(input_list) if x == 0]

    # Determine how many zeros to keep and randomly sample their indices
    zeros_to_remove = min(int(len(zero_indices)*throw_away_ratio), max_zeros_to_remove)
    zeros_to_keep_count = len(zero_indices) - zeros_to_remove
    kept_zero_indices = random.sample(zero_indices, zeros_to_keep_count)

    # Combine the indices of non-zeros and kept zeros
    combined_indices = non_zero_indices + kept_zero_indices

    return sorted(combined_indices)

def inverse_transform_data(y, transform_method, scaler, eff0=None):
    if 'asinh' in transform_method:
        if eff0 is None:
            raise ValueError('eff0 is required for asinh transformation')
        y_with_possible_nan = inv_smooth_linlog(scaler.inverse_transform(y), eff0)
    elif 'log' in transform_method:
        y_with_possible_nan = scaler.inverse_transform(10**y)
    else:
        y_with_possible_nan = scaler.inverse_transform(y)        
    return np.nan_to_num(y_with_possible_nan, nan=0, neginf=0, posinf=0)

def make_weights_dict(y_dict,
                      rr_key,                 # the output name for rain rate in y_* dicts
                      rr_threshold_raw,       # raw-unit threshold you use to define "no rain"
                      asinh_scale, t_mean, t_std,  # transform params used to make y_dict
                      w_zero=0.2, w_pos=1.0,
                      smooth_alpha=0.0):
    """
    Build per-output weight arrays matching shapes of y_dict[output].
    All outputs get ones except rr_key, which is down-weighted for 'no-rain'.
    y_dict values are already TRANSFORMED (asinh -> standardize).
    """
    # transformed threshold t0 for the raw threshold
    t0 = (np.arcsinh(rr_threshold_raw / asinh_scale) - t_mean) / (t_std)

    sw = {}
    for k, y in y_dict.items():
        w = np.ones_like(y, dtype=np.float32)
        if k == rr_key:
            if smooth_alpha and smooth_alpha > 0.0:
                ramp = 1.0 / (1.0 + np.exp(-smooth_alpha * (y - t0)))
                w = w_zero + (w_pos - w_zero) * ramp
            else:
                w = np.where(y <= t0, w_zero, w_pos).astype(np.float32)
        sw[k] = w
    return sw

# instead of importing tensorflow:
def softplus(x):
    return np.log(1 + np.exp(x))

smooth_linlog = lambda y, eff0: eff0*np.arcsinh(y/eff0)
inv_smooth_linlog = lambda y, eff0: eff0*np.sinh(y/eff0)
boxcox = lambda y, lam: (y**lam - 1)/lam if lam != 0 else np.log(y)
