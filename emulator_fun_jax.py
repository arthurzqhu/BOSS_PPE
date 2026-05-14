import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn import preprocessing
import sklearn.model_selection as mod_sec
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import socket
import random

# JAX equivalents
import jax
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
from tinygp.helpers import JAXArray
import optax
import tuning_fun_jax as tu_jax


ncol_max = 4
T = 21600.0

softplus = lambda x: np.log(1 + np.exp(x))
smooth_linlog = lambda y, eff0: eff0 * np.arcsinh(jnp.clip(y, a_min=1e-35) / eff0)
inv_smooth_linlog = lambda y, eff0: eff0 * np.sinh(jnp.clip(y / eff0, a_min=-50, a_max=50))
boxcox = lambda y, lam: (y**lam - 1)/lam if lam != 0 else np.log(y)
blowup_tan = lambda t: np.tan(np.pi * (t / T - 0.5))
blowup_tan_inv = lambda y: T * (0.5 + (1.0/np.pi) * np.arctan(y))


def get_param_interest_idx(dataset, return_perturbed_groupname=False):
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

    if return_perturbed_groupname:
        return param_interest_idx, perturbed_pgroup
    else:
        return param_interest_idx

def get_params(basepath, filename):
    """
    Get the parameters of interest.
    If the netCDF file has a 'select_params_idx' global attribute (written for
    '_select' PPE runs), those indices are used to slice params_PPE instead of
    the default param_interest_idx derived from is_perturbed_* flags.
    """
    dataset = nc.Dataset(basepath + filename, mode='r')
    vars_vn = dataset.getncattr('init_var')

    param_interest_idx = np.concatenate(get_param_interest_idx(dataset))

    # Override with select_params_idx if present
    if 'select_params_idx' in dataset.ncattrs():
        param_interest_idx = np.array(dataset.getncattr('select_params_idx'), dtype=int)

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

def filter_zeros_and_get_indices(input_list, throw_away_ratio, seed=None):
    if seed:
        random.seed(seed)

    nppe = len(input_list)
    max_zeros_to_remove = int(nppe * throw_away_ratio)

    non_zero_indices = [i for i, x in enumerate(input_list) if any(x != 0)]
    zero_indices = [i for i, x in enumerate(input_list) if all(x == 0)]

    zeros_to_remove = min(int(len(zero_indices)*throw_away_ratio), max_zeros_to_remove)
    zeros_to_keep_count = len(zero_indices) - zeros_to_remove
    kept_zero_indices = random.sample(zero_indices, zeros_to_keep_count)

    combined_indices = non_zero_indices + kept_zero_indices

    return sorted(combined_indices)

def inverse_transform_data(y, transform_method, scaler, eff0=None):
    # Ensure y is 2D for sklearn if it's 1D
    is_1d = (y.ndim == 1)
    if is_1d:
        y_for_scaler = y.reshape(-1, 1)
    else:
        y_for_scaler = y
        
    if 'asinh' in transform_method or 'softplus' in transform_method:
        if eff0 is None:
            raise ValueError('eff0 is required for asinh/softplus transformation')
        # Inverse transform and then inverse asinh/softplus
        y_scaled_inv = scaler.inverse_transform(y_for_scaler)
        y_with_possible_nan = inv_smooth_linlog(y_scaled_inv, eff0)
    elif 'log' in transform_method:
        # Inverse transform and then 10** (assuming log10)
        y_scaled_inv = scaler.inverse_transform(y_for_scaler)
        y_with_possible_nan = 10**y_scaled_inv
    elif 'standard' in transform_method:
        y_with_possible_nan = scaler.inverse_transform(y_for_scaler)
    else:
        y_with_possible_nan = y_for_scaler
    
    y_with_possible_nan = np.nan_to_num(y_with_possible_nan, nan=0, neginf=0, posinf=0)
    if is_1d:
        return y_with_possible_nan.flatten()
    return y_with_possible_nan


def get_train_val_tgt_data(basepath, filename, param_train, transform_methods,
                           l_multi_output=False, test_size=0.2, random_state=1,
                           set_nan_to_neg1001=True, var_select=None, var_limit_zero=None,
                           throw_away_ratio=0., hurdle_threshold=0.3):
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
    var_constraints = getattr(dataset, 'var_constraints')
    
    if 'thresholds_eff0' in dataset.ncattrs():
        eff0s = getattr(dataset, 'thresholds_eff0')
    else:
        eff0s = []
    
    for ivar, varcon in enumerate(var_constraints):
        if ('prate_dm' in varcon or 'precip_max_dm' in varcon) and len(eff0s) > 0:
            eff0s[ivar] = 1e-4
        elif 'prate_ss_std' in varcon and len(eff0s) > 0:
            eff0s[ivar] = 1e-4
    
    if var_select is not None:
        var_missing = [v for v in var_select if v not in var_constraints]
        if var_missing:
            raise KeyError(f"Missing keys in var_constraints: {var_missing}")
        else:
            var_indices = [list(var_constraints).index(v) for v in var_select]
            var_constraints = [var_constraints[i] for i in var_indices]
            eff0s = [eff0s[i] for i in var_indices]
            
    nvar = len(var_constraints)
    ppe_var_names = ['ppe_' + i for i in var_constraints]
    ppe_raw_vals = []
    for i in ppe_var_names:
        val = dataset.variables[i][:]
        if isinstance(val, np.ma.MaskedArray):
            val = val.filled(np.nan)
        ppe_raw_vals.append(val)

    tgt_var_names = ['tgt_' + i for i in var_constraints]
    tgt_raw_vals = []
    for i in tgt_var_names:
        val = dataset.variables[i][:]
        if isinstance(val, np.ma.MaskedArray):
            val = val.filled(np.nan)
        tgt_raw_vals.append(val)

    if var_limit_zero is not None:
        idx = filter_zeros_and_get_indices(ppe_raw_vals[var_constraints.index(var_limit_zero)], throw_away_ratio, seed=random_state)
        ppe_vals_temp = ppe_raw_vals.copy()
        ppe_raw_vals = [i[idx] for i in ppe_vals_temp]
        param_train['vals'] = param_train['vals'][idx,:]

    minmaxscale = preprocessing.MinMaxScaler().fit(param_train['vals'])
    x_all = minmaxscale.transform(param_train['vals'])
    scalers['x'] = minmaxscale
    scalers['y'] = []

    y_train = {}
    y_val = {}

    ppe_info['nppe'], ppe_info['nparam_init'] = param_train['vals'].shape
    ppe_info['nobs'] = [int(np.prod(i.shape[1:])) for i in ppe_raw_vals]
    ppe_info['ncases'] = tgt_raw_vals[0].shape[0]
    try:
        ppe_info['npert'] = dataset.getncattr('npert')
    except AttributeError:
        ppe_info['npert'] = 1
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
        if f'tgt_unc_{varcon}' in dataset.variables:
            tgt_unc.append(dataset.variables[f'tgt_unc_{varcon}'][:])

        if isinstance(transform_methods, str):
            transform_method = transform_methods
        else:
            transform_method = transform_methods[ivar]

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
        elif transform_method == 'standard_scaler_blowup_tan':
            val = blowup_tan(ppe_raw_val_reshaped)
            val[val>100] = np.nan
            ppe_norm.append(val)
            val = blowup_tan(tgt_raw_val_reshaped)
            val[val>100] = np.nan
            tgt_norm.append(val)
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

    if len(ppe_norm) > 0:
        for i, iscale in enumerate(scalers['y']):
            dat = iscale.transform(ppe_norm[i])
            ppe_data.append(dat)
            dat[np.isinf(dat)] = np.nan
            dat = iscale.transform(tgt_norm[i])
            dat[np.isnan(tgt_norm[i])] = np.nan
            if ppe_info['npert'] > 1:
                dat = np.reshape(dat, (ppe_info['ncases'], ppe_info['npert']))
            tgt_data.append(dat)

    # Detect zero-inflated (hurdle) variables and compute transformed-zero values
    hurdle_vars = set()
    zero_fracs = []
    transformed_zeros = []
    for ivar, varcon in enumerate(var_constraints):
        zero_frac = 1.0 - np.mean(ppe_var_presence[ivar])
        zero_fracs.append(zero_frac)
        if zero_frac > hurdle_threshold:
            hurdle_vars.add(varcon)
        # Compute the transformed value of zero for hurdle combination
        eff0 = eff0s[ivar] if ivar < len(eff0s) else 0
        if isinstance(transform_methods, str):
            tm = transform_methods
        else:
            tm = transform_methods[ivar]
        if 'asinh' in tm:
            zero_normed = smooth_linlog(np.array([[0.0]]), eff0)
        else:
            zero_normed = np.array([[0.0]])
        if ivar < len(scalers['y']):
            transformed_zeros.append(float(scalers['y'][ivar].transform(zero_normed).flatten()[0]))
        else:
            transformed_zeros.append(0.0)
    ppe_info['hurdle_vars'] = hurdle_vars
    ppe_info['zero_fracs'] = zero_fracs
    ppe_info['transformed_zeros'] = transformed_zeros
    if hurdle_vars:
        print(f"Hurdle variables (>{hurdle_threshold*100:.0f}% zeros): {sorted(hurdle_vars)}")
        for ivar, varcon in enumerate(var_constraints):
            if varcon in hurdle_vars:
                print(f"  {varcon}: {zero_fracs[ivar]*100:.1f}% zeros, transformed_zero={transformed_zeros[ivar]:.4f}")

    for ivar, (ppe_varr_tmp, ppe_varp_tmp) in enumerate(zip(ppe_data, ppe_var_presence)):
        varcon = var_constraints[ivar]
        if test_size > 0:
            x_train, x_val, y_train_rawv_single_tmp, y_val_rawv_single_tmp = \
                mod_sec.train_test_split(x_all, ppe_varr_tmp, test_size=test_size, random_state=random_state)
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

        # Always store presence data (needed for hurdle GP training)
        if test_size > 0:
            _, _, y_train_wpresence_single, y_val_wpresence_single = \
                mod_sec.train_test_split(x_all, ppe_varp_tmp, test_size=test_size, random_state=random_state)
        else:
            y_train_wpresence_single = ppe_varp_tmp
            y_val_wpresence_single = None
        y_train[f'presence_{varcon}'] = y_train_wpresence_single
        y_val[f'presence_{varcon}'] = y_val_wpresence_single
    
    dataset.close()
    return x_train, x_val, y_train, y_val, tgt_data, tgt_unc, tgt_initvar_matrix, ppe_info, scalers


def _predict_gp(theta, x_train_valid, y_train_valid, x_val):
    """Build GP, condition on training data, predict at x_val. Returns (mean, variance)."""
    gp = tu_jax.build_gp(theta, x_train_valid)
    cond_gp = gp.condition(y_train_valid, jnp.asarray(x_val)).gp
    return np.array(cond_gp.loc), np.array(cond_gp.variance)

def get_model_results_gp(models, x_val, y_val, ppe_info, transform_methods, scalers, x_train, y_train_dict):
    eff0s = ppe_info['eff0s']
    nvar = ppe_info['nvar']
    nobs = ppe_info['nobs']
    var_constraints = ppe_info['var_constraints']
    h_vars = ppe_info.get('hurdle_vars', set())
    t_zeros = ppe_info.get('transformed_zeros', [0.0] * nvar)

    y_mdl_unc_inv = []
    y_mdl_unc = []
    y_mdl_inv = []
    y_mdl = []
    y_tgt_inv = []
    y_tgt = []

    for ivar, varcon in enumerate(var_constraints):
        if isinstance(transform_methods, str):
            transform_method = transform_methods
        else:
            transform_method = transform_methods[ivar]

        y_val_raw = y_val[varcon][:, :nobs[ivar]]
        y_tgt_inv.append(inverse_transform_data(y_val_raw, transform_method, scalers['y'][ivar], eff0s[ivar]))
        y_tgt.append(y_val_raw)

        y_t = y_train_dict[varcon][:, :nobs[ivar]]
        y_t_1d = jnp.asarray(y_t)[:, 0]
        valid_mask = np.array(y_t_1d) > -999

        if varcon in h_vars:
            # Presence GP prediction
            presence_targets = y_train_dict[f'presence_{varcon}'][:, 0].astype(np.float64)
            p_x = jnp.asarray(x_train)[valid_mask]
            p_y = jnp.asarray(presence_targets[valid_mask])
            p_loc, p_var = _predict_gp(models[f'presence_{varcon}'], p_x, p_y, x_val)
            p_loc = np.clip(p_loc, 0.01, 0.99)
            p_var = np.clip(p_var, 1e-12, None)

            # Value-given-present GP prediction
            present_mask = valid_mask & (np.array(presence_targets) > 0.5)
            v_x = jnp.asarray(x_train)[present_mask]
            v_y = y_t_1d[present_mask]
            v_loc, v_var = _predict_gp(models[f'value_{varcon}'], v_x, v_y, x_val)

            # Combine: E[Y] = P * V + (1-P) * tz
            tz = t_zeros[ivar]
            loc = p_loc * v_loc + (1.0 - p_loc) * tz
            # Var[Y] = P^2 * Var[V] + (V - tz)^2 * Var[P] + Var[P] * Var[V]
            var = p_loc**2 * v_var + (v_loc - tz)**2 * p_var + p_var * v_var
        else:
            # Standard single GP
            x_train_valid = jnp.asarray(x_train)[valid_mask]
            y_t_valid = y_t_1d[valid_mask]
            loc, var = _predict_gp(models[varcon], x_train_valid, y_t_valid, x_val)

        y_mdl.append(loc)
        unc = np.sqrt(np.clip(var, a_min=1e-12, a_max=None))
        y_mdl_unc.append(unc)

        y_mdl_inv.append(inverse_transform_data(loc, transform_method, scalers['y'][ivar], eff0s[ivar]))

        # Propagate uncertainty to raw space using finite difference
        y_up_inv = inverse_transform_data(loc + unc, transform_method, scalers['y'][ivar], eff0s[ivar])
        y_down_inv = inverse_transform_data(loc - unc, transform_method, scalers['y'][ivar], eff0s[ivar])
        y_mdl_unc_inv.append(np.abs(y_up_inv - y_down_inv) / 2.0)

    for var_tmp in y_tgt:
        var_tmp[var_tmp < -999] = np.nan

    return y_mdl_inv, y_tgt_inv, y_mdl, y_tgt, y_mdl_unc, y_mdl_unc_inv

def apply_model_gp(models, x_input, x_train, y_train_dict, var_constraints):
    outputs = {}
    for varcon in var_constraints:
        theta = models[varcon]
        y_t = y_train_dict[varcon]

        y_t_1d = jnp.asarray(y_t)[:, 0]
        valid_mask = np.array(y_t_1d) > -999
        x_train_valid = jnp.asarray(x_train)[valid_mask]
        y_t_valid = y_t_1d[valid_mask]

        gp = tu_jax.build_gp(theta, x_train_valid)
        cond_gp = gp.condition(y_t_valid, jnp.asarray(x_input)).gp
        
        loc = np.array(cond_gp.loc)
        var = np.array(cond_gp.variance)
        outputs[varcon] = np.concatenate([loc, var], axis=-1)
        
    return outputs

def plot_scatter(y_tgt, y_mdl, ppe_info, title, l_plot_log, savefig=False, prefix="", y_mdl_unc=None):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']
    
    ncol = min(ncol_max, nvar)
    nrow = int(np.ceil(nvar/ncol_max))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, nrow*2))
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
    
    for i, (yt_tmp, yp_tmp) in enumerate(zip(y_tgt, y_mdl)):
        # Robustly handle non-finite values
        yt_tmp = np.array(yt_tmp).flatten()
        yp_tmp = np.array(yp_tmp).flatten()
        v = np.isfinite(yt_tmp) & np.isfinite(yp_tmp)
        xt = yt_tmp[v]
        xp = yp_tmp[v]
        
        if len(xt) == 0:
            axs[i].text(0.5, 0.5, 'No Data', transform=axs[i].transAxes, ha='center')
            continue

        if y_mdl_unc is not None:
            axs[i].set_aspect('equal')
            yunc = np.array(y_mdl_unc[i]).flatten()[v]
            axs[i].errorbar(xt, xp, yerr=yunc, fmt='o', alpha=0.3, markersize=1, elinewidth=0.5, capsize=0)
        else:
            axs[i].set_aspect('equal')
            axs[i].scatter(xt, xp, alpha=0.1)
        axs[i].set_title(var_constraints[i])
        if title == 'Raw values' and l_plot_log:
            # Avoid setting log scale if we have non-positive values
            if np.any(xt <= 0) or np.any(xp <= 0):
                axs[i].set_xlabel('target output')
                axs[i].set_ylabel('emulator output')
            else:
                axs[i].set_xlabel('log10 target output')
                axs[i].set_ylabel('log10 emulator output')
                axs[i].set_xscale('log')
                axs[i].set_yscale('log')
        else:
            axs[i].set_xlabel('target output')
            axs[i].set_ylabel('emulator output')
        
        # Robustly get limits
        xlim = axs[i].get_xlim()
        ylim = axs[i].get_ylim()
        
        # Filter out non-finite values from limits
        valid_lims = [l for l in list(xlim) + list(ylim) if np.isfinite(l)]
        if len(valid_lims) >= 2:
            ax_min = max([xlim[0], ylim[0]])
            ax_max = min([xlim[1], ylim[1]])
            
            # Use data range if limits are degenerate
            if not np.isfinite(ax_min) or not np.isfinite(ax_max) or ax_min >= ax_max:
                ax_min = min(xt.min(), xp.min())
                ax_max = max(xt.max(), xp.max())

            if np.isfinite(ax_min) and np.isfinite(ax_max) and ax_min < ax_max:
                axs[i].plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange')
                axs[i].set_xlim(ax_min, ax_max)
                axs[i].set_ylim(ax_min, ax_max)
    
    fig.suptitle(title)
    try:
        fig.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed for {title}: {e}")
    if savefig:
        filename = f"{prefix}_" if prefix else ""
        plt.savefig(f'plots/ppe/emulator_scatter_{filename}{title.replace(" ", "_")}.pdf')

def plot_2dhist_unc(y_tgt, y_mdl, y_mdl_unc, title, ppe_info, savefig=False, prefix=""):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']

    ncol = min(ncol_max, nvar)
    nrow = int(np.ceil(nvar/ncol_max))
    fig, axs = plt.subplots(nrow, ncol, figsize=(14, nrow*3))
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, (yt_tmp, yp_tmp, yunc_tmp) in enumerate(zip(y_tgt, y_mdl, y_mdl_unc)):
        # Robustly handle non-finite values and flatten
        yt_tmp = np.array(yt_tmp).flatten()
        yp_tmp = np.array(yp_tmp).flatten()
        yunc_tmp = np.array(yunc_tmp).flatten()
        vpoint = np.isfinite(yt_tmp) & np.isfinite(yp_tmp) & np.isfinite(yunc_tmp)
        
        if not np.any(vpoint):
            axs[i].text(0.5, 0.5, 'No Data', transform=axs[i].transAxes, ha='center')
            continue

        axs[i].set_aspect('equal')
        x_tmp = yt_tmp[vpoint]
        y_tmp = yp_tmp[vpoint]
        unc_tmp = yunc_tmp[vpoint]
        
        xmin, xmax = np.min(x_tmp), np.max(x_tmp)
        ymin, ymax = np.min(y_tmp), np.max(y_tmp)
        
        # Robustly handle degenerate ranges
        if xmin == xmax:
            xmin -= 0.1
            xmax += 0.1
        if ymin == ymax:
            ymin -= 0.1
            ymax += 0.1
            
        xbins = np.linspace(xmin, xmax, 51)
        ybins = np.linspace(ymin, ymax, 61)

        xidx_tmp = np.digitize(x_tmp, xbins) - 1
        yidx_tmp = np.digitize(y_tmp, ybins) - 1

        mean_unc = np.full((len(xbins)-1, len(ybins)-1), np.nan)
        counts = np.zeros_like(mean_unc)

        valid = (xidx_tmp >= 0) & (xidx_tmp < mean_unc.shape[0]) & (yidx_tmp >= 0) & (yidx_tmp < mean_unc.shape[1])
        flat_idx_tmp = xidx_tmp[valid] * mean_unc.shape[1] + yidx_tmp[valid]
        unc_valid_tmp = unc_tmp[valid]

        sum_unc = np.bincount(flat_idx_tmp, weights=unc_valid_tmp, minlength=mean_unc.size)
        count_unc = np.bincount(flat_idx_tmp, minlength=mean_unc.size)

        with np.errstate(invalid='ignore', divide='ignore'):
            mean_vals = sum_unc / count_unc
        mean_unc[:,:] = mean_vals.reshape(mean_unc.shape)
        counts[:,:] = count_unc.reshape(mean_unc.shape)

        pcm = axs[i].pcolor(xbins, ybins, mean_unc.T, cmap='magma', shading='auto')
        plt.colorbar(pcm, ax=axs[i], label='std dev')

        ax_min = min(xmin, ymin)
        ax_max = max(xmax, ymax)
        if np.isfinite(ax_min) and np.isfinite(ax_max) and ax_min < ax_max:
            axs[i].plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange', linestyle='--')
            axs[i].set_xlim(ax_min, ax_max)
            axs[i].set_ylim(ax_min, ax_max)

        axs[i].set_title(var_constraints[i])
        suffix = " (normalized)" if title == "Normalized" else ""
        axs[i].set_xlabel('target' + suffix)
        axs[i].set_ylabel('emulator' + suffix)

    fig.suptitle(f'Uncertainty: {title}')
    try:
        fig.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed: {e}")
    if savefig:
        filename = f"{prefix}_" if prefix else ""
        plt.savefig(f'plots/ppe/emulator_2dhist_unc_{filename}{title.replace(" ", "_")}.pdf')
        

def plot_2dhist(y_tgt, y_mdl, ppe_info, title, l_plot_log, savefig=False, prefix=""):
    nvar = ppe_info['nvar']
    var_constraints = ppe_info['var_constraints']

    ncol = min(ncol_max, nvar)
    nrow = int(np.ceil(nvar/ncol_max))
    fig, axs = plt.subplots(nrow, ncol, figsize=(14, nrow*3))
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
    
    for i, (yt_tmp, yp_tmp) in enumerate(zip(y_tgt, y_mdl)):
        # Robustly handle non-finite values
        yt_tmp = np.array(yt_tmp).flatten()
        yp_tmp = np.array(yp_tmp).flatten()
        vpoint = np.isfinite(yt_tmp) & np.isfinite(yp_tmp)
        if not np.any(vpoint):
            continue
            
        axs[i].set_aspect('equal')

        if title == 'Normalized':
            hist, xedges, yedges = np.histogram2d(yt_tmp[vpoint], yp_tmp[vpoint], bins=[50, 60])
        elif title == 'Raw values':
            v_positive = vpoint & (yt_tmp > 0) & (yp_tmp > 0)
            if l_plot_log and np.any(v_positive):
                hist, xedges, yedges = np.histogram2d(np.log10(yt_tmp[v_positive]), np.log10(yp_tmp[v_positive]), bins=[50, 60])
            else:
                hist, xedges, yedges = np.histogram2d(yt_tmp[vpoint], yp_tmp[vpoint], bins=[50, 60])

        hist_min = 1e-6
        s = hist.sum()
        if s > 0:
            hist = hist / s
        if l_plot_log:
            hist = np.log10(np.maximum(hist, hist_min)).T
        else:
            hist = np.maximum(hist, hist_min).T
        pcm = axs[i].pcolor(xedges, yedges, hist, cmap='viridis', shading='auto')
        plt.colorbar(pcm, ax=axs[i], label='density')

        ax_min = min(xedges.min(), yedges.min())
        ax_max = max(xedges.max(), yedges.max())
        if np.isfinite(ax_min) and np.isfinite(ax_max) and ax_min < ax_max:
            axs[i].plot([ax_min, ax_max], [ax_min, ax_max], color='tab:orange', linestyle='--')
            axs[i].set_xlim(ax_min, ax_max)
            axs[i].set_ylim(ax_min, ax_max)
            
        axs[i].set_title(var_constraints[i])
        if title == 'Raw values' and l_plot_log:
             axs[i].set_xlabel('log10 target')
             axs[i].set_ylabel('log10 emulator')
        else:
             suffix = " (normalized)" if title == "Normalized" else ""
             axs[i].set_xlabel('target' + suffix)
             axs[i].set_ylabel('emulator' + suffix)
    
    fig.suptitle(title)
    try:
        fig.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed for {title}: {e}")
    if savefig:
        filename = f"{prefix}_" if prefix else ""
        plt.savefig(f'plots/ppe/emulator_2dhist_{filename}{title.replace(" ", "_")}.pdf')



def plot_emulator_results_gp(x_val, y_val, models, ppe_info, transform_methods, scalers, x_train, y_train_dict,
                          l_plot_uncertainty=False, l_plot_log=True, l_plot_scatter=False, savefig=False, prefix=""):

    y_mdl_inv, y_tgt_inv, y_mdl, y_tgt, y_mdl_unc, y_mdl_unc_inv = get_model_results_gp(models, x_val, y_val, ppe_info, transform_methods, scalers, x_train, y_train_dict)

    if l_plot_scatter:
        plot_scatter(y_tgt, y_mdl, ppe_info, 'Normalized', l_plot_log, savefig, prefix, y_mdl_unc=y_mdl_unc if l_plot_uncertainty else None)
        plot_scatter(y_tgt_inv, y_mdl_inv, ppe_info, 'Raw values', l_plot_log, savefig, prefix, y_mdl_unc=y_mdl_unc_inv if l_plot_uncertainty else None)
    else:
        plot_2dhist(y_tgt, y_mdl, ppe_info, 'Normalized', l_plot_log, savefig, prefix)
        plot_2dhist(y_tgt_inv, y_mdl_inv, ppe_info, 'Raw values', l_plot_log, savefig, prefix)
        
    if l_plot_uncertainty:
        plot_2dhist_unc(y_tgt, y_mdl, y_mdl_unc, 'Normalized', ppe_info, savefig, prefix)
        plot_2dhist_unc(y_tgt_inv, y_mdl_inv, y_mdl_unc_inv, 'Raw values', ppe_info, savefig, prefix)
