import numpy as np
import pandas as pd
import tensorflow as tf
import netCDF4 as nc
# from numba import cuda
import gc
from sklearn import preprocessing
import sklearn.model_selection as mod_sec

def get_params(basepath, filename, param_interest_idx):
    dataset = nc.Dataset(basepath + filename, mode='r')
    vars_vn = dataset.getncattr('init_var')
    if isinstance(vars_vn, str):
        vars_vn = [vars_vn]
    PPE_vns = [istr + "_PPE" for istr in vars_vn]
    initvar_matrix = []
    
    for PPE_vn in PPE_vns:
        initvar_matrix.append(np.expand_dims(dataset.variables[PPE_vn][:], axis=1))
    
    params_PPE = dataset.variables['params_PPE'][:][:, np.arange(28,40)]
    initvar_matrix.append(params_PPE)
    params_train = np.concatenate(initvar_matrix, axis=1)

    return {'pnames': dataset.variables['param_names'][:],
            'vals': params_train}

# def get_params(basepath, filename, param_interest_idx):
#     input_interest_idx = np.concatenate((np.arange(2), param_interest_idx + 2)) + 1
#     param_table = pd.read_csv(basepath + filename)
#     param_names = param_table.columns[1:].to_list()
#     param_vals = param_table.values[:, input_interest_idx].astype('float64')
#     # param_vals_log = np.log(param_vals)
#     # masked_pvl = np.ma.masked_invalid(param_vals_log)

#     return {'pnames': param_names,
#             'vals': param_vals}
#             # 'log_vals': masked_pvl}

def get_vars(basepath, filename):
    dataset = nc.Dataset(basepath + filename, mode='r')
    eff0 = getattr(dataset, 'thresholds_eff0')
    var_constraints = getattr(dataset, 'var_constraints')
    ppe_var_names = ['boss_' + i for i in var_constraints]
    ppe_raw_vals = [dataset.variables[i][:] for i in ppe_var_names]
    tgt_var_names = ['bin_' + i for i in var_constraints]
    tgt_raw_vals = [dataset.variables[i][:] for i in tgt_var_names]

    ppe_var_presence = []
    ppe_asinh = []
    ppe_all = []
    tgt_var_presence = []
    tgt_asinh = []
    tgt_all = []
    for i in range(len(var_constraints)):
        ppe_var_presence.append((ppe_raw_vals[i] > eff0[i]/100).astype('float32'))
        ppe_raw_vals[i][ppe_raw_vals[i] < eff0[i]/100] = np.nan
        ppe_raw_vals[i][~np.isfinite(ppe_raw_vals[i])] = np.nan
        ppe_asinh.append(smooth_linlog(ppe_raw_vals[i], eff0[i]))
        tgt_var_presence.append((tgt_raw_vals[i] > eff0[i]/100).astype('float32'))
        tgt_raw_vals[i][tgt_raw_vals[i] < eff0[i]/100] = np.nan
        tgt_raw_vals[i][~np.isfinite(tgt_raw_vals[i])] = np.nan
        tgt_asinh.append(smooth_linlog(tgt_raw_vals[i], eff0[i]))
        standscale = preprocessing.StandardScaler().fit(ppe_asinh[i].reshape(-1, 1))
        ppe_all.append(standscale.transform(ppe_asinh[i]).reshape(-1, 1))
        tgt_all.append(standscale.transform(tgt_asinh[i]).reshape(-1, 1))

    return ppe_var_names, ppe_var_presence, ppe_all, \
           tgt_var_names, tgt_var_presence, tgt_all

# def get_vars(basepath, filename, slim_down=[]):
#     var_table = pd.read_csv(basepath + filename)
#     var_vals = var_table.values[:,1:].astype('float64')
#     var_vals_log = np.log10(var_vals)
#     # masked_vvl = np.ma.masked_invalid(var_vals_log)

#     if len(slim_down) == 0:
#         slim_down = np.arange(var_vals.shape[1])

#     var_names = var_table.columns[slim_down+1].to_list()
#     return {'var_names': var_names,
#             'vals': var_vals[:, slim_down],
#             'log_vals': var_vals_log[:, slim_down]}

smooth_linlog = lambda y, eff0: eff0*np.arcsinh(y/eff0)
    