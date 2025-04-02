import numpy as np
import pandas as pd
import tensorflow as tf
# from numba import cuda
import gc

def get_params(basepath, filename, param_interest_idx):
    input_interest_idx = np.concatenate((np.arange(2), param_interest_idx + 2)) + 1
    param_table = pd.read_csv(basepath + filename)
    param_names = param_table.columns[1:].to_list()
    param_vals = param_table.values[:, input_interest_idx].astype('float64')
    # param_vals_log = np.log(param_vals)
    # masked_pvl = np.ma.masked_invalid(param_vals_log)

    return {'pnames': param_names,
            'vals': param_vals}
            # 'log_vals': masked_pvl}

def get_vars(basepath, filename, slim_down=[]):
    var_table = pd.read_csv(basepath + filename)
    var_vals = var_table.values[:,1:].astype('float64')
    var_vals_log = np.log10(var_vals)
    # masked_vvl = np.ma.masked_invalid(var_vals_log)

    if len(slim_down) == 0:
        slim_down = np.arange(var_vals.shape[1])

    var_names = var_table.columns[slim_down+1].to_list()
    return {'var_names': var_names,
            'vals': var_vals[:, slim_down],
            'log_vals': var_vals_log[:, slim_down]}

smooth_linlog = lambda y, eff0: eff0*np.arcsinh(y/eff0)
    