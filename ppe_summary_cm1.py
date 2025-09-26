"""
PPE summary processing script
Note that some hard-coded variables are used in this script.
"""

import cm1_load_utils as cl
import load_ppe_fun as lp
import numpy as np
import matplotlib.pyplot as plt
import os
# from tqdm import tqdm
from time import sleep
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import itertools
import util_fun as uf
import re
import pandas as pd
import netCDF4 as nc

import sys
from tqdm.auto import tqdm

# Disable the background monitor thread entirely
tqdm.monitor_interval = 0  # <- important

# Only show bars on a TTY (prevents odd behaviour in batch)
tqdm_disable = not sys.stderr.isatty()

def main():
    """Main function to process PPE data with memory efficiency"""
    # Configuration
    nikki = ''
    target_nikki = 'target'
    sim_config = 'fullmp_ppe_r1_sedflux'
    target_sim_config = 'fullmp_with_sedflux'
    
    if not os.path.exists(lp.nc_dir):
        os.makedirs(lp.nc_dir)
    l_cic = True
    
    n_init = 1
    target_mp = 'BIN-TAU'
    train_mp = 'SLC-BOSS'
    mconfigs = os.listdir(cl.output_dir + nikki)
    vars_strs, vars_vn = lp.get_dics(cl.output_dir, target_nikki, target_sim_config, n_init)
    var_interest = []
    # var_interest += ['M0_last2hrmean', 'M3_last2hrmean', 'M4_last2hrmean', 'M6_last2hrmean'] # domain-mean path
    var_interest += ['M0_path_last2hrmean', 'M3_path_last2hrmean', 'M4_path_last2hrmean', 'M6_path_last2hrmean', 'prate_dm_last2hrmean'] # domain-mean path
    var_interest += ['sfM0_per5lvl_last2hrmean', 'sfM3_per5lvl_last2hrmean', 'sfM4_per5lvl_last2hrmean', 'sfM6_per5lvl_last2hrmean'] # domain-mean fluxes
    print("Memory usage at start:")
    uf.detailed_memory_analysis()
    
    # Process data
    
    file_info = {'dir': cl.output_dir, 
                'date': nikki,
                'vars_vn': vars_vn}

    if 'nc_dict' not in globals():
        nc_dict = {}

    # Load PPE data
    print("\nLoading PPE data...")
    file_info.update({'sim_config': sim_config,
                    'date': nikki,
                    'mp_config': train_mp})
    ppe_idx = cl.get_ppe_idx(file_info)
    ppe_idx = [int(i) for i in ppe_idx]
    for ippe in tqdm(ppe_idx, desc='loading BOSS data'):
        cl.load_cm1(file_info, var_interest, nc_dict, True, ippe=ippe)

    # Load target data
    print("\nLoading target data...")
    for initcond_combo in tqdm(itertools.product(*vars_strs), desc='loading BIN data'):
        # ic_str = "".join(initcond_combo)
        file_info.update({'sim_config': target_sim_config,
                        'vars_str': list(initcond_combo),
                        'date': target_nikki,
                        'mp_config': target_mp})
        cl.load_cm1(file_info, var_interest, nc_dict, False)
    
    print("\nMemory usage after loading data:")
    uf.detailed_memory_analysis()


    plot_dir = f"plots/{nikki}/{sim_config}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
    axs = axs.flatten()
    na = []
    for initcond_combo in itertools.product(*vars_strs):
        ic_str = "".join(initcond_combo)
        na.append(nc_dict[ic_str]['BIN-TAU']['na'])

    na = np.array(na)

    for ivar, var_name in enumerate(var_interest[:5]):
        tgt_data = []
        train_data = []
        na_train = []
        for initcond_combo in itertools.product(*vars_strs):
            ic_str = "".join(initcond_combo)
            tgt_data.append(nc_dict[ic_str]['BIN-TAU'][var_name]['value'])
        for ippe in ppe_idx:
            ippe = int(ippe)
            train_data.append(nc_dict['cic']['SLC-BOSS'][ippe][var_name]['value'])
            na_train.append(nc_dict['cic']['SLC-BOSS'][ippe]['na'])
        tgt_data = np.array(tgt_data)
        train_data = np.array(train_data)
        na_train = np.array(na_train)
        axs[ivar].plot(na, tgt_data, label=ic_str, linewidth=2, marker='o')
        axs[ivar].scatter(na_train, train_data, label=ic_str, s=5, color='tab:orange', alpha=0.5)
        axs[ivar].set_title(cl.output_var_set[var_name]['longname'])
        axs[ivar].set_yscale('log')

    plt.savefig(f"{plot_dir}{sim_config}_dm_path.png")

    ncase = 1
    ncase_respective = [len(i) for i in vars_strs]
    for i in ncase_respective:
        ncase *= i
    
    dims = {
        'scalar_var': 1,
        'ncase': ncase,
        'nppe': len(ppe_idx),
    }

    global_attrs = {
        'description': 'PPE data for ' + sim_config,
        'date_simulated': nikki,
    }
    
    
    # Create variable structure
    ncvars = create_nc_variables_structure(nc_dict, vars_vn, var_interest)
    
    print("\nProcessing target data...")
    process_target_data(nc_dict, vars_vn, vars_strs, var_interest, ncvars, dims, target_mp)
                        
    # Processing PPE data
    print("\nProcessing PPE data...")
    ncvars, dims = process_ppe_data(nc_dict, ppe_idx, vars_vn, var_interest, ncvars, dims, nikki, sim_config, train_mp)
    
    print("\nMemory usage after processing data:")
    uf.detailed_memory_analysis()
    
    # Set up global attributes
    global_attrs['thresholds_eff0'] = []
    var_constraints = []
    for ivar in var_interest:
        var_constraints.append(ivar)   
    global_attrs['var_constraints'] = np.array(var_constraints)
    global_attrs['init_var'] = np.array(vars_vn)
    for var_vn in vars_vn:
        global_attrs[var_vn + '_units'] = nc_dict[var_vn + '_units']
    global_attrs['n_init'] = n_init
    global_attrs['n_param_nevp'] = nc_dict['n_param_nevp']
    global_attrs['n_param_condevp'] = nc_dict['n_param_condevp']
    global_attrs['n_param_coal'] = nc_dict['n_param_coal']
    global_attrs['n_param_sed'] = nc_dict['n_param_sed']
    global_attrs['is_perturbed_nevp'] = nc_dict['is_perturbed_nevp']
    global_attrs['is_perturbed_condevp'] = nc_dict['is_perturbed_condevp']
    global_attrs['is_perturbed_coal'] = nc_dict['is_perturbed_coal']
    global_attrs['is_perturbed_sed'] = nc_dict['is_perturbed_sed']

    # Calculate thresholds
    for ivar in var_interest:
        value_greater_0 = ncvars['ppe_' + ivar]['data'][ncvars['ppe_' + ivar]['data'] > 0]
        if 'V_M' in ivar:
            global_attrs['thresholds_eff0'].append(0.1)
        else:
            global_attrs['thresholds_eff0'].append(np.nanpercentile(value_greater_0, 1))
    
    print("Thresholds:", global_attrs['thresholds_eff0'])
    
    # Write netCDF file
    print("\nWriting netCDF file...")

    # Check if file exists and handle overwrite
    nc_filename = f"{lp.nc_dir}{sim_config}_last2hrmean_N{len(ppe_idx)}.nc"
    if os.path.exists(nc_filename):
        print(f"\nFile '{nc_filename}' already exists.")
        user_choice = input("Do you want to replace it (r) or keep both (k)? [r/k]: ").strip().lower()
        if user_choice == 'k':
            base, ext = os.path.splitext(nc_filename)
            suffix = 1
            new_filename = f"{base}_copy{suffix}{ext}"
            while os.path.exists(new_filename):
                suffix += 1
                new_filename = f"{base}_copy{suffix}{ext}"
            nc_filename = new_filename
            print(f"Saving as '{nc_filename}' instead.")
        elif user_choice == 'r':
            try:
                os.remove(nc_filename)
                print(f"Removed existing file '{nc_filename}'.")
            except Exception as e:
                print(f"Could not remove file '{nc_filename}': {e}")
                print("Exiting without saving.")
                try:
                    nc_file.close()
                except Exception:
                    pass
        else:
            print("Invalid input. Exiting without saving.")
            try:
                nc_file.close()
            except Exception:
                pass

    nc_file = nc.Dataset(nc_filename, 'w', format='NETCDF4')
    write_netcdf(nc_file, ncvars, dims, global_attrs)
    nc_file.close()
    
    # Clear all large variables
    del ncvars, nc_file, dims, global_attrs, nc_dict
    
    print("\nFinal memory usage:")
    uf.detailed_memory_analysis()
    print("\nProcessing complete!")

def create_nc_variables_structure(nc_dict, vars_vn, var_interest):
    """Create the netCDF variable structure without loading data"""
    ncvars = {}
    # Initialize PPE variables
    for var_vn in vars_vn:
        ncvars[var_vn + '_PPE'] = {
            'dims': ('nppe',),
            'units': nc_dict[var_vn + '_units'],
            'data': None  # Will be filled later
        }
    
    # Initialize summary variables
    for ivar in var_interest:
        var_units = cl.output_var_set[ivar]['var_unit']

        if 'per5lvl' in ivar:
            ncvars['ppe_' + ivar] = {
                'dims': ('nppe', 'nlevel'),
                'units': var_units,
                'data': None
            }

            ncvars['tgt_' + ivar] = {
                'dims': ('ncase', 'nlevel'),
                'units': var_units,
                'data': None
            }

        else:
            ncvars['ppe_' + ivar] = {
                'dims': ('nppe',),
                'units': var_units,
                'data': None
            }

            ncvars['tgt_' + ivar] = {
                'dims': ('ncase',),
                'units': var_units,
                'data': None
            }

    # Initialize case variables
    for var_vn in vars_vn:
        ncvars['case_' + var_vn] = {
            'dims': ('ncase',),
            'units': nc_dict[var_vn + '_units'],
            'data': None
        }
    
    return ncvars

def process_ppe_data(nc_dict, ppe_idx, vars_vn, var_interest, ncvars, dims, nikki, sim_config, train_mp):
    """Load PPE data with memory cleanup"""
    ic_str = 'cic'
    
    # Load PPE parameters
    for ippe, ppe in enumerate(tqdm(ppe_idx, desc='loading params')):
        param_df = pd.read_csv(f"{cl.output_dir}{nikki}/{sim_config}/{train_mp}/{ppe}/params.csv")
        if ippe == 0:  # First iteration
            header = param_df.columns
            param_names = np.array([a.strip() for a in header])
            dims['nparams'] = len(header)
            
            # Initialize parameter arrays
            ncvars['param_names'] = {
                'dims': ('nparams',),
                'data': param_names,
                'units': ''
            }
            ncvars['params_PPE'] = {
                'dims': ('nppe','nparams',),
                'units': '',
                'data': np.zeros((len(ppe_idx), dims['nparams']))
            }
        
        # Store parameters
        ncvars['params_PPE']['data'][ippe, :] = np.array(param_df)
        
        # Store initial conditions
        for var_vn in vars_vn:
            if ncvars[var_vn + '_PPE']['data'] is None:
                ncvars[var_vn + '_PPE']['data'] = np.zeros((len(ppe_idx),))
            ncvars[var_vn + '_PPE']['data'][ippe] = nc_dict[ic_str][train_mp][ppe][var_vn]
    
    # Load summary variables
    for ivar in var_interest:
        ncvars['ppe_' + ivar]['data'] = np.array([nc_dict[ic_str][train_mp][ppe][ivar]['value'] for ppe in ppe_idx])

    return ncvars, dims

def process_target_data(nc_dict, vars_vn, vars_strs, var_interest, ncvars, dims, target_mp):
    """Process target data with memory cleanup"""
    ncase = dims['ncase']
    
    for ivar in var_interest:
        if 'nlevel' in ncvars['tgt_' + ivar]['dims']:
            if 'nlevel' not in dims:
                dims['nlevel'] = nc_dict[ic_str][target_mp][ivar]['value'].shape[0]
            ncvars['tgt_' + ivar]['data'] = np.zeros((ncase, dims['nlevel']))
        else:
            ncvars['tgt_' + ivar]['data'] = np.zeros(ncase)
        icase = 0
        for combo in itertools.product(*vars_strs):
            ic_str = "".join(combo)
            # print(ivar, nc_dict[ic_str][target_mp][ivar]['value'])
            ncvars['tgt_' + ivar]['data'][icase] = nc_dict[ic_str][target_mp][ivar]['value']
            icase += 1

    for var_vn in vars_vn:
        ncvars['case_' + var_vn]['data'] = np.zeros(ncase)
    for icase, combo in enumerate(itertools.product(*vars_strs)):
        ic_str = "".join(combo)
        for i_init, var_vn in enumerate(vars_vn):
            ncvars['case_' + var_vn]['data'][icase] = nc_dict[ic_str][target_mp][var_vn]
    
def write_netcdf(nc_file, ncvars, dims, global_attrs):
    """Write netCDF data with memory cleanup"""
    # Create dimensions
    for dim_name, dim in dims.items():
        if dim_name not in nc_file.dimensions:
            nc_file.createDimension(dim_name, dim)

    # Save global attributes
    for attr_name, attr_value in global_attrs.items():
        if isinstance(attr_value, list):
            nc_file.setncattr(attr_name, np.array(attr_value))
        else:
            nc_file.setncattr(attr_name, attr_value)

    # Write variables
    outnc_dict = {}
    for var_name, var in ncvars.items():
        if var_name not in nc_file.variables:
            if all(isinstance(item, str) for item in var['data']):
                outnc_dict[var_name] = nc_file.createVariable(var_name, str, var['dims'])
            else:
                outnc_dict[var_name] = nc_file.createVariable(var_name, np.float64, var['dims'])
            
            if 'data' in var and var['data'] is not None:
                outnc_dict[var_name][:] = var['data']
            
            try:
                outnc_dict[var_name].units = var['units']
            except:
                outnc_dict[var_name].units = ""

if __name__ == "__main__":
    main() 
