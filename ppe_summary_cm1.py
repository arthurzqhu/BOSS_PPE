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
import re
import socket
import pandas as pd
import netCDF4 as nc
import dask
from dask.distributed import Client, progress

import sys
from tqdm.auto import tqdm

# Disable the background monitor thread entirely
tqdm.monitor_interval = 0  # <- important

# Only show bars on a TTY (prevents odd behaviour in batch)
tqdm_disable = not sys.stderr.isatty()

# Perlmutter/Lustre fix: Disable HDF5 file locking
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def main():
    """Main function to process PPE data with memory efficiency"""
    # Configuration
    l_parallel = False
    l_testing = True
    n_test = 3
    nikki = ''
    target_nikki = 'target'
    # sim_config = 'NCE_dycoms_aphase_lhs'
    # target_sim_config = 'NCE_dycoms'
    # steady_state_hrs = 2
    # sim_config = 'fullmp_joint_dycoms_eo6_r1_lhs'
    # target_sim_config = 'fullmp_dycoms_t_onset_1e-4'
    sim_config = 'fullmp_joint_rico_eo6_r1_lhs'
    target_sim_config = 'fullmp_rico_t_onset_1e-4'
    steady_state_hrs = 4
    # sim_config = 'fullmp_joint_rico_eo4_lhs'
    # target_sim_config = 'fullmp_rico_t_onset_1e-4'
    lwp_threshold = 0.02
    print('lwp_threshold:', lwp_threshold)
    print('PPE directory:', sim_config)
    print('target directory:', target_sim_config)
    
    if not os.path.exists(lp.nc_dir):
        os.makedirs(lp.nc_dir)
    l_cic = True
    
    n_init = 1
    target_mp = 'BIN-TAU'
    train_mp = 'SLC-BOSS'
    mconfigs = os.listdir(cl.output_dir + nikki)
    vars_strs, vars_vn = lp.get_dics(cl.output_dir, target_nikki, target_sim_config, n_init)
    var_interest = []
    var_interest += [
                     'M0_dmpath_ss_mean', 'M3_dmpath_ss_mean', 'M4_dmpath_ss_mean', 'M6_dmpath_ss_mean', 
                     'M0_path_ss_std', 'M3_path_ss_std', 'M4_path_ss_std', 'M6_path_ss_std', 
                     'M6_ss_99th_prctl', 'meanD_dm_03_ss_mean',
                     'prate_dm_ss_mean', 'prate_ss_std', 'v_precip_onset', 'precip_max_dm',
                    ] # domain-mean path
    # var_interest += ['M0_path_ss_mean', 'M3_path_ss_mean', 'M4_path_ss_mean', 'M6_path_ss_mean',] # domain-mean path
    # var_interest += ['M0_per5lvl_ss_mean', 'M3_per5lvl_ss_mean', 'M4_per5lvl_ss_mean', 'M6_per5lvl_ss_mean']
    # var_interest += ['sfM0_per5lvl_ss_mean', 'sfM3_per5lvl_ss_mean', 'sfM4_per5lvl_ss_mean', 'sfM6_per5lvl_ss_mean'] # domain-mean fluxes
    var_interest += [
            # 'sfM0_per5lvl', 'sfM3_per5lvl', 'sfM4_per5lvl', 'sfM6_per5lvl',
            'sfM0_dm_10m_ss_mean',  'sfM3_dm_10m_ss_mean',  'sfM4_dm_10m_ss_mean',  'sfM6_dm_10m_ss_mean',
            'sfM0_dm_100m_ss_mean', 'sfM3_dm_100m_ss_mean', 'sfM4_dm_100m_ss_mean', 'sfM6_dm_100m_ss_mean',
            'sfM0_dm_250m_ss_mean', 'sfM3_dm_250m_ss_mean', 'sfM4_dm_250m_ss_mean', 'sfM6_dm_250m_ss_mean',
            # 'sfM0_dm_500m_ss_mean', 'sfM3_dm_500m_ss_mean', 'sfM4_dm_500m_ss_mean', 'sfM6_dm_500m_ss_mean',
]

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

    if l_testing:
        ppe_idx = ppe_idx[:n_test]
        if n_test < len(vars_strs[0]):
            vars_strs = [vars_strs[0][:n_test]]
        else:
            vars_strs = [vars_strs[0]]

    if l_testing:
        nc_filename = f"{lp.nc_dir}{sim_config}_momval_lwp{lwp_threshold}_test_N{n_test}.nc"
    else:
        nc_filename = f"{lp.nc_dir}{sim_config}_momval_lwp{lwp_threshold}_N{len(ppe_idx)}.nc"

    if l_parallel:
        # On compute nodes, use processes (True) for library isolation (NetCDF is often not thread-safe).
        # We limit workers to avoid memory/IO pressure on the Lustre filesystem.
        dask_scratch = os.path.join(os.environ.get('PSCRATCH', '/tmp'), 'dask-scratch-space')
        
        # Using a reasonable number of workers (e.g., 16) even on a full compute node 
        # is usually safer and faster for I/O bound NetCDF tasks.
        client = Client(n_workers=32, threads_per_worker=1, processes=True, local_directory=dask_scratch)
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print(f"Using 16 Processes. Scratch: {dask_scratch}")

        tasks = []
        for ippe in ppe_idx:
            # We pass None as nc_dict so it returns a new one for each member
            task = dask.delayed(cl.load_cm1)(
                file_info, var_interest, None, True, 
                ss_hrs=steady_state_hrs, ippe=ippe, lwp_threshold=lwp_threshold
            )
            tasks.append(task)

        print("Computing PPE data in parallel...")
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)

        for r in tqdm(results, desc='merging PPE results'):
            cl.deep_merge(nc_dict, r)
    else:
        for ippe in tqdm(ppe_idx, desc='loading BOSS data'):
            cl.load_cm1(file_info, var_interest, nc_dict, True, ss_hrs=steady_state_hrs, ippe=ippe, lwp_threshold=lwp_threshold)

    # Load target data
    print("\nLoading target data...")
    if l_parallel:
        tasks = []
        for initcond_combo in itertools.product(*vars_strs):
            # Create a separate file_info for each combo to avoid mutation issues
            finfo_target = file_info.copy()
            finfo_target.update({
                'sim_config': target_sim_config,
                'vars_str': list(initcond_combo),
                'date': target_nikki,
                'mp_config': target_mp
            })
            task = dask.delayed(cl.load_cm1)(
                finfo_target, var_interest, None, False, 
                ss_hrs=steady_state_hrs, lwp_threshold=lwp_threshold
            )
            tasks.append(task)

        print("Computing target data in parallel...")
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)
        
        for r in tqdm(results, desc='merging target results'):
            cl.deep_merge(nc_dict, r)
        
        # Shutdown client
        client.close()
    else:
        for initcond_combo in tqdm(itertools.product(*vars_strs), desc='loading BIN data'):
            file_info.update({'sim_config': target_sim_config,
                            'vars_str': list(initcond_combo),
                            'date': target_nikki,
                            'mp_config': target_mp})
            cl.load_cm1(file_info, var_interest, nc_dict, False, ss_hrs=steady_state_hrs, lwp_threshold=lwp_threshold)

#     plot_dir = f"plots/{nikki}/{sim_config}/"
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)

#     fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True)
#     axs = axs.flatten()
#     na = []
#     for initcond_combo in itertools.product(*vars_strs):
#         ic_str = "".join(initcond_combo)
#         na.append(nc_dict[ic_str]['BIN-TAU']['na'])

#     na = np.array(na)

#     for ivar, var_name in enumerate(var_interest[:5]):
#         tgt_data = []
#         train_data = []
#         na_train = []
#         for initcond_combo in itertools.product(*vars_strs):
#             ic_str = "".join(initcond_combo)
#             tgt_data.append(nc_dict[ic_str]['BIN-TAU'][var_name]['value'])
#         for ippe in ppe_idx:
#             ippe = int(ippe)
#             train_data.append(nc_dict['cic']['SLC-BOSS'][ippe][var_name]['value'])
#             na_train.append(nc_dict['cic']['SLC-BOSS'][ippe]['na'])
#         tgt_data = np.array(tgt_data)
#         train_data = np.array(train_data)
#         na_train = np.array(na_train)
#         axs[ivar].plot(na, tgt_data, label=ic_str, linewidth=2, marker='o')
#         axs[ivar].scatter(na_train, train_data, label=ic_str, s=5, color='tab:orange', alpha=0.5)
#         axs[ivar].set_title(cl.output_var_set[var_name]['longname'])
#         axs[ivar].set_yscale('log')

#     plt.savefig(f"{plot_dir}{sim_config}_dm_path.png")

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
        elif 'prate' in ivar:
            global_attrs['thresholds_eff0'].append(1e-4)
        else:
            global_attrs['thresholds_eff0'].append(np.nanpercentile(value_greater_0, 10))
    
    print("Thresholds:", global_attrs['thresholds_eff0'])
    
    # Write netCDF file
    print("\nWriting netCDF file...")

    # Check if file exists and handle overwrite
    if os.path.exists(nc_filename):
        print(f"\nFile '{nc_filename}' already exists.")
        # user_choice = input("Do you want to replace it (r) or keep both (k)? [r/k]: ").strip().lower()
        # if user_choice == 'k':
        base, ext = os.path.splitext(nc_filename)
        suffix = 1
        new_filename = f"{base}_copy{suffix}{ext}"
        while os.path.exists(new_filename):
            suffix += 1
            new_filename = f"{base}_copy{suffix}{ext}"
        nc_filename = new_filename
        print(f"Saving as '{nc_filename}' instead.")
        # elif user_choice == 'r':
        #     try:
        #         os.remove(nc_filename)
        #         print(f"Removed existing file '{nc_filename}'.")
        #     except Exception as e:
        #         print(f"Could not remove file '{nc_filename}': {e}")
        #         print("Exiting without saving.")
        #         try:
        #             nc_file.close()
        #         except Exception:
        #             pass
        # else:
        #     print("Invalid input. Exiting without saving.")
        #     try:
        #         nc_file.close()
        #     except Exception:
        #         pass

    nc_file = nc.Dataset(nc_filename, 'w', format='NETCDF4')
    write_netcdf(nc_file, ncvars, dims, global_attrs)
    nc_file.close()
    
    # Clear all large variables
    del ncvars, nc_file, dims, global_attrs, nc_dict
    
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

        # assuming it's a time series var if 'last' is not in the name
        if 'ss' not in ivar and 'max' not in ivar and 'onset' not in ivar:
            ncvars['ppe_' + ivar]['dims'] += ('ntime',)
            ncvars['tgt_' + ivar]['dims'] += ('ntime',)

        if 'lvl' in ivar:
            ncvars['ppe_' + ivar]['dims'] += ('nlevel',)
            ncvars['tgt_' + ivar]['dims'] += ('nlevel',)

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
            # nparams = nc_dict['n_param_nevp'] + nc_dict['n_param_condevp'] + \
            #           nc_dict['n_param_coal'] + nc_dict['n_param_sed']
            if param_df.shape[0] > 5:
                nparams = param_df.shape[0]
                is_vertical = True
            else:
                nparams = param_df.shape[1]
                is_vertical = False

            dims['nparams'] = nparams

            # is_vertical = nparams == param_df.shape[0]

            if is_vertical:
                param_names = param_df.iloc[:, 0].to_numpy()
            else:
                param_names = np.array([a.strip() for a in param_df.columns])

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
        if is_vertical:
            ncvars['params_PPE']['data'][ippe, :] = np.array(param_df.iloc[:, 1].to_numpy())
        else:
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
    
    combo = list(itertools.product(*vars_strs))[0]
    ic_str = "".join(combo)

    for ivar in var_interest:
        tgt_dims = (ncase,)
        if 'ntime' in ncvars['tgt_' + ivar]['dims']:
            dims['ntime'] = nc_dict[ic_str][target_mp][ivar]['value'].shape[0]
            tgt_dims += (dims['ntime'],)
        if 'nlevel' in ncvars['tgt_' + ivar]['dims']:
            dims['nlevel'] = nc_dict[ic_str][target_mp][ivar]['value'].shape[-1]
            tgt_dims += (dims['nlevel'],)

        ncvars['tgt_' + ivar]['data'] = np.zeros(tgt_dims)

        icase = 0
        for combo in itertools.product(*vars_strs):
            ic_str = "".join(combo)
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
