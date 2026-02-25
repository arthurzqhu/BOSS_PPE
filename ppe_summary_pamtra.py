"""
PPE summary processing script for Pamtra
Mimics ppe_summary_cm1.py structure.
"""

import pamtra_load_util as pl
import load_ppe_fun as lp
import numpy as np
import os
import itertools
import netCDF4 as nc
import dask
from dask.distributed import Client, progress
import sys
import pandas as pd
import warnings
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Disable the background monitor thread entirely
tqdm.monitor_interval = 0
tqdm_disable = not sys.stderr.isatty()
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def main():
    """Main function to process PPE data with memory efficiency"""
    # Configuration
    l_parallel = True
    l_testing = not l_parallel
    n_test = 3
    nikki = ''
    target_nikki = 'target'

    target_mp = 'BIN-TAU'
    train_mp = 'SLC-BOSS'

    # Example config (user can modify)
    sim_config = 'fullmp_joint_dycoms_narrower_prior_lhs'
    target_sim_config = 'fullmp_dycoms_taudist'
    steady_state_hrs = 2
    
    ze_threshold = -60.0
    print('ze_threshold:', ze_threshold)
    print('PPE directory:', sim_config)
    print('target directory:', target_sim_config)
    
    if not os.path.exists(lp.nc_dir):
        os.makedirs(lp.nc_dir)
    
    n_init = 1

    # Vars strings setup (reusing cm1 setup or similar)
    # Note: This relies on lp.get_dics pointing to correct dirs.
    # If using Pamtra output, ensure lp functions work or override.
    # Assuming lp.get_dics works for structure
    # If load_ppe_fun not compatible with pamtra structure, we might need manual list.
    vars_strs, vars_vn = lp.get_dics(pl.output_dir, target_nikki, target_sim_config, n_init)
    var_interest = [
        'Ze_dmcolmax_ss_mean', 'Ze_colmax_ss_std',
        'Ze_dm_surface_ss_mean', 'Ze_surface_ss_std',
        'PIA_dm_ss_mean', 'PIA_ss_std',
        'Specific_Attenuation_dmcolmax_ss_mean', 'Specific_Attenuation_colmax_ss_std',
        'Radar_MeanDopplerVel_dmcolmean_ss_mean', 'Radar_MeanDopplerVel_colmean_ss_std',
        'Radar_SpectrumWidth_dmcolmean_ss_mean', 'Radar_SpectrumWidth_colmean_ss_std',
        'tb_dm_ss_mean', 'tb_ss_std'
    ]

    # Process data
    file_info = {'dir': pl.output_dir, 
                'date': nikki,
                'vars_vn': vars_vn}

    if 'nc_dict' not in globals():
        nc_dict = {}

    # Load PPE data
    print("\nLoading PPE data...")
    file_info.update({'sim_config': sim_config,
                    'date': nikki,
                    'mp_config': train_mp})
    
    ppe_idx = pl.get_ppe_idx(file_info)
    ppe_idx = [int(i) for i in ppe_idx]

    if l_testing:
        ppe_idx = ppe_idx[:n_test]
        if n_test < len(vars_strs[0]):
            vars_strs = [vars_strs[0][:n_test]]
        else:
            vars_strs = [vars_strs[0]]

    nc_filename = f"{pl.nc_dir}{sim_config}_pamtra_bin_ze{ze_threshold}_N{len(ppe_idx)}.nc"

    if l_parallel:
        dask_scratch = os.path.join(os.environ.get('PSCRATCH', '/tmp'), 'dask-scratch-space')
        client = Client(n_workers=16, threads_per_worker=1, processes=True, local_directory=dask_scratch)
        tasks = []
        for ippe in ppe_idx:
            task = dask.delayed(pl.load_pamtra)(
                file_info, var_interest, None, True, 
                ss_hrs=steady_state_hrs, ippe=ippe, ze_threshold=ze_threshold
            )
            tasks.append(task)
        
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)
        
        for r in tqdm(results, desc='merging PPE results'):
            pl.deep_merge(nc_dict, r)
    else:
        for ippe in tqdm(ppe_idx, desc='loading BOSS data'):
            pl.load_pamtra(file_info, var_interest, nc_dict, True, ss_hrs=steady_state_hrs, ippe=ippe, ze_threshold=ze_threshold)

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
            # Use load_pamtra for target data
            task = dask.delayed(pl.load_pamtra)(
                finfo_target, var_interest, None, False, 
                ss_hrs=steady_state_hrs, ze_threshold=ze_threshold
            )
            tasks.append(task)
            
        print("Computing target data in parallel...")
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)
        
        for r in tqdm(results, desc='merging target results'):
            pl.deep_merge(nc_dict, r)
        
        # Shutdown client
        # client.close() # Keep open if needed or close here
    else:
        for initcond_combo in tqdm(itertools.product(*vars_strs), desc='loading Target data'):
            file_info.update({'sim_config': target_sim_config,
                            'vars_str': list(initcond_combo),
                            'date': target_nikki,
                            'mp_config': target_mp})
            pl.load_pamtra(file_info, var_interest, nc_dict, False, ss_hrs=steady_state_hrs, ze_threshold=ze_threshold)
    
    ncase = 1
    # ncase logic depends on valid vars_strs
    if isinstance(vars_strs, list) and len(vars_strs) > 0:
        ncase_respective = [len(i) for i in vars_strs]
        for i in ncase_respective: ncase *= i
    else:
        ncase = 1
    
    dims = {
        'scalar_var': 1,
        'ncase': ncase,
        'nppe': len(ppe_idx),
    }

    global_attrs = {
        'description': 'PPE Pamtra data for ' + sim_config,
        'ze_threshold': ze_threshold
    }

    # Set up global attributes (copied from cm1)
    global_attrs['thresholds_eff0'] = []
    var_constraints = []
    for ivar in var_interest:
        var_constraints.append(ivar)   
    global_attrs['var_constraints'] = np.array(var_constraints)
    global_attrs['init_var'] = np.array(vars_vn)
    
    for var_vn in vars_vn:
        global_attrs[var_vn + '_units'] = nc_dict.get(var_vn + '_units', '')
        
    global_attrs['n_init'] = n_init
    global_attrs['n_param_nevp'] = nc_dict.get('n_param_nevp', 4)
    global_attrs['n_param_condevp'] = nc_dict.get('n_param_condevp', 9)
    global_attrs['n_param_coal'] = nc_dict.get('n_param_coal', 12)
    global_attrs['n_param_sed'] = nc_dict.get('n_param_sed', 12)
    global_attrs['is_perturbed_nevp'] = nc_dict.get('is_perturbed_nevp', 0)
    global_attrs['is_perturbed_condevp'] = nc_dict.get('is_perturbed_condevp', 0)
    global_attrs['is_perturbed_coal'] = nc_dict.get('is_perturbed_coal', 1)
    global_attrs['is_perturbed_sed'] = nc_dict.get('is_perturbed_sed', 1)

    ncvars = create_nc_variables_structure(nc_dict, vars_vn, var_interest)
    
    # Process PPE data into ncvars
    ncvars, dims = process_ppe_data(nc_dict, ppe_idx, vars_vn, var_interest, ncvars, dims, nikki, sim_config, train_mp)

    print("\nProcessing target data...")
    process_target_data(nc_dict, vars_vn, vars_strs, var_interest, ncvars, dims, target_sim_config, target_mp)

    # Calculate thresholds
    for ivar in var_interest:
        data = ncvars['ppe_' + ivar]['data']
        valid_data = data[np.isfinite(data)]
        
        if 'Ze_dm_surface_ss_mean' in ivar:
            # Special case for Ze at surface and column max, proxy for rain fall and M6
            global_attrs['thresholds_eff0'].append(8e-5) # equivalent to ~1e-4 mm/hr
        elif 'PIA_dm_ss_mean' in ivar:
            global_attrs['thresholds_eff0'].append(0.1)
        elif 'Specific_Attenuation_dmcolmax_ss_mean' in ivar:
            global_attrs['thresholds_eff0'].append(0.1)
        else:
            # Default fallback
            global_attrs['thresholds_eff0'].append(np.percentile(valid_data, 10))
    
    print("Thresholds:", global_attrs['thresholds_eff0'])


    print("\nWriting netCDF file...")
    nc_file = nc.Dataset(nc_filename, 'w', format='NETCDF4')
    write_netcdf(nc_file, ncvars, dims, global_attrs)
    nc_file.close()

    print("\nProcessing complete!")

def create_nc_variables_structure(nc_dict, vars_vn, var_interest):
    ncvars = {}
    for ivar in var_interest:
        # Determine units from load_util
        var_units = ''
        if ivar in pl.output_var_set:
            var_units = pl.output_var_set[ivar]['var_unit']
            
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
    # Initialize case variables and PPE variables for initial conditions
    for var_vn in vars_vn:
        # Case variable (for target/control)
        ncvars['case_' + var_vn] = {
            'dims': ('ncase',),
            'units': nc_dict.get(var_vn + '_units', ''),
            'data': None
        }
        # PPE variable (for ensemble members)
        ncvars[var_vn + '_PPE'] = {
            'dims': ('nppe',),
            'units': nc_dict.get(var_vn + '_units', ''),
            'data': None
        }

    return ncvars

def process_ppe_data(nc_dict, ppe_idx, vars_vn, var_interest, ncvars, dims, nikki, sim_config, train_mp):
    ic_str = 'cic'
    
    # Load PPE parameters
    # Mimicking cm1 logic
    for ippe, ppe in enumerate(tqdm(ppe_idx, desc='loading params')):

        got_params = False
        
        # Check if params loaded from ppe_summary.nc
        params = nc_dict[sim_config][train_mp][ic_str][ppe]['params']
        pnames = nc_dict[sim_config][train_mp][ic_str][ppe]['param_names']
        
        # Only init on first found or first iter
        if ippe == 0:
            nparams = len(pnames)
            dims['nparams'] = nparams
            
            ncvars['param_names'] = {
                'dims': ('nparams',),
                'data': pnames,
                'units': ''
            }
            ncvars['params_PPE'] = {
                'dims': ('nppe','nparams',),
                'units': '',
                'data': np.zeros((len(ppe_idx), nparams))
            }

        # Store
        if 'params_PPE' in ncvars:
            ncvars['params_PPE']['data'][ippe, :] = params
            got_params = True

        if not got_params:
            # Fallback to params.csv
            warnings.warn(f"PPE {ppe} params not found in nc_dict.")
            
            if os.path.exists(param_path):
                 import pandas as pd
                 param_df = pd.read_csv(param_path)
                 if ippe == 0 and 'params_PPE' not in ncvars:  # First iteration init
                     if param_df.shape[0] > 5:
                         nparams = param_df.shape[0]
                         is_vertical = True
                     else:
                         nparams = param_df.shape[1]
                         is_vertical = False
        
                     dims['nparams'] = nparams
        
                     if is_vertical:
                         param_names = param_df.iloc[:, 0].to_numpy().astype(str)
                     else:
                         param_names = np.array([a.strip() for a in param_df.columns]).astype(str)
        
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
                 if 'params_PPE' in ncvars:
                     # Re-determine verticality if needed, or assume consistent
                     # Safe assumption: consistent layout
                     is_vertical = (param_df.shape[0] > 5)
                     if is_vertical:
                         ncvars['params_PPE']['data'][ippe, :] = np.array(param_df.iloc[:, 1].to_numpy())
                     else:
                         ncvars['params_PPE']['data'][ippe, :] = np.array(param_df)
            
        # Store initial conditions
        for var_vn in vars_vn:
            if ncvars[var_vn + '_PPE']['data'] is None:
                ncvars[var_vn + '_PPE']['data'] = np.zeros((len(ppe_idx),))
            # Access via standard path
            # Access via standard path
            ncvars[var_vn + '_PPE']['data'][ippe] = nc_dict[sim_config][train_mp][ic_str][ppe][var_vn]

    # Load Summary Variables
    for ivar in var_interest:
        data_list = []
        for ppe in ppe_idx:
            # Access via sim_config -> mp -> ic_str -> ppe
            val = nc_dict[sim_config][train_mp][ic_str][ppe][ivar]['value']
            data_list.append(val)
        ncvars['ppe_' + ivar]['data'] = np.array(data_list)
        
    return ncvars, dims

def process_target_data(nc_dict, vars_vn, vars_strs, var_interest, ncvars, dims, target_sim_config, target_mp):
    """Process target data with memory cleanup"""
    ncase = dims['ncase']
    
    # We need to construct ic_str same way as load_pamtra does: "".join(vars_str)
    # vars_strs is a list of lists of options. itertools.product gives a tuple.
    
    combo = list(itertools.product(*vars_strs))[0]
    ic_str = "".join(combo)

    # Check existence to avoid crashes
    if target_sim_config not in nc_dict or target_mp not in nc_dict[target_sim_config]:
         raise ValueError("Warning: Target data not found in nc_dict structure.")

    for ivar in var_interest:
        tgt_dims = (ncase,)
        # Check if we need to update expected dims based on data
        # Skipping dynamic dim update for now as we initialized consistently
        
        ncvars['tgt_' + ivar]['data'] = np.zeros(tgt_dims)

        icase = 0
        for combo in itertools.product(*vars_strs):
            ic_str = "".join(combo)
            val = nc_dict[target_sim_config][target_mp][ic_str][ivar]['value']
            ncvars['tgt_' + ivar]['data'][icase] = val
            icase += 1

    for var_vn in vars_vn:
        ncvars['case_' + var_vn]['data'] = np.zeros(ncase)
    
    for icase, combo in enumerate(itertools.product(*vars_strs)):
        ic_str = "".join(combo)
        # We need to retrieve the param values. 
        # load_pamtra stores 'vn' attributes in the same dict lvl as 'value' keys?
        # No, load_pamtra: nc_dict[fsim_config][mp][ic_str][vn] = ds0.getncattr(vn)
        for i_init, var_vn in enumerate(vars_vn):
             # Check for case_var_vn first (from ppe_summary.nc)
             # Note: load_pamtra loads ppe_summary.nc variables into the dict
             val = nc_dict[target_sim_config][target_mp][ic_str].get('case_' + var_vn)
             if val is None:
                 # Check for var_vn (fallback, e.g. from attributes or if ppe_summary.nc used original names)
                 val = nc_dict[target_sim_config][target_mp][ic_str][var_vn]
             ncvars['case_' + var_vn]['data'][icase] = val

def write_netcdf(nc_file, ncvars, dims, global_attrs):
    for dim_name, dim in dims.items():
        if dim_name not in nc_file.dimensions:
            nc_file.createDimension(dim_name, dim)

    for attr_name, attr_value in global_attrs.items():
        if isinstance(attr_value, list):
             nc_file.setncattr(attr_name, np.array(attr_value))
        else:
             nc_file.setncattr(attr_name, attr_value)

    # Write variables
    outnc_dict = {}
    for var_name, var in ncvars.items():
        if var_name not in nc_file.variables:
            # Check if data contains strings
            is_str = False
            if var['data'] is not None:
                if len(var['data']) > 0:
                     flat_data = np.ravel(var['data'])
                     if isinstance(flat_data[0], (str, np.str_)):
                         is_str = True
            
            if is_str:
                outnc_dict[var_name] = nc_file.createVariable(var_name, str, var['dims'])
            else:
                outnc_dict[var_name] = nc_file.createVariable(var_name, np.float64, var['dims'])
            
            if 'data' in var and var['data'] is not None:
                outnc_dict[var_name][:] = var['data']
            
            try:
                outnc_dict[var_name].units = var.get('units', '')
            except:
                outnc_dict[var_name].units = ""

if __name__ == "__main__":
    main()
