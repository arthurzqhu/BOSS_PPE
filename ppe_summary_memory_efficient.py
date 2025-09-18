"""
PPE summary processing script
Note that some hard-coded variables are used in this script.
"""

import load_ppe_fun as lp
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
import re
import netCDF4 as nc
import itertools
import util_fun as uf


def main():
    """Main function to process PPE data with memory efficiency"""
    # Configuration
    vnum = '0001'
    nikki = '2025-09-16'
    target_nikki = 'target'
    momxy = '46'
    
    ppe_config = 'condevap_ppe'
    target_simconfig = 'condevap'
    
    if not os.path.exists(lp.nc_dir):
        os.makedirs(lp.nc_dir)
    l_cic = True
    
    n_init = 2
    target_name = 'BIN_TAU' + momxy
    
    vars_strs, vars_vn = lp.get_dics(lp.output_dir, target_nikki, target_simconfig, n_init)
    mps, _ = lp.get_mps(lp.output_dir, nikki, ppe_config, l_cic, vars_strs, momxy)
    mps = lp.sort_strings_by_number(mps)
    

    # snapshot_var_idx = [4, 86, 119, 120] # LWP, LNP, Mx_path, My_path
    # snapshot_var_idx = [86] # LNP, Mx_path, My_path
    snapshot_var_idx = [115, 116, 117, 118] # M3, M0, Mx, My
    # snapshot_var_idx = [116, 117, 118] # M0, Mx, My
    # snapshot_var_idx = [135, 136, 137, 138]
    # snapshot_var_idx = []
    # summary_var_idx = [144, 145, 146, 147]
    summary_var_idx = []
    # summary_var_idx = [95, 107, 121, 122]
    # snapshot_var_idx = [21, 97, 98, 99] # dm3_sed, dm0_sed, dm6_sed, dm9_sed
    # snapshot_var_idx = [21, 97, 143, 98] # dm3_sed, dm0_sed, dm4_sed, dm6_sed
    # summary_var_idx = [131, 132, 133, 134]
    # snapshot_var_idx = [135, 136, 137, 138] # V_M0, V_M3, V_Mx, V_My
    # snapshot_var_idx = [87, 88, 89] # dm0_coal, dmx_coal, dmy_coal
    # summary_var_idx = [95, 107, 121, 122, 123, 124, 125, 126]
    # summary_var_idx = [108, 95, 107, 121, 122]
    
    print("Memory usage at start:")
    uf.detailed_memory_analysis()
    
    # Process data
    var_interest = snapshot_var_idx + summary_var_idx
    
    file_info = {
        'dir': lp.output_dir, 
        'date': nikki, 
        'version_number': vnum,
        'vars_vn': vars_vn
    }
    
    # Load PPE data
    print("\nLoading PPE data...")
    nc_dict = {}
    nc_dict = load_ppe_data(nc_dict, mps, ppe_config, file_info, var_interest)

    # Load target data
    print("\nLoading target data...")
    nc_dict = load_target_data(nc_dict, vars_strs, file_info, var_interest, target_simconfig, target_nikki, target_name)
    
    print("\nMemory usage after loading data:")
    uf.detailed_memory_analysis()
    
    # Setup dimensions and variables
    diag_dt = max(1., nc_dict['time'][1] - nc_dict['time'][0])
    diag_dz = 50
    
    dt = nc_dict['time'][1] - nc_dict['time'][0]
    dz = nc_dict['z'][1] - nc_dict['z'][0]
    
    diag_ts = np.array([x for x in nc_dict['time'] if x % diag_dt == 0])
    diag_zs = np.array([x for x in nc_dict['z'][1:] if x % diag_dz == 0])
    # diag_zs = np.arange(1000, 3001, diag_dz)
    
    ncase = 1
    ncase_respective = [len(i) for i in vars_strs]
    for i in ncase_respective:
        ncase *= i
    
    dims = {
        'ntime': len(diag_ts),
        'nz': len(diag_zs),
        'scalar_var': 1,
        'ncase': ncase,
        'nppe': len(mps),
    }
    
    ncvars = {
        'diag_ts': {
            'data': diag_ts,
            'dims': ('ntime',),
            'units': 's',
        },
        'diag_zs': {
            'data': diag_zs,
            'dims': ('nz',),
            'units': 'm',
        },
    }
    
    global_attrs = {
        'description': 'PPE data for ' + ppe_config,
        'date_simulated': nikki,
    }
    
    # Create variable structure
    ncvars = create_nc_variables_structure(ncvars, nc_dict, vars_vn, snapshot_var_idx, summary_var_idx)
    
    print("\nProcessing target data...")
    process_target_data(nc_dict, vars_vn, vars_strs, snapshot_var_idx, summary_var_idx, 
                        ncvars, diag_ts, diag_zs, dims, target_name)
                        
    # Processing PPE data
    print("\nProcessing PPE data...")
    ncvars, dims = process_ppe_data(nc_dict, mps, vars_vn, snapshot_var_idx, summary_var_idx, 
                                    ncvars, diag_ts, diag_zs, dims, nikki, ppe_config)
    
    print("\nMemory usage after processing data:")
    uf.detailed_memory_analysis()
    
    # Set up global attributes
    global_attrs['thresholds_eff0'] = []
    var_constraints = []
    for ivar in summary_var_idx + snapshot_var_idx:
        var_constraints.append(lp.indvar_ename_set[ivar])   
    global_attrs['var_constraints'] = np.array(var_constraints)
    global_attrs['init_var'] = np.array(vars_vn)
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
    for isumm in summary_var_idx + snapshot_var_idx:
        var_ename = lp.indvar_ename_set[isumm]
        value_greater_0 = ncvars['ppe_' + var_ename]['data'][ncvars['ppe_' + var_ename]['data'] > 0]
        if 'V_M' in var_ename:
            global_attrs['thresholds_eff0'].append(0.1)
        else:
            global_attrs['thresholds_eff0'].append(np.nanpercentile(value_greater_0, 10))
    
    print("Thresholds:", global_attrs['thresholds_eff0'])
    
    # Write netCDF file
    print("\nWriting netCDF file...")

    # Check if file exists and handle overwrite
    nc_filename = f"{lp.nc_dir}{ppe_config}_{momxy}_momvals_N{len(mps)}_dt{diag_dt}.nc"
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

def load_ppe_data(nc_dict, mps, ppe_config, file_info, var_interest):
    """Process PPE data with memory cleanup"""
    file_info.update({'sim_config': ppe_config})
    data_range = {}
    
    for mp in tqdm(mps, desc='loading PPEs'):
        file_info['mp_config'] = mp
        _ = lp.load_KiD(file_info, var_interest, nc_dict, data_range, 
                                continuous_ic=True, set_OOB_as_NaN=False, set_NaN_to_0=True)
    
    return nc_dict

def load_target_data(nc_dict, vars_strs, file_info, var_interest, target_simconfig, target_nikki, target_name):
    """Process target data with memory cleanup"""
    data_range = {}
    file_info.update({'sim_config': target_simconfig})
    # Generate all combinations
    all_combos = list(itertools.product(*vars_strs))
    
    for combo in tqdm(all_combos, desc='loading target'):
        file_info.update({
            'sim_config': target_simconfig,
            'vars_str': list(combo),
            'date': target_nikki,
            'mp_config': target_name
        })
        _ = lp.load_KiD(file_info, var_interest, nc_dict, data_range, 
                                                        continuous_ic=False, set_OOB_as_NaN=False, set_NaN_to_0=True)
    
    return nc_dict

def create_nc_variables_structure(ncvars, nc_dict, vars_vn, snapshot_var_idx, summary_var_idx):
    """Create the netCDF variable structure without loading data"""
    # Initialize PPE variables
    for var_vn in vars_vn:
        ncvars[var_vn + '_PPE'] = {
            'dims': ('nppe',),
            'units': nc_dict[var_vn + '_units'],
            'data': None  # Will be filled later
        }
    
    # Initialize summary variables
    for isumm in summary_var_idx:
        var_ename = lp.indvar_ename_set[isumm]
        var_units = lp.indvar_units_set[isumm]
        if var_units != '':
            var_units = var_units[2:-1]
        
        ncvars['ppe_' + var_ename] = {
            'dims': ('nppe',),
            'units': var_units,
            'data': None
        }
        
        ncvars['tgt_' + var_ename] = {
            'dims': ('ncase',),
            'units': var_units,
            'data': None
        }
    
    mps = list(nc_dict['cic'].keys())
    nz = len(nc_dict['z'])
    nt = len(nc_dict['time'])
    
    # Initialize snapshot variables
    for isnap in snapshot_var_idx:
        var_ename = lp.indvar_ename_set[isnap]
        var_units = lp.indvar_units_set[isnap]
        if var_units != '':
            var_units = var_units[2:-1]
        
        vshape = nc_dict['cic'][mps[0]][var_ename].shape
        if len(vshape) == 1 and vshape[0] == nt:
            ncvars['ppe_' + var_ename] = {
                'dims': ('nppe', 'ntime',),
                'units': var_units,
                'data': None
            }
            ncvars['tgt_' + var_ename] = {
                'dims': ('ncase', 'ntime',),
                'units': var_units,
                'data': None
            }
        elif vshape[0] == nz and vshape[1] == nt:
            ncvars['ppe_' + var_ename] = {
                'dims': ('nppe', 'nz', 'ntime',),
                'units': var_units,
                'data': None
            }
            ncvars['tgt_' + var_ename] = {
                'dims': ('ncase', 'nz', 'ntime',),
                'units': var_units,
                'data': None
            }
        else:
            raise ValueError(f"Variable {var_ename} has unexpected shape {vshape}")
    
    # Initialize case variables
    for var_vn in vars_vn:
        ncvars['case_' + var_vn] = {
            'dims': ('ncase',),
            'units': nc_dict[var_vn + '_units'],
            'data': None
        }
    
    return ncvars

def process_ppe_data(nc_dict, mps, vars_vn, snapshot_var_idx, summary_var_idx, 
                   ncvars, diag_ts, diag_zs, dims, nikki, ppe_config):
    """Load PPE data with memory cleanup"""
    ic_str = 'cic'
    
    dt = nc_dict['time'][1] - nc_dict['time'][0]
    dz = nc_dict['z'][1] - nc_dict['z'][0]
    nz = len(nc_dict['z'])
    
    # Load PPE parameters
    for imp, mp in enumerate(tqdm(mps, desc='loading params')):
        param_df = pd.read_csv(f"{lp.output_dir}{nikki}/{ppe_config}/{mp}/params.csv")
        if imp == 0:  # First iteration
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
                'data': np.zeros((len(mps), dims['nparams']))
            }
        
        # Store parameters
        ncvars['params_PPE']['data'][imp, :] = np.array(param_df)
        
        # Store initial conditions
        for var_vn in vars_vn:
            if ncvars[var_vn + '_PPE']['data'] is None:
                ncvars[var_vn + '_PPE']['data'] = np.zeros((len(mps),))
            ncvars[var_vn + '_PPE']['data'][imp] = nc_dict[ic_str][mp][var_vn]
    
    # Load summary variables
    for isumm in summary_var_idx:
        var_ename = lp.indvar_ename_set[isumm]
        
        ncvars['ppe_' + var_ename]['data'] = np.array([nc_dict[ic_str][mp][var_ename] for mp in mps])

    # Load snapshot variables
    for isnap in snapshot_var_idx:
        var_ename = lp.indvar_ename_set[isnap]
        
        # Determine dimensions
        vshape = nc_dict[ic_str][mps[0]][var_ename].shape
        if vshape[0] == nz:
            ncvars['ppe_' + var_ename]['data'] = np.zeros((len(mps), len(diag_zs), len(diag_ts)))
        else:
            ncvars['ppe_' + var_ename]['data'] = np.zeros((len(mps), len(diag_ts)))
        
        # Load data
        for imp, mp in enumerate(tqdm(mps, desc=f'processing PPEs - {var_ename}')):
            for idgt, diag_t in enumerate(diag_ts):
                it = int(diag_t / dt) - 1
                if nc_dict[ic_str][mp][var_ename].shape[0] == nz:
                    for idgz, diag_z in enumerate(diag_zs):
                        iz = int(diag_z / dz) - 1
                        ncvars['ppe_' + var_ename]['data'][imp, idgz, idgt] = nc_dict[ic_str][mp][var_ename][iz, it]
                else:
                    ncvars['ppe_' + var_ename]['data'][imp, idgt] = nc_dict[ic_str][mp][var_ename][it]
    
    return ncvars, dims

def process_target_data(nc_dict, vars_vn, vars_strs, snapshot_var_idx, summary_var_idx, 
                        ncvars, diag_ts, diag_zs, dims, target_name):
    """Process target data with memory cleanup"""
    dt = nc_dict['time'][1] - nc_dict['time'][0]
    dz = nc_dict['z'][1] - nc_dict['z'][0]
    nz = len(nc_dict['z'])
    ncase = dims['ncase']
    
    for isumm in summary_var_idx:
        var_ename = lp.indvar_ename_set[isumm]
        var_units = lp.indvar_units_set[isumm]
        if var_units != '':
            var_units = var_units[2:-1]
        ncvars['tgt_' + var_ename]['data'] = np.zeros(ncase)
        icase = 0
        for combo in itertools.product(*vars_strs):
            ic_str = "".join(combo)
            ncvars['tgt_' + var_ename]['data'][icase] = nc_dict[ic_str][target_name][var_ename]
            icase += 1

    for isnap in snapshot_var_idx:
        var_ename = lp.indvar_ename_set[isnap]
        var_units = lp.indvar_units_set[isnap]
        if var_units != '':
            var_units = var_units[2:-1]
        # ic_str inherits from above
        ic_str = "".join(list(itertools.product(*vars_strs))[0])
        if nc_dict[ic_str][target_name][var_ename].shape[0] == nz:
            ncvars['tgt_' + var_ename]['data'] = np.zeros((ncase, len(diag_zs), len(diag_ts)))
        else:
            ncvars['tgt_' + var_ename]['data'] = np.zeros((ncase, len(diag_ts)))

        icase = 0
        for combo in itertools.product(*vars_strs):
            ic_str = "".join(combo)
            for idgt, diag_t in enumerate(diag_ts):
                it = int(diag_t / dt) - 1
                if nc_dict[ic_str][target_name][var_ename].shape[0] == nz:
                    for idgz, diag_z in enumerate(diag_zs):
                        iz = int(diag_z / dz) - 1
                        ncvars['tgt_' + var_ename]['data'][icase, idgz, idgt] = nc_dict[ic_str][target_name][var_ename][iz, it]
                else:
                    ncvars['tgt_' + var_ename]['data'][icase, idgt] = nc_dict[ic_str][target_name][var_ename][it]
            icase += 1

    for var_vn in vars_vn:
        ncvars['case_' + var_vn]['data'] = np.zeros(ncase)
    for icase, combo in enumerate(itertools.product(*vars_strs)):
        for i_init, var_vn in enumerate(vars_vn):
            ncvars['case_' + var_vn]['data'][icase] = float(re.search(r'[+-]?\d*\.?\d+', combo[i_init]).group())
    
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
