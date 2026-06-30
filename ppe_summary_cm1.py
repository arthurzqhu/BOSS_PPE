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
import joblib
import argparse
import sys
from tqdm.auto import tqdm
import platform

hostname = socket.gethostname()
if hostname == "simurgh":
    n_workers = 32
else:
    n_workers = 8

# Disable the background monitor thread entirely
tqdm.monitor_interval = 0  # <- important

# Only show bars on a TTY (prevents odd behaviour in batch)
tqdm_disable = not sys.stderr.isatty()

# Perlmutter/Lustre fix: Disable HDF5 file locking
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Stinginess levels: HI, MID, LOW 
# HI: HI includes all samples (not throwing away any potentially useless samples, therefore stingy)
# LOW: LOW includes only samples within the bound of the most recent PPE (last entry of the ppe_basename)
# MID: MID is somewhere in between, including all samples with parameters within the range of the 
# most recent PPE plus a buffer size.
sting_lvl = 'HI'
buffer_size = 2
l_dry_run = False

def main(camp='dycoms'):
    """Main function to process PPE data"""
    # Configuration
    l_parallel = True
    l_testing = False
    n_test = n_workers

    ppe_basename = [
                    'fullmp_dycoms_offpolicy_r5_bky_lhs',
                    ]

    sim_configs = [bn.replace('dycoms', camp) for bn in ppe_basename]
    # target_sim_config = f'NCE_{camp}_tgt'

    if 'NCE' in ppe_basename[0]:
        target_sim_config = f'NCE_{camp}_tgt'
        l_pert = False
    else:
        target_sim_config = f'fullmp_{camp}_tgt_pert_oldcoalkernel'
        l_pert = True


    if camp == 'rico':
        steady_state_hrs = 5
        min_files = 33
        onset_skip_hrs = 1.5
    elif camp == 'dycoms':
        steady_state_hrs = 2
        min_files = 25
        onset_skip_hrs = 0.5
    else:
        min_files = None
        onset_skip_hrs = 0.0

    lwp_threshold = 0.02 # equiv to 50 g/m2
    print('lwp_threshold:', lwp_threshold)
    print('PPE directory:', sim_configs)
    print('target directory:', target_sim_config)
    
    if not os.path.exists(lp.nc_dir):
        os.makedirs(lp.nc_dir)
    l_cic = True
    
    n_init = 1
    target_mp = 'BIN-TAU'
    train_mp = 'SLC-BOSS'
    datedir = 'ppe'
    target_datedir = 'target'
    # CSV listing parameters actually perturbed (first column = param names);
    # only used when sim_config_str ends with '_select'
    select_params_csv = os.path.join(os.path.dirname(__file__),
                                     'perturbation_design/test62_perturbation_parameters.csv')

    # mconfigs = os.listdir(cl.output_dir + datedir) # this line seems unused and might fail if sim_configs is list, but it uses os.listdir(cl.output_dir + datedir) which is fine if datedir is empty string and output_dir is base. 
    # But wait, cl.output_dir + datedir is just the date dir. 
    # mconfigs is unused in the script? Let's check. 
    # It is unused. I will leave it or comment it out if it causes issues.
    
    vars_strs, vars_vn = lp.get_dics(cl.output_dir, target_datedir, target_sim_config, n_init)
    var_interest = []
    var_interest += [
                     'M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss', 
                     'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss', 
                     # 'KX_dm_ss', 'KY346_dm_ss',
                    ] # domain-mean path

    if 'fullmp' in ppe_basename[0]:
        var_interest += [
                     'meanD_dm_03_ss', 'v_precip_onset', 'precip_frac_ss', 'prate_dm_ss',
                ]

#     var_interest += [
#             # 'sfM0_per5lvl', 'sfM3_per5lvl', 'sfM4_per5lvl', 'sfM6_per5lvl',
#             'sfM0_dm_10m_ss',  'sfM3_dm_10m_ss',  'sfM4_dm_10m_ss',  'sfM6_dm_10m_ss',
#             'sfM0_dm_100m_ss', 'sfM3_dm_100m_ss', 'sfM4_dm_100m_ss', 'sfM6_dm_100m_ss',
#             'sfM0_dm_250m_ss', 'sfM3_dm_250m_ss', 'sfM4_dm_250m_ss', 'sfM6_dm_250m_ss',
#             # 'sfM0_dm_500m_ss', 'sfM3_dm_500m_ss', 'sfM4_dm_500m_ss', 'sfM6_dm_500m_ss',
# ]

    # Process data
    file_info = {'dir': cl.output_dir, 
                'date': datedir,
                'vars_vn': vars_vn}
    
    # Initialize nc_dict default if not present
    if 'nc_dict' not in globals():
        nc_dict = {}

    file_info.update({'sim_config': sim_configs,
                    'date': datedir,
                    'mp_config': train_mp,
                    'min_files': min_files,
                    'onset_skip_hrs': onset_skip_hrs})
    ppe_idx = cl.get_pert_idx(file_info)

    # Filter PPE members based on stinginess level
    ppe_idx = cl.filter_ppe_by_stinginess(
        ppe_idx, sim_configs, sting_lvl, buffer_size,
        cl.output_dir, datedir, train_mp
    )

    print(f"Total PPE members to process: {len(ppe_idx)}")

    if l_dry_run:
        print("\n*** DRY RUN: exiting before loading data ***")
        return

    if l_testing:
        ppe_idx = ppe_idx[:n_test]
        if n_test < len(vars_strs[0]):
            vars_strs = [vars_strs[0][:n_test]]
        else:
            vars_strs = [vars_strs[0]]

    # Update logic for filename if sim_configs is list
    sim_config_str = sim_configs[-1] if isinstance(sim_configs, list) else sim_configs
    # If list, maybe join names or just use first one? User said "each directory... will still count from 1 to however many ensembles".
    # The output filename uses `sim_configs`. I should probably adjust this if it's a list. 
    # But for now, let's just use the first one or a combined name.
    # The user didn't specify filename format change, only internal handling.
    # I'll use the first config name for the file or join them. 
    # Let's use sim_configs[0] for the filename prefix if it's a list.
    
    if l_testing:
        nc_filename = f"{lp.nc_dir}{sim_config_str}_momval_lwp{round(lwp_threshold*1e3)}_test_N{n_test}.nc"
    else:
        nc_filename = f"{lp.nc_dir}{sim_config_str}_momval_lwp{round(lwp_threshold*1e3)}_N{len(ppe_idx)}.nc"

    if l_parallel:
        # On compute nodes, use processes (True) for library isolation (NetCDF is often not thread-safe).
        # We limit workers to avoid memory/IO pressure on the Lustre filesystem.
        dask_scratch = os.path.join(os.environ.get('PSCRATCH', '/home/arthurhu/tmp'), 'dask-scratch-space')

        # Using a reasonable number of workers (e.g., 16) even on a full compute node 
        # is usually safer and faster for I/O bound NetCDF tasks.
        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True, local_directory=dask_scratch)
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print(f"Using {n_workers} Processes. Scratch: {dask_scratch}")

    def _sanitize_sim_payload(payload, expected_sim_config):
        """
        Strip cross-contamination from a per-sim_config cache payload.
        Earlier buggy runs sometimes saved a dict whose top level contained
        OTHER sim_config keys (e.g. {'configA': {...}, 'configB': {...}})
        instead of the expected {'SLC-BOSS': {...}, 'z': ..., ...} layout.
        If we detect that pattern, unwrap to the inner dict matching this
        sim_config (and drop the rest).
        """
        if not isinstance(payload, dict):
            return payload
        nested_self = expected_sim_config in payload and isinstance(payload[expected_sim_config], dict)
        nested_others = any(
            (k != expected_sim_config and isinstance(v, dict)
             and ('SLC-BOSS' in v or 'BIN-TAU' in v))
            for k, v in payload.items()
        )
        if nested_self and (nested_others or 'SLC-BOSS' not in payload):
            inner = payload[expected_sim_config]
            print(f"  [sanitize] unwrapping nested cache for {expected_sim_config}; "
                  f"dropped extra keys: {[k for k in payload if k != expected_sim_config]}")
            return inner
        # Drop any stray sibling sim_config keys without unwrapping
        strays = [k for k, v in payload.items()
                  if isinstance(v, dict) and k != expected_sim_config
                  and ('SLC-BOSS' in v or 'BIN-TAU' in v)]
        if strays:
            print(f"  [sanitize] dropping stray sim_config keys from cache: {strays}")
            for k in strays:
                del payload[k]
        return payload

    for sim_config in sim_configs:
        # NB: one joblib cache per sim_config — using sim_config_str (= sim_configs[-1])
        # here caused every iteration to share the last config's cache file.
        train_jl_path = f"{cl.output_dir}/{datedir}/joblibs/{sim_config}_lwpthres{round(lwp_threshold*1e3)}.joblib"
        if os.path.exists(train_jl_path):
            print(f"Loading {sim_config} data from {train_jl_path}")
            nc_dict[sim_config] = cl.invalidate_stale_vars(
                _sanitize_sim_payload(joblib.load(train_jl_path), sim_config)
            )

            # Load global attributes from the first valid member
            ppe_idx_sim = [ippe for ippe in ppe_idx if (isinstance(ippe, dict) and ippe['sim_config'] == sim_config) or (not isinstance(ippe, dict) and sim_config == sim_configs[0])]
            if ppe_idx_sim:
                print(f"Loading global attributes for {sim_config}")
                finfo_attr = file_info.copy()
                finfo_attr['sim_config'] = sim_config
                cl.load_cm1_attrs(finfo_attr, nc_dict=nc_dict, ipert=ppe_idx_sim[0], continuous_ic=True)

            # Check for missing variables AND missing members in cached data.
            # The cache may have been built from a smaller ppe_idx, or earlier
            # corrupt-file crashes may have left some global_ids absent.
            try:
                cached = nc_dict[sim_config][train_mp]['cic']
            except KeyError:
                cached = {}
            ppe_vars_to_load = set()
            members_to_reload = []
            for ippe in ppe_idx_sim:
                gid = ippe['global_id'] if isinstance(ippe, dict) else (int(ippe) if str(ippe).isdigit() else None)
                member_dict = cached.get(gid)
                if member_dict is None:
                    members_to_reload.append(ippe)
                    ppe_vars_to_load.update(var_interest)
                    continue
                missing = [v for v in var_interest if v not in member_dict]
                if missing:
                    members_to_reload.append(ippe)
                    ppe_vars_to_load.update(missing)
            ppe_vars_to_load = list(ppe_vars_to_load)

            if members_to_reload:
                print(f"Reloading {len(members_to_reload)}/{len(ppe_idx_sim)} members for {sim_config}; vars: {ppe_vars_to_load}")
                if l_parallel:
                    tasks = []
                    for ippe in members_to_reload:
                        task = dask.delayed(cl.load_cm1)(
                            file_info, ppe_vars_to_load, steady_state_hrs, nc_dict=None, continuous_ic=True,
                            ipert=ippe, lwp_threshold=lwp_threshold
                        )
                        tasks.append(task)
                    futures = client.compute(tasks)
                    progress(futures)
                    results = client.gather(futures)
                    for r in tqdm(results, desc=f'merging missing PPE vars for {sim_config}'):
                        cl.deep_merge(nc_dict, r)
                else:
                    pbar = tqdm(members_to_reload, desc=f'loading missing PPE vars for {sim_config}')
                    for ippe in pbar:
                        cl.load_cm1(file_info, ppe_vars_to_load, steady_state_hrs, nc_dict=nc_dict, continuous_ic=True, ipert=ippe, lwp_threshold=lwp_threshold, pbar=pbar)
                nc_dict[sim_config] = _sanitize_sim_payload(nc_dict[sim_config], sim_config)
                joblib.dump(nc_dict[sim_config], train_jl_path)
                print(f"Updated cache saved to {train_jl_path}")

        else:
            ppe_idx_sim = [ippe for ippe in ppe_idx if (isinstance(ippe, dict) and ippe['sim_config'] == sim_config) or (not isinstance(ippe, dict) and sim_config == sim_configs[0])]
            
            if l_parallel:
                tasks = []
                for ippe in ppe_idx_sim:
                    # We pass None as nc_dict so it returns a new one for each member
                    # ippe is now a dictionary
                    task = dask.delayed(cl.load_cm1)(
                        file_info, var_interest, steady_state_hrs, nc_dict=None, continuous_ic=True, 
                        ipert=ippe, lwp_threshold=lwp_threshold
                    )
                    tasks.append(task)

                print(f"Computing PPE data for {sim_config} in parallel...")
                futures = client.compute(tasks)
                progress(futures)
                results = client.gather(futures)

                for r in tqdm(results, desc=f'merging PPE results for {sim_config}'):
                    cl.deep_merge(nc_dict, r)

            else:
                pbar = tqdm(ppe_idx_sim, desc=f'loading BOSS data for {sim_config}')
                for ippe in pbar:
                    cl.load_cm1(file_info, var_interest, steady_state_hrs, nc_dict=nc_dict, continuous_ic=True, ipert=ippe, lwp_threshold=lwp_threshold, pbar=pbar)

            nc_dict[sim_config] = _sanitize_sim_payload(nc_dict[sim_config], sim_config)
            joblib.dump(nc_dict[sim_config], train_jl_path)
            print(f"Dictionary saved to {train_jl_path}")

    # Load target data
    print("\nLoading target data...")
    tgt_jl_path = f"{cl.output_dir}/{target_datedir}/joblibs/{target_sim_config}_lwpthres{round(lwp_threshold*1e3)}.joblib"
    vars_to_load = var_interest
    if os.path.exists(tgt_jl_path):
        print(f"Loading target data from {tgt_jl_path}")
        nc_dict[target_sim_config] = cl.invalidate_stale_vars(joblib.load(tgt_jl_path))
        
        # Load global attributes
        print(f"Loading global attributes for {target_sim_config}")
        finfo_target = file_info.copy()
        first_combo = list(itertools.product(*vars_strs))[0]
        finfo_target.update({
            'sim_config': target_sim_config,
            'vars_str': list(first_combo),
            'date': target_datedir,
            'mp_config': target_mp,
            'l_pert': l_pert
        })
        if l_pert:
            try:
                target_pert_idx = cl.get_pert_idx(finfo_target)
                if target_pert_idx:
                    cl.load_cm1_attrs(finfo_target, nc_dict=nc_dict, ipert=target_pert_idx[0], continuous_ic=False)
            except Exception as e:
                print(f"Could not load target global attributes (pert): {e}")
        else:
            try:
                cl.load_cm1_attrs(finfo_target, nc_dict=nc_dict, continuous_ic=False)
            except Exception as e:
                print(f"Could not load target global attributes (no pert): {e}")
        
        # Check if all variables exist
        try:
            first_ic = "".join(list(itertools.product(*vars_strs))[0])
            if l_pert:
                first_pert = list(nc_dict[target_sim_config][target_mp][first_ic].keys())[0]
                existing_vars = nc_dict[target_sim_config][target_mp][first_ic][first_pert].keys()
            else:
                existing_vars = nc_dict[target_sim_config][target_mp][first_ic].keys()
            vars_to_load = [v for v in var_interest if v not in existing_vars]
        except (KeyError, IndexError):
            vars_to_load = var_interest

    if vars_to_load:
        if vars_to_load != var_interest:
            print(f"Missing variables in target data: {vars_to_load}. Loading missing ones...")
        else:
            print(f"Target data not found or being fully reloaded at {tgt_jl_path}")

        if l_parallel:
            tasks = []
            for initcond_combo in itertools.product(*vars_strs):
                finfo_target = file_info.copy()
                finfo_target.update({
                    'sim_config': target_sim_config,
                    'vars_str': list(initcond_combo),
                    'date': target_datedir,
                    'mp_config': target_mp,
                    'l_pert': l_pert
                })
                if l_pert:
                    target_pert_idx = cl.get_pert_idx(finfo_target)
                    for ipert in target_pert_idx:
                        task = dask.delayed(cl.load_cm1)(
                            finfo_target, vars_to_load, steady_state_hrs, nc_dict=None, continuous_ic=False, 
                            ipert=ipert, lwp_threshold=lwp_threshold
                        )
                        tasks.append(task)
                else:
                    task = dask.delayed(cl.load_cm1)(
                        finfo_target, vars_to_load, steady_state_hrs, nc_dict=None, continuous_ic=False, 
                        lwp_threshold=lwp_threshold
                    )
                    tasks.append(task)

            print(f"Computing {len(vars_to_load)} target variables in parallel...")
            futures = client.compute(tasks)
            progress(futures)
            results = client.gather(futures)
            
            for r in tqdm(results, desc='merging target results'):
                cl.deep_merge(nc_dict, r)
        else:
            pbar = tqdm(itertools.product(*vars_strs), desc='loading BIN data')
            for initcond_combo in pbar:
                finfo_target = file_info.copy()
                finfo_target.update({
                    'sim_config': target_sim_config,
                    'vars_str': list(initcond_combo),
                    'date': target_datedir,
                    'mp_config': target_mp,
                    'l_pert': l_pert
                })
                if l_pert:
                    target_pert_idx = cl.get_pert_idx(finfo_target)
                    for ipert in target_pert_idx:
                        cl.load_cm1(finfo_target, vars_to_load, steady_state_hrs, nc_dict=nc_dict, continuous_ic=False, ipert=ipert, lwp_threshold=lwp_threshold, pbar=pbar)
                else:
                    cl.load_cm1(finfo_target, vars_to_load, steady_state_hrs, nc_dict=nc_dict, continuous_ic=False, lwp_threshold=lwp_threshold, pbar=pbar)
        
        # Save updated nc_dict to joblib
        joblib.dump(nc_dict[target_sim_config], tgt_jl_path)
    else:
        print("All variables of interest already exist in target data.")


    ncase = 1
    ncase_respective = [len(i) for i in vars_strs]
    for i in ncase_respective:
        ncase *= i
    
    dims = {
        'scalar_var': 1,
        'ncase': ncase,
        'nppe': len(ppe_idx),
    }

    if l_pert:
        first_combo = list(itertools.product(*vars_strs))[0]
        first_ic = "".join(first_combo)
        dims['npert'] = len(nc_dict[target_sim_config][target_mp][first_ic].keys())

    global_attrs = {
        'description': 'PPE data for ' + sim_config_str,
        'date_simulated': datedir,
    }
    
    
    # Create variable structure
    ncvars = create_nc_variables_structure(nc_dict, vars_vn, var_interest, l_pert)
    
    print("\nProcessing target data...")
    if l_pert:
        tgt_pert_list = cl.get_pert_idx(file_info)
    else:
        tgt_pert_list = None
    process_target_data(nc_dict, vars_vn, vars_strs, var_interest, ncvars, dims, target_mp, target_sim_config, l_pert, tgt_pert_list)
                        
    # Processing PPE data
    print("\nProcessing PPE data...")
    ncvars, dims = process_ppe_data(nc_dict, ppe_idx, vars_vn, var_interest, ncvars, dims, datedir, sim_configs, train_mp)

    # If this is a '_select' PPE, record which parameters are actually perturbed
    if '_select' in sim_config_str:
        if os.path.exists(select_params_csv):
            sel_df = pd.read_csv(select_params_csv)
            sel_param_names = sel_df.iloc[:, 0].to_numpy()
            all_param_names = ncvars['param_names']['data']
            # Find index of each selected param in the full param_names array
            name_to_idx = {name: idx for idx, name in enumerate(all_param_names)}
            select_params_idx = np.sort(np.array([name_to_idx[n] for n in sel_param_names if n in name_to_idx]))
            global_attrs['select_params_idx'] = select_params_idx
            print(f"select_params_idx: {select_params_idx}")
        else:
            print(f"Warning: select_params_csv '{select_params_csv}' not found; skipping select_params_idx.")
    
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
    if 'npert' in dims:
        global_attrs['npert'] = dims['npert']
    else:
        global_attrs['npert'] = 1
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
        elif 'precip_frac' in ivar:
            global_attrs['thresholds_eff0'].append(0.01)
        elif 'onset' in ivar or 'M3_' in ivar:
            global_attrs['thresholds_eff0'].append(np.nan)
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

def create_nc_variables_structure(nc_dict, vars_vn, var_interest, l_pert):
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
            'dims': ('ncase', 'npert') if l_pert else ('ncase',),
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

_PPE_FLAG_RE = re.compile(r'^\s*l_ppe_(\w+)\s*=\s*\.(true|false)\.', re.I | re.M)

def _unperturbed_params(member_dir):
    """Return the set of base-param names whose `l_ppe_<name>` flag is .false.
    in {member_dir}/CM1.nml. These rows in params.csv carry placeholder values
    and should be dropped from the perturbation summary."""
    nml_path = os.path.join(member_dir, 'CM1.nml')
    if not os.path.exists(nml_path):
        return set()
    with open(nml_path) as f:
        text = f.read()
    return {name for name, val in _PPE_FLAG_RE.findall(text) if val.lower() == 'false'}

def process_ppe_data(nc_dict, ppe_idx, vars_vn, var_interest, ncvars, dims, datedir, sim_configs, train_mp):
    """Load PPE data with memory cleanup"""
    ic_str = 'cic'

    # Load PPE parameters
    for i, ppe_item in enumerate(tqdm(ppe_idx, desc='loading params')):
        # Check if ppe_item is dict or legacy
        if isinstance(ppe_item, dict):
            current_config = ppe_item['sim_config']
            member = ppe_item['member']
            global_id = ppe_item['global_id']
        else:
            current_config = sim_configs[0] if isinstance(sim_configs, list) else sim_configs
            member = str(ppe_item)
            global_id = int(member) if member.isdigit() else i

        # Construct path using the correct config
        member_dir = f"{cl.output_dir}{datedir}/{current_config}/{train_mp}/{member}"
        param_df = pd.read_csv(f"{member_dir}/params.csv")

        # Drop rows for params whose l_ppe_<name> flag is .false. in CM1.nml.
        # Those rows hold placeholder values (no real perturbation), so including
        # them would pollute params_PPE with a constant column.
        unperturbed = _unperturbed_params(member_dir)
        if unperturbed:
            if param_df.shape[0] > param_df.shape[1]:  # vertical layout
                names_col = param_df.columns[0]
                mask = ~param_df[names_col].isin(unperturbed)
                param_df = param_df.loc[mask].reset_index(drop=True)
            else:  # horizontal layout: param names are column headers
                cols_to_drop = [c for c in param_df.columns if c.strip() in unperturbed]
                if cols_to_drop:
                    param_df = param_df.drop(columns=cols_to_drop)


        if i == 0:  # First iteration
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
            ncvars['params_PPE']['data'][i, :] = np.array(param_df.iloc[:, 1].to_numpy())
        else:
            ncvars['params_PPE']['data'][i, :] = np.array(param_df)
        
        # Store initial conditions
        for var_vn in vars_vn:
            if ncvars[var_vn + '_PPE']['data'] is None:
                ncvars[var_vn + '_PPE']['data'] = np.zeros((len(ppe_idx),))

            # Retrieve value from nc_dict using global dict key
            ncvars[var_vn + '_PPE']['data'][i] = nc_dict[current_config][train_mp][ic_str][global_id][var_vn]

    # Load summary variables
    for ivar in var_interest:
        # Re-construct list of values using ppe indices (which are global IDs in nc_dict)
        data_list = []
        for ppe_item in ppe_idx:
            if isinstance(ppe_item, dict):
                idx = ppe_item['global_id']
                item_config = ppe_item['sim_config']
            else:
                idx = int(ppe_item)
                item_config = sim_configs[0] if isinstance(sim_configs, list) else sim_configs
            data_list.append(nc_dict[item_config][train_mp][ic_str][idx][ivar]['value'])

        ncvars['ppe_' + ivar]['data'] = np.array(data_list)

    return ncvars, dims

def process_target_data(nc_dict, vars_vn, vars_strs, var_interest, ncvars, dims, target_mp, target_sim_config, l_pert, tgt_pert_list):
    """Process target data with memory cleanup"""
    ncase = dims['ncase']
    
    combo = list(itertools.product(*vars_strs))[0]
    ic_str = "".join(combo)

    def get_tgt_val(ic_str_key, var_key):
        if l_pert:
            available_idxs = list(nc_dict[target_sim_config][target_mp][ic_str_key].keys())
            vals = []
            for idx in available_idxs:
                vals.append(nc_dict[target_sim_config][target_mp][ic_str_key][idx][var_key]['value'])
            return np.array(vals)
        else:
            return nc_dict[target_sim_config][target_mp][ic_str_key][var_key]['value']

    for ivar in var_interest:
        tgt_dims = (ncase, dims['npert']) if l_pert else (ncase,)
        ref_val = get_tgt_val(ic_str, ivar)
        if 'ntime' in ncvars['tgt_' + ivar]['dims']:
            dims['ntime'] = ref_val.shape[1] if l_pert else ref_val.shape[0]
            tgt_dims += (dims['ntime'],)
        if 'nlevel' in ncvars['tgt_' + ivar]['dims']:
            dims['nlevel'] = ref_val.shape[-1]
            tgt_dims += (dims['nlevel'],)

        ncvars['tgt_' + ivar]['data'] = np.zeros(tgt_dims)

        icase = 0
        for combo in itertools.product(*vars_strs):
            ic_str_cur = "".join(combo)
            ncvars['tgt_' + ivar]['data'][icase] = get_tgt_val(ic_str_cur, ivar)
            icase += 1

    for var_vn in vars_vn:
        tgt_case_dims = (ncase,)
        ncvars['case_' + var_vn]['data'] = np.zeros(tgt_case_dims)
    for icase, combo in enumerate(itertools.product(*vars_strs)):
        ic_str_cur = "".join(combo)
        for i_init, var_vn in enumerate(vars_vn):
            if l_pert:
                available_idxs = list(nc_dict[target_sim_config][target_mp][ic_str_cur].keys())
                val = nc_dict[target_sim_config][target_mp][ic_str_cur][available_idxs[0]][var_vn]
            else:
                val = nc_dict[target_sim_config][target_mp][ic_str_cur][var_vn]
            ncvars['case_' + var_vn]['data'][icase] = val

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('camp', nargs='?', default='dycoms',
                        help='Campaign name (e.g., dycoms, rico). Default: dycoms')
    args = parser.parse_args()
    main(camp=args.camp)
