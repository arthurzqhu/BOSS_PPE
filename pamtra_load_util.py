import os
import re
import numpy as np
import netCDF4 as nc
from glob import glob
import platform
import socket
import load_ppe_fun as lp
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import sys
import matplotlib.pyplot as plt

# Platform specific paths (copied from cm1_load_utils logic)
if 'macOS' in platform.platform():
    output_dir = '/Volumes/ESSD/research/serpentine/' # Guessing path structure
    bossppe_dir = '/Users/arthurhu/github/BOSS_PPE/'
    nc_dir = '/Users/arthurhu/github/BOSS_PPE/summary_ncs/'
elif 'Linux' in platform.platform():
    hostname = socket.gethostname()
    if hostname == "simurgh":
        output_dir = '/data1/arthurhu/serpentine/' # Guessing path structure
        nc_dir = '/home/arthurhu/BOSS_PPE/summary_ncs/'
        bossppe_dir = '/home/arthurhu/BOSS_PPE/'
    else:
        output_dir = '/pscratch/sd/a/arthurhu/serpentine/' # Guessing path structure
        nc_dir = '/pscratch/sd/a/arthurhu/BOSS_PPE/summary_ncs/'
        bossppe_dir = '/pscratch/sd/a/arthurhu/BOSS_PPE/'

output_var_set = {
    'Ze_dmcolmax': {
        'var_source': 'Ze', 'var_unit': 'dBZ', 'longname': 'Domain-Mean Column-Max Ze'
    },
    'Ze_dmcolmax_ss_mean': {
        'var_source': 'Ze', 'var_unit': 'mm6/m3', 'longname': 'Steady State Domain-Mean Column-Max Ze'
    },
    'Ze_colmax_ss_std': {
        'var_source': 'Ze', 'var_unit': 'dBZ', 'longname': 'Steady State Std Column-Max Ze'
    },
    'Ze_dm_surface': {
        'var_source': 'Ze', 'var_unit': 'dBZ', 'longname': 'Domain-Mean Surface Ze'
    },
    'Ze_dm_surface_ss_mean': {
        'var_source': 'Ze', 'var_unit': 'mm6/m3', 'longname': 'Steady State Domain-Mean Surface Ze'
    },
    'Ze_surface_ss_std': {
        'var_source': 'Ze', 'var_unit': 'dBZ', 'longname': 'Steady State Std Surface Ze'
    },
    'PIA_dm': {
        'var_source': 'Attenuation_Hydrometeors', 'var_unit': 'dB', 'longname': 'Domain-Mean Integrated Attenuation'
    },
    'PIA_dm_ss_mean': {
        'var_source': 'Attenuation_Hydrometeors', 'var_unit': '', 'longname': 'Steady State Domain-Mean Integrated Attenuation'
    },
    'PIA_ss_std': {
        'var_source': 'Attenuation_Hydrometeors', 'var_unit': 'dB', 'longname': 'Steady State Std Integrated Attenuation'
    },
    'Specific_Attenuation_dmcolmax': {
        'var_source': 'Attenuation_Hydrometeors', 'var_unit': 'dB/km', 'longname': 'Domain-Mean Column-Max Specific Attenuation'
    },
    'Specific_Attenuation_dmcolmax_ss_mean': {
        'var_source': 'Attenuation_Hydrometeors', 'var_unit': 'dB/km', 'longname': 'Steady State Domain-Mean Column-Max Specific Attenuation'
    },
    'Specific_Attenuation_colmax_ss_std': {
        'var_source': 'Attenuation_Hydrometeors', 'var_unit': 'dB/km', 'longname': 'Steady State Std Column-Max Specific Attenuation'
    },
    'Radar_MeanDopplerVel_dmcolmean': {
        'var_source': 'Radar_MeanDopplerVel', 'var_unit': 'm/s', 'longname': 'Domain-Mean Ze-Weighted Column-Mean Doppler Vel'
    },
    'Radar_MeanDopplerVel_dmcolmean_ss_mean': {
        'var_source': 'Radar_MeanDopplerVel', 'var_unit': 'm/s', 'longname': 'Steady State Domain-Mean Ze-Weighted Column-Mean Doppler Vel'
    },
    'Radar_MeanDopplerVel_colmean_ss_std': {
        'var_source': 'Radar_MeanDopplerVel', 'var_unit': 'm/s', 'longname': 'Steady State Std Ze-Weighted Column-Mean Doppler Vel'
    },
    'Radar_SpectrumWidth_dmcolmean': {
        'var_source': 'Radar_SpectrumWidth', 'var_unit': 'm/s', 'longname': 'Domain-Mean Ze-Weighted Column-Mean Spectrum Width'
    },
    'Radar_SpectrumWidth_dmcolmean_ss_mean': {
        'var_source': 'Radar_SpectrumWidth', 'var_unit': 'm/s', 'longname': 'Steady State Domain-Mean Ze-Weighted Column-Mean Spectrum Width'
    },
    'Radar_SpectrumWidth_colmean_ss_std': {
        'var_source': 'Radar_SpectrumWidth', 'var_unit': 'm/s', 'longname': 'Steady State Std Ze-Weighted Column-Mean Spectrum Width'
    },
    'tb_dm': {
        'var_source': 'tb', 'var_unit': 'K', 'longname': 'Domain-Mean Brightness Temperature'
    },
    'tb_dm_ss_mean': {
        'var_source': 'tb', 'var_unit': 'K', 'longname': 'Steady State Domain-Mean Brightness Temperature'
    },
    'tb_ss_std': {
        'var_source': 'tb', 'var_unit': 'K', 'longname': 'Steady State Std Brightness Temperature'
    },
}

def get_ppe_idx(file_info):
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    mp = file_info['mp_config']
    # Check if directory exists, return empty list if not to avoid crash, or handle elsewhere
    path = f"{output_dir}{fdate}/{fsim_config}/{mp}"
    if not os.path.exists(path):
        print(f"Warning: Directory not found: {path}")
        return []
    ppe_idx = os.listdir(path)
    ppe_idx = lp.sort_strings_by_number(ppe_idx)
    return ppe_idx

def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def load_pamtra(file_info, var_interest, nc_dict=None, continuous_ic=True, ss_hrs=2, ippe=0, ze_threshold=-50.0):
    if nc_dict is None:
        nc_dict = {}
    mp          = file_info['mp_config']
    vars_vn     = file_info['vars_vn']
    fdir        = file_info['dir']
    fdate       = file_info['date']
    fsim_config = file_info['sim_config']
    fn_prefix, fn_suffix = "pamtra_out", ".nc" # Guessing filename pattern

    if continuous_ic:
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{mp}/{ippe}/{fn_prefix}*{fn_suffix}"
        ic_str = 'cic'
    else:
        ic_str = "".join(file_info['vars_str'])
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{fn_prefix}*{fn_suffix}"

    file_paths = sorted(glob(file_pattern), key=last_number_key)
    if not file_paths:
        # Fallback to check if it's just one file or different naming
        # print(f"No files match: {file_pattern}")
        # Return empty/None to handle gracefully?
         # For now raise error similar to cm1
        raise FileNotFoundError(f"No files match: {file_pattern}")

    # time vector initialization
    if 'time' not in nc_dict:
        nc_dict['time'] = np.empty(len(file_paths), dtype=float)

    # Get dt
    if len(file_paths) >= 2:
        try:
            t0 = float(file_paths[0].split('_')[-1].split('.')[0]) * 3600
            t1 = float(file_paths[1].split('_')[-1].split('.')[0]) * 3600
            dt = float(t1 - t0)
        except:
            dt = np.nan
    else:
        dt = np.nan

    # open first file to grab coords/attrs
    with nc.Dataset(file_paths[0], 'r') as ds0:
        nc_dict.setdefault(fsim_config, {})
        nc_dict[fsim_config].setdefault(mp, {})
        nc_dict[fsim_config][mp].setdefault(ic_str, {})
        nc_dict['init_var'] = vars_vn
        if ippe > 0:
            nc_dict[fsim_config][mp][ic_str].setdefault(ippe, {})

        # vn attributes
        for vn in vars_vn:
            # Pamtra files might not have these attributes, wrap in try/except or default
            try:
                nc_dict[vn + '_units'] = ds0.getncattr(vn + '_units')
                keydst = nc_dict[fsim_config][mp][ic_str] if ippe == 0 else nc_dict[fsim_config][mp][ic_str][ippe]
                keydst[vn] = ds0.getncattr(vn)
            except:
                pass 
        
        # coords
        nc_dict['z'] = ds0.variables['height'][:][0, 0, :]
        zf = np.zeros(len(nc_dict['z']) + 1)
        zf[0] = 0
        for i in range(len(nc_dict['z'])):
            zf[i+1] = 2 * nc_dict['z'][i] - zf[i]
        # Calculate dz
        dz = np.diff(zf)

    n_needed = int(np.ceil((ss_hrs * 3600) / dt) + 1) if np.isfinite(dt) and dt > 0 else 1

    # Pre-parse meta and setup collectors
    var_meta = {vn: parse_var_meta(vn) for vn in var_interest}
    raw_collector = {vn: [] for vn in var_interest}

    for ifp, fp in enumerate(file_paths):
        with nc.Dataset(fp, 'r') as ds:
            t_val = float(fp.split('_')[-1].split('.')[0]) * 3600
            nc_dict['time'][ifp] = t_val
            
            # Need Ze for thresholding and weighting
            # Assuming 'Ze' is in the file.
            # Handle potential name variations? 
            # Standard Pamtra: 'Ze' or 'refl'?
            if 'Ze' in ds.variables:
                ze_var = ds.variables['Ze'][...]
            else:
                # Fallback, maybe look for something else or error
                # Making dummy if testing
                ze_var = None

            for vn in var_interest:
                meta = var_meta[vn]
                is_ss_file = (ifp >= len(file_paths) - n_needed)
                
                if not meta['is_ss'] or is_ss_file:
                    val = extract_and_reduce(vn, ds, dz, ze_var, ze_threshold)
                    raw_collector[vn].append(val)

    # Final aggregation
    for vn in var_interest:
        if continuous_ic:
            dst = nc_dict[fsim_config][mp][ic_str][ippe]
        else:
            dst = nc_dict[fsim_config][mp][ic_str]

        dst.setdefault(vn, {})
        
        dst[vn]['value'] = aggregate_timeseries(vn, raw_collector[vn], var_meta[vn])
        # dst[vn]['units'] = output_var_set[vn]['var_unit']
        # Handle lookup safely
        if vn in output_var_set:
            dst[vn]['units'] = output_var_set[vn]['var_unit']
        elif any(k in vn for k in output_var_set):
             # Try to find partial match
             for k, v in output_var_set.items():
                 if k in vn:
                     dst[vn]['units'] = v['var_unit']
                     break


    # Load ppe_summary.nc if exists (contains params and attributes from CM1 loop)
    # Path: output_dir/fdate/fsim_config/mp/ippe/ppe_summary.nc
    # Use fdir from file_info? fdir usually ends with /
    if continuous_ic:
        summary_path = f"{fdir}{fdate}/{fsim_config}/{mp}/{ippe}/ppe_summary.nc"
    else:
        # Target case (vars_str structure)
        # Directory structure: output_dir/target/fsim_config/vars_dir/mp/ppe_summary.nc
        # Need to reconstruct vars_dir
        if 'vars_str' in file_info:
            vars_dir = "/".join(file_info['vars_str'])
            # Note: fdir is usually output_dir/ (e.g. .../serpentine/)
            # Target path seems to be .../serpentine/target/...
            # Let's try to construct it based on pamtra_cm1 logic:
            # /data1/arthurhu/serpentine/target/{target_sim_config}/{vars_dir}/{target_mp}/ppe_summary.nc
            
            # Assuming fdir points to .../serpentine/
            summary_path = f"{fdir}target/{fsim_config}/{vars_dir}/{mp}/ppe_summary.nc"
        else:
            summary_path = ""

    if os.path.exists(summary_path):
        with nc.Dataset(summary_path, 'r') as snc:
            # Load Global Attributes
            for attr in snc.ncattrs():
                nc_dict[attr] = snc.getncattr(attr)
            
            if continuous_ic:
                dst = nc_dict[fsim_config][mp][ic_str][ippe]
            else:
                dst = nc_dict[fsim_config][mp][ic_str]
            
            # Load Variables
            for vn in snc.variables:
                var = snc.variables[vn]
                
                if vn == 'params':
                        dst['params'] = var[:]
                elif vn == 'param_names':
                        # Handle string array (VLEN or CHAR)
                        if hasattr(var.dtype, 'char') and var.dtype.char == 'S1':
                             dst['param_names'] = nc.chartostring(var[:])
                        else:
                             dst['param_names'] = var[:]
                else:
                        # Assume scalar variable (initial conditions)
                        val = var[:]
                        # scalar extraction if single element
                        if val.size == 1:
                            val = val.item()
                        dst[vn] = val
                
                # Units
                if hasattr(var, 'units'):
                        nc_dict[vn + '_units'] = var.units

    return nc_dict

def parse_var_meta(var_name):
    re_ss_prc  = re.search(r'ss_(\d+)th_prct', var_name)
    is_ss_mean = bool(re.search(r'ss_mean', var_name))
    is_ss_std  = bool(re.search(r'ss_std', var_name))
    is_ss_prc  = bool(re_ss_prc)
    # is_dm      = bool(re.search(r'_dm', var_name))
    is_dm      = True # Always domain mean for these summary stats usually
    is_ss      = max(is_ss_mean, is_ss_std, is_ss_prc)
    nth_prctl  = int(re_ss_prc.group(1)) if is_ss_prc else None
    return {'is_ss': is_ss, 'is_ss_mean': is_ss_mean, 'is_ss_std': is_ss_std, 
            'is_ss_prc': is_ss_prc, 'is_dm': is_dm, 'nth_prctl': nth_prctl}

from_db = lambda db: 10**(db/10.)
# from_db = lambda db: db

def extract_and_reduce(var_name, ds, dz, ze, ze_threshold):
    # Map requested variable to source variable
    base_key = None
    for k in output_var_set:
        if var_name.startswith(k) and (len(k) > len(base_key) if base_key else True):
            base_key = k
    
    if not base_key:
        return np.nan

    vsource = output_var_set[base_key]['var_source']
    scale = output_var_set[base_key].get('scale', 1.0)
    
    # Get raw data and slice to (x, y, z) or (x, y)
    # Pamtra dims: (grid_x, grid_y, heightbins/outlevels, [freq], [pol], [peak/angle])
    
    if vsource not in ds.variables:
        return np.nan
        
    raw_var = ds.variables[vsource]
    chk_shape = raw_var.shape
    
    # Handle Ze for mask (x, y, z)
    if ze.ndim > 3:
        ze_3d = ze[:, :, :, 0, 0, 0]
    else:
        ze_3d = ze

    ze_3d[ze_3d > 100] = np.nan
    
    # helper to slice to 3D (x, y, z)
    # usually taking index 0 for extra dims, except tb
    if 'tb' in var_name:
        # (x, y, outlevels, angles, freq, pol)
        # Take Top level (-1), Angle 0, Freq 0, Pol 0
        # Result: (x, y)
        if raw_var.ndim >= 3:
            # outlevels = [833000, 0] by default, so 1 is 0m
            data = raw_var[:, :, 1, -1, 0, 0] * scale
        else:
            data = raw_var[...] * scale
    elif 'Ze' in var_name:
        # (x, y, z, freq, pol, peak)
        data = raw_var[:, :, :, 0, 0, 0] * scale
        # Fill masked values with NaN first if it's a masked array
        if np.ma.is_masked(data):
            data = data.filled(np.nan)
        # Mask out fill values / unphysical large values
        data[ze_3d > 100] = np.nan
    elif 'Radar' in var_name:
         # (x, y, z, freq, pol, peak)
         data = raw_var[:, :, :, 0, 0, 0] * scale
         if np.ma.is_masked(data):
            data = data.filled(np.nan)
         # Mask out fill values / unphysical large values
         data[ze_3d > 100] = np.nan
    elif 'PIA' in var_name:
         # (x, y, z, freq, pol)
         data = raw_var[:, :, :, 0, 0] * scale
         if np.ma.is_masked(data):
            data = data.filled(np.nan)
    elif 'Specific_Attenuation' in var_name:
         # (x, y, z, freq, pol)
         data = raw_var[:, :, :, 0, 0] * scale / (dz/1e3)
         if np.ma.is_masked(data):
            data = data.filled(np.nan)
    else:
         # Fallback
         data = raw_var[...] * scale
         if np.ma.is_masked(data):
            data = data.filled(np.nan)
    
    # ColMax Mask
    # Axis 2 is height
    ze_colmax = np.nanmax(ze_3d, axis=2) # (x, y)
    mask_col = ze_colmax < ze_threshold

    # Apply mask for Column-based variables
    # If data is 3D (x, y, z)
    if data.ndim == 3:
        # Broadcast mask (x, y) -> (x, y, 1)
        # Actually (x, y, z)
        # apply to all z
        data[mask_col, :] = np.nan
    elif data.ndim == 2:
        # (x, y)
        data[mask_col] = np.nan

    # Reduction Logic
    if 'colmax' in var_name:
        # Column Max: (x, y, z) -> (x, y) -> scalar
        if data.ndim == 3:
            col_res = np.nanmax(data, axis=2) # (x, y)
            
            if 'dm' in var_name:
                if 'Ze' in var_name:
                    col_res = from_db(col_res)
                res = np.nanmean(col_res)
                if np.isnan(res):
                    res = 0
            else:
                res = col_res
        else:
            raise ValueError(f"Error: {var_name} requested colmax but data ndim is {data.ndim}")
    elif 'surface' in var_name:
        # Surface: take lowest level (index 0)
        if data.ndim == 3:
            surf_res = np.nanmax(data[:, :, :20], axis=2) # (x, y)
            if 'dm' in var_name:
                if 'Ze' in var_name:
                    surf_res = from_db(surf_res)
                res = np.nanmean(surf_res)
                if np.isnan(res):
                    res = 0
            else:
                res = surf_res
        else:
            raise ValueError(f"Error: {var_name} requested surface but data ndim is {data.ndim}")

    elif 'path' in var_name or 'PIA' in var_name:
        # Column Integral: sum(data * dz)
        if data.ndim == 3:
            # dz is 1D (z). Broadcast shape: (1, 1, z)
            dz_b = dz[None, None, :]
            # Check length of dz vs data.shape[2]
            # dz might be len(z) or len(z)-1?
            # In load_pamtra, dz is diff(zf), so same len as heightbins if calculated right.
            # But let's slice just in case
            n_z = data.shape[2]
            dz_use = dz_b[:, :, :n_z]
            
            col_res = np.nansum(data * dz_use, axis=2) # (x, y)

            if 'dm' in var_name:
                if 'PIA' in var_name:
                    col_res = from_db(col_res)
                res = np.nanmean(col_res)
            else:
                res = col_res
        else:
            raise ValueError(f"Error: {var_name} requested path but data ndim is {data.ndim}")

    elif 'colmean' in var_name:
        # Ze-weighted Column Mean
        if data.ndim == 3:
            z_lin = 10**(0.1 * ze_3d) # (x, y, z)
            dz_b = dz[None, None, :data.shape[2]]
            
            # Weighted sum
            # Note: ze_3d might have NaNs. data might have NaNs.
            prod = data * z_lin * dz_b
            weight = z_lin * dz_b
            
            num = np.nansum(prod, axis=2)
            den = np.nansum(weight, axis=2)
            
            res = num / den
            res[den == 0] = np.nan
            if 'dm' in var_name:
                r_sum = np.nansum(res)
                r_count = np.sum(~np.isnan(res))
                res = r_sum / r_count if r_count > 0 else np.nan
        else:
            raise ValueError(f"Error: {var_name} requested colmean but data ndim is {data.ndim}")

    elif '_dm' in var_name:
        # Just domain mean (scalar)
        res = np.nanmean(data, axis=(0, 1))
    else:
        res = data

    return res

def aggregate_timeseries(var_name, ts, meta):
    if not ts: return np.nan
    
    # Check if elements are scalars
    if np.ndim(ts[0]) == 0:
        # If masked scalars, convert to array with NaNs
        arg = np.array([float(x) if x is not np.ma.masked else np.nan for x in ts])
    else:
        arg = np.stack(ts, axis=-1) # (TotalTime,)
    
    if meta['is_ss_std']:
        arg = np.nanstd(arg, axis=(0, 1) if arg.ndim > 1 else 0)
        if np.all(np.isnan(arg)):
            arg[:] = 0
    
    if meta['is_ss']:
        arg = np.nanmean(arg, axis=0) if arg.ndim > 0 else arg
    
    return arg

def last_number_key(s):
    matches = re.findall(r'(\d+)(?!.*\d)', s)
    return int(matches[0]) if matches else 0
