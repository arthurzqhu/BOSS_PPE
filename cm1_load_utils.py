from fileinput import filename
import os
import re
import numpy as np
import netCDF4 as nc
from glob import glob
import platform
import socket
import load_ppe_fun as lp
from tqdm import tqdm

M3toQ = np.pi/6*1e3
QtoM3 = 1/M3toQ

if 'macOS' in platform.platform():
    output_dir = '/Volumes/ESSD/research/cm1/'
    bossppe_dir = '/Users/arthurhu/github/BOSS_PPE/'
    nc_dir = '/Users/arthurhu/github/BOSS_PPE/summary_ncs/'
elif 'Linux' in platform.platform():
    hostname = socket.gethostname()
    if hostname == "simurgh":
        output_dir = '/data1/arthurhu/cm1/'
        nc_dir = '/home/arthurhu/BOSS_PPE/summary_ncs/'
        bossppe_dir = '/home/arthurhu/BOSS_PPE/'
    else:
        output_dir = '/pscratch/sd/a/arthurhu/cm1/'
        nc_dir = '/pscratch/sd/a/arthurhu/BOSS_PPE/summary_ncs/'
        bossppe_dir = '/pscratch/sd/a/arthurhu/BOSS_PPE/'

output_var_set = {
                  'M0_path': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'LNP'}, 
                  'M3_path': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'LWP'}, 
                  'M4_path': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'M4 Path'},
                  'M5_path': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'M5 Path'},
                  'M6_path': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'M6 Path'},
                  'M9_path': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'M9 Path'},
                  'M0_dmprof': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Domain-Mean LNP'},
                  'M3_dmprof': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Domain-Mean LWC'},
                  'M4_dmprof': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Domain-Mean M4'},
                  'M5_dmprof': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Domain-Mean M5'},
                  'M6_dmprof': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Domain-Mean M6'},
                  'M9_dmprof': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Domain-Mean M9'},
                  'M0_dmpath': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Domain-Mean LNP'}, 
                  'M3_dmpath': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Domain-Mean LWP'}, 
                  'M4_dmpath': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Domain-Mean M4 Path'},
                  'M5_dmpath': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Domain-Mean M5 Path'},
                  'M6_dmpath': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Domain-Mean M6 Path'},
                  'M9_dmpath': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Domain-Mean M9 Path'},
                  'M0_curtain_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNC'}, 
                  'M3_curtain_mean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4_curtain_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_curtain_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_curtain_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_curtain_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_path_last4hrmean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Last 4hr Mean LNP'}, 
                  'M3_path_last4hrmean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Last 4hr Mean LWC'}, 
                  'M4_path_last4hrmean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean M4'},
                  'M5_path_last4hrmean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Last 4hr Mean M5'},
                  'M6_path_last4hrmean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean M6'},
                  'M9_path_last4hrmean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Last 4hr Mean M9'},
                  'M0': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNP'}, 
                  'M3': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'u_dmprof': {'var_source': 'uinterp', 'var_unit': 'm/s', 'longname': 'Horizontal Wind (x) Domain-Mean Profile'},
                  'v_dmprof': {'var_source': 'vinterp', 'var_unit': 'm/s', 'longname': 'Horizontal Wind (y) Domain-Mean Profile'},
                  'w_dmprof': {'var_source': 'winterp', 'var_unit': 'm/s', 'longname': 'Vertical Wind (z) Domain-Mean Profile'},
                  'u_curtain_mean': {'var_source': 'uinterp', 'var_unit': 'm/s', 'longname': 'Horizontal Wind (x) Curtain Mean'},
                  'v_curtain_mean': {'var_source': 'vinterp', 'var_unit': 'm/s', 'longname': 'Horizontal Wind (y) Curtain Mean'},
                  'w_curtain_mean': {'var_source': 'winterp', 'var_unit': 'm/s', 'longname': 'Vertical Wind (z) Curtain Mean'},
                  'u_curtain_slice': {'var_source': 'uinterp', 'var_unit': 'm/s', 'longname': 'Horizontal Wind (x) Curtain Slice'},
                  'v_curtain_slice': {'var_source': 'vinterp', 'var_unit': 'm/s', 'longname': 'Horizontal Wind (y) Curtain Slice'},
                  'w_curtain_slice': {'var_source': 'winterp', 'var_unit': 'm/s', 'longname': 'Vertical Wind (z) Curtain Slice'},
                  'w': {'var_source': 'winterp', 'var_unit': 'm/s', 'longname': 'Vertical Wind (z)'},
                  'prate_dm': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Domain-Mean Precipitation Rate'},
                  'prate_dm_last4hrmean': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Last 4hr Mean Domain-Mean Precipitation Rate'},
                  'prate_dm_last4hrstd': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Last 4hr Mean Domain-Std Precipitation Rate'},
                  'sedflux_m0': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Sedflux M0'},
                  'sedflux_m3': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Sedflux M3'},
                  'sedflux_m4': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Sedflux M4'},
                  'sedflux_m6': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Sedflux M6'},
                  'sfM0_last4hrmean': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Last 4hr Mean Sedflux M0'},
                  'sfM3_last4hrmean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Last 4hr Mean Sedflux M3'},
                  'sfM4_last4hrmean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean Sedflux M4'},
                  'sfM6_last4hrmean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean Sedflux M6'},
                  'sfM0_per5lvl_last4hrmean': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Last 4hr Mean Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl_last4hrmean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Last 4hr Mean Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl_last4hrmean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl_last4hrmean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean Sedflux M6 per 5 levels'},
                  'M0_per5lvl_last4hrmean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Last 4hr Mean LNP'}, 
                  'M3_per5lvl_last4hrmean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Last 4hr Mean LWC'}, 
                  'M4_per5lvl_last4hrmean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean M4'},
                  'M5_per5lvl_last4hrmean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Last 4hr Mean M5'},
                  'M6_per5lvl_last4hrmean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean M6'},
                  'M9_per5lvl_last4hrmean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Last 4hr Mean M9'},
                  'sfM0_10m_last4hrmean': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Last 4hr Mean Sedflux M0 10m'},
                  'sfM3_10m_last4hrmean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Last 4hr Mean Sedflux M3 10m'},
                  'sfM4_10m_last4hrmean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean Sedflux M4 10m'},
                  'sfM6_10m_last4hrmean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean Sedflux M6 10m'},
                  'sfM0_250m_last4hrmean': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Last 4hr Mean Sedflux M0 250m'},
                  'sfM3_250m_last4hrmean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Last 4hr Mean Sedflux M3 250m'},
                  'sfM4_250m_last4hrmean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean Sedflux M4 250m'},
                  'sfM6_250m_last4hrmean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean Sedflux M6 250m'},
                  'sfM0_500m_last4hrmean': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Last 4hr Mean Sedflux M0 500m'},
                  'sfM3_500m_last4hrmean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Last 4hr Mean Sedflux M3 500m'},
                  'sfM4_500m_last4hrmean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean Sedflux M4 500m'},
                  'sfM6_500m_last4hrmean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean Sedflux M6 500m'},
                  'sfM0_750m_last4hrmean': {'var_source': 'sedflux_M0', 'var_unit': '1/m^2/s', 'longname': 'Last 4hr Mean Sedflux M0 750m'},
                  'sfM3_750m_last4hrmean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/m^2/s', 'scale': M3toQ, 'longname': 'Last 4hr Mean Sedflux M3 750m'},
                  'sfM4_750m_last4hrmean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/m^2/s', 'scale': 1e-4**4, 'longname': 'Last 4hr Mean Sedflux M4 750m'},
                  'sfM6_750m_last4hrmean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/m^2/s', 'scale': 1e-4**6, 'longname': 'Last 4hr Mean Sedflux M6 750m'},
                  'precip_onset':{'var_source': 'prate', 'var_unit': 'hr', 'longname': 'Precipitation Onset'},
                  'precip_max_dm':{'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Peak Precipitation'},
                  }

def get_ppe_idx(file_info):
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    mp = file_info['mp_config']
    ppe_idx = os.listdir(f"{output_dir}{fdate}/{fsim_config}/{mp}")
    ppe_idx = lp.sort_strings_by_number(ppe_idx)
    return ppe_idx

def load_cm1(file_info, var_interest, nc_dict, continuous_ic, ippe=0):
    mp          = file_info['mp_config']
    vars_vn     = file_info['vars_vn']
    fdir        = file_info['dir']
    fdate       = file_info['date']
    fsim_config = file_info['sim_config']
    fn_prefix, fn_suffix = "cm1out_0", ".nc"

    if continuous_ic:
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{mp}/{ippe}/{fn_prefix}*{fn_suffix}"
        ic_str = 'cic'
    else:
        ic_str = "".join(file_info['vars_str'])
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{fn_prefix}*{fn_suffix}"

    file_paths = sorted(glob(file_pattern), key=last_number_key)
    if not file_paths:
        raise FileNotFoundError(f"No files match: {file_pattern}")

    # time vector
    if 'time' not in nc_dict:
        time_array = np.empty(len(file_paths), dtype=float)
        for i_file, fp in enumerate(file_paths):
            with nc.Dataset(fp, 'r') as ds:
                t = np.asarray(ds['time'][:]).ravel()
                time_array[i_file] = t[0] if t.size else np.nan
        nc_dict['time'] = time_array

    # open first file to grab coords/attrs, then close it
    with nc.Dataset(file_paths[0], 'r') as ds0:
        nc_dict.setdefault(ic_str, {})
        nc_dict[ic_str].setdefault(mp, {})
        nc_dict['init_var'] = vars_vn
        if ippe > 0:
            nc_dict[ic_str][mp].setdefault(ippe, {})

        # vn attributes (variable names)
        for vn in vars_vn:
            nc_dict[vn + '_units'] = ds0.getncattr(vn + '_units')
            keydst = nc_dict[ic_str][mp] if ippe == 0 else nc_dict[ic_str][mp][ippe]
            keydst[vn] = ds0.getncattr(vn)
        
        # coords
        if 'z' not in nc_dict: nc_dict['z'] = np.asarray(ds0['zh'][:]).copy()
        if 'x' not in nc_dict: nc_dict['x'] = np.asarray(ds0['xh'][:]).copy()
        if 'y' not in nc_dict: nc_dict['y'] = np.asarray(ds0['yh'][:]).copy()
        
        zf = np.asarray(ds0['zf'][:]).copy() * 1e3

        # optional BOSS attrs
        if 'BOSS' in mp:
            nc_dict['n_param_nevp']    = ds0.getncattr('boss_n_param_nevp')
            nc_dict['n_param_condevp'] = ds0.getncattr('boss_n_param_condevp')
            nc_dict['n_param_coal']    = ds0.getncattr('boss_n_param_coal')
            nc_dict['n_param_sed']     = ds0.getncattr('boss_n_param_sed')
            if bool(ds0.getncattr('boss_is_ppe')):
                nc_dict['is_perturbed_nevp']    = ds0.getncattr('boss_param_perturbed_nevp')
                nc_dict['is_perturbed_condevp'] = ds0.getncattr('boss_param_perturbed_condevp')
                nc_dict['is_perturbed_coal']    = ds0.getncattr('boss_param_perturbed_coal')
                nc_dict['is_perturbed_sed']     = ds0.getncattr('boss_param_perturbed_sed')

    # dt from time (guard single-file case)
    t = nc_dict['time']
    dt = float(t[1] - t[0]) if t.size >= 2 else np.nan

    # assume constant density since it's not an output of the model currently
    rho = 1.1 # kg/m^3
    # variables of interest
    for var_name in var_interest:
        dst = nc_dict[ic_str][mp] if ippe == 0 else nc_dict[ic_str][mp][ippe]
        dst.setdefault(var_name, {})
        dst[var_name]['value'] = var2phys(var_name, file_paths, dt, zf, rho)
        dst[var_name]['units'] = output_var_set[var_name]['var_unit']


def var2phys(var_name, file_paths, dt, zf, rho):
    val_timeseries = []
    dz = zf[1:] - zf[:-1]
    z = (zf[1:] + zf[:-1])/2
    # Handle ..._lastXXhrmean
    re_lasthrmean = re.search(r'_last(\d+)hrmean', var_name)
    re_lasthrstd  = re.search(r'_last(\d+)hrstd', var_name)
    is_lasthrmean = bool(re_lasthrmean)
    is_lasthrstd  = bool(re_lasthrstd)
    is_steadystate = max(is_lasthrmean, is_lasthrstd)
    if is_steadystate:
        if is_lasthrmean: n_last_hr = int(re_lasthrmean.group(1)) + 1
        if is_lasthrstd: n_last_hr = int(re_lasthrstd.group(1)) + 1

        n_needed  = int(np.ceil((n_last_hr * 3600) / dt)) if np.isfinite(dt) and dt > 0 else 1
        for fp in file_paths[-n_needed:]:
            with nc.Dataset(fp, 'r') as ds:
                vname   = output_var_set[var_name]['var_source']
                rawdata = ds.variables[vname][...]
                if 'path' in var_name:
                    dz_broadcast = dz[None, :, None, None]  # (1, z, 1, 1)
                    if is_lasthrmean: rawdata = np.sum(rawdata * dz_broadcast * rho, axis=(0, 1))  # sum over z, shape (time, y, x)
                elif 'per5lvl' in var_name:
                    rawdata = np.mean(rawdata[:, :66:5, :, :], axis=(0, 2, 3))
                elif '_10m_' in var_name:
                    rawdata = np.mean(rawdata[:, 1, :, :])
                elif '_250m_' in var_name:
                    rawdata = np.mean(rawdata[:, 20, :, :])
                elif '_500m_' in var_name:
                    rawdata = np.mean(rawdata[:, 30, :, :])
                elif '_750m_' in var_name:
                    rawdata = np.mean(rawdata[:, 44, :, :])

                if 'scale' in output_var_set[var_name]:
                    rawdata = rawdata * output_var_set[var_name]['scale']
                val_timeseries.append(np.asarray(rawdata))
        val_timeseries = np.stack(val_timeseries)
        val_timeavg = np.mean(val_timeseries, axis=0)
        if 'per5lvl' in var_name:
            out = val_timeavg
        else:
            if is_lasthrstd: out = np.std(np.mean(val_timeseries, axis=(1,2)))
            if is_lasthrmean: out = np.mean(val_timeavg)
        
        return out

    # General case: iterate all files, always close the handle
    for fp in file_paths:
        with nc.Dataset(fp, 'r') as ds:
            vname   = output_var_set[var_name]['var_source']
            rawdata = ds.variables[vname][...]
            if 'scale' in output_var_set[var_name]:
                rawdata = rawdata * output_var_set[var_name]['scale']
            if '_path' in var_name:
                dz_broadcast = dz[None, :, None, None]  # (1, z, 1, 1)
                val_timeseries.append(np.sum(rawdata * dz_broadcast * rho, axis=1))
            elif '_dmprof' in var_name:
                val_timeseries.append(np.mean(rawdata, axis=(0, 2, 3)))
            elif '_dmpath' in var_name:
                dz_broadcast = dz[None, :, None, None]  # (1, z, 1, 1)
                path = np.sum(rawdata * dz_broadcast * rho, axis=1)  # sum over z, shape (time, y, x)
                val_timeseries.append(np.mean(path, axis=(1, 2)))  # mean over (y, x), shape (time,)
            elif '_curtain_mean' in var_name:
                val_timeseries.append(np.mean(rawdata, axis=(0, 2)))
            elif '_curtain_slice' in var_name:
                # guard index 64
                yidx = rawdata.shape[2] // 2
                val_timeseries.append(rawdata[0, :, yidx, :])
            elif 'prate_dm' in var_name or 'precip_max_dm' or 'precip_onset':
                val_timeseries.append(np.mean(rawdata))
            else:
                val_timeseries.append(rawdata[0, ...])

    arr = np.array(val_timeseries)  # ensure shapes are consistent across files
    if 'precip_max_dm' in var_name:
        arr = np.max(arr)
    if 'precip_onset' in var_name:
        onset_idx = np.argmax(arr > max(arr)/1e4)
        with nc.Dataset(file_paths[onset_idx], 'r') as ds:
            arr = ds['time'][:]
    if '_runmean' in var_name:
        arr = np.mean(arr)
    return arr

def last_number_key(s):
    matches = re.findall(r'(\d+)(?!.*\d)', s)
    return int(matches[0]) if matches else 0
