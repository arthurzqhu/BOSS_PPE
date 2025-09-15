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
                  'M0_last2hrmean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Last 2hr Mean LNP'}, 
                  'M3_last2hrmean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Last 2hr Mean LWC'}, 
                  'M4_last2hrmean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Last 2hr Mean M4'},
                  'M5_last2hrmean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Last 2hr Mean M5'},
                  'M6_last2hrmean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Last 2hr Mean M6'},
                  'M9_last2hrmean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Last 2hr Mean M9'},
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
                  'prate_dm_last2hrmean': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Last 2hr Mean Domain-Mean Precipitation Rate'},
                  }

def get_ppe_idx(file_info):
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    mp = file_info['mp_config']
    ppe_idx = os.listdir(f"{output_dir}{fdate}/{fsim_config}/{mp}")
    ppe_idx = lp.sort_strings_by_number(ppe_idx)
    return ppe_idx

def load_cm1(file_info, var_interest, nc_dict, continuous_ic, ippe=0):
    mp = file_info['mp_config']
    vars_vn = file_info['vars_vn']
    fdir = file_info['dir']
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    fn_prefix = "cm1out_0"
    fn_suffix = ".nc"

    if continuous_ic:
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{mp}/{ippe}/{fn_prefix}*{fn_suffix}"
        ic_str = 'cic' # = continuous initial condition
    else:
        ic_str = "".join(file_info['vars_str'])
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{fn_prefix}*{fn_suffix}"

    try:
        file_paths = glob(file_pattern)
        file_paths = sorted(file_paths, key=last_number_key)
    except IndexError:
        print('perhaps no such file at:' + file_pattern)

    if 'time' not in nc_dict.keys():
        time_len = len(file_paths)
        time_array = np.zeros(time_len)
        for i_file, file_path in enumerate(file_paths):
            time_array[i_file] = nc.Dataset(file_path)['time'][:]
        nc_dict['time'] = time_array

    dataset = nc.Dataset(file_paths[0])
    # no need to do two tries anymore

    nc_dict.setdefault(ic_str, {})
    nc_dict[ic_str].setdefault(mp, {})
    if ippe > 0:
        nc_dict[ic_str][mp].setdefault(ippe, {})

    for vn in vars_vn:
        if ippe == 0:
            try:
                nc_dict[vn + '_units'] = dataset.getncattr('nanew2_units')
                nc_dict[ic_str][mp][vn] = dataset.getncattr('nanew2')
            except:
                nc_dict[vn + '_units'] = dataset.getncattr('na_units')            
                nc_dict[ic_str][mp][vn] = dataset.getncattr('na')
        else:
            nc_dict[vn + '_units'] = dataset.getncattr('na_units')            
            nc_dict[ic_str][mp][ippe][vn] = dataset.getncattr('na')

    dt = nc_dict['time'][1] - nc_dict['time'][0]

    if 'z' not in nc_dict.keys():
        nc_dict['z'] = dataset['zh'][:]

    if 'x' not in nc_dict.keys():
        nc_dict['x'] = dataset['xh'][:]

    if 'y' not in nc_dict.keys():
        nc_dict['y'] = dataset['yh'][:]
    
    if 'BOSS' in mp:
        nc_dict['n_param_nevp'] = dataset.getncattr('boss_n_param_nevp')
        nc_dict['n_param_condevp'] = dataset.getncattr('boss_n_param_condevp')
        nc_dict['n_param_coal'] = dataset.getncattr('boss_n_param_coal')
        nc_dict['n_param_sed'] = dataset.getncattr('boss_n_param_sed')
        
        is_ppe = bool(dataset.getncattr('boss_is_ppe'))
        if is_ppe:
            nc_dict['is_perturbed_nevp'] = dataset.getncattr('boss_param_perturbed_nevp')
            nc_dict['is_perturbed_condevp'] = dataset.getncattr('boss_param_perturbed_condevp')
            nc_dict['is_perturbed_coal'] = dataset.getncattr('boss_param_perturbed_coal')
            nc_dict['is_perturbed_sed'] = dataset.getncattr('boss_param_perturbed_sed')

    for var_name in var_interest:
        if ippe == 0:
            nc_dict[ic_str][mp].setdefault(var_name, {})
            nc_dict[ic_str][mp][var_name]['value'] = var2phys(var_name, file_paths, dt)
            nc_dict[ic_str][mp][var_name]['units'] = output_var_set[var_name]['var_unit']
        else:
            nc_dict[ic_str][mp][ippe].setdefault(var_name, {})
            nc_dict[ic_str][mp][ippe][var_name]['value'] = var2phys(var_name, file_paths, dt)
            nc_dict[ic_str][mp][ippe][var_name]['units'] = output_var_set[var_name]['var_unit']

def var2phys(var_name, file_paths, dt):
    var_data = []
    if re.search(r'_last\d+hrmean', var_name):
        n_last_hr = int(re.search(r'_last(\d+)hrmean', var_name).group(1)) + 1
        fp_read = file_paths[-n_last_hr*int(3600/dt):]
        for filepath in fp_read:
            dataset = nc.Dataset(filepath)
            rawdata = dataset.variables[output_var_set[var_name]['var_source']][:]
            if 'scale' in output_var_set[var_name].keys():
                rawdata *= output_var_set[var_name]['scale']
            var_data.append(rawdata)
        var_data = np.array(var_data)
        var_data = np.mean(var_data[-n_last_hr*int(3600/dt):,...])
    else:
        for filepath in file_paths:
            dataset = nc.Dataset(filepath)
            rawdata = dataset.variables[output_var_set[var_name]['var_source']][:]
            # if 'qc6' in output_var_set[var_name]['var_source']:
            #     rawdata = rawdata*10**9.5
            # if 'qc9' in output_var_set[var_name]['var_source']:
            #     rawdata = rawdata*10**9.5
            # 4d variable in [time, z, y, x]
            if 'scale' in output_var_set[var_name].keys():
                rawdata *= output_var_set[var_name]['scale']
            if '_path' in var_name:
                var_data.append(np.sum(rawdata, axis=1))
            elif '_dmprof' in var_name:
                var_data.append(np.mean(rawdata, axis=(0,2,3)))
            elif '_dmpath' in var_name:
                var_data.append(np.mean(np.sum(rawdata, axis=(1)), axis=(1,2)))
            elif '_curtain_mean' in var_name:
                var_data.append(np.mean(rawdata, axis=(0,2)))
            elif '_curtain_slice' in var_name:
                var_data.append(rawdata[0, :, 64, :])
            else:
                var_data.append(rawdata[0, ...])
        var_data = np.array(var_data)

    if '_runmean' in var_name:
        var_data = np.mean(var_data)

    return var_data

def last_number_key(s):
    matches = re.findall(r'(\d+)(?!.*\d)', s)
    return int(matches[0]) if matches else 0