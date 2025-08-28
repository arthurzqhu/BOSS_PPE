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
                  }

def load_cm1(file_info, var_interest, nc_dict, continuous_ic):
    mp = file_info['mp_config']
    vars_vn = file_info['vars_vn']
    fdir = file_info['dir']
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    fn_prefix = "cm1out_0"
    fn_suffix = ".nc"

    if continuous_ic:
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{mp}/{fn_prefix}*{fn_suffix}"
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

    if 'z' not in nc_dict.keys():
        nc_dict['z'] = nc.Dataset(file_paths[0])['zh'][:]

    if 'x' not in nc_dict.keys():
        nc_dict['x'] = nc.Dataset(file_paths[0])['xh'][:]

    if 'y' not in nc_dict.keys():
        nc_dict['y'] = nc.Dataset(file_paths[0])['yh'][:]
    
    nc_dict.setdefault(ic_str, {})
    nc_dict[ic_str].setdefault(mp, {})

    for var_name in tqdm(var_interest, desc=f'Loading {ic_str} {mp} variables...'):
        nc_dict[ic_str][mp].setdefault(var_name, {})
        nc_dict[ic_str][mp][var_name]['value'] = var2phys(var_name, file_paths)
        nc_dict[ic_str][mp][var_name]['units'] = output_var_set[var_name]['var_unit']

def var2phys(var_name, file_paths):
    var_data = []
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
        elif 'dmprof' in var_name:
            var_data.append(np.mean(rawdata, axis=(0,2,3)))
        elif '_dmpath' in var_name:
            var_data.append(np.mean(np.sum(rawdata, axis=(1)), axis=(1,2)))
        elif '_curtain_mean' in var_name:
            var_data.append(np.mean(rawdata, axis=(0,2)))
        elif '_curtain_slice' in var_name:
            var_data.append(rawdata[0, :, 64, :])
        else:
            var_data.append(rawdata[0, ...])
    return np.array(var_data)

def last_number_key(s):
    matches = re.findall(r'(\d+)(?!.*\d)', s)
    return int(matches[0]) if matches else 0