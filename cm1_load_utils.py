import os
import re
import numpy as np
import netCDF4 as nc
from glob import glob
import platform
import socket
import load_ppe_fun as lp
from tqdm import tqdm
import sys
import warnings
from scipy.fft import fft2, ifft2, fftshift

M3toQ = np.pi/6*1e3
QtoM3 = 1/M3toQ

def calc_rho(ds):
    if all(var in ds.variables for var in ['prs', 'th', 'qv']):
        prs = ds.variables['prs'][...]
        th  = ds.variables['th'][...]
        qv  = ds.variables['qv'][...]
        return prs / (287.04 * th * (prs / 1e5)**(287.04 / 1004) * (1 + 0.61 * qv))
    else:
        return 1.15

def calc_lwp(ds, dz, rho=None):
    if rho is None:
        rho = calc_rho(ds)
    lwc = ds.variables['qc3'][...] * M3toQ
    dz_broadcast = dz[None, :, None, None]
    return np.sum(lwc * dz_broadcast * rho, axis=(0, 1))

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

# TODO: refactor this dictionary into a class ...
output_var_set = {
                  'M0_path': {'var_source': 'qc0', 'var_unit': '1/m$^2$', 'longname': 'LNP'}, 
                  'M3_path': {'var_source': 'qc3', 'var_unit': 'kg/m$^2$', 'scale': M3toQ, 'longname': 'LWP'}, 
                  'M4_path': {'var_source': 'qc4', 'var_unit': 'm$^4$/m$^2$', 'scale': 1e-4**4, 'longname': 'M4 Path'},
                  'M5_path': {'var_source': 'qc5', 'var_unit': 'm$^5$/m$^2$', 'scale': 1e-4**5, 'longname': 'M5 Path'},
                  'M6_path': {'var_source': 'qc6', 'var_unit': 'm$^6$/m$^2$', 'scale': 1e-4**6, 'longname': 'M6 Path'},
                  'M9_path': {'var_source': 'qc9', 'var_unit': 'm$^9$/m$^2$', 'scale': 1e-4**9, 'longname': 'M9 Path'},
                  'M0_dmprof': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Domain-Mean LNC'},
                  'M3_dmprof': {'var_source': 'qc3', 'var_unit': 'g/kg', 'scale': M3toQ*1e3, 'longname': 'Domain-Mean LWC'},
                  'M4_dmprof': {'var_source': 'qc4', 'var_unit': 'm$^4$/kg', 'scale': 1e-4**4, 'longname': 'Domain-Mean M4'},
                  'M5_dmprof': {'var_source': 'qc5', 'var_unit': 'm$^5$/kg', 'scale': 1e-4**5, 'longname': 'Domain-Mean M5'},
                  'M6_dmprof': {'var_source': 'qc6', 'var_unit': 'm$^6$/kg', 'scale': 1e-4**6, 'longname': 'Domain-Mean M6'},
                  'M9_dmprof': {'var_source': 'qc9', 'var_unit': 'm$^9$/kg', 'scale': 1e-4**9, 'longname': 'Domain-Mean M9'},
                  'M0_dmpath': {'var_source': 'qc0', 'var_unit': '1/m$^2$', 'longname': 'Domain-Mean LNP'}, 
                  'M3_dmpath': {'var_source': 'qc3', 'var_unit': 'kg/m$^2$', 'scale': M3toQ, 'longname': 'Domain-Mean LWP'}, 
                  'M4_dmpath': {'var_source': 'qc4', 'var_unit': 'm$^4$/m$^2$', 'scale': 1e-4**4, 'longname': 'Domain-Mean M4 Path'},
                  'M5_dmpath': {'var_source': 'qc5', 'var_unit': 'm$^5$/m$^2$', 'scale': 1e-4**5, 'longname': 'Domain-Mean M5 Path'},
                  'M6_dmpath': {'var_source': 'qc6', 'var_unit': 'm$^6$/m$^2$', 'scale': 1e-4**6, 'longname': 'Domain-Mean M6 Path'},
                  'M9_dmpath': {'var_source': 'qc9', 'var_unit': 'm$^9$/m$^2$', 'scale': 1e-4**9, 'longname': 'Domain-Mean M9 Path'},
                  'M0_curtain_slice': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNC'}, 
                  'M3_curtain_slice': {'var_source': 'qc3', 'var_unit': 'g/kg', 'scale': M3toQ*1e3, 'longname': 'LWC'}, 
                  'M4_curtain_slice': {'var_source': 'qc4', 'var_unit': 'm$^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_curtain_slice': {'var_source': 'qc5', 'var_unit': 'm$^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_curtain_slice': {'var_source': 'qc6', 'var_unit': 'm$^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_curtain_slice': {'var_source': 'qc9', 'var_unit': 'm$^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_curtain_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNC'}, 
                  'M3_curtain_mean': {'var_source': 'qc3', 'var_unit': 'g/kg', 'scale': M3toQ*1e3, 'longname': 'LWC'}, 
                  'M4_curtain_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_curtain_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_curtain_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_curtain_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_path_ss': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Steady State LNP'}, 
                  'M3_path_ss': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_path_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_path_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_path_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_path_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_dmpath_ss': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Steady State LNP'}, 
                  'M3_dmpath_ss': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_dmpath_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_dmpath_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_dmpath_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_dmpath_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_dspath_ss': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Steady State DS LNP'}, 
                  'M3_dspath_ss': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Steady State DS LWC'}, 
                  'M4_dspath_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Steady State DS M4'},
                  'M5_dspath_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Steady State DS M5'},
                  'M6_dspath_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Steady State DS M6'},
                  'M9_dspath_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Steady State DS M9'},
                  'M0_10m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_10m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_10m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_10m_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_10m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_10m_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_250m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_250m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_250m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_250m_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_250m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_250m_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'LNC': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNC'}, 
                  'LWC': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNP'}, 
                  'M3_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
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
                  'prate_dm': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Domain-Mean Rain Rate'},
                  'prate_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State Rain Rate'},
                  'prate_dm_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State Domain-Mean Rain Rate'},
                  'prate_ds_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State Domain-Std Rain Rate'},
                  'M6_99th_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6 99th percentile', 'lwc_threshold': 1e-5},
                  'M6_ds_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6 Standard Deviation', 'lwc_threshold': 1e-5},
                  'prate_10th_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State 10th percentile Rain Rate'},
                  'prate_90th_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State 90th percentile Rain Rate'},
                  'sedflux_m0': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Sedflux M0'},
                  'sedflux_m3': {'var_source': 'sedflux_M3', 'var_unit': 'mm/hr', 'scale': M3toQ*3600, 'longname': 'Rain flux'},
                  'sedflux_m4': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Sedflux M4'},
                  'sedflux_m6': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Sedflux M6'},
                  'sfM0_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0'},
                  'sfM3_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3'},
                  'sfM4_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4'},
                  'sfM6_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6'},
                  'sfM0_per5lvl': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 per 5 levels'},
                  'sfM0_per5lvl_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 per 5 levels'},
                  'M0_per5lvl': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_per5lvl': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_per5lvl': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_per5lvl': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_per5lvl': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_per5lvl': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_per5lvl_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_per5lvl_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_per5lvl_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_per5lvl_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_per5lvl_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_per5lvl_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'sfM0_dm_10m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 10m'},
                  'sfM3_dm_10m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 10m'},
                  'sfM4_dm_10m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 10m'},
                  'sfM6_dm_10m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 10m'},
                  'sfM0_dm_100m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 100m'},
                  'sfM3_dm_100m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 100m'},
                  'sfM4_dm_100m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 100m'},
                  'sfM6_dm_100m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 100m'},
                  'sfM0_dm_250m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 250m'},
                  'sfM3_dm_250m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 250m'},
                  'sfM4_dm_250m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 250m'},
                  'sfM6_dm_250m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 250m'},
                  'sfM0_dm_500m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 500m'},
                  'sfM3_dm_500m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 500m'},
                  'sfM4_dm_500m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 500m'},
                  'sfM6_dm_500m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 500m'},
                  'sfM0_dm_750m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 750m'},
                  'sfM3_dm_750m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 750m'},
                  'sfM4_dm_750m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 750m'},
                  'sfM6_dm_750m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 750m'},
                  'sfM0_10m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 10m'},
                  'sfM3_10m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 10m'},
                  'sfM4_10m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 10m'},
                  'sfM6_10m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 10m'},
                  'sfM0_100m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 100m'},
                  'sfM3_100m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 100m'},
                  'sfM4_100m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 100m'},
                  'sfM6_100m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 100m'},
                  'sfM0_250m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 250m'},
                  'sfM3_250m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 250m'},
                  'sfM4_250m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 250m'},
                  'sfM6_250m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 250m'},
                  'sfM0_500m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 500m'},
                  'sfM3_500m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 500m'},
                  'sfM4_500m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 500m'},
                  'sfM6_500m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 500m'},
                  'sfM0_750m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 750m'},
                  'sfM3_750m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 750m'},
                  'sfM4_750m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 750m'},
                  'sfM6_750m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 750m'},
                  'v_precip_onset':{'var_source': 't_precip_onset', 'var_unit': '1/hr', 'longname': 'Rain Onset Speed'},
                  't_precip_onset':{'var_source': 't_precip_onset', 'var_unit': 'hr', 'longname': 'Rain Onset Time'},
                  'precip_max_dm':{'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Peak Rain Rate'},
                  'meanD_dm_03_10m_ss':  {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 10m'},
                  'meanD_dm_03_100m_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 100m'},
                  'meanD_dm_03_250m_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 250m'},
                  'meanD_dm_03_500m_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 500m'},
                  'meanD_dm_36_10m_ss':  {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 10m'},
                  'meanD_dm_36_100m_ss': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 100m'},
                  'meanD_dm_36_250m_ss': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 250m'},
                  'meanD_dm_36_500m_ss': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 500m'},
                  'meanD_dm_03_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam'},
                  'tempK': {'var_source': ['th', 'prs'], 'var_unit': 'K', 'longname': 'Temperature (K)'},
                  'RH': {'var_source': ['th', 'prs', 'qv'], 'var_unit': '%', 'longname': 'Relative Humidity (%)'},
                  'pressure': {'var_source': 'prs', 'var_unit': 'Pa', 'longname': 'Pressure (Pa)'},
                  'u_10m': {'var_source': 'uinterp', 'var_unit': 'm/s', 'longname': '10m Wind Speed (m/s)'},
                  'v_10m': {'var_source': 'vinterp', 'var_unit': 'm/s', 'longname': '10m Wind Speed (m/s)'},
                  'v_hori': {'var_source': ['uinterp', 'vinterp'], 'var_unit': 'm/s', 'longname': 'Horizontal Wind Speed (m/s)'},
                  'w': {'var_source': 'winterp', 'var_unit': 'm/s', 'longname': 'Vertical Wind Speed (m/s)'},
                  'prate': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Rain Rate (mm/hr)'},
                  'reff': {'var_source': 'reff', 'var_unit': 'μm', 'scale': 1e6, 'longname': 'Effective radius (μm)'},
                  # LWP is already given as an input so no var_source is needed
                  'decorr_length_ss': {'var_source': [], 'var_unit': 'm', 'longname': 'Decorrelation Length (m)'}, 
                  }

def get_pert_idx(file_info):
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    mp = file_info['mp_config']
    fdir = file_info.get('dir', output_dir)
    l_pert = file_info.get('l_pert', False)
    
    if l_pert and 'vars_str' in file_info:
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        base_path_template = f"{fdir}{fdate}/{{config}}/{vars_dir}/{mp}"
    else:
        base_path_template = f"{fdir}{fdate}/{{config}}/{mp}"
    
    if isinstance(fsim_config, list):
        pert_idx_list = []
        global_id_counter = 0
        for config in fsim_config:
            search_dir = base_path_template.format(config=config)
            if not os.path.exists(search_dir):
                raise FileNotFoundError(f"Directory {search_dir} not found for config: {config}")
            member_dirs = os.listdir(search_dir)
            member_dirs = lp.sort_strings_by_number(member_dirs)
            for member in member_dirs:
                pert_idx_list.append({
                    'sim_config': config,
                    'member': member,
                    'global_id': global_id_counter
                })
                global_id_counter += 1
        if not pert_idx_list:
            raise ValueError(f"No perturbations found for sim_config={fsim_config}, mp_config={mp}, date={fdate} at base path.")
        return pert_idx_list
    else:
        # Legacy/Single config behavior: conform to new list-of-dicts structure for consistency
        search_dir = base_path_template.format(config=fsim_config)
        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"Directory {search_dir} not found")
        pert_idx = os.listdir(search_dir)
        pert_idx = lp.sort_strings_by_number(pert_idx)
        result = []
        for i, m in enumerate(pert_idx):
            global_id = int(m) if m.isdigit() else i
            result.append({
                'sim_config': fsim_config,
                'member': m,
                'global_id': global_id
            })
        if not result:
            raise ValueError(f"No perturbations found for sim_config={fsim_config}, mp_config={mp}, date={fdate} at base path {search_dir}.")
        return result

def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1.
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def load_cm1(file_info, var_interest, ss_hrs, nc_dict=None, continuous_ic=True, ipert=0, lwp_threshold=0.02, pbar=None):
    if nc_dict is None:
        nc_dict = {}
        
    # Unpack ipert if it's a dictionary (new structure)
    if isinstance(ipert, dict):
        current_config = ipert['sim_config']
        member = ipert['member']
        global_id = ipert['global_id']
        is_dict_ipert = True
    else:
        current_config = file_info['sim_config']
        member = str(ipert)
        global_id = ipert
        is_dict_ipert = False

    mp          = file_info['mp_config']
    vars_vn     = file_info['vars_vn']
    fdir        = file_info['dir']
    fdate       = file_info['date']
    l_pert      = file_info.get('l_pert', False)
    fsim_config = current_config 
    
    fn_prefix, fn_suffix = "cm1out_0", ".nc"

    if continuous_ic:
        file_pattern = f"{fdir}{fdate}/{fsim_config}/{mp}/{member}/{fn_prefix}*{fn_suffix}"
        ic_str = 'cic'
    else:
        ic_str = "".join(file_info['vars_str'])
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        if l_pert:
            file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{member}/{fn_prefix}*{fn_suffix}"
        else:
            file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{fn_prefix}*{fn_suffix}"

    file_paths = sorted(glob(file_pattern), key=last_number_key)
    if not file_paths:
        raise FileNotFoundError(f"No files match: {file_pattern}")

    # Get dt for n_needed
    if len(file_paths) >= 2:
        with nc.Dataset(file_paths[0], 'r') as ds_a, nc.Dataset(file_paths[1], 'r') as ds_b:
            t0 = ds_a.variables['time'][0]
            t1 = ds_b.variables['time'][0]
            dt = float(t1 - t0)
    else:
        dt = np.nan

    # open first file to grab coords/attrs, then close it
    with nc.Dataset(file_paths[0], 'r') as ds0:
        nc_dict.setdefault(fsim_config, {})
        nc_dict[fsim_config].setdefault(mp, {})
        nc_dict[fsim_config][mp].setdefault(ic_str, {})
        nc_dict['init_var'] = vars_vn
        if continuous_ic or l_pert:
            nc_dict[fsim_config][mp][ic_str].setdefault(global_id, {})

        # time vector initialization
        if 'time' not in nc_dict[fsim_config]:
            nc_dict[fsim_config]['time'] = np.empty(len(file_paths), dtype=float)

        # vn attributes (variable names)
        for vn in vars_vn:
            nc_dict[vn + '_units'] = ds0.getncattr(vn + '_units')
            keydst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
            keydst[vn] = ds0.getncattr(vn)
        
        # coords
        if 'z' not in nc_dict[fsim_config]: nc_dict[fsim_config]['z'] = np.round(ds0['zh'][:] * 1e3, decimals=1)
        if 'x' not in nc_dict[fsim_config]: nc_dict[fsim_config]['x'] = np.round(ds0['xh'][:] * 1e3, decimals=1)
        if 'y' not in nc_dict[fsim_config]: nc_dict[fsim_config]['y'] = np.round(ds0['yh'][:] * 1e3, decimals=1)
        
        zf = np.asarray(ds0['zf'][:]).copy() * 1e3

        # optional BOSS attrs
        if 'SLC-BOSS' in mp:
            nc_dict['n_param_nevp']    = ds0.getncattr('boss_n_param_nevp')
            nc_dict['n_param_condevp'] = ds0.getncattr('boss_n_param_condevp')
            nc_dict['n_param_coal']    = ds0.getncattr('boss_n_param_coal')
            nc_dict['n_param_sed']     = ds0.getncattr('boss_n_param_sed')
            if bool(ds0.getncattr('boss_is_ppe')):
                nc_dict['is_perturbed_nevp']    = ds0.getncattr('boss_param_perturbed_nevp')
                nc_dict['is_perturbed_condevp'] = ds0.getncattr('boss_param_perturbed_condevp')
                nc_dict['is_perturbed_coal']    = ds0.getncattr('boss_param_perturbed_coal')
                nc_dict['is_perturbed_sed']     = ds0.getncattr('boss_param_perturbed_sed')

    dz = zf[1:] - zf[:-1]
    z = (zf[1:] + zf[:-1])/2
    dx = nc_dict[fsim_config]['x'][1] - nc_dict[fsim_config]['x'][0]

    n_needed = int(np.ceil((ss_hrs * 3600) / dt) + 1) if np.isfinite(dt) and dt > 0 else 1

    # Pre-parse meta and setup collectors
    var_meta = {vn: parse_var_meta(vn) for vn in var_interest}
    raw_collector = {vn: [] for vn in var_interest}
    lwp_pcts = np.zeros(len(file_paths))
    
    # Main single-pass loop
    for ifp, fp in enumerate(file_paths):
        with nc.Dataset(fp, 'r') as ds:
            # Time tracking
            t_val = np.asarray(ds.variables['time'][:]).item()
            nc_dict[fsim_config]['time'][ifp] = t_val
            
            # Physics helpers
            rho = calc_rho(ds)
            lwp = calc_lwp(ds, dz, rho=rho)
            lwp_pcts[ifp] = np.mean(lwp > lwp_threshold) * 100

            for vn in var_interest:
                meta = var_meta[vn]
                
                # Normal variable extraction
                is_ss_file = (ifp >= len(file_paths) - n_needed)
                if not meta['is_ss'] or is_ss_file:
                    val = extract_and_reduce(vn, ds, rho, lwp, dz, z, dx, lwp_threshold)
                    raw_collector[vn].append(val)

    if pbar is not None:
        mean_pct = np.mean(lwp_pcts)
        pbar.set_postfix(lwp_pct=f"{mean_pct:.2f}%")

    # Final aggregation and assignment
    for vn in var_interest:
        dst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
        dst.setdefault(vn, {})
        
        dst[vn]['value'] = aggregate_timeseries(vn, raw_collector[vn], var_meta[vn])
        dst[vn]['units'] = output_var_set[vn]['var_unit']

    return nc_dict

def parse_var_meta(var_name):
    # is percentile
    re_prc  = re.search(r'(\d+)th', var_name)
    is_prc  = bool(re_prc)
    nth_prctl  = int(re_prc.group(1)) if is_prc else None

    # is domain mean/std (spatial mean/std)
    is_dm   = bool(re.search(r'_dm', var_name))
    is_ds   = bool(re.search(r'_ds', var_name))

    # is steady state (temporal mean of the last x hr)
    is_ss   = bool(re.search(r'ss', var_name))

    return {'is_prc': is_prc, 'nth_prctl': nth_prctl, 'is_dm': is_dm, 'is_ds': is_ds, 'is_ss': is_ss}

def extract_and_reduce(var_name, ds, rho, lwp, dz, z, dx, lwp_threshold):
    vsource = output_var_set[var_name]['var_source']
    scale = output_var_set[var_name].get('scale', 1.0)
    lwc_thresh = output_var_set[var_name].get('lwc_threshold')
    
    def get_masked_data(vn):
        data = ds.variables[vn][...]
        if lwc_thresh is not None:
            # Mask based on qc3 (LWC)
            lwc = ds.variables['qc3'][:] * M3toQ
            mask = lwc <= lwc_thresh
            data[mask] = np.nan
        else:
            # Mask based on column LWP
            mask = lwp <= lwp_threshold
            if data.ndim >= 2:
                data[..., mask] = np.nan
        return data

    if isinstance(vsource, list):
        data = [get_masked_data(vn) for vn in vsource]
        raw_data = [ds.variables[vn][...] for vn in vsource]
    else:
        data = get_masked_data(vsource)
        raw_data = ds.variables[vsource][...]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Reduction
        if 'prof' in var_name:
            res = np.nanmean(data, axis=(0, 2, 3)) * scale
        elif 'path' in var_name:
            dz_b = dz[None, :, None, None]
            path = np.nansum(data * dz_b * rho, axis=1) # (time, y, x)
            # Ensure columns that are all NaN stay NaN instead of 0.0
            if np.all(np.isnan(data), axis=1).any():
                path[np.all(np.isnan(data), axis=1)] = np.nan
            res = path * scale # (time,)
        elif 'per5lvl' in var_name:
            res = np.nanmean(data[:, :55:5, :, :], axis=(0, 2, 3)) * scale
        elif 'per10lvl' in var_name:
            res = np.nanmean(data[:, :55:10, :, :], axis=(0, 2, 3)) * scale
        elif 'meanD_dm' in var_name:
            # Tuple for aggregate ratio calculation
            h_match = re.search(r'_(\d+(?:\.\d+)?)m', var_name)
            if bool(h_match):
                target_h = float(h_match.group(1))
                h_idx = np.argmin(np.abs(z - target_h))
                res = (np.nanmean(data[1][:, h_idx, :, :]), np.nanmean(data[0][:, h_idx, :, :]))
            else:
                res = (np.nanmean(data[1]), np.nanmean(data[0]))
        elif re.search(r'_(\d+(?:\.\d+)?)m', var_name):
            h_match = re.search(r'_(\d+(?:\.\d+)?)m', var_name)
            target_h = float(h_match.group(1))
            h_idx = np.argmin(np.abs(z - target_h))
            res = data[:, h_idx, :, :] * scale
        elif '_curtain_mean' in var_name:
            res = np.nanmean(data, axis=(0, 2)) * scale
        elif '_curtain_slice' in var_name:
            yidx = data.shape[2] // 2
            res = data[0, :, yidx, :] * scale
        # elif 'prate_dm' in var_name or 'precip_max_dm' in var_name:
        #     res = np.nanmean(data) * scale
        elif 'tempK' in var_name:
            res = get_tempK_from_theta_p(raw_data[0], raw_data[1])
        elif 'v_hori' in var_name:
            res = np.sqrt(raw_data[0]**2 + raw_data[1]**2)
        elif 'RH' in var_name:
            theta = raw_data[0]
            press = raw_data[1]
            qv    = raw_data[2]
            tempK = get_tempK_from_theta_p(theta, press)
            eps = 0.622
            vap_prs = (qv * press) / (eps + qv * (1 - eps))
            vap_prs_sat = saturation_vapor_pressure_liquid(tempK)
            res = vap_prs / vap_prs_sat * 100
        elif 'v_precip_onset' in var_name:
            res = 3600./data
        elif 't_precip_onset' in var_name:
            res = data/3600.
        elif 'decorr_length' in var_name:
            lags, radial_R = get_spatial_autocorrelation(lwp, dx)
            threshold = np.exp(-1)
            idx = np.argmax(radial_R < threshold)
            if idx == 0 and radial_R[0] > threshold:
                # It never dropped below 1/e (e.g., highly uniform stratus)
                res = lags[-1]
            else:
                res = lags[idx]
        else:
            res = raw_data[0, ...] * scale

    return res

def get_spatial_autocorrelation(lwp_field, dx):
    """
    Calculates the 1D radial spatial autocorrelation of a 2D LWP field.
    
    Args:
        lwp_field: 2D numpy array of Liquid Water Path
        dx: Grid spacing in meters (assumes dx = dy)
        
    Returns:
        lags: 1D array of distance lags in meters
        radial_R: 1D array of autocorrelation values
    """
    # 1. Anomaly and Variance
    lwp_anom = lwp_field - np.mean(lwp_field)
    var = np.var(lwp_field)
    
    if var == 0: # Handle clear sky edge-cases
        return np.array([0]), np.array([1.0])

    # 2. FFT to get Autocovariance (LES is periodic, no padding needed)
    F = fft2(lwp_anom)
    psd = np.abs(F)**2
    autocov = np.real(ifft2(psd)) / lwp_field.size
    
    # 3. Normalize and shift zero-lag to the center of the array
    autocorr_2d = fftshift(autocov) / var
    
    # 4. Radial Averaging
    ny, nx = lwp_field.shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    
    # Calculate distance of each pixel from the center (in grid units)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int) # Bin distances to nearest integer
    
    # Average the autocorrelation values within each radial bin
    tbin = np.bincount(r.ravel(), autocorr_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_R = tbin / nr
    
    # Convert lags from grid units to physical distance
    lags = np.arange(len(radial_R)) * dx
    
    # Keep only the first half of the domain (Nyquist/periodic bounds)
    max_lag_idx = min(nx, ny) // 2
    
    return lags[:max_lag_idx], radial_R[:max_lag_idx]

def aggregate_timeseries(var_name, ts, meta):
    if not ts: return np.nan
    
    if 'meanD_dm' in var_name:
        num = np.nanmean([x[0] for x in ts])
        den = np.nanmean([x[1] for x in ts])
        res = (num / den)**(1/3) * 1e6 if den > 0 else np.nan
        return res
    
    arr = np.squeeze(np.stack(ts))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if meta['is_ds']:
            # is domain std
            arr = np.nanstd(arr, axis=(1, 2))
        
        if meta['is_dm']:
            # is domain mean
            arr = np.nanmean(arr, axis=(1,2))
        
        if meta['is_prc']:
            # is percentile
            valid = arr[~np.isnan(arr) & (arr > 0)]
            if valid.size == 0:
                arr = np.nan
            else:
                arr = np.percentile(valid, meta['nth_prctl'])

        # handle special cases
        if 'v_precip_onset' in var_name:
            return arr.max()
        if 't_precip_onset' in var_name:
            return arr.min()

        # Temporal average
        if meta['is_ss']:
            res = np.nanmean(arr, axis=0)
        else:
            res = arr

        if 'precip_max_dm' in var_name:
            return np.nanmax(res)

        if '_runmean' in var_name:
            return np.nanmean(res)
            
        if 'dmpath' in var_name:
            return res
        else:
            return np.nan_to_num(res, nan=0.0) if not isinstance(res, tuple) else res

def last_number_key(s):
    matches = re.findall(r'(\d+)(?!.*\d)', s)
    return int(matches[0]) if matches else 0

def get_tempK_from_theta_p(theta, p):
    
    """
    where Rd is the gas constant for dry air and cp is the specific heat capacity at constant pressure.
    Rd = 287.058 # J/(kg K)
    cp = 1005.7 # J/(kg K)
    return theta * (p/1000)**(Rd/cp)
    """
    Rd = 287.058 # J/(kg K)
    cp = 1005.7 # J/(kg K)
    T = theta * (p/100000)**(Rd/cp)
    return T

def saturation_vapor_pressure_liquid(T):
    """
    Calculates the saturation vapor pressure (es) with respect to liquid water.
    This function is a translation of the polysvp2 Fortran function 
    for i_type=0 (liquid saturation). It uses a polynomial fit (Flatau et al. 1992)
    for T >= 202.0 K and a modified Goff-Gratch equation for T < 202.0 K.

    Args:
        T (array_like): Absolute temperature (K).

    Returns:
        array_like: Saturation vapor pressure (Pa).
    """
    
    # Ensure T is a NumPy array for vectorized operations
    T = np.asarray(T)
    
    # Constants for saturation over liquid (a0, a1, ..., a8) from the Fortran code
    A = np.array([
        6.11239921, 
        0.443987641, 
        0.142986287e-1, 
        0.264847430e-3, 
        0.302950461e-5, 
        0.206739458e-7, 
        0.640689451e-10, 
        -0.952447341e-13, 
        -0.976195544e-15
    ])
    
    T_CELSIUS = T - 273.15  # Temperature in Celsius (dt in the Fortran code)
    
    # Initialize the output array
    es = np.zeros_like(T, dtype=float)
    
    # --- Part 1: Polynomial (Flatau et al. 1992) for T >= 202.0 K ---
    
    # Create a mask for the temperature range
    mask_poly = (T >= 202.0)
    
    if np.any(mask_poly):
        T_C_poly = T_CELSIUS[mask_poly]
        
        # Horner's method for polynomial evaluation (efficiently handles the nested multiplication)
        # P(x) = a0 + x*(a1 + x*(a2 + ... + x*(a7 + x*a8)...))
        poly_val = (A[8] * T_C_poly + A[7]) * T_C_poly + A[6]
        poly_val = (poly_val * T_C_poly + A[5]) * T_C_poly + A[4]
        poly_val = (poly_val * T_C_poly + A[3]) * T_C_poly + A[2]
        poly_val = (poly_val * T_C_poly + A[1]) * T_C_poly + A[0]

        # The result is in hPa, so multiply by 100 to get Pa
        es[mask_poly] = poly_val * 100.0

    # --- Part 2: Modified Goff-Gratch for T < 202.0 K ---
    
    mask_gg = (T < 202.0)
    
    if np.any(mask_gg):
        T_gg = T[mask_gg]
        
        # Fortran: polysvp2 = 10.**(-7.90298*(373.16/t-1.) + ... + alog10(1013.246)) * 100.
        # Python: uses np.log10 for alog10 and ** for exponentiation
        
        # Constants from the Goff-Gratch part of the Fortran code
        T_STAR = 373.16  # Boiling point of water (K)
        P_STAR = 1013.246 # Standard pressure (hPa)
        
        exponent = (
            -7.90298 * (T_STAR / T_gg - 1.0)
            + 5.02808 * np.log10(T_STAR / T_gg)
            - 1.3816e-7 * (10.0**(11.344 * (1.0 - T_gg / T_STAR)) - 1.0)
            + 8.1328e-3 * (10.0**(-3.49149 * (T_STAR / T_gg - 1.0)) - 1.0)
            + np.log10(P_STAR)
        )
        
        # The result is 10^exponent in hPa, so multiply by 100 to get Pa
        es[mask_gg] = 10.0**exponent * 100.0

    return es
