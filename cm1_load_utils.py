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

M3toQ = np.pi/6*1e3
QtoM3 = 1/M3toQ

def calc_rho(ds):
    prs = ds.variables['prs'][...]
    th  = ds.variables['th'][...]
    qv  = ds.variables['qv'][...]
    return prs / (287.04 * th * (prs / 1e5)**(287.04 / 1004) * (1 + 0.61 * qv))

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
                  'M0_path_ss_mean': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Steady State LNP'}, 
                  'M3_path_ss_mean': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_path_ss_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_path_ss_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_path_ss_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_path_ss_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_dmpath_ss_mean': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Steady State LNP'}, 
                  'M3_dmpath_ss_mean': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_dmpath_ss_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_dmpath_ss_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_dmpath_ss_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_dmpath_ss_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_path_ss_std': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'Steady State DS LNP'}, 
                  'M3_path_ss_std': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'Steady State DS LWC'}, 
                  'M4_path_ss_std': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'Steady State DS M4'},
                  'M5_path_ss_std': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'Steady State DS M5'},
                  'M6_path_ss_std': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'Steady State DS M6'},
                  'M9_path_ss_std': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'Steady State DS M9'},
                  'M0_10m_ss_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_10m_ss_mean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_10m_ss_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_10m_ss_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_10m_ss_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_10m_ss_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_250m_ss_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_250m_ss_mean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_250m_ss_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_250m_ss_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_250m_ss_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_250m_ss_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNP'}, 
                  'M3': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_ss_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNP'}, 
                  'M3_ss_mean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4_ss_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_ss_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_ss_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_ss_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
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
                  'prate_ss_mean': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State Rain Rate'},
                  'prate_dm_ss_mean': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State Domain-Mean Rain Rate'},
                  'prate_ss_std': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State Domain-Std Rain Rate'},
                  'M6_ss_99th_prctl': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6 99th percentile', 'lwc_threshold': 1e-5},
                  'M6_ss_std': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6 Standard Deviation', 'lwc_threshold': 1e-5},
                  'prate_ss_10th_prct': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State 10th percentile Rain Rate'},
                  'prate_ss_90th_prct': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Steady State 90th percentile Rain Rate'},
                  'sedflux_m0': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Sedflux M0'},
                  'sedflux_m3': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Sedflux M3'},
                  'sedflux_m4': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Sedflux M4'},
                  'sedflux_m6': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Sedflux M6'},
                  'sfM0_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0'},
                  'sfM3_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3'},
                  'sfM4_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4'},
                  'sfM6_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6'},
                  'sfM0_per5lvl': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 per 5 levels'},
                  'sfM0_per5lvl_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 per 5 levels'},
                  'M0_per5lvl': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_per5lvl': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_per5lvl': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_per5lvl': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_per5lvl': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_per5lvl': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'M0_per5lvl_ss_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'Steady State LNP'}, 
                  'M3_per5lvl_ss_mean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Steady State LWC'}, 
                  'M4_per5lvl_ss_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'Steady State M4'},
                  'M5_per5lvl_ss_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'Steady State M5'},
                  'M6_per5lvl_ss_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'Steady State M6'},
                  'M9_per5lvl_ss_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'Steady State M9'},
                  'sfM0_dm_10m_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 10m'},
                  'sfM3_dm_10m_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 10m'},
                  'sfM4_dm_10m_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 10m'},
                  'sfM6_dm_10m_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 10m'},
                  'sfM0_dm_100m_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 100m'},
                  'sfM3_dm_100m_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 100m'},
                  'sfM4_dm_100m_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 100m'},
                  'sfM6_dm_100m_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 100m'},
                  'sfM0_dm_250m_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 250m'},
                  'sfM3_dm_250m_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 250m'},
                  'sfM4_dm_250m_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 250m'},
                  'sfM6_dm_250m_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 250m'},
                  'sfM0_dm_500m_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 500m'},
                  'sfM3_dm_500m_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 500m'},
                  'sfM4_dm_500m_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 500m'},
                  'sfM6_dm_500m_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 500m'},
                  'sfM0_dm_750m_ss_mean': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 750m'},
                  'sfM3_dm_750m_ss_mean': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 750m'},
                  'sfM4_dm_750m_ss_mean': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 750m'},
                  'sfM6_dm_750m_ss_mean': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 750m'},
                  'sfM0_10m_ss_std': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 10m'},
                  'sfM3_10m_ss_std': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 10m'},
                  'sfM4_10m_ss_std': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 10m'},
                  'sfM6_10m_ss_std': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 10m'},
                  'sfM0_100m_ss_std': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 100m'},
                  'sfM3_100m_ss_std': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 100m'},
                  'sfM4_100m_ss_std': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 100m'},
                  'sfM6_100m_ss_std': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 100m'},
                  'sfM0_250m_ss_std': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 250m'},
                  'sfM3_250m_ss_std': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 250m'},
                  'sfM4_250m_ss_std': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 250m'},
                  'sfM6_250m_ss_std': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 250m'},
                  'sfM0_500m_ss_std': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 500m'},
                  'sfM3_500m_ss_std': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 500m'},
                  'sfM4_500m_ss_std': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 500m'},
                  'sfM6_500m_ss_std': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 500m'},
                  'sfM0_750m_ss_std': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Steady State Sedflux M0 750m'},
                  'sfM3_750m_ss_std': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Steady State Sedflux M3 750m'},
                  'sfM4_750m_ss_std': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Steady State Sedflux M4 750m'},
                  'sfM6_750m_ss_std': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Steady State Sedflux M6 750m'},
                  'v_precip_onset':{'var_source': 'prate', 'var_unit': '1/hr', 'longname': 'Rain Onset Speed'},
                  'precip_max_dm':{'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Peak Rain Rate'},
                  'meanD_dm_03_10m_ss_mean':  {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 10m'},
                  'meanD_dm_03_100m_ss_mean': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 100m'},
                  'meanD_dm_03_250m_ss_mean': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 250m'},
                  'meanD_dm_03_500m_ss_mean': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam 500m'},
                  'meanD_dm_36_10m_ss_mean':  {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 10m'},
                  'meanD_dm_36_100m_ss_mean': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 100m'},
                  'meanD_dm_36_250m_ss_mean': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 250m'},
                  'meanD_dm_36_500m_ss_mean': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'Steady State M6-meandiam 500m'},
                  'meanD_dm_03_ss_mean': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'Steady State mass-meandiam'},
                  }

def get_ppe_idx(file_info):
    fdate = file_info['date']
    fsim_config = file_info['sim_config']
    mp = file_info['mp_config']
    ppe_idx = os.listdir(f"{output_dir}{fdate}/{fsim_config}/{mp}")
    ppe_idx = lp.sort_strings_by_number(ppe_idx)
    return ppe_idx

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

def load_cm1(file_info, var_interest, nc_dict=None, continuous_ic=True, ss_hrs=2, ippe=0, lwp_threshold=0.01):
    import netCDF4 as nc
    if nc_dict is None:
        nc_dict = {}
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

    # time vector initialization
    if 'time' not in nc_dict:
        nc_dict['time'] = np.empty(len(file_paths), dtype=float)

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

    dz = zf[1:] - zf[:-1]
    z = (zf[1:] + zf[:-1])/2

    n_needed = int(np.ceil((ss_hrs * 3600) / dt)) if np.isfinite(dt) and dt > 0 else 1

    # Pre-parse meta and setup collectors
    var_meta = {vn: parse_var_meta(vn) for vn in var_interest}
    raw_collector = {vn: [] for vn in var_interest}
    
    # Special trackers for v_precip_onset
    precip_onset_threshold = 1e-4 / 3600
    onset_t_tmp = {vn: 28800. for vn in var_interest if 'v_precip_onset' in vn}
    onset_found_first = {vn: False for vn in var_interest if 'v_precip_onset' in vn}
    onset_finished = {vn: False for vn in var_interest if 'v_precip_onset' in vn}

    # Main single-pass loop
    for ifp, fp in enumerate(file_paths):
        with nc.Dataset(fp, 'r') as ds:
            # Time tracking
            t_val = np.asarray(ds.variables['time'][:]).item()
            nc_dict['time'][ifp] = t_val
            
            # Physics helpers
            rho = calc_rho(ds)
            lwp = calc_lwp(ds, dz, rho=rho)
            
            for vn in var_interest:
                meta = var_meta[vn]
                
                # Precipitation onset logic
                if 'v_precip_onset' in vn and not onset_finished[vn]:
                    prate = ds.variables[output_var_set[vn]['var_source']][:]
                    prate[..., lwp <= lwp_threshold] = np.nan
                    mean_prate = np.nanmean(prate)
                    if mean_prate > precip_onset_threshold:
                        if not onset_found_first[vn]:
                            onset_found_first[vn] = True
                        else:
                            onset_t_tmp[vn] = t_val
                            onset_finished[vn] = True
                
                # Normal variable extraction
                is_ss_file = (ifp >= len(file_paths) - n_needed)
                if not meta['is_ss'] or is_ss_file:
                    val = extract_and_reduce(vn, ds, rho, lwp, dz, z, lwp_threshold)
                    raw_collector[vn].append(val)

    # Final aggregation and assignment
    for vn in var_interest:
        dst = nc_dict[ic_str][mp] if ippe == 0 else nc_dict[ic_str][mp][ippe]
        dst.setdefault(vn, {})
        
        if 'v_precip_onset' in vn:
            dst[vn]['value'] = 3600.0 / onset_t_tmp[vn]
        else:
            dst[vn]['value'] = aggregate_timeseries(vn, raw_collector[vn], var_meta[vn])
        dst[vn]['units'] = output_var_set[vn]['var_unit']

    return nc_dict

def parse_var_meta(var_name):
    re_ss_prc  = re.search(r'ss_(\d+)th_prct', var_name)
    is_ss_mean = bool(re.search(r'ss_mean', var_name))
    is_ss_std  = bool(re.search(r'ss_std', var_name))
    is_ss_prc  = bool(re_ss_prc)
    is_dm      = bool(re.search(r'_dm', var_name))
    is_ss      = max(is_ss_mean, is_ss_std, is_ss_prc)
    nth_prctl  = int(re_ss_prc.group(1)) if is_ss_prc else None
    return {'is_ss': is_ss, 'is_ss_mean': is_ss_mean, 'is_ss_std': is_ss_std, 
            'is_ss_prc': is_ss_prc, 'is_dm': is_dm, 'nth_prctl': nth_prctl}

def extract_and_reduce(var_name, ds, rho, lwp, dz, z, lwp_threshold):
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
            data[..., mask] = np.nan
        return data

    if isinstance(vsource, list):
        data = [get_masked_data(vn) for vn in vsource]
    else:
        data = get_masked_data(vsource)

    # Reduction
    if 'path' in var_name:
        dz_b = dz[None, :, None, None]
        res = np.nansum(data * dz_b * rho, axis=(0, 1)) * scale
        # Ensure columns that are all NaN stay NaN instead of 0.0
        if np.all(np.isnan(data), axis=(0, 1)).any():
            res[np.all(np.isnan(data), axis=(0, 1))] = np.nan
    elif 'per5lvl' in var_name:
        res = np.nanmean(data[:, :55:5, :, :], axis=(0, 2, 3)) * scale
    elif 'per10lvl' in var_name:
        res = np.nanmean(data[:, :55:10, :, :], axis=(0, 2, 3)) * scale
    elif 'meanD_dm' in var_name:
        # Tuple for aggregate ratio calculation
        h_match = re.search(r'_(\d+(?:\.\d+)?)m_', var_name)
        if bool(h_match):
            target_h = float(h_match.group(1))
            h_idx = np.argmin(np.abs(z - target_h))
            res = (np.nanmean(data[1][:, h_idx, :, :]), np.nanmean(data[0][:, h_idx, :, :]))
        else:
            res = (np.nanmean(data[1]), np.nanmean(data[0]))
    elif re.search(r'_(\d+(?:\.\d+)?)m_', var_name):
        h_match = re.search(r'_(\d+(?:\.\d+)?)m_', var_name)
        target_h = float(h_match.group(1))
        h_idx = np.argmin(np.abs(z - target_h))
        res = np.nanmean(data[:, h_idx, :, :]) * scale
    elif '_dmprof' in var_name:
        res = np.nanmean(data, axis=(0, 2, 3)) * scale
    elif '_dmpath' in var_name:
        dz_b = dz[None, :, None, None]
        path = np.nansum(data * dz_b * rho, axis=1) # (time, y, x)
        # Ensure columns that are all NaN stay NaN instead of 0.0
        if np.all(np.isnan(data), axis=1).any():
            path[np.all(np.isnan(data), axis=1)] = np.nan
        res = np.nanmean(path, axis=(1, 2)) * scale # (time,)
    elif '_curtain_mean' in var_name:
        res = np.nanmean(data, axis=(0, 2)) * scale
    elif '_curtain_slice' in var_name:
        yidx = data.shape[2] // 2
        res = data[0, :, yidx, :] * scale
    elif 'prate_dm' in var_name or 'precip_max_dm' in var_name:
        res = np.nanmean(data) * scale
    else:
        res = data[0, ...] * scale

    return res

def aggregate_timeseries(var_name, ts, meta):
    if not ts: return np.nan
    
    if 'meanD_dm' in var_name:
        num = np.nanmean([x[0] for x in ts])
        den = np.nanmean([x[1] for x in ts])
        res = (num / den)**(1/3) * 1e6 if den > 0 else np.nan
        return res
    
    arr = np.squeeze(np.stack(ts))
    
    if 'per5lvl' in var_name or 'per10lvl' in var_name or '_dmprof' in var_name:
        return np.nanmean(arr, axis=0) if arr.ndim > 1 else arr # Profile average
    
    if meta['is_ss_std']:
        return np.nanmean(np.nanstd(arr, axis=(1, 2))) if arr.ndim >= 3 else np.nanstd(arr)
    
    if meta['is_ss_prc']:
        valid = arr[~np.isnan(arr) & (arr > 0)]
        return np.percentile(valid, meta['nth_prctl']) if valid.size > 0 else 0.0
    
    # Temporal average
    res = np.nanmean(arr, axis=0)
    if meta['is_dm']:
        res = np.nanmean(res)
    
    if 'precip_max_dm' in var_name:
        return np.nanmax(arr)

    if 'precip_max_dm' in var_name:
        return np.nanmax(arr)
    
    if '_runmean' in var_name:
        return np.nanmean(arr)
        
    if 'dmpath' in var_name:
        return res
    else:
        return np.nan_to_num(res, nan=0.0) if not isinstance(res, tuple) else res

def last_number_key(s):
    matches = re.findall(r'(\d+)(?!.*\d)', s)
    return int(matches[0]) if matches else 0
