import os
import re
import numpy as np
import netCDF4 as nc

def _open_nc(path, mode='r'):
    try:
        return nc.Dataset(path, mode)
    except OSError as e:
        raise OSError(f"Failed to open NetCDF file: {path} ({e})") from e

def _open_last_valid(file_paths, mode='r'):
    """Try files from the end backward and return the first one that opens.
    Returns (path, Dataset) or (None, None) if every file is corrupt."""
    for path in reversed(file_paths):
        try:
            ds = nc.Dataset(path, mode)
            return path, ds
        except OSError:
            print(f"[_open_last_valid] skipping corrupt file: {path}")
            continue
    return None, None

# Vars whose computation method has changed; force a reload by stripping any
# stale cached values so the missing-var check in callers picks them up.
# Repopulate this tuple temporarily after changing how a cached var is computed
# (e.g. ('v_precip_onset',)) so a single run sweeps the live caches, then clear.
INVALIDATE_VARS = ()
# INVALIDATE_VARS = ('v_precip_onset','precip_frac_ss')

def invalidate_stale_vars(payload, vars_to_drop=INVALIDATE_VARS):
    """Recursively walk a cache dict and pop entries named in vars_to_drop
    whose value looks like {'value': ..., 'units': ...}. Layout-agnostic:
    works for continuous_ic, per-pert target, and multi-sim_config caches."""
    if not isinstance(payload, dict) or not vars_to_drop:
        return payload
    for k in list(payload.keys()):
        v = payload[k]
        if k in vars_to_drop and isinstance(v, dict) and 'value' in v:
            payload.pop(k)
        elif isinstance(v, dict):
            invalidate_stale_vars(v, vars_to_drop)
    return payload

def _diagnose_precip_onset(file_paths, dt, skip_hrs, threshold_mmhr=1e-5):
    """Walk files in order starting after skip_hrs; return the first time
    (in hours) at which the raw domain-mean surface prate exceeds the
    threshold (mm/hr). Returns np.nan if onset is never detected.

    Early files outside the steady-state window are sometimes corrupt;
    those are skipped with a print, matching _open_last_valid's pattern.
    """
    if not np.isfinite(dt) or dt <= 0 or not file_paths:
        return np.nan
    # Index of the earliest file we should consider (1-based cm1out_000001 ↔ t≈dt).
    start_idx = max(0, int(np.floor(skip_hrs * 3600.0 / dt)) - 1)
    for fp in file_paths[start_idx:]:
        try:
            with nc.Dataset(fp, 'r') as ds:
                t_s = float(np.asarray(ds.variables['time'][:]).item())
                if t_s / 3600.0 < skip_hrs:
                    continue
                prate = np.asarray(ds.variables['prate'][...]) * 3600.0  # mm/hr
                if np.nanmean(prate) > threshold_mmhr:
                    return t_s / 3600.0
        except OSError:
            print(f"[_diagnose_precip_onset] skipping corrupt file: {fp}")
            continue
    return np.nan

from glob import glob
import platform
import socket
import load_ppe_fun as lp
from tqdm import tqdm
import sys
import warnings
from scipy.fft import fft2, ifft2, fftshift
import pandas as pd

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

def calc_path(ds, vn, dz, scale=1.0, rho=None):
    """Column-integrated path of moment field `vn`, mirroring calc_lwp's
    integration (rho-weighted, dz-weighted vertical sum). Returns (ny, nx)."""
    if rho is None:
        rho = calc_rho(ds)
    field = np.asarray(ds.variables[vn][...]) * scale
    dz_broadcast = dz[None, :, None, None]
    return np.nansum(field * dz_broadcast * rho, axis=(0, 1))

# Transient diagnostics that the steady-state window cannot see (their signal
# lives in the spin-up/onset window). Computed by a single full-run scan
# (_diagnose_run_transients), wired into load_cm1 like the onset vars, and
# skipped by the normal extract_and_reduce/aggregate path.
TRANSIENT_VARS = ('M6_dmpath_overshoot', 'prate_dm_overshoot', 'lwp_persist_ss')

def _diagnose_run_transients(file_paths, dt, dz, ss_hrs):
    """Single full-run scan (like _diagnose_precip_onset) that builds the
    domain-mean time series of M6 path, LWP, and surface rain rate, then
    returns diagnostics the steady-state window cannot capture:
      - M6_dmpath_overshoot / prate_dm_overshoot: peak-over-run divided by the
        steady-state mean. >1 flags the early spike-then-decay in SLC-BOSS.
      - lwp_persist_ss: steady-state mean divided by peak LWP. <1 flags cloud
        rain-out / collapse (the cloud depletes after peaking).
    Steady state = mean over the last ss_hrs. Corrupt files are skipped.
    Returns a dict; missing/failed quantities are np.nan.
    """
    out = {v: np.nan for v in TRANSIENT_VARS}
    if not np.isfinite(dt) or dt <= 0 or not file_paths:
        return out
    scale6 = 1e-4 ** 6
    times, m6p, lwpp, prt = [], [], [], []
    for fp in file_paths:
        try:
            with nc.Dataset(fp, 'r') as ds:
                t_hr = float(np.asarray(ds.variables['time'][:]).item()) / 3600.0
                rho = calc_rho(ds)
                lwp = calc_lwp(ds, dz, rho=rho)
                m6_path = calc_path(ds, 'qc6', dz, scale=scale6, rho=rho)
                prate = np.asarray(ds.variables['prate'][0]) * 3600.0
                times.append(t_hr)
                m6p.append(np.nanmean(m6_path))
                lwpp.append(np.nanmean(lwp))
                prt.append(np.nanmean(prate))
        except (OSError, KeyError):
            # corrupt file, or an early spin-up file missing moment/prate vars
            continue
    if len(times) == 0:
        return out
    times = np.asarray(times, dtype=float)
    m6p = np.asarray(m6p, dtype=float)
    lwpp = np.asarray(lwpp, dtype=float)
    prt = np.asarray(prt, dtype=float)
    t_end = np.nanmax(times)
    ss_mask = times >= (t_end - ss_hrs)

    def _ratio(peak, base):
        return float(peak / base) if (np.isfinite(base) and base > 0) else np.nan

    out['M6_dmpath_overshoot'] = _ratio(np.nanmax(m6p), np.nanmean(m6p[ss_mask]))
    out['prate_dm_overshoot'] = _ratio(np.nanmax(prt), np.nanmean(prt[ss_mask]))
    lwp_peak = np.nanmax(lwpp)
    out['lwp_persist_ss'] = _ratio(np.nanmean(lwpp[ss_mask]), lwp_peak)
    return out

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
                  'M3_dmprof': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'Domain-Mean LWC'},
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
                  'M3_curtain_slice': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4_curtain_slice': {'var_source': 'qc4', 'var_unit': 'm$^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_curtain_slice': {'var_source': 'qc5', 'var_unit': 'm$^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_curtain_slice': {'var_source': 'qc6', 'var_unit': 'm$^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_curtain_slice': {'var_source': 'qc9', 'var_unit': 'm$^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_curtain_mean': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNC'}, 
                  'M3_curtain_mean': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'}, 
                  'M4_curtain_mean': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M5_curtain_mean': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'M5'},
                  'M6_curtain_mean': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'M9_curtain_mean': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'M9'},
                  'M0_path_ss': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'SS LNP'}, 
                  'M3_path_ss': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'SS LWP'}, 
                  'M4_path_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'SS M4'},
                  'M5_path_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'SS M5'},
                  'M6_path_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'SS M6'},
                  'M9_path_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'SS M9'},
                  'M0_dmpath_ss': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'SS DM LNP'}, 
                  'M3_dmpath_ss': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'SS DM LWP'}, 
                  'M4_dmpath_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'SS DM M4'},
                  'M5_dmpath_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'SS DM M5'},
                  'M6_dmpath_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'SS DM M6'},
                  'M9_dmpath_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'SS DM M9'},
                  'M0_dspath_ss': {'var_source': 'qc0', 'var_unit': '1/$m^2$', 'longname': 'SS DS LNP'}, 
                  'M3_dspath_ss': {'var_source': 'qc3', 'var_unit': 'kg/$m^2$', 'scale': M3toQ, 'longname': 'SS DS LWC'}, 
                  'M4_dspath_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/$m^2$', 'scale': 1e-4**4, 'longname': 'SS DS M4'},
                  'M5_dspath_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/$m^2$', 'scale': 1e-4**5, 'longname': 'SS DS M5'},
                  'M6_dspath_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/$m^2$', 'scale': 1e-4**6, 'longname': 'SS DS M6'},
                  'M9_dspath_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/$m^2$', 'scale': 1e-4**9, 'longname': 'SS DS M9'},
                  'M0_10m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP'}, 
                  'M3_10m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC'}, 
                  'M4_10m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4'},
                  'M5_10m_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'SS M5'},
                  'M6_10m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6'},
                  'M9_10m_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'SS M9'},
                  'M0_250m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP'}, 
                  'M3_250m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC'}, 
                  'M4_250m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4'},
                  'M5_250m_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'SS M5'},
                  'M6_250m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6'},
                  'M9_250m_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'SS M9'},
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
                  'prate_dm': {'var_source': 'prate', 'var_unit': 'mm/day', 'scale': 3600*24, 'longname': 'Domain-Mean Rain Rate'},
                  'prate_ds': {'var_source': 'prate', 'var_unit': 'mm/day', 'scale': 3600*24, 'longname': 'Domain-Std Rain Rate'},
                  'prate_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS Rain Rate'},
                  'prate_dm_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS DM Rain Rate'},
                  'prate_dme_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS Domain-Median Rain Rate'},
                  'prate_ds_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS DS Rain Rate'},
                  'prate_tsdm_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS Temporal-STD of DM Rain Rate'},
                  'cloud_thickness_dm_ss': {'var_source': 'qc3', 'var_unit': 'm', 'longname': 'SS DM cloud thickness'},
                  'M6_99th_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6 99th percentile', 'lwc_threshold': 1e-5},
                  'M6_ds_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6 Standard Deviation', 'lwc_threshold': 1e-5},
                  # Numerical broadening diagnostics: KY(036) = M0*M6/M3^2 (dimensionless, >=1 by Cauchy-Schwarz)
                  # Broadening widens the DSD (reduces shape param nu), increasing KY toward ~20 (exponential limit)
                  'KY036_dm_ss':   {'var_source': ['qc0', 'qc3', 'qc6'], 'var_unit': '-', 'lwc_threshold': 1e-5, 'longname': 'SS in-cloud mean KY(036)'},
                  'KY036_99th_ss': {'var_source': ['qc0', 'qc3', 'qc6'], 'var_unit': '-', 'lwc_threshold': 1e-5, 'longname': 'SS in-cloud 99th pct KY(036)'},
                  # Numerical broadening diagnostics: KY(346) = M3^(4/3) * M6^(2/3) / M4^2 (dimensionless, >=1 by Cauchy-Schwarz)
                  # Matches Fortran get_k(mom_set, 3, 4, 6) in module_mp_p3_slc.F
                  'KY346_dm_ss':   {'var_source': ['qc3', 'qc4', 'qc6'], 'var_unit': '-', 'lwc_threshold': 1e-5, 'longname': 'SS in-cloud mean KY(346)'},
                  'KY346_99th_ss': {'var_source': ['qc3', 'qc4', 'qc6'], 'var_unit': '-', 'lwc_threshold': 1e-5, 'longname': 'SS in-cloud 99th pct KY(346)'},
                  # Numerical broadening diagnostics: KX = M0^0.5 * M4^1.5 / M3^2 (dimensionless, >=1 by Cauchy-Schwarz)
                  # Uses qc0 (M0), qc3 (M3), qc4 (M4 in (100um)^4/m^3 units); M4_phys = qc4 * 1e-16
                  'KX_dm_ss':   {'var_source': ['qc0', 'qc3', 'qc4'], 'var_unit': '-', 'lwc_threshold': 1e-5, 'longname': 'SS in-cloud mean KX'},
                  'KX_99th_ss': {'var_source': ['qc0', 'qc3', 'qc4'], 'var_unit': '-', 'lwc_threshold': 1e-5, 'longname': 'SS in-cloud 99th pct KX'},
                  'prate_10th_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS 10th percentile Rain Rate'},
                  'prate_90th_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS 90th percentile Rain Rate'},
                  'prate_99th_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'SS 99th percentile Rain Rate'},
                  'sedflux_m0': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Sedflux M0'},
                  'sedflux_m3': {'var_source': 'sedflux_M3', 'var_unit': 'mm/hr', 'scale': M3toQ*3600, 'longname': 'Rain flux'},
                  'sedflux_m4': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Sedflux M4'},
                  'sedflux_m6': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Sedflux M6'},
                  # domain-mean evap rate profiles (time, z)
                  'evap_M0_dmprof': {'var_source': 'evap_M0', 'var_unit': '1/kg/s', 'longname': 'DM Evap M0'},
                  'evap_M3_dmprof': {'var_source': 'evap_M3', 'var_unit': 'kg/kg/s', 'scale': M3toQ, 'longname': 'DM Evap M3',
                                    'fallback_var_source': 'condevapqr', 'fallback_scale': 1.0},
                  'evap_M4_dmprof': {'var_source': 'evap_M4', 'var_unit': '$m^4$/kg/s', 'scale': 1e-4**4, 'longname': 'DM Evap M4'},
                  'evap_M6_dmprof': {'var_source': 'evap_M6', 'var_unit': '$m^6$/kg/s', 'scale': 1e-4**6, 'longname': 'DM Evap M6'},
                  # domain-mean advection tendency profiles (time, z)
                  'adv_M0_dmprof': {'var_source': 'adv_M0', 'var_unit': '1/kg/s', 'longname': 'DM Adv M0'},
                  'adv_M3_dmprof': {'var_source': 'adv_M3', 'var_unit': 'kg/kg/s', 'scale': M3toQ, 'longname': 'DM Adv M3'},
                  'adv_M4_dmprof': {'var_source': 'adv_M4', 'var_unit': '$m^4$/kg/s', 'scale': 1e-4**4, 'longname': 'DM Adv M4'},
                  'adv_M6_dmprof': {'var_source': 'adv_M6', 'var_unit': '$m^6$/kg/s', 'scale': 1e-4**6, 'longname': 'DM Adv M6'},
                  # domain-mean sed flux profiles (time, z)
                  'sedflux_M0_dmprof': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'DM Sedflux M0'},
                  'sedflux_M3_dmprof': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'DM Sedflux M3'},
                  'sedflux_M4_dmprof': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'DM Sedflux M4'},
                  'sedflux_M6_dmprof': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'DM Sedflux M6'},
                  # domain-mean fall speed profiles (time, z); m/s, no scale
                  'vfall_M0_dmprof': {'var_source': 'vfall_M0', 'var_unit': 'm/s', 'longname': 'DM Vfall M0'},
                  'vfall_M3_dmprof': {'var_source': 'vfall_M3', 'var_unit': 'm/s', 'longname': 'DM Vfall M3'},
                  'vfall_M4_dmprof': {'var_source': 'vfall_M4', 'var_unit': 'm/s', 'longname': 'DM Vfall M4'},
                  'vfall_M6_dmprof': {'var_source': 'vfall_M6', 'var_unit': 'm/s', 'longname': 'DM Vfall M6'},
                  # per-cell mean drop diameter profiles (time, z); all in microns
                  # D_03 = (M3/M0)^(1/3), D_34 = M4/M3, D_36 = (M6/M3)^(1/3), D_06 = (M6/M0)^(1/6)
                  'meanD_03_dmprof': {'var_source': ['qc0', 'qc3'],        'var_unit': '$\\mu$m', 'longname': 'DM Mean Diam (M0,M3)'},
                  'meanD_34_dmprof': {'var_source': ['qc3', 'qc4'],        'var_unit': '$\\mu$m', 'longname': 'DM Mean Diam (M3,M4)'},
                  'meanD_36_dmprof': {'var_source': ['qc3', 'qc6'],        'var_unit': '$\\mu$m', 'longname': 'DM Mean Diam (M3,M6)'},
                  'meanD_06_dmprof': {'var_source': ['qc0', 'qc6'],        'var_unit': '$\\mu$m', 'longname': 'DM Mean Diam (M0,M6)'},
                  # last-timestep curtain (z, x), y-averaged. Stored as 2D arrays.
                  'M0_curtainlast': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'LNC'},
                  'M3_curtainlast': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'LWC'},
                  'M4_curtainlast': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'M4'},
                  'M6_curtainlast': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'M6'},
                  'adv_M0_curtainlast': {'var_source': 'adv_M0', 'var_unit': '1/kg/s', 'longname': 'Adv M0'},
                  'adv_M3_curtainlast': {'var_source': 'adv_M3', 'var_unit': 'kg/kg/s', 'scale': M3toQ, 'longname': 'Adv M3'},
                  'adv_M4_curtainlast': {'var_source': 'adv_M4', 'var_unit': '$m^4$/kg/s', 'scale': 1e-4**4, 'longname': 'Adv M4'},
                  'adv_M6_curtainlast': {'var_source': 'adv_M6', 'var_unit': '$m^6$/kg/s', 'scale': 1e-4**6, 'longname': 'Adv M6'},
                  'evap_M0_curtainlast': {'var_source': 'evap_M0', 'var_unit': '1/kg/s', 'longname': 'Evap M0'},
                  'evap_M3_curtainlast': {'var_source': 'evap_M3', 'var_unit': 'kg/kg/s', 'scale': M3toQ, 'longname': 'Evap M3',
                                         'fallback_var_source': 'condevapqr', 'fallback_scale': 1.0},
                  'evap_M4_curtainlast': {'var_source': 'evap_M4', 'var_unit': '$m^4$/kg/s', 'scale': 1e-4**4, 'longname': 'Evap M4'},
                  'evap_M6_curtainlast': {'var_source': 'evap_M6', 'var_unit': '$m^6$/kg/s', 'scale': 1e-4**6, 'longname': 'Evap M6'},
                  'sedflux_M0_curtainlast': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'Sedflux M0'},
                  'sedflux_M3_curtainlast': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'Sedflux M3'},
                  'sedflux_M4_curtainlast': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'Sedflux M4'},
                  'sedflux_M6_curtainlast': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'Sedflux M6'},
                  'vfall_M0_curtainlast': {'var_source': 'vfall_M0', 'var_unit': 'm/s', 'longname': 'Vfall M0'},
                  'vfall_M3_curtainlast': {'var_source': 'vfall_M3', 'var_unit': 'm/s', 'longname': 'Vfall M3'},
                  'vfall_M4_curtainlast': {'var_source': 'vfall_M4', 'var_unit': 'm/s', 'longname': 'Vfall M4'},
                  'vfall_M6_curtainlast': {'var_source': 'vfall_M6', 'var_unit': 'm/s', 'longname': 'Vfall M6'},
                  'meanD_03_curtainlast': {'var_source': ['qc0', 'qc3'], 'var_unit': '$\\mu$m', 'longname': 'Mean Diam (M0,M3)'},
                  'meanD_34_curtainlast': {'var_source': ['qc3', 'qc4'], 'var_unit': '$\\mu$m', 'longname': 'Mean Diam (M3,M4)'},
                  'meanD_36_curtainlast': {'var_source': ['qc3', 'qc6'], 'var_unit': '$\\mu$m', 'longname': 'Mean Diam (M3,M6)'},
                  'meanD_06_curtainlast': {'var_source': ['qc0', 'qc6'], 'var_unit': '$\\mu$m', 'longname': 'Mean Diam (M0,M6)'},
                  'sfM0_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0'},
                  'sfM3_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3'},
                  'sfM4_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4'},
                  'sfM6_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6'},
                  'sfM0_per5lvl': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 per 5 levels'},
                  'sfM0_per5lvl_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 per 5 levels'},
                  'sfM3_per5lvl_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 per 5 levels'},
                  'sfM4_per5lvl_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 per 5 levels'},
                  'sfM6_per5lvl_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 per 5 levels'},
                  'M0_per5lvl': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP'}, 
                  'M3_per5lvl': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC'}, 
                  'M4_per5lvl': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4'},
                  'M5_per5lvl': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'SS M5'},
                  'M6_per5lvl': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6'},
                  'M9_per5lvl': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'SS M9'},
                  'M0_per5lvl_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP'}, 
                  'M3_per5lvl_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC'}, 
                  'M4_per5lvl_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4'},
                  'M5_per5lvl_ss': {'var_source': 'qc5', 'var_unit': '$m^5$/kg', 'scale': 1e-4**5, 'longname': 'SS M5'},
                  'M6_per5lvl_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6'},
                  'M9_per5lvl_ss': {'var_source': 'qc9', 'var_unit': '$m^9$/kg', 'scale': 1e-4**9, 'longname': 'SS M9'},
                  'M0_dm_10m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP 10m'},
                  'M3_dm_10m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC 10m'},
                  'M4_dm_10m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4 10m'},
                  'M6_dm_10m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6 10m'},
                  'M0_dm_100m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP 100m'},
                  'M3_dm_100m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC 100m'},
                  'M4_dm_100m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4 100m'},
                  'M6_dm_100m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6 100m'},
                  'M0_dm_250m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP 250m'},
                  'M3_dm_250m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC 250m'},
                  'M4_dm_250m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4 250m'},
                  'M6_dm_250m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6 250m'},
                  'M0_dm_500m_ss': {'var_source': 'qc0', 'var_unit': '1/kg', 'longname': 'SS LNP 500m'},
                  'M3_dm_500m_ss': {'var_source': 'qc3', 'var_unit': 'kg/kg', 'scale': M3toQ, 'longname': 'SS LWC 500m'},
                  'M4_dm_500m_ss': {'var_source': 'qc4', 'var_unit': '$m^4$/kg', 'scale': 1e-4**4, 'longname': 'SS M4 500m'},
                  'M6_dm_500m_ss': {'var_source': 'qc6', 'var_unit': '$m^6$/kg', 'scale': 1e-4**6, 'longname': 'SS M6 500m'},
                  'sfM0_dm_10m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 10m'},
                  'sfM3_dm_10m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 10m'},
                  'sfM4_dm_10m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 10m'},
                  'sfM6_dm_10m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 10m'},
                  'sfM0_dm_100m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 100m'},
                  'sfM3_dm_100m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 100m'},
                  'sfM4_dm_100m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 100m'},
                  'sfM6_dm_100m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 100m'},
                  'sfM0_dm_250m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 250m'},
                  'sfM3_dm_250m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 250m'},
                  'sfM4_dm_250m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 250m'},
                  'sfM6_dm_250m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 250m'},
                  'sfM0_dm_500m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 500m'},
                  'sfM3_dm_500m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 500m'},
                  'sfM4_dm_500m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 500m'},
                  'sfM6_dm_500m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 500m'},
                  'sfM0_dm_750m_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 750m'},
                  'sfM3_dm_750m_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 750m'},
                  'sfM4_dm_750m_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 750m'},
                  'sfM6_dm_750m_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 750m'},
                  'sfM0_10m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 10m'},
                  'sfM3_10m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 10m'},
                  'sfM4_10m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 10m'},
                  'sfM6_10m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 10m'},
                  'sfM0_100m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 100m'},
                  'sfM3_100m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 100m'},
                  'sfM4_100m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 100m'},
                  'sfM6_100m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 100m'},
                  'sfM0_250m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 250m'},
                  'sfM3_250m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 250m'},
                  'sfM4_250m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 250m'},
                  'sfM6_250m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 250m'},
                  'sfM0_500m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 500m'},
                  'sfM3_500m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 500m'},
                  'sfM4_500m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 500m'},
                  'sfM6_500m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 500m'},
                  'sfM0_750m_ds_ss': {'var_source': 'sedflux_M0', 'var_unit': '1/$m^2$/s', 'longname': 'SS Sedflux M0 750m'},
                  'sfM3_750m_ds_ss': {'var_source': 'sedflux_M3', 'var_unit': 'kg/$m^2$/s', 'scale': M3toQ, 'longname': 'SS Sedflux M3 750m'},
                  'sfM4_750m_ds_ss': {'var_source': 'sedflux_M4', 'var_unit': '$m^4$/$m^2$/s', 'scale': 1e-4**4, 'longname': 'SS Sedflux M4 750m'},
                  'sfM6_750m_ds_ss': {'var_source': 'sedflux_M6', 'var_unit': '$m^6$/$m^2$/s', 'scale': 1e-4**6, 'longname': 'SS Sedflux M6 750m'},
                  'v_precip_onset':{'var_source': 't_precip_onset', 'var_unit': '1/hr', 'longname': 'Rain Onset Speed'},
                  't_precip_onset':{'var_source': 't_precip_onset', 'var_unit': 'hr', 'longname': 'Rain Onset Time'},
                  'precip_max_dm':{'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'longname': 'Peak Rain Rate'},
                  'meanD_dm_03_10m_ss':  {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'SS mass-meandiam 10m'},
                  'meanD_dm_03_100m_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'SS mass-meandiam 100m'},
                  'meanD_dm_03_250m_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'SS mass-meandiam 250m'},
                  'meanD_dm_03_500m_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'SS mass-meandiam 500m'},
                  'meanD_dm_36_10m_ss':  {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'SS M6-meandiam 10m'},
                  'meanD_dm_36_100m_ss': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'SS M6-meandiam 100m'},
                  'meanD_dm_36_250m_ss': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'SS M6-meandiam 250m'},
                  'meanD_dm_36_500m_ss': {'var_source': ['qc3', 'qc6'], 'var_unit': 'μm', 'longname': 'SS M6-meandiam 500m'},
                  'meanD_dm_03_ss': {'var_source': ['qc0', 'qc3'], 'var_unit': 'μm', 'longname': 'SS mass-meandiam'},
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
                  'precip_frac_ss': {'var_source': 'prate', 'var_unit': '', 'scale': 3600, 'longname': 'Rain Area Fraction'},
                  # Rain-rate exceedance curve. precip_frac_ss above is the >1e-3 mm/hr
                  # point; these add three more. Bulk schemes matching bin at 'hi' while
                  # undershooting at 'lo' is a direct light-rain coverage deficit.
                  'precip_frac_lo_ss': {'var_source': 'prate', 'var_unit': '', 'scale': 3600, 'precip_frac_thr': 1e-2, 'longname': 'Rain Area Frac >0.01 mm/hr'},
                  'precip_frac_mid_ss': {'var_source': 'prate', 'var_unit': '', 'scale': 3600, 'precip_frac_thr': 1e-1, 'longname': 'Rain Area Frac >0.1 mm/hr'},
                  'precip_frac_hi_ss': {'var_source': 'prate', 'var_unit': '', 'scale': 3600, 'precip_frac_thr': 1.0, 'longname': 'Rain Area Frac >1 mm/hr'},
                  # Rain intensity conditioned on raining columns only. prate_dm_ss averages
                  # over the LWP-union gate instead, so it mixes intensity with coverage.
                  'prate_cond_dm_ss': {'var_source': 'prate', 'var_unit': 'mm/hr', 'scale': 3600, 'prate_threshold': 1e-3, 'longname': 'SS Rain Rate | Raining'},
                  # DSD tail-shape constraint: in-cloud mean tail diameter (M6/M4)^(1/2)
                  # in microns. Isolates the M6 tail relative to M4 (the highest matched
                  # moment), the exact axis of the "M6 high, M4 fine" bias.
                  'Dtail_dm_ss': {'var_source': ['qc4', 'qc6'], 'var_unit': 'μm', 'lwc_threshold': 1e-5, 'longname': 'SS tail diameter (M6/M4)^0.5'},
                  # Transient diagnostics (full-run scan; see _diagnose_run_transients):
                  # overshoot = peak/steady (penalizes early spike-then-decay),
                  # persistence = steady/peak LWP (penalizes rain-out collapse).
                  'M6_dmpath_overshoot': {'var_source': 'qc6', 'var_unit': '-', 'longname': 'M6 path peak/steady overshoot'},
                  'prate_dm_overshoot': {'var_source': 'prate', 'var_unit': '-', 'longname': 'Rain rate peak/steady overshoot'},
                  'lwp_persist_ss': {'var_source': 'qc3', 'var_unit': '-', 'longname': 'LWP steady/peak persistence'},
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
                global_id = int(member) if member.isdigit() else global_id_counter
                pert_idx_list.append({
                    'sim_config': config,
                    'member': member,
                    'global_id': global_id
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

def filter_ppe_by_stinginess(ppe_idx, sim_configs, sting_lvl, buffer_size,
                              base_dir, datedir, train_mp):
    """
    Filter PPE members based on stinginess level.

    HI:  Keep all samples (no filtering).
    LOW: Keep only samples within the parameter bounds of the most recent
         (last) ensemble.
    MID: Keep samples within the bounds of the most recent ensemble plus
         a buffer (buffer_size * range) on each side.

    Members of the reference (last) ensemble are always kept.
    Only checks parameters that vary (range > 0) in the reference ensemble.

    Returns filtered ppe_idx (preserving original order).
    """
    ref_config = sim_configs[-1]
    earlier_configs = sim_configs[:-1]

    print(f"\n{'='*60}")
    print(f"Stinginess level  : {sting_lvl}")
    print(f"Reference ensemble: {ref_config}")
    if sting_lvl == 'MID': 
        print(f"Buffer size       : {buffer_size}")
    print(f"Total PPE before filtering: {len(ppe_idx)}")

    if sting_lvl == 'HI':
        print(f"sting_lvl='HI': keeping all {len(ppe_idx)} members")
        print(f"{'='*60}\n")
        return ppe_idx

    if sting_lvl not in ('MID', 'LOW'):
        raise ValueError(f"Invalid sting_lvl: {sting_lvl}. Must be 'HI', 'MID', or 'LOW'.")

    # --- Read params from reference ensemble to get bounds ---
    ref_members = [p for p in ppe_idx if p['sim_config'] == ref_config]
    ref_params_list = []
    for m in ref_members:
        path = f"{base_dir}{datedir}/{m['sim_config']}/{train_mp}/{m['member']}/params.csv"
        df = pd.read_csv(path)
        ref_params_list.append(df.set_index('param_name')['pvalue_mean'])

    ref_df = pd.DataFrame(ref_params_list)

    # Identify varying parameters (range > 0)
    param_min = ref_df.min()
    param_max = ref_df.max()
    param_range = param_max - param_min
    varying_params = param_range[param_range > 0].index.tolist()

    print(f"Varying parameters ({len(varying_params)}):")

    # Compute bounds
    bounds_lo = param_min[varying_params].copy()
    bounds_hi = param_max[varying_params].copy()

    if sting_lvl == 'MID':
        buf = param_range[varying_params] * buffer_size
        bounds_lo -= buf
        bounds_hi += buf
        print(f"  (bounds extended by {buffer_size*100:.0f}% of range on each side)")

    for p in varying_params:
        print(f"  {p:20s}: [{bounds_lo[p]:12.6f}, {bounds_hi[p]:12.6f}]")

    # --- Filter: keep all reference members, filter earlier ensembles ---
    filtered_ppe_idx = []
    n_kept = {cfg: 0 for cfg in earlier_configs}
    n_total = {cfg: 0 for cfg in earlier_configs}

    for p in ppe_idx:
        if p['sim_config'] == ref_config:
            filtered_ppe_idx.append(p)
            continue

        cfg = p['sim_config']
        n_total[cfg] += 1

        path = f"{base_dir}{datedir}/{cfg}/{train_mp}/{p['member']}/params.csv"
        df = pd.read_csv(path)
        params = df.set_index('param_name')['pvalue_mean']

        in_bounds = all(
            bounds_lo[vp] <= params[vp] <= bounds_hi[vp]
            for vp in varying_params
        )

        if in_bounds:
            filtered_ppe_idx.append(p)
            n_kept[cfg] += 1

    # --- Summary ---
    print(f"\nFiltering summary:")
    for cfg in earlier_configs:
        print(f"  {cfg}: kept {n_kept[cfg]}/{n_total[cfg]} members")
    print(f"  {ref_config}: kept {len(ref_members)}/{len(ref_members)} members (reference, all kept)")
    print(f"Total after filtering: {len(filtered_ppe_idx)}")
    print(f"{'='*60}\n")

    return filtered_ppe_idx

def load_cm1_attrs(file_info, nc_dict, ipert=0, continuous_ic=True):
    """
    Lightweight function to load only global attributes and coordinate info 
    from a NetCDF file without loading full variable data.
    """
    if isinstance(ipert, dict):
        current_config = ipert['sim_config']
        member = ipert['member']
        is_dict_ipert = True
    else:
        current_config = file_info['sim_config']
        member = str(ipert)
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
    else:
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        if l_pert:
            file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{member}/{fn_prefix}*{fn_suffix}"
        else:
            file_pattern = f"{fdir}{fdate}/{fsim_config}/{vars_dir}/{mp}/{fn_prefix}*{fn_suffix}"

    file_paths = sorted(glob(file_pattern), key=last_number_key)
    if not file_paths:
        raise FileNotFoundError(f"No files match: {file_pattern}")

    # Walk from the end backward; tolerate corrupt files. If every file is bad,
    # skip the attr/coord population entirely (caller will fill NaN downstream).
    _used_path, _ds0 = _open_last_valid(file_paths)
    if _ds0 is None:
        print(f"[load_cm1_attrs] no readable files for {fsim_config}/{member}; skipping attr load.")
        nc_dict.setdefault(fsim_config, {})
        nc_dict[fsim_config].setdefault(mp, {})
        nc_dict['init_var'] = vars_vn
        return nc_dict
    with _ds0 as ds0:
        nc_dict.setdefault(fsim_config, {})
        nc_dict[fsim_config].setdefault(mp, {})
        nc_dict['init_var'] = vars_vn

        # vn attributes (variable names)
        for vn in vars_vn:
            nc_dict[vn + '_units'] = ds0.getncattr(vn + '_units')
        
        # coords
        if 'z' not in nc_dict[fsim_config]: nc_dict[fsim_config]['z'] = np.round(ds0['zh'][:] * 1e3, decimals=1)
        if 'x' not in nc_dict[fsim_config]: nc_dict[fsim_config]['x'] = np.round(ds0['xh'][:] * 1e3, decimals=1)
        if 'y' not in nc_dict[fsim_config]: nc_dict[fsim_config]['y'] = np.round(ds0['yh'][:] * 1e3, decimals=1)
        
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
    
    return nc_dict

def _is_full_timeseries_var(var_name, meta):
    """True for time-resolved profile/path variables that cm1_viz plots against
    the full simulation time (e.g. *_dmprof, *_dmpath, prate_dm). These benefit
    from backfilling the pre-steady-state window so the time series isn't blank
    during spin-up. SS means, curtains, onset, and percentile diagnostics are
    excluded — they are intentionally restricted to the steady-state window.
    """
    if var_name in ('v_precip_onset', 't_precip_onset'):
        return False
    if var_name in TRANSIENT_VARS:
        return False
    if meta['is_ss'] or meta['is_prc']:
        return False
    if 'curtainlast' in var_name or '_curtain' in var_name:
        return False
    if any(k in var_name for k in ('KY036', 'KY346', 'KX_')):
        return False
    return True

def _load_early_timeseries(early_paths, full_ts_vars, var_meta, dz, z, dx, lwp_threshold):
    """Extract the pre-steady-state portion of time-resolved variables so the
    full time series can be reconstructed. Returns {vn: [slice or None, ...]}
    with one entry per early file (in time order); corrupt/unreadable files
    yield None placeholders so leading-axis time alignment is preserved.
    """
    early = {vn: [] for vn in full_ts_vars}
    for fp in early_paths:
        try:
            ds = nc.Dataset(fp, 'r')
        except OSError:
            print(f"[_load_early_timeseries] skipping corrupt file: {fp}")
            for vn in full_ts_vars:
                early[vn].append(None)
            continue
        try:
            rho = calc_rho(ds)
            lwp = calc_lwp(ds, dz, rho=rho)
            for vn in full_ts_vars:
                try:
                    val = extract_and_reduce(vn, ds, rho, lwp, dz, z, dx, lwp_threshold)
                    early[vn].append(aggregate_timeseries(vn, [val], var_meta[vn]))
                except Exception:
                    early[vn].append(None)
        finally:
            ds.close()
    return early

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

    # Handle early-error runs (crashed CM1): fewer than min_files cm1out files.
    # Fill ic attrs, then set all summary variables to NaN and return.
    min_files = file_info.get('min_files', None)
    if (min_files is not None) and (len(file_paths) < min_files):
        load_cm1_attrs(file_info, nc_dict, ipert=ipert, continuous_ic=continuous_ic)
        nc_dict[fsim_config][mp].setdefault(ic_str, {})
        if continuous_ic or l_pert:
            nc_dict[fsim_config][mp][ic_str].setdefault(global_id, {})
        _used_path, _ds0 = _open_last_valid(file_paths)
        if _ds0 is not None:
            with _ds0 as ds0:
                for vn in vars_vn:
                    keydst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
                    keydst[vn] = ds0.getncattr(vn)
        else:
            for vn in vars_vn:
                keydst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
                keydst[vn] = np.nan
        for vn in var_interest:
            dst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
            dst.setdefault(vn, {})
            dst[vn]['value'] = np.nan
            dst[vn]['units'] = output_var_set[vn]['var_unit']
        # NaN-fill perturbed BOSS params so downstream can detect the bad member.
        # Read names from params.csv (still written at run init even if CM1 crashed).
        try:
            member_dir = os.path.dirname(file_paths[0])
            params_csv = os.path.join(member_dir, 'params.csv')
            if os.path.exists(params_csv):
                pdf = pd.read_csv(params_csv)
                if pdf.shape[0] > pdf.shape[1]:  # vertical layout: rows = params
                    pnames = pdf.iloc[:, 0].astype(str).tolist()
                else:                            # horizontal layout: cols = params
                    pnames = [c.strip() for c in pdf.columns]
                params_nan = {name: np.nan for name in pnames}
            else:
                params_nan = {}
        except Exception as e:
            print(f"[load_cm1] could not read params.csv for {fsim_config}/{member}: {e}")
            params_nan = {}
        dst_root = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
        dst_root['params'] = params_nan
        print(f"[load_cm1] {fsim_config}/{member}: only {len(file_paths)} files (<{min_files}); filling NaN (vars + params).")
        return nc_dict

    # Get dt from the LAST two files (always within the SS window).
    # Early files may be corrupted; we never want to open them.
    if len(file_paths) >= 2:
        with _open_nc(file_paths[-2]) as ds_a, _open_nc(file_paths[-1]) as ds_b:
            t0 = ds_a.variables['time'][0]
            t1 = ds_b.variables['time'][0]
            dt = float(t1 - t0)
    else:
        dt = np.nan

    n_needed = int(np.ceil((ss_hrs * 3600) / dt) + 1) if np.isfinite(dt) and dt > 0 else 1
    # Cap to available files and restrict to last n_needed — anything earlier is
    # outside the steady-state window and not opened.
    n_needed = min(n_needed, len(file_paths))
    files_to_use = file_paths[-n_needed:]

    # Diagnose rain onset: first time after `onset_skip_hrs` that the domain-mean
    # prate exceeds a fixed threshold (mm/hr).
    needs_onset = any(v in var_interest for v in ('v_precip_onset', 't_precip_onset'))
    if needs_onset:
        t_onset_hr = _diagnose_precip_onset(
            file_paths, dt, file_info.get('onset_skip_hrs', 0.0)
        )
    else:
        t_onset_hr = np.nan

    # Extract attributes and coordinates (uses last file)
    load_cm1_attrs(file_info, nc_dict, ipert=ipert, continuous_ic=continuous_ic)

    # Transient diagnostics (overshoot / persistence) require the full run,
    # since their signal lives in the spin-up/onset window; scan once here.
    needs_transients = any(v in var_interest for v in TRANSIENT_VARS)

    with _open_nc(files_to_use[-1]) as ds0:
        nc_dict[fsim_config][mp].setdefault(ic_str, {})
        if continuous_ic or l_pert:
            nc_dict[fsim_config][mp][ic_str].setdefault(global_id, {})

        # Full-simulation time vector (length = len(file_paths)) derived
        # arithmetically from dt and the last file's time, so we don't have
        # to open early (often-corrupt) files. cm1_viz expects this shape.
        if 'time' not in nc_dict[fsim_config]:
            n_total = len(file_paths)
            t_last_s = float(np.asarray(ds0.variables['time'][:]).item())
            if np.isfinite(dt) and dt > 0:
                nc_dict[fsim_config]['time'] = (
                    t_last_s - (n_total - 1 - np.arange(n_total)) * dt
                )
            else:
                nc_dict[fsim_config]['time'] = np.full(n_total, np.nan)

        # var-specific ic values
        for vn in vars_vn:
            keydst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
            keydst[vn] = ds0.getncattr(vn)

        zf = np.asarray(ds0['zf'][:]).copy() * 1e3

    dz = zf[1:] - zf[:-1]
    z = (zf[1:] + zf[:-1])/2
    dx = nc_dict[fsim_config]['x'][1] - nc_dict[fsim_config]['x'][0]

    transients = _diagnose_run_transients(file_paths, dt, dz, ss_hrs) if needs_transients else {}

    # Pre-parse meta and setup collectors
    var_meta = {vn: parse_var_meta(vn) for vn in var_interest}
    raw_collector = {vn: [] for vn in var_interest}
    lwp_pcts = np.zeros(len(files_to_use))
    # Peak surface rain rate anywhere in the SS window. Exactly zero means the
    # member never rains, so its rain statistics are undefined rather than
    # legitimately zero (see the rain_free branch in the aggregation loop).
    prate_max = 0.0

    # Main single-pass loop over the SS window only.
    # The full-sim `time` vector is computed arithmetically above, so we no
    # longer need to read time from each file here.
    for ifp, fp in enumerate(files_to_use):
        with nc.Dataset(fp, 'r') as ds:
            # Physics helpers
            rho = calc_rho(ds)
            lwp = calc_lwp(ds, dz, rho=rho)
            lwp_pcts[ifp] = np.mean(lwp > lwp_threshold) * 100
            if 'prate' in ds.variables:
                prate_max = max(prate_max, float(np.max(ds.variables['prate'][...])))

            for vn in var_interest:
                # Onset vars and transient diagnostics are diagnosed via
                # separate full-run scans above, not the SS-window loop.
                if vn in ('v_precip_onset', 't_precip_onset') or vn in TRANSIENT_VARS:
                    continue
                # Every file in files_to_use is within the SS window,
                # so unconditionally extract. KeyError means the source variable
                # is absent from this MP scheme's output (e.g. 2CAT-BOSS doesn't
                # output evap_M*/vfall_M*); store None so aggregate_timeseries
                # returns NaN for those variables.
                try:
                    val = extract_and_reduce(vn, ds, rho, lwp, dz, z, dx, lwp_threshold)
                    raw_collector[vn].append(val)
                except KeyError:
                    raw_collector[vn].append(None)

    if pbar is not None:
        mean_pct = np.mean(lwp_pcts)
        pbar.set_postfix(lwp_pct=f"{mean_pct:.2f}%")

    # Number of files skipped at the start (outside the SS window).
    # Per-time arrays from raw_collector cover only files_to_use; pad with
    # leading NaN so their leading axis matches len(file_paths) — the same
    # length as nc_dict[fsim_config]['time'] — which is what cm1_viz expects.
    n_skip = len(file_paths) - len(files_to_use)

    # Optionally backfill the early (pre-SS-window) portion of time-resolved
    # profile/path variables so cm1_viz time series cover the full simulation
    # instead of being NaN during spin-up. Gated behind 'full_timeseries' so
    # callers that only want SS diagnostics (e.g. ppe_summary_cm1) don't pay the
    # extra early-file reads. SS means, curtains, onset, and percentiles are
    # unaffected — they still use the SS window only.
    full_ts_early = {}
    if file_info.get('full_timeseries', False) and n_skip > 0:
        full_ts_vars = [vn for vn in var_interest
                        if vn not in ('v_precip_onset', 't_precip_onset')
                        and _is_full_timeseries_var(vn, var_meta[vn])]
        if full_ts_vars:
            full_ts_early = _load_early_timeseries(
                file_paths[:n_skip], full_ts_vars, var_meta, dz, z, dx, lwp_threshold
            )

    # Final aggregation and assignment
    rain_free = (prate_max == 0.0)
    for vn in var_interest:
        dst = nc_dict[fsim_config][mp][ic_str][global_id] if (continuous_ic or l_pert) else nc_dict[fsim_config][mp][ic_str]
        dst.setdefault(vn, {})

        if vn == 't_precip_onset':
            dst[vn]['value'] = t_onset_hr if np.isfinite(t_onset_hr) else np.nan
        elif vn == 'v_precip_onset':
            # 1/t in hr^-1; never-rains → 0.0 (slowest possible signal for NN/GP)
            dst[vn]['value'] = (1.0 / t_onset_hr) if (np.isfinite(t_onset_hr) and t_onset_hr > 0) else 0.0
        elif vn in TRANSIENT_VARS:
            dst[vn]['value'] = transients.get(vn, np.nan)
        else:
            val = aggregate_timeseries(vn, raw_collector[vn], var_meta[vn])
            # Align time-axis-leading arrays with full-sim `time`: backfill the
            # early window with real data when available (full_timeseries mode),
            # otherwise pad with leading NaN as before.
            if (n_skip > 0
                    and isinstance(val, np.ndarray)
                    and val.ndim >= 1
                    and val.shape[0] == len(files_to_use)):
                tail = val.shape[1:]
                early_slices = full_ts_early.get(vn)
                if early_slices:
                    early_arr = np.stack([
                        np.asarray(s, dtype=float) if s is not None
                        else np.full(tail, np.nan)
                        for s in early_slices
                    ], axis=0)
                    val = np.concatenate([early_arr, val], axis=0)
                else:
                    pad_shape = (n_skip,) + tail
                    val = np.concatenate([np.full(pad_shape, np.nan), val], axis=0)
            dst[vn]['value'] = val
        dst[vn]['units'] = output_var_set[vn]['var_unit']

    return nc_dict

def parse_var_meta(var_name):
    # is percentile
    re_prc  = re.search(r'(\d+)th', var_name)
    is_prc  = bool(re_prc)
    nth_prctl  = int(re_prc.group(1)) if is_prc else None

    # is domain mean/median/std (spatial mean/median/std). '_dme' (domain median)
    # would otherwise also match the '_dm' substring, so is_dm excludes it.
    is_dme  = bool(re.search(r'_dme', var_name))
    is_dm   = bool(re.search(r'_dm(?!e)', var_name))
    is_ds   = bool(re.search(r'_ds', var_name))

    # is temporal std of the domain mean (over the SS window)
    is_tsdm = bool(re.search(r'_tsdm', var_name))

    # is SS (temporal mean of the last x hr)
    is_ss   = bool(re.search(r'_ss', var_name))

    return {'is_prc': is_prc, 'nth_prctl': nth_prctl, 'is_dm': is_dm, 'is_dme': is_dme,
            'is_ds': is_ds, 'is_tsdm': is_tsdm, 'is_ss': is_ss}

def extract_and_reduce(var_name, ds, rho, lwp, dz, z, dx, lwp_threshold):
    vsource = output_var_set[var_name]['var_source']
    scale = output_var_set[var_name].get('scale', 1.0)
    lwc_thresh = output_var_set[var_name].get('lwc_threshold')
    prate_thresh = output_var_set[var_name].get('prate_threshold')

    # Raise KeyError early if any source variable is absent from this file
    # (allows callers to silently skip variables not produced by this MP scheme).
    # If a fallback_var_source is defined in output_var_set, try that first before raising.
    _sources = vsource if isinstance(vsource, list) else ([vsource] if vsource else [])
    _missing = [s for s in _sources if s not in ds.variables]
    if _missing:
        _fb_src = output_var_set[var_name].get('fallback_var_source')
        _fb_scl = output_var_set[var_name].get('fallback_scale', scale)
        if _fb_src and _fb_src in ds.variables:
            vsource = _fb_src
            scale = _fb_scl
        else:
            raise KeyError(f"{var_name}: source variable(s) not in file: {_missing}")

    def get_masked_data(vn):
        data = ds.variables[vn][...]
        if lwc_thresh is not None:
            # Mask based on qc3 (LWC)
            lwc = ds.variables['qc3'][:] * M3toQ
            mask = lwc <= lwc_thresh
            data[mask] = np.nan
        elif prate_thresh is not None:
            # Mask on surface rain rate (mm/hr): keep only raining columns. Decouples
            # rain intensity from rain coverage, which the default LWP-union gate below
            # conflates (thick non-raining columns pass on the LWP leg and dilute the
            # mean with prate ~ 0).
            mask = ds.variables['prate'][0, ...]*3600 <= prate_thresh
            if data.ndim >= 2:
                data[..., mask] = np.nan
        else:
            # Mask based on column LWP
            lwp_mask = lwp <= lwp_threshold
            rain_mask = ds.variables['prate'][0, ...]*3600 <= 1e-5
            mask = lwp_mask & rain_mask
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
        if 'meanD_' in var_name and ('dmprof' in var_name or 'curtainlast' in var_name):
            # per-cell mean drop diameter (microns) from a moment pair.
            # qc3 is m^3/kg natively; qc4, qc6 are stored in mxscale/myscale.
            if '_03_' in var_name:
                m0, m3 = data
                m0s = np.where(np.isfinite(m0) & (m0 > 0), m0, np.nan)
                m3s = np.where(np.isfinite(m3) & (m3 > 0), m3, np.nan)
                d_m = (m3s / m0s) ** (1.0 / 3.0)
            elif '_34_' in var_name:
                m3, m4 = data
                m3s = np.where(np.isfinite(m3) & (m3 > 0), m3, np.nan)
                m4_phys = np.where(np.isfinite(m4) & (m4 > 0), m4, np.nan) * (1e-4 ** 4)
                d_m = m4_phys / m3s
            elif '_36_' in var_name:
                m3, m6 = data
                m3s = np.where(np.isfinite(m3) & (m3 > 0), m3, np.nan)
                m6_phys = np.where(np.isfinite(m6) & (m6 > 0), m6, np.nan) * (1e-4 ** 6)
                d_m = (m6_phys / m3s) ** (1.0 / 3.0)
            elif '_06_' in var_name:
                m0, m6 = data
                m0s = np.where(np.isfinite(m0) & (m0 > 0), m0, np.nan)
                m6_phys = np.where(np.isfinite(m6) & (m6 > 0), m6, np.nan) * (1e-4 ** 6)
                d_m = (m6_phys / m0s) ** (1.0 / 6.0)
            else:
                raise ValueError(f"unknown meanD variant: {var_name}")
            res = d_m * 1e6  # m -> micron
        elif 'curtainlast' in var_name:
            # keep full (1, z, y, x); aggregator picks last time and means over y.
            # Use raw (unmasked) field so curtain visualizations show the full
            # spatial structure, not just above-LWP-threshold columns.
            res = raw_data * scale
        elif 'prof' in var_name:
            res = data * scale
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
            # raw (unmasked): curtain visualizations show full field, not just
            # above-threshold columns
            res = np.nanmean(raw_data, axis=(0, 2)) * scale
        elif '_curtain_slice' in var_name:
            yidx = raw_data.shape[2] // 2
            res = raw_data[0, :, yidx, :] * scale
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
        elif 'decorr_length' in var_name:
            lags, radial_R = get_spatial_autocorrelation(lwp, dx)
            threshold = np.exp(-1)
            idx = np.argmax(radial_R < threshold)
            if idx == 0 and radial_R[0] > threshold:
                # It never dropped below 1/e (e.g., highly uniform stratus)
                res = lags[-1]
            else:
                res = lags[idx]
        elif 'precip_frac' in var_name:
            # Exceedance threshold in mm/hr; defaults to the original 1e-3. Several
            # thresholds together trace the rain-rate exceedance curve, separating a
            # light-rain coverage deficit from a heavy-rain one.
            thr = output_var_set[var_name].get('precip_frac_thr', 1e-3)
            res = np.mean(raw_data * scale > thr)
        elif 'cloud_thickness' in var_name:
            # qc3 (3rd moment) is mass-equivalent after *M3toQ (kg/kg). In SLC everything
            # is "cloud", so qc3 already includes rain/drizzle 3rd-moment contributions.
            qc3_arr = ds.variables['qc3'][...]
            qc_kgkg = np.asarray(qc3_arr) * M3toQ  # (1, nz, ny, nx)
            cloud_mask = qc_kgkg > 1e-5
            cm = cloud_mask[0]  # (nz, ny, nx)
            # Cloud top / base indices per column
            top_idx = cm.shape[0] - 1 - np.argmax(cm[::-1, :, :], axis=0)
            base_idx = np.argmax(cm, axis=0)
            thickness = (z[top_idx] - z[base_idx]).astype(np.float64)  # (ny, nx)
            # Same column gate as other *_dm_ss vars
            lwp_mask = lwp > lwp_threshold
            rain_mask = ds.variables['prate'][0, ...] * 3600 > 1e-5
            column_gate = lwp_mask | rain_mask
            valid = column_gate & cm.any(axis=0)
            thickness[~valid] = np.nan
            res = thickness
        elif 'KY036' in var_name:
            m0, m3, m6_raw = data  # (ntime, nz, ny, nx) each; masked to NaN outside cloud
            m3_safe = np.where(m3 > 0, m3, np.nan)
            # KY = M0 * M6 / M3^2; M6_phys = m6_raw * (1e-4)^6 = m6_raw * 1e-24
            ky = m0 * m6_raw * 1e-24 / m3_safe**2
            # Use .filled(np.nan) to convert MaskedArray fill values to NaN before isfinite check;
            # boolean indexing on a MaskedArray uses fill_value=True for masked positions,
            # which would include all non-cloud fill values in the valid array.
            ky_plain = np.array(ky.filled(np.nan) if hasattr(ky, 'filled') else ky, dtype=np.float64)
            valid = ky_plain[np.isfinite(ky_plain)]
            if valid.size == 0:
                res = np.nan
            else:
                res = np.mean(valid)
        elif 'Dtail' in var_name:
            m4_raw, m6_raw = data  # (ntime, nz, ny, nx) each; masked to NaN outside cloud
            # tail diameter (M6/M4)^(1/2): M4_phys = m4_raw*1e-16, M6_phys = m6_raw*1e-24 (SI)
            m4_phys = np.where(m4_raw > 0, m4_raw, np.nan) * 1e-16
            m6_phys = np.where(m6_raw > 0, m6_raw, np.nan) * 1e-24
            dtail = np.sqrt(m6_phys / m4_phys) * 1e6  # m -> micron
            # Use .filled(np.nan) to avoid MaskedArray fill_value contamination (see KY036 comment)
            dt_plain = np.array(dtail.filled(np.nan) if hasattr(dtail, 'filled') else dtail, dtype=np.float64)
            valid = dt_plain[np.isfinite(dt_plain)]
            res = np.nan if valid.size == 0 else np.mean(valid)
        elif 'KY346' in var_name:
            m3, m4_raw, m6_raw = data  # (ntime, nz, ny, nx) each; masked to NaN outside cloud
            m3_safe = np.where(m3 > 0, m3, np.nan)
            # KY(346) = M3^(4/3) * M6^(2/3) / M4^2; M4_phys = m4_raw * 1e-16, M6_phys = m6_raw * 1e-24
            m4_phys = np.where(m4_raw > 0, m4_raw, np.nan) * 1e-16
            m6_phys = m6_raw * 1e-24
            ky = (m3_safe**(4.0/3.0)) * (m6_phys**(2.0/3.0)) / m4_phys**2
            # Use .filled(np.nan) to avoid MaskedArray fill_value contamination (see KY036 comment)
            ky_plain = np.array(ky.filled(np.nan) if hasattr(ky, 'filled') else ky, dtype=np.float64)
            valid = ky_plain[np.isfinite(ky_plain)]
            if valid.size == 0:
                res = np.nan
            else:
                res = np.mean(valid)
        elif 'KX_' in var_name:
            m0, m3, m4_raw = data  # (ntime, nz, ny, nx) each; masked to NaN outside cloud
            m3_safe = np.where(m3 > 0, m3, np.nan)
            # KX = M0^0.5 * M4^1.5 / M3^2; M4_phys = m4_raw * (1e-4)^4 = m4_raw * 1e-16
            kx = (m0**0.5) * ((m4_raw * 1e-16)**1.5) / m3_safe**2
            # Use .filled(np.nan) to avoid MaskedArray fill_value contamination (see KY036 comment)
            kx_plain = np.array(kx.filled(np.nan) if hasattr(kx, 'filled') else kx, dtype=np.float64)
            valid = kx_plain[np.isfinite(kx_plain)]
            if valid.size == 0:
                res = np.nan
            else:
                res = np.mean(valid)
        elif 'pressure' in var_name:
            res = raw_data
        elif prate_thresh is not None:
            # Must use the masked array: the default below reads raw_data, so a
            # prate_threshold entry would otherwise be silently ignored.
            res = data.squeeze() * scale
        else:
            res = raw_data.squeeze() * scale

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
    ts = [x for x in ts if x is not None]
    if not ts: return np.nan
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if 'meanD_dm' in var_name:
            num = np.nanmean([x[0] for x in ts])
            den = np.nanmean([x[1] for x in ts])
            res = (num / den)**(1/3) * 1e6 if den > 0 else np.nan
            return res

        if 'KY036' in var_name or 'KY346' in var_name or 'KX_' in var_name or 'Dtail' in var_name:
            # Each timestep already returned a scalar (mean or 99th pct); just average over ss timesteps
            return np.nanmean(ts)

        if 'curtainlast' in var_name:
            # ts[-1] is (1, z, y, x); take last timestep, mean over y, leave (z, x)
            last = np.asarray(ts[-1])
            if last.ndim == 4:
                last = last[0]
            # last is now (z, y, x); mean over y axis
            return np.nanmean(last, axis=1)

        arr = np.squeeze(np.stack(ts))
        # netCDF reads come back as MaskedArrays. A fully-masked reduction below
        # returns the read-only np.ma.masked singleton, which np.nanmean then
        # tries to write into ("output array is read-only"). Drop to plain nan.
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan)

        if meta['is_tsdm']:
            # temporal std of the domain-mean value over the SS window: spatial
            # mean per SS-window timestep first, then std across those timesteps.
            dm_series = np.nanmean(arr, axis=(-2, -1))
            return np.nanstd(dm_series, axis=0)

        if meta['is_ds']:
            # is domain std — normalized by the domain mean (coefficient of
            # variation). Zero domain mean (e.g. a member that never rains)
            # gives nan, which nan_to_num below turns into 0.
            # mean_xy = np.nanmean(arr, axis=(-2, -1))
            # arr = np.nanstd(arr, axis=(-2, -1)) / np.where(mean_xy == 0, np.nan, mean_xy)
            arr = np.nanstd(arr, axis=(-2, -1))

        if meta['is_dme']:
            # is domain median — median over last two spatial dims (y, x)
            arr = np.nanmedian(arr, axis=(-2, -1))

        if meta['is_dm']:
            # is domain mean — mean over last two spatial dims (y, x)
            arr = np.nanmean(arr, axis=(-2, -1))

        # Temporal average
        if meta['is_ss']:
            arr = np.nanmean(arr, axis=0)

        if meta['is_prc']:
            # is percentile
            valid = arr[~np.isnan(arr) & (arr > 0)]
            if valid.size == 0:
                arr = np.nan
            else:
                arr = np.percentile(valid, meta['nth_prctl'])

        # handle special cases
        if 'precip_max_dm' in var_name:
            return np.nanmax(arr)

        if '_runmean' in var_name:
            return np.nanmean(arr)
            
        if 'dmpath' in var_name:
            return arr
        else:
            return np.nan_to_num(arr, nan=0.0) if not isinstance(arr, tuple) else arr

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
