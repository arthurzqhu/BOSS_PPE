#!/usr/bin/env python
# coding: utf-8

# # setup

# In[1]:


import multiprocessing as _mp
try:
    _mp.set_start_method('fork', force=True)
except RuntimeError:
    pass  # already set

import cm1_load_utils as cl
import load_ppe_fun as lp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from time import sleep
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator
import itertools
import importlib
import dask
dask.config.set({"distributed.worker.multiprocessing-method": "fork"})
from dask.distributed import Client, progress
import joblib
import socket


# In[ ]:

hostname = socket.gethostname()
if hostname == "simurgh":
    n_workers = 32
else:
    n_workers = 8


import argparse as _argparse
import datetime as _datetime
_p = _argparse.ArgumentParser(add_help=False)
_p.add_argument('--nikki', type=str, default=_datetime.date.today().strftime('%Y-%m-%d'))
_p.add_argument('--camp', choices=['rico', 'dycoms'], default=None,
                help="Campaign name. If omitted, auto-detected from sim_config.")
_p.add_argument('--sim_config', type=str, default=None,
                help="Run config name, e.g. 'fullmp_rico_test_resol'. "
                     "Defaults to 'fullmp_<camp>_test_resol'.")
_p.add_argument('--target_sim_config', type=str, default=None,
                help="Defaults to 'fullmp_{camp}_tgt_pert_oldcoalkernel'.")
_p.add_argument('--target_only', action='store_true',
                help="Plot only the target run; skip loading/plotting sim_config entirely.")
_a, _ = _p.parse_known_args()

nikki = _a.nikki
target_nikki = 'target'
target_only = _a.target_only

camps = ['rico', 'dycoms']
if _a.camp is not None:
    camp = _a.camp
elif _a.sim_config is not None or (target_only and _a.target_sim_config is not None):
    _detect_from = _a.sim_config if _a.sim_config is not None else _a.target_sim_config
    _matches = [c for c in camps if c in _detect_from]
    if len(_matches) != 1:
        raise ValueError(f"Could not auto-detect camp from {_detect_from!r}; "
                         f"matched {_matches}. Pass --camp explicitly.")
    camp = _matches[0]
else:
    camp = 'dycoms'
sim_config = _a.sim_config if _a.sim_config is not None else f'fullmp_{camp}_test_resol'
# sim_configs = [simr2_config]
l_pert = False
lwp_threshold = 0.02


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

# target_sim_config = _a.target_sim_config if _a.target_sim_config is not None else f'NCE_{camp}_tgt'
target_sim_config = _a.target_sim_config if _a.target_sim_config is not None else f'fullmp_{camp}_tgt_pert_oldcoalkernel'
if camp == 'rico':
    steady_state_hrs = 5
    min_files = 33
elif camp == 'dycoms':
    steady_state_hrs = 2
    min_files = 25
else:
    min_files = None

sim_configs = [] if target_only else [sim_config]

# In target-only mode nothing is read from / written under sim_config, so key the
# plot directory and figure filenames off the target run instead.
fig_prefix = target_sim_config if target_only else sim_config
plot_dir = f"plots/{nikki}/{fig_prefix}/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

n_init = 1
target_mp = 'BIN-TAU'
mconfigs = [] if target_only else os.listdir(cl.output_dir + nikki)
# Initial conditions (Na subdirs) come from whichever run we actually plot.
if target_only:
    vars_strs, vars_vn = lp.get_dics(cl.output_dir, target_nikki, target_sim_config, n_init)
else:
    vars_strs, vars_vn = lp.get_dics(cl.output_dir, nikki, sim_config, n_init)
# Auto-detect train MP(s) from the first Na subdirectory under sim_config
if target_only:
    train_mps = []
else:
    _first_na = vars_strs[0][0]
    _mp_search_dir = os.path.join(cl.output_dir, nikki, sim_config, _first_na)
    if os.path.isdir(_mp_search_dir):
        train_mps = sorted(d for d in os.listdir(_mp_search_dir)
                           if os.path.isdir(os.path.join(_mp_search_dir, d)) and d != target_mp)
    else:
        train_mps = []
    if not train_mps:
        train_mps = ['SLC-BOSS']
train_mp = train_mps[0] if train_mps else None
var_interest = ['M0_dmpath', 'M3_dmpath', 'M4_dmpath', 'M6_dmpath', 'prate_dm',
                'M0_dmprof', 'M3_dmprof', 'M4_dmprof', 'M6_dmprof',
                # 'adv_M0_dmprof', 'adv_M3_dmprof', 'adv_M4_dmprof', 'adv_M6_dmprof',
                # 'evap_M0_dmprof', 'evap_M3_dmprof', 'evap_M4_dmprof', 'evap_M6_dmprof',
                # 'sedflux_M0_dmprof', 'sedflux_M3_dmprof', 'sedflux_M4_dmprof', 'sedflux_M6_dmprof',
                # 'vfall_M0_dmprof', 'vfall_M3_dmprof', 'vfall_M4_dmprof', 'vfall_M6_dmprof',
                # 'meanD_03_dmprof', 'meanD_34_dmprof', 'meanD_36_dmprof', 'meanD_06_dmprof',
                # 'M0_curtainlast', 'M3_curtainlast', 'M4_curtainlast', 'M6_curtainlast',
                # 'adv_M0_curtainlast', 'adv_M3_curtainlast', 'adv_M4_curtainlast', 'adv_M6_curtainlast',
                # 'evap_M0_curtainlast', 'evap_M3_curtainlast', 'evap_M4_curtainlast', 'evap_M6_curtainlast',
                # 'sedflux_M0_curtainlast', 'sedflux_M3_curtainlast', 'sedflux_M4_curtainlast', 'sedflux_M6_curtainlast',
                # 'vfall_M0_curtainlast', 'vfall_M3_curtainlast', 'vfall_M4_curtainlast', 'vfall_M6_curtainlast',
                # 'meanD_03_curtainlast', 'meanD_34_curtainlast', 'meanD_36_curtainlast', 'meanD_06_curtainlast',
                'M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss',
                'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss',
                # 'prate_dm_ss', 'prate_ds_ss', 'v_precip_onset', 'precip_frac_ss',
                 ] # domain-mean path
# var_interest += ['M0_curtain_mean', 'M3_curtain_mean', 'M4_curtain_mean', 'M6_curtain_mean'] # curtain
                # ] # last 2 hr mean path

# var_interest += ['sedflux_m3']
# var_interest += ['Qc_dmpath', 'Qr_dmpath', 'Nc_dmpath', 'Nr_dmpath']
# var_interest += ['w', 'w_dmprof', 'w_curtain_slice', 'w_curtain_mean'] # 4D var
# var_interest += ['u_dmprof', 'v_dmprof', 'w_dmprof']
# var_interest += ['M0', 'M3']

train_file_info = {'dir': cl.output_dir, 
                   'date': nikki,
                   'vars_vn': vars_vn,
                   'l_pert': False,
                   'sim_config': sim_config,
                   'mp_config': train_mp,
                   'onset_skip_hrs': onset_skip_hrs,
                   'full_timeseries': True,
                  }

tgt_file_info = {'dir': cl.output_dir,
                 'date': target_nikki,
                 'vars_vn': vars_vn,
                 'l_pert': l_pert,
                 'sim_config': target_sim_config,
                 'mp_config': target_mp,
                 'onset_skip_hrs': onset_skip_hrs,
                 'full_timeseries': True,
                }

if 'nc_dict' not in globals():
    nc_dict = {}


# In[5]:


# load BOSS data

if not target_only:
    for initcond_combo in tqdm(itertools.product(*[vars_strs[0]])):
        train_file_info.update({'vars_str': initcond_combo})
        for _sc in sim_configs:
            for _tmp in train_mps:
                train_file_info.update({'sim_config': _sc, 'mp_config': _tmp})
                cl.load_cm1(train_file_info, var_interest, steady_state_hrs, nc_dict, False, lwp_threshold=lwp_threshold)
    # for ref_mp in ref_mps:
    #     train_file_info_2cat = train_file_info.copy()
    #     train_file_info_2cat.update({'sim_config': sim2cat_config, 'mp_config': ref_mp})
    #     cl.load_cm1(train_file_info_2cat, var_interest, steady_state_hrs, nc_dict, False, lwp_threshold=lwp_threshold)
else:
    print("target_only: skipping load of sim_config data")


# In[6]:


tgt_jl_path = f"{cl.output_dir}/{target_nikki}/joblibs/{target_sim_config}_lwpthres{round(lwp_threshold*1e3)}.joblib"
vars_to_load = var_interest
if os.path.exists(tgt_jl_path):
    print(f"Loading target data from {tgt_jl_path}")
    nc_dict[target_sim_config] = joblib.load(tgt_jl_path)
    # Load global attributes
    print(f"Loading global attributes for {target_sim_config}")
    finfo_target = tgt_file_info.copy()
    first_combo = list(itertools.product(*vars_strs))[0]
    finfo_target.update({
        'vars_str': list(first_combo),
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
    # Check if all variables exist. Scan EVERY ic / perturbation, not just the
    # first one: a variable can be present in first_ic but missing in a later ic
    # or pert, which previously slipped through and KeyError'd at plot time.
    # Any variable missing from any ic/pert gets recomputed below and merged into
    # the cached joblib (deep_merge preserves already-cached data).
    try:
        missing = set()
        for initcond_combo in itertools.product(*vars_strs):
            ic = "".join(initcond_combo)
            try:
                ic_dict = nc_dict[target_sim_config][target_mp][ic]
            except KeyError:
                missing.update(var_interest)  # whole ic absent
                continue
            if l_pert:
                pert_keys = [k for k in ic_dict.keys() if isinstance(k, int)]
                if not pert_keys:
                    missing.update(var_interest)
                for pk in pert_keys:
                    missing.update(v for v in var_interest if v not in ic_dict[pk])
            else:
                missing.update(v for v in var_interest if v not in ic_dict)
        # preserve var_interest ordering
        vars_to_load = [v for v in var_interest if v in missing]
    except (KeyError, IndexError):
        vars_to_load = var_interest

if vars_to_load:
    if vars_to_load != var_interest:
        print(f"Missing variables in target data: {vars_to_load}. Loading missing ones...")
    else:
        print(f"Target data not found or being fully reloaded at {tgt_jl_path}")
    dask_scratch = os.path.join(os.environ.get('PSCRATCH', '~/tmp'), 'dask-scratch-space')
    client = Client(n_workers=n_workers, threads_per_worker=1, processes=True, local_directory=dask_scratch)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    print(f"Using {n_workers} processes. Scratch: {dask_scratch}")
    tasks = []
    for initcond_combo in itertools.product(*vars_strs):
        # CRITICAL: Create a separate copy for each combo to avoid mutating the shared reference
        finfo_target = tgt_file_info.copy()
        finfo_target.update({ 'vars_str': list(initcond_combo) })

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

    # Save the specific dictionary key
    joblib.dump(nc_dict[target_sim_config], tgt_jl_path)
    print(f"Dictionary saved to {tgt_jl_path}")
    # Shutdown client
    client.close()
else:
    print("All variables of interest already exist in target data.")


# # basics

# In[7]:


main_sim_config = target_sim_config if target_only else sim_config
time = nc_dict[main_sim_config]['time']/3600

_tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
if target_only:
    # single row / single curve: the target only
    all_sim_configs = [target_sim_config]
    mp_labels = [target_mp]
    mp_markers = ['*']
    color_order = ['tab:orange']
else:
    all_sim_configs = [sim_config, target_sim_config]
    mp_labels = list(train_mps) + [target_mp]
    mp_markers = ['o'] * len(train_mps) + ['*']
    color_order = _tab_colors[:len(mp_labels)]
_n_total = len(mp_labels)
idx_to_plot = list(range(len(all_sim_configs)))
line_styles = ['-'] * _n_total
line_widths = [2.5] * _n_total

# color_order = ['cornflowerblue', 'mediumblue', 'navy', 'tab:orange']
# all_sim_configs = [simr0_config, simr1_config, simr2_config, target_sim_config]
# idx_to_plot = [2, 3]
# mp_labels = ['SLC mean prior', 'SLC MAP round 1', 'SLC MAP round 2', 'target (BIN-TAU)']
# color_order = [color_order[i] for i in idx_to_plot]
# mp_labels = [mp_labels[i] for i in idx_to_plot]

# color_order = ['navy', 'tab:orange', 'firebrick', 'black', 'dimgray']
# mp_labels = ['SLC-BOSS', 'target (BIN-TAU)', '2CAT-BOSS', '2CAT-KK2000', '2CAT-SB2001']
# all_sim_configs = [sim_config, target_sim_config, sim2cat_config]
# idx_to_plot = [0, 1, 2]
# line_styles = ['-', '-', '-', '-.', '-.']
# line_widths = [2.5, 2.5, 2.5, 1.5, 1.5]

x = nc_dict[main_sim_config]['x']
z = nc_dict[main_sim_config]['z']
plt.rcParams['font.size'] = 12


# ## paths

# In[8]:


varsplot = ['M0_dmpath', 'M4_dmpath', 'M3_dmpath', 'M6_dmpath', 'prate_dm']

for case in vars_strs[0]:
    fig, axs = plt.subplots(2, 3, figsize=(12, 5), sharex=True)
    axs = axs.T.flatten()
    for iax, var in enumerate(varsplot):
        if iax == 4:
            iax += 1
        units = cl.output_var_set[var]['var_unit']
        longname = cl.output_var_set[var]['longname']
        i = 0
        for idx in idx_to_plot:
            sc = all_sim_configs[idx]
            for mp in nc_dict[sc].keys():
                if any(x in mp for x in ['time', 'x', 'y', 'z']):
                    continue

                # if '2CAT' in mp and ('M4' in var or 'M6' in var):
                #     continue
                if 'BIN' in mp and l_pert:
                    tgt_data = []
                    for ipert in range(len(nc_dict[sc][mp][case].keys())):
                        tgt_data.append(nc_dict[sc][mp][case][ipert+1][var]['value'])
                    tgt_data = np.stack(tgt_data)
                    mean_tgt = np.mean(tgt_data, axis=0)
                    min_tgt = np.min(tgt_data, axis=0)
                    max_tgt = np.max(tgt_data, axis=0)
                    axs[iax].plot(time, mean_tgt, label=mp_labels[i], linewidth=3, alpha=0.8, color=color_order[i])
                    axs[iax].fill_between(time, min_tgt, max_tgt, linewidth=3, alpha=0.3, color=color_order[i])
                else:
                    axs[iax].plot(time, nc_dict[sc][mp][case][var]['value'],
                                label=mp_labels[i],
                                linewidth=3,
                                alpha=0.8,
                                color=color_order[i])
                i += 1
        axs[iax].set_title(longname, fontsize=16)
        is_even = (iax % 2 == 0)
        if not is_even:
            axs[iax].set_xlabel('Time [hr]')
        axs[iax].set_ylabel(f"[{units}]")
        axs[iax].set_xlim(0, np.max(time))
        # axs[iax].legend(loc='lower right')
        time_int = np.arange(np.max(time)+1)
        axs[iax].set_xticks(time_int)
        axs[iax].set_xticklabels([f"{val:.0f}" for val in time_int], fontsize=12)
        axs[iax].grid(True)
        axs[iax].yaxis.get_offset_text().set_position((-0.15, 0.8))
        axs[iax].set_yscale('log')
    # After plotting all lines, add a single, consolidated legend for the entire figure.
    # Place it in the empty subplot slot (axs[-2]).
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-2].legend(handles, labels, loc='center', fontsize=14, frameon=True, edgecolor='black', fancybox=True)
    axs[-2].axis('off')  # Hide the axes in the empty subplot

    plt.tight_layout()
    # Use bbox_inches='tight' to ensure the legend is fully captured in the saved figure
    plt.savefig(f"{plot_dir}{case}_dm_path_r1.png", bbox_inches='tight')
    plt.close(fig)


# ## profs

# In[9]:


def _plot_prof_panel(varsplot, fname_suffix, use_abs=False, sentinel_thresh=1e20,
                     linear=False, vlim=None, zero_thresh=0.0):
    """4-moment profile pcolormesh figure, modeled on the original profs block.

    varsplot: list of 4 var_names (e.g. M*_dmprof / evap_M*_dmprof / sedflux_M*_dmprof).
    fname_suffix: filename suffix written after sim_config_<case>_.
    use_abs: if True, plot |data| (evap/sedflux can be signed). Masks |data| > sentinel_thresh as NaN.
    linear: linear color scale (default log).
    vlim: (vmin, vmax) override; otherwise limits are auto-derived from the union of
          all rows' data so SLC and BIN share identical limits per column.
    zero_thresh: cells with |data| <= zero_thresh are masked to NaN (rendered white).
                 Only applied in linear mode (log mode already masks <=0).
    """
    def _clean(arr):
        a = np.asarray(arr).copy().astype(float)
        if use_abs:
            a = np.abs(a)
        a[~np.isfinite(a)] = np.nan
        a[np.abs(a) > sentinel_thresh] = np.nan
        return a

    # build a colormap whose 'bad' (NaN) color is white so empty cells appear blank
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')

    def _collect_rowdata(case, var):
        """Return ordered list of (j, data) for the rows in figure order."""
        rowdata = []
        j = 0
        for idx in idx_to_plot:
            sc = all_sim_configs[idx]
            for mp in nc_dict[sc].keys():
                if any(x in mp for x in ['time', 'x', 'y', 'z']):
                    continue
                case_dict = nc_dict[sc][mp].get(case, {})
                if 'BIN' in mp and l_pert:
                    entry = case_dict.get(1, {}).get(var)
                else:
                    entry = case_dict.get(var)
                data = _clean(entry['value'].T) if entry is not None else None
                rowdata.append((j, data))
                j += 1
        return rowdata

    for case in vars_strs[0]:
        fig, axs = plt.subplots(len(color_order), len(varsplot),
                                figsize=(max(4, 3 * len(varsplot)), 5),
                                sharex=True, sharey=True, squeeze=False)
        for i, var in enumerate(varsplot):
            if var not in cl.output_var_set:
                continue
            units = cl.output_var_set[var]['var_unit']
            longname = cl.output_var_set[var]['longname']
            shortname = longname.replace("Domain-Mean", "DM")
            im_thiscol = [None] * len(color_order)

            rowdata = _collect_rowdata(case, var)

            # Shared color limits per column, derived from the union of all rows.
            if vlim is not None:
                vmin, vmax = vlim
            else:
                all_vals = [d.ravel() for _, d in rowdata if d is not None]
                if not all_vals:
                    continue
                stacked = np.concatenate(all_vals)
                if linear:
                    valid = stacked[np.isfinite(stacked)]
                else:
                    valid = stacked[np.isfinite(stacked) & (stacked > 0)]
                if valid.size == 0:
                    continue
                if linear:
                    vmin = float(np.nanmin(valid))
                    vmax = float(np.nanmax(valid))
                    if vmin == vmax:
                        vmax = vmin + 1e-12
                else:
                    vmin = float(np.percentile(valid, 1))
                    vmax = float(np.nanmax(valid))

            for j, data in rowdata:
                if data is None:
                    continue
                if linear:
                    data = np.where((np.abs(data) > zero_thresh) & np.isfinite(data),
                                    data, np.nan)
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    data = np.where((data > 0) & np.isfinite(data), data, np.nan)
                    data = np.where(data > vmin, data, np.nan)
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                im = axs[j, i].pcolormesh(time, z, data,
                                          norm=norm,
                                          cmap=cmap,
                                          rasterized=True)
                axs[j, i].annotate(mp_labels[j], xy=(0.97, 0.85),
                                   xycoords='axes fraction', ha='right', va='bottom',
                                   fontsize=12, fontweight='bold')
                im_thiscol[j] = im
                if j == 0:
                    axs[j, i].set_title(longname, fontsize=14)
                if i == 0:
                    axs[j, i].set_ylabel('Height [m]', fontsize=12)
                if j == len(color_order) - 1:
                    axs[j, i].set_xlabel('Time [hr]', fontsize=12)

            bb_row0 = axs[0, i].get_position(fig)
            bb_row1 = axs[-1, i].get_position(fig)
            cb_left = min(bb_row0.xmin, bb_row1.xmin)
            cb_right = max(bb_row0.xmax, bb_row1.xmax)
            cb_width = cb_right - cb_left
            cb_height = 0.025
            cb_pad = 0.15
            cb_bottom = bb_row1.ymin - cb_pad - cb_height
            cb_ax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
            last_im = next((im for im in im_thiscol if im is not None), None)
            if last_im is not None:
                cbar = fig.colorbar(last_im, cax=cb_ax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=10)
                prefix = '|' if use_abs else ''
                suffix = '|' if use_abs else ''
                cbar.set_label(f"{prefix}{shortname}{suffix} [{units}]", fontsize=12)
        plt.savefig(f"{plot_dir}{fig_prefix}_{case}_{fname_suffix}.png", bbox_inches='tight')
        plt.close(fig)


_plot_prof_panel(['M0_dmprof', 'M3_dmprof', 'M4_dmprof', 'M6_dmprof'],
                 fname_suffix='dm_prof')
# _plot_prof_panel(['adv_M0_dmprof', 'adv_M3_dmprof', 'adv_M4_dmprof', 'adv_M6_dmprof'],
#                  fname_suffix='adv_dmprof', use_abs=True)
# _plot_prof_panel(['evap_M0_dmprof', 'evap_M3_dmprof', 'evap_M4_dmprof', 'evap_M6_dmprof'],
#                  fname_suffix='evap_dmprof', use_abs=True)
# _plot_prof_panel(['sedflux_M0_dmprof', 'sedflux_M3_dmprof', 'sedflux_M4_dmprof', 'sedflux_M6_dmprof'],
#                  fname_suffix='sedflux_dmprof', use_abs=True)
# _plot_prof_panel(['vfall_M0_dmprof', 'vfall_M3_dmprof', 'vfall_M4_dmprof', 'vfall_M6_dmprof'],
#                  fname_suffix='vfall_dmprof', use_abs=True, linear=True, zero_thresh=1e-12)
# _plot_prof_panel(['meanD_03_dmprof', 'meanD_34_dmprof', 'meanD_36_dmprof', 'meanD_06_dmprof'],
#                  fname_suffix='meanD_dmprof', zero_thresh=1e-6)


# ## curtain plots (last timestep, y-averaged) — (z, x)


def _plot_curtain_panel(varsplot, fname_suffix, use_abs=False, sentinel_thresh=1e20,
                        linear=False, vlim=None, zero_thresh=0.0):
    """Last-time curtain figure: x on horizontal axis, z on vertical.

    Same conventions as _plot_prof_panel. Each variable is loaded as a (z, x)
    array (y-averaged at the final timestep).
    """
    def _clean(arr):
        a = np.asarray(arr).copy().astype(float)
        if use_abs:
            a = np.abs(a)
        a[~np.isfinite(a)] = np.nan
        a[np.abs(a) > sentinel_thresh] = np.nan
        return a

    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')

    def _collect_rowdata(case, var):
        rowdata = []
        j = 0
        for idx in idx_to_plot:
            sc = all_sim_configs[idx]
            for mp in nc_dict[sc].keys():
                if any(x in mp for x in ['time', 'x', 'y', 'z']):
                    continue
                case_dict = nc_dict[sc][mp].get(case, {})
                if 'BIN' in mp and l_pert:
                    entry = case_dict.get(1, {}).get(var)
                else:
                    entry = case_dict.get(var)
                data = _clean(entry['value']) if entry is not None else None
                rowdata.append((j, data))
                j += 1
        return rowdata

    for case in vars_strs[0]:
        fig, axs = plt.subplots(len(color_order), len(varsplot),
                                figsize=(max(4, 3 * len(varsplot)), 5),
                                sharex=True, sharey=True, squeeze=False)
        for i, var in enumerate(varsplot):
            if var not in cl.output_var_set:
                continue
            units = cl.output_var_set[var]['var_unit']
            longname = cl.output_var_set[var]['longname']
            shortname = longname.replace("Domain-Mean", "DM")
            im_thiscol = [None] * len(color_order)

            rowdata = _collect_rowdata(case, var)

            if vlim is not None:
                vmin, vmax = vlim
            else:
                all_vals = [d.ravel() for _, d in rowdata if d is not None]
                if not all_vals:
                    continue
                stacked = np.concatenate(all_vals)
                if linear:
                    valid = stacked[np.isfinite(stacked)]
                else:
                    valid = stacked[np.isfinite(stacked) & (stacked > 0)]
                if valid.size == 0:
                    continue
                if linear:
                    vmin = float(np.nanmin(valid))
                    vmax = float(np.nanmax(valid))
                    if vmin == vmax:
                        vmax = vmin + 1e-12
                else:
                    vmin = float(np.percentile(valid, 1))
                    vmax = float(np.nanmax(valid))

            for j, data in rowdata:
                if data is None:
                    continue
                # per-row x coord (in case domains differ between SLC and BIN)
                sc_j = all_sim_configs[idx_to_plot[j]]
                x_j = nc_dict[sc_j]['x']
                z_j = nc_dict[sc_j]['z']
                if linear:
                    data = np.where((np.abs(data) > zero_thresh) & np.isfinite(data),
                                    data, np.nan)
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    data = np.where((data > 0) & np.isfinite(data), data, np.nan)
                    data = np.where(data > vmin, data, np.nan)
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                im = axs[j, i].pcolormesh(x_j, z_j, data,
                                          norm=norm, cmap=cmap, rasterized=True)
                axs[j, i].annotate(mp_labels[j], xy=(0.97, 0.85),
                                   xycoords='axes fraction', ha='right', va='bottom',
                                   fontsize=12, fontweight='bold')
                im_thiscol[j] = im
                if j == 0:
                    axs[j, i].set_title(longname, fontsize=14)
                if i == 0:
                    axs[j, i].set_ylabel('Height [m]', fontsize=12)
                if j == len(color_order) - 1:
                    axs[j, i].set_xlabel('x [m]', fontsize=12)

            bb_row0 = axs[0, i].get_position(fig)
            bb_row1 = axs[-1, i].get_position(fig)
            cb_left = min(bb_row0.xmin, bb_row1.xmin)
            cb_right = max(bb_row0.xmax, bb_row1.xmax)
            cb_width = cb_right - cb_left
            cb_height = 0.025
            cb_pad = 0.15
            cb_bottom = bb_row1.ymin - cb_pad - cb_height
            cb_ax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
            last_im = next((im for im in im_thiscol if im is not None), None)
            if last_im is not None:
                cbar = fig.colorbar(last_im, cax=cb_ax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=10)
                prefix = '|' if use_abs else ''
                suffix = '|' if use_abs else ''
                cbar.set_label(f"{prefix}{shortname}{suffix} [{units}]", fontsize=12)
        plt.savefig(f"{plot_dir}{fig_prefix}_{case}_{fname_suffix}.png", bbox_inches='tight')
        plt.close(fig)


# _plot_curtain_panel(['M0_curtainlast', 'M3_curtainlast', 'M4_curtainlast', 'M6_curtainlast'],
                    # fname_suffix='dm_curtainlast')
# _plot_curtain_panel(['adv_M0_curtainlast', 'adv_M3_curtainlast', 'adv_M4_curtainlast', 'adv_M6_curtainlast'],
#                     fname_suffix='adv_curtainlast', use_abs=True)
# _plot_curtain_panel(['evap_M0_curtainlast', 'evap_M3_curtainlast', 'evap_M4_curtainlast', 'evap_M6_curtainlast'],
#                     fname_suffix='evap_curtainlast', use_abs=True)
# _plot_curtain_panel(['sedflux_M0_curtainlast', 'sedflux_M3_curtainlast', 'sedflux_M4_curtainlast', 'sedflux_M6_curtainlast'],
#                     fname_suffix='sedflux_curtainlast', use_abs=True)
# _plot_curtain_panel(['vfall_M0_curtainlast', 'vfall_M3_curtainlast', 'vfall_M4_curtainlast', 'vfall_M6_curtainlast'],
#                     fname_suffix='vfall_curtainlast', use_abs=True, linear=True, zero_thresh=1e-12)
# _plot_curtain_panel(['meanD_03_curtainlast', 'meanD_34_curtainlast', 'meanD_36_curtainlast', 'meanD_06_curtainlast'],
#                     fname_suffix='meanD_curtainlast', zero_thresh=1e-6)


# # comparison between cases

# In[17]:


time = nc_dict[target_sim_config]['time']/3600
# color_order / mp_labels / mp_markers / idx_to_plot are already target_only-aware
# (set in the "basics" block above); reuse them so this figure matches the others.
mps = list(mp_labels)
x = nc_dict[target_sim_config]['x']
z = nc_dict[target_sim_config]['z']*1e3
plt.rc('font', size=16)
cases = vars_strs[0]
var_na_sens = [
               'M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss',
               'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss',
               # 'prate_dm_ss', 'prate_ds_ss', 'v_precip_onset', 'precip_frac_ss',
                ] # last 2 hr mean path


# In[12]:


plotdata = np.zeros((len(cases), len(mps)))
Na = np.zeros(len(cases))
for i, case in enumerate(cases):
    if l_pert:
        Na[i] = nc_dict[target_sim_config][target_mp][case][1]['na']/1e6
    else:
        Na[i] = nc_dict[target_sim_config][target_mp][case]['na']/1e6

fig, axs = plt.subplots(len(var_na_sens)//4, 4, figsize=(15, 10))
axs = axs.flatten()

for ivar, varname in enumerate(var_na_sens):
    units = cl.output_var_set[varname]['var_unit']
    longname = cl.output_var_set[varname]['longname']
    i = 0
    for idx in idx_to_plot:
        sc = all_sim_configs[idx]
        for mp in nc_dict[sc].keys():
            if any(x in mp for x in ['time', 'x', 'y', 'z']):
                continue
            for j, case in enumerate(cases):
                if 'BIN' in mp and l_pert:
                    plotdata[j, i] = nc_dict[sc][mp][case][1][varname]['value']
                else:
                    plotdata[j, i] = nc_dict[sc][mp][case][varname]['value']
            axs[ivar].plot(Na, plotdata[:, i], linewidth=3, alpha=0.8, label=mp_labels[i], 
                        color=color_order[i], marker=mp_markers[i], markersize=10)
            # axs[ivar].set_title(longname)
            axs[ivar].set_xlabel('$n_{aero}$ [$10^6$ kg$^{-1}$]')
            axs[ivar].set_title(f'{longname} [{units}]', fontsize=12)
            axs[ivar].set_xscale('log')
            if 'onset' in varname or 'frac' in varname:
                axs[ivar].set_yscale('linear')
            else:
                axs[ivar].set_yscale('log')
            axs[ivar].tick_params(axis='both', which='both', labelsize=12)
            i += 1

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, mp_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), fontsize=16, ncol=3)
fig.suptitle("Sensitivity of Domain-Mean Variables on $n_{aero}$", fontsize=20)
plt.tight_layout()

plt.savefig(f"{plot_dir}{fig_prefix}_na_sensitivity.png", bbox_inches='tight')
