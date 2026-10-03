#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[14]:


import cm1_load_utils as cl
import load_ppe_fun as lp
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from time import sleep
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import itertools
import importlib
import pandas as pd
import joblib
import dask
from dask.distributed import Client, progress
import socket
import platform

hostname = socket.gethostname()
if hostname == "simurgh":
    n_workers = 32
elif "nid" in hostname:
    n_workers = 128
else:
    n_workers = 8

# Optional target-band filter (mirrors ppe_summary_cm1.py; keep the two in sync
# to preview exactly what the summary NetCDF will contain).
# Members whose FILTER_VAR value falls outside the target's spread across
# realizations - the light-orange band below - at their own Na are marked as
# rejected: they keep their LWP-ratio color but are drawn faded and outline-free.
#   FILTER_RATIO = 0    -> filter disabled, plots unchanged
#   FILTER_RATIO = 1    -> strictly inside the target min/max band
#   FILTER_RATIO = 0.5  -> [0.5 * band_low, 2.0 * band_high]
# The ratio is symmetric, so r and 1/r mean the same thing (0.5 == 2.0).
FILTER_VAR = 'M3_dmpath_ss'
FILTER_RATIO = 0.
FILTER_INIT_VN = 'na'

# Cross-campaign consistency, matching ppe_summary_cm1.py's FILTER_JOIN_CAMPS.
# With this set, a member counts as "in band" only if it is in band for EVERY
# listed campaign, so the full-color points here are exactly the members that
# reach the joint-calibration summary NetCDFs. DYCOMS mostly undershoots the
# LWP target and RICO mostly overshoots, so the intersection is much smaller
# than either campaign's own in-band set - that gap is the point of showing it.
# Other campaigns are read from their joblib caches only, never raw CM1 output.
# Set to None (or ()) to judge each campaign in isolation.
FILTER_JOIN_CAMPS = ('dycoms', 'rico')

def main(camp='dycoms'):
    nikki = 'ppe'
    target_nikki = 'target'
    lwp_threshold = 0.0
    sting_lvl = 'HI'
    buffer_size = .5

    ppe_basename = [
                    # 'fullmp_offline_km12_D64bound_dycoms_lhs',
                    # 'fullmp_offline_km12_D64bound_e1_dycoms_lhs',
                    # 'fullmp_offline_km12_D64bound_e2_dycoms_lhs',
                    # 'fullmp_offline_km12_D64bound_pm_dycoms_lhs',
                    # 'fullmp_offline_km12_D64bound_pm_e1_dycoms_lhs',
                    # 'fullmp_D64bound_isect_coarse_r1_dycoms_arviz',
                    # 'fullmp_D64bound_isect_coarse_r1_e1_dycoms_arviz',
                    # 'fullmp_D64bound_isect_coarse_r1_e2_dycoms_arviz',
                    # 'fullmp_D64bound_r1_pm_dycoms_arviz',
                    'fullmp_test0_offline_qmom2_dycoms_lhs',
                    'fullmp_test0_offline_qmom2_e1_dycoms_lhs',
                    'fullmp_test0_offline_qmom2_e2_dycoms_lhs',
                    'fullmp_test0_offline_qmom2_pm_dycoms_lhs',
                    ]
    sim_configs = [bn.replace('dycoms', camp) for bn in ppe_basename]
    # sim_config_str is used for joblib paths, plot dirs, and saved-file names
    sim_config_str = sim_configs[-1]

    if 'NCE' in ppe_basename[0]:
        target_sim_config = f'NCE_tgt_{camp}_pert'
        l_pert = True
    else:
        target_sim_config = f'fullmp_tgt_NewCoalKernel_{camp}_pert'
        l_pert = True

    if camp == 'rico':
        steady_state_hrs = 8
        min_files = 49
        onset_skip_hrs = 1.5
    elif camp == 'dycoms':
        steady_state_hrs = 2
        min_files = 25
        onset_skip_hrs = 0.5
    else:
        min_files = None
        onset_skip_hrs = 0.0

    plot_dir = f"plots/{nikki}/{sim_config_str}_lwpthres{round(lwp_threshold*1e3)}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    n_init = 1
    target_mp = 'BIN-TAU'
    # Auto-detect train_mp from the sim_config directory (mirrors cm1_viz.py).
    # For multiple sim_configs they must share the same train_mp.
    def _detect_train_mp(sc):
        d = os.path.join(cl.output_dir, nikki, sc)
        if not os.path.isdir(d):
            return None
        mps = sorted(x for x in os.listdir(d)
                     if os.path.isdir(os.path.join(d, x)) and x != target_mp)
        return mps[0] if mps else None
    detected = {sc: _detect_train_mp(sc) for sc in sim_configs}
    unique_mps = {m for m in detected.values() if m is not None}
    if len(unique_mps) > 1:
        raise ValueError(
            f"sim_configs have inconsistent train_mp: {detected}. "
            "All sim_configs in ppe_basename must share the same train_mp."
        )
    if not unique_mps:
        print(f"[cm1_ppe] warning: no train_mp directory found under {sim_configs}; defaulting to 'SLC-BOSS'")
        train_mp = 'SLC-BOSS'
    else:
        train_mp = unique_mps.pop()
        print(f"[cm1_ppe] auto-detected train_mp = '{train_mp}'")
    mconfigs = os.listdir(cl.output_dir + nikki)
    vars_strs, vars_vn = lp.get_dics(cl.output_dir, target_nikki, target_sim_config, n_init)
    var_interest = []

    var_interest += [
                     'M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss',
                     'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss',
    ]

    if 'fullmp' in ppe_basename[0]:
        var_interest += [
                         'M6_99th_ss', 'meanD_dm_03_ss', 'v_precip_onset', 'precip_frac_ss',
                         'prate_dm_ss', 'prate_tsdm_ss', 'prate_ds_ss', 'cloud_thickness_dm_ss',
                         # Rain-rate distribution: exceedance curve (precip_frac_ss is the
                         # >1e-3 mm/hr point) plus intensity conditioned on raining columns.
                         # Separates a light-rain coverage deficit from a heavy-rain one.
                         'cloud_cover_10_ss', 'cloud_cover_30_ss', 'cloud_cover_50_ss',
                         'precip_frac_lo_ss', 'precip_frac_mid_ss', 'precip_frac_hi_ss', 'prate_cond_dm_ss',
                         'Dtail_dm_ss', 'M6_dmpath_overshoot', 'prate_dm_overshoot','lwp_persist_ss',
                         # 'sfM0_dm_10m_ss', 'sfM3_dm_10m_ss', 'sfM4_dm_10m_ss', 'sfM6_dm_10m_ss',
                         # 'sfM0_dm_100m_ss', 'sfM3_dm_100m_ss', 'sfM4_dm_100m_ss', 'sfM6_dm_100m_ss',
                         # 'sfM0_dm_250m_ss', 'sfM3_dm_250m_ss', 'sfM4_dm_250m_ss', 'sfM6_dm_250m_ss',
                         # 'sfM0_dm_500m_ss', 'sfM3_dm_500m_ss', 'sfM4_dm_500m_ss', 'sfM6_dm_500m_ss',
                         # 'M0_dm_10m_ss', 'M3_dm_10m_ss', 'M4_dm_10m_ss', 'M6_dm_10m_ss',
                         # 'M0_dm_100m_ss', 'M3_dm_100m_ss', 'M4_dm_100m_ss', 'M6_dm_100m_ss',
                         # 'M0_dm_250m_ss', 'M3_dm_250m_ss', 'M4_dm_250m_ss', 'M6_dm_250m_ss',
                         # 'M0_dm_500m_ss', 'M3_dm_500m_ss', 'M4_dm_500m_ss', 'M6_dm_500m_ss',
                ]

    train_file_info = {'dir': cl.output_dir,
                       'date': nikki,
                       'vars_vn': vars_vn,
                       'l_pert': False,
                       'sim_config': sim_configs,  # list — get_pert_idx handles multi-config
                       'mp_config': train_mp,
                       'min_files': min_files,
                       'onset_skip_hrs': onset_skip_hrs,
                      }

    tgt_file_info = {'dir': cl.output_dir,
                     'date': target_nikki,
                     'vars_vn': vars_vn,
                     'l_pert': l_pert,
                     'sim_config': target_sim_config,
                     'mp_config': target_mp,
                     'onset_skip_hrs': onset_skip_hrs,
                    }

    nc_dict = {}


    # # load data

    # In[16]:


    importlib.reload(cl)
    # load BOSS data — train_file_info['sim_config'] is a list; get_pert_idx returns
    # members with per-member 'sim_config' fields when given a list
    ppe_idx = cl.get_pert_idx(train_file_info)

    ppe_idx = cl.filter_ppe_by_stinginess(
        ppe_idx, sim_configs, sting_lvl, buffer_size,
        cl.output_dir, 'ppe', train_mp
    )

    # One joblib per sim_config (mirrors ppe_summary_cm1.py). Each per-config
    # joblib stores nc_dict[sc] directly (NOT wrapped in {sc: ...}).
    client = None
    def _get_client():
        nonlocal client
        if client is None:
            dask_scratch = os.path.join(os.environ.get('PSCRATCH', '/home/arthurhu/tmp'), 'dask-scratch-space')
            client = Client(n_workers=n_workers, threads_per_worker=1, processes=True, local_directory=dask_scratch)
            print(f"Dask dashboard available at: {client.dashboard_link}")
            print(f"Using {n_workers} Processes. Scratch: {dask_scratch}")
        return client

    for sim_config in sim_configs:
        train_jl_path = f"{cl.output_dir}/{nikki}/joblibs/{sim_config}_lwpthres{round(lwp_threshold*1e3)}.joblib"
        # Subset of ppe_idx that belongs to this sim_config
        ppe_idx_sim = [ippe for ippe in ppe_idx
                       if (isinstance(ippe, dict) and ippe['sim_config'] == sim_config)]

        if os.path.exists(train_jl_path):
            print(f"Loading {sim_config} data from {train_jl_path}")
            saved = cl.invalidate_stale_vars(joblib.load(train_jl_path))
            # Legacy fallback: if the joblib was written in the old multi-config
            # format ({sc: {...}, ...}), unwrap the relevant inner dict.
            if isinstance(saved, dict) and sim_config in saved and 'SLC-BOSS' not in saved:
                saved = saved[sim_config]
            nc_dict[sim_config] = saved

            # Load global attributes from the first valid member of this sim_config
            if ppe_idx_sim:
                finfo_attr = train_file_info.copy()
                finfo_attr['sim_config'] = sim_config
                cl.load_cm1_attrs(finfo_attr, nc_dict=nc_dict, ipert=ppe_idx_sim[0], continuous_ic=True)

            # Check for missing vars/members in cached data
            try:
                cached = nc_dict[sim_config][train_mp]['cic']
            except KeyError:
                cached = {}
            ppe_vars_to_load = set()
            members_to_reload = []
            for ippe in ppe_idx_sim:
                gid = ippe['global_id'] if isinstance(ippe, dict) else int(ippe)
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
                _get_client()
                tasks = [
                    dask.delayed(cl.load_cm1)(
                        train_file_info, ppe_vars_to_load, steady_state_hrs,
                        nc_dict=None, continuous_ic=True, ipert=ippe, lwp_threshold=lwp_threshold,
                    )
                    for ippe in members_to_reload
                ]
                futures = client.compute(tasks)
                progress(futures)
                results = client.gather(futures)
                for r in tqdm(results, desc=f'merging missing vars for {sim_config}'):
                    cl.deep_merge(nc_dict, r)
                joblib.dump(nc_dict[sim_config], train_jl_path)
                print(f"Updated cache saved to {train_jl_path}")
            else:
                print(f"All variables of interest already exist for {sim_config}.")
        else:
            print(f"No cache for {sim_config}; loading all members.")
            _get_client()
            tasks = [
                dask.delayed(cl.load_cm1)(
                    train_file_info, var_interest, steady_state_hrs,
                    nc_dict=None, continuous_ic=True, ipert=ippe, lwp_threshold=lwp_threshold,
                )
                for ippe in ppe_idx_sim
            ]
            print(f"Computing PPE data for {sim_config} in parallel...")
            futures = client.compute(tasks)
            progress(futures)
            results = client.gather(futures)
            for r in tqdm(results, desc=f'merging PPE results for {sim_config}'):
                cl.deep_merge(nc_dict, r)
            joblib.dump(nc_dict[sim_config], train_jl_path)
            print(f"Dictionary saved to {train_jl_path}")


    # In[17]:


    params_ppe = []
    for ippe, ppe in enumerate(tqdm(ppe_idx, desc='loading params')):
        sc = ppe['sim_config']   # per-member config (handles multi-config correctly)
        member = ppe['member']
        gid = ppe['global_id']
        # Always read params.csv to preserve param ordering & count, then
        # NaN it out if the member is a load_cm1 early-error case. Signature:
        # 'params' key in nc_dict (new path) OR all var_interest values NaN
        # (covers older cached joblibs from before the 'params' marker existed).
        param_df = pd.read_csv(f"{cl.output_dir}{nikki}/{sc}/{train_mp}/{member}/params.csv")
        member_rec = nc_dict.get(sc, {}).get(train_mp, {}).get('cic', {}).get(gid, {})
        is_bad = False
        if isinstance(member_rec, dict):
            if 'params' in member_rec:
                is_bad = True
            elif var_interest:
                vals = []
                for v in var_interest:
                    entry = member_rec.get(v)
                    if isinstance(entry, dict):
                        vals.append(entry.get('value'))
                if vals and all(
                    (val is None) or (isinstance(val, float) and np.isnan(val))
                    for val in vals
                ):
                    is_bad = True
        row = param_df.values[:, 1].astype(float).copy()
        if is_bad:
            row[:] = np.nan
        params_ppe.append(row)
    params_ppe = np.array(params_ppe)


    # In[18]:


    importlib.reload(cl)

    tgt_jl_path = f"{cl.output_dir}/{target_nikki}/joblibs/{target_sim_config}_lwpthres{round(lwp_threshold*1e3)}.joblib"
    vars_to_load = var_interest
    if os.path.exists(tgt_jl_path):
        print(f"Loading target data from {tgt_jl_path}")
        nc_dict[target_sim_config] = cl.invalidate_stale_vars(joblib.load(tgt_jl_path))
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
        # first one: a variable can be present in first_ic/first_pert but missing in
        # a later ic or pert, which previously slipped through and KeyError'd when the
        # data was consumed. Any variable missing from any ic/pert is recomputed below
        # and merged into the cached joblib (deep_merge preserves already-cached data).
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
        print(f"Using {n_workers} Processes. Scratch: {dask_scratch}")
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


    # # visualize

    # In[19]:

    print('plotting ...')
    var_interest_blk1 = var_interest[:]
    # var_interest_blk2 = var_interest[16:32]
    # var_interest_blk3 = var_interest[32:]
    var_interest_blks = [var_interest_blk1]
    # var_interest_blks = [var_interest_blk1, var_interest_blk2, var_interest_blk3]
    block_names = ['summary', 'sedfluxes', 'moments']


    na = []
    for initcond_combo in itertools.product(*vars_strs):
        ic_str = "".join(initcond_combo)
        tgt_file_info.update({ 'vars_str': list(initcond_combo), })
        if l_pert:
            target_pert_idx = cl.get_pert_idx(tgt_file_info)
            first_pert_id = target_pert_idx[0]['global_id']
            na.append(nc_dict[target_sim_config][target_mp][ic_str][first_pert_id]['na'])
        else:
            na.append(nc_dict[target_sim_config][target_mp][ic_str]['na'])

    na = np.array(na)

    # --- Optional target-band filter. Same band math as ppe_summary_cm1.py, so
    # the members shown in full color here are exactly the ones that survive
    # into the summary NetCDF for the same FILTER_VAR / FILTER_RATIO /
    # FILTER_JOIN_CAMPS.
    if FILTER_VAR and FILTER_RATIO and FILTER_VAR not in var_interest:
        raise ValueError(
            f"FILTER_VAR '{FILTER_VAR}' is not in var_interest; it must be one "
            "of the loaded constraint variables."
        )
    band_edges = cl.target_band_edges(
        nc_dict, vars_strs, target_sim_config, target_mp, l_pert,
        FILTER_VAR, FILTER_RATIO, init_vn=FILTER_INIT_VN,
    )

    def _cached_camp_keep_keys(other_camp):
        """In-band member keys for another campaign, from its joblib caches.

        Reads cached data only, so the other campaign must have been loaded at
        least once. Raises with an actionable message if a cache is absent.
        """
        oc_sim_configs = [bn.replace('dycoms', other_camp) for bn in ppe_basename]
        if 'NCE' in ppe_basename[0]:
            oc_target = f'NCE_tgt_{other_camp}_pert'
        else:
            oc_target = f'fullmp_tgt_NewCoalKernel_{other_camp}_pert'
        oc_vars_strs, oc_vars_vn = lp.get_dics(cl.output_dir, target_nikki,
                                               oc_target, n_init)

        oc_nc = {}
        for sc in oc_sim_configs:
            jl = (f"{cl.output_dir}/{nikki}/joblibs/"
                  f"{sc}_lwpthres{round(lwp_threshold*1e3)}.joblib")
            if not os.path.exists(jl):
                raise FileNotFoundError(
                    f"FILTER_JOIN_CAMPS includes '{other_camp}' but its PPE cache "
                    f"is missing: {jl}\nRun `python cm1_ppe.py {other_camp}` first, "
                    "or set FILTER_JOIN_CAMPS = None."
                )
            saved = cl.invalidate_stale_vars(joblib.load(jl))
            # Same legacy unwrap as the main load loop above.
            if isinstance(saved, dict) and sc in saved and 'SLC-BOSS' not in saved:
                saved = saved[sc]
            oc_nc[sc] = saved
        oc_tgt_jl = (f"{cl.output_dir}/{target_nikki}/joblibs/"
                     f"{oc_target}_lwpthres{round(lwp_threshold*1e3)}.joblib")
        if not os.path.exists(oc_tgt_jl):
            raise FileNotFoundError(
                f"FILTER_JOIN_CAMPS includes '{other_camp}' but its target cache "
                f"is missing: {oc_tgt_jl}\nRun `python cm1_ppe.py {other_camp}` "
                "first, or set FILTER_JOIN_CAMPS = None."
            )
        oc_nc[oc_target] = cl.invalidate_stale_vars(joblib.load(oc_tgt_jl))

        oc_finfo = {'dir': cl.output_dir, 'date': nikki, 'vars_vn': oc_vars_vn,
                    'l_pert': False, 'sim_config': oc_sim_configs,
                    'mp_config': train_mp, 'min_files': min_files,
                    'onset_skip_hrs': onset_skip_hrs}
        oc_ppe_idx = cl.filter_ppe_by_stinginess(
            cl.get_pert_idx(oc_finfo), oc_sim_configs, sting_lvl, buffer_size,
            cl.output_dir, nikki, train_mp)

        return cl.band_keep_keys(
            oc_nc, oc_ppe_idx, oc_vars_strs, oc_target, target_mp, train_mp,
            l_pert, other_camp, FILTER_VAR, FILTER_RATIO, init_vn=FILTER_INIT_VN)

    join_camps = [c for c in (FILTER_JOIN_CAMPS or ())]
    band_keep_by_member = {}
    band_n_self = None          # in-band count for THIS campaign alone
    if band_edges is not None:
        _r = cl.canonical_band_ratio(FILTER_RATIO)
        self_keep, self_seen = cl.band_keep_keys(
            nc_dict, ppe_idx, vars_strs, target_sim_config, target_mp, train_mp,
            l_pert, camp, FILTER_VAR, FILTER_RATIO, init_vn=FILTER_INIT_VN)
        band_n_self = len(self_keep)
        print(f"[band filter] {FILTER_VAR} ratio {FILTER_RATIO} "
              f"(effective factor {_r:g} / {1.0 / _r:g})")

        if join_camps:
            if camp not in join_camps:
                raise ValueError(
                    f"camp '{camp}' is not in FILTER_JOIN_CAMPS "
                    f"{tuple(join_camps)}; add it or set FILTER_JOIN_CAMPS = None."
                )
            print(f"[band filter] cross-campaign intersection over {tuple(join_camps)}")
            keep_keys, seen_sets = None, []
            for jc in join_camps:
                ks, seen = ((self_keep, self_seen) if jc == camp
                            else _cached_camp_keep_keys(jc))
                seen_sets.append(seen)
                print(f"[band filter]   {jc:8s}: {len(ks)}/{len(seen)} members in band")
                keep_keys = ks if keep_keys is None else (keep_keys & ks)
            # A member absent from any campaign's pool cannot be used jointly.
            common = set.intersection(*seen_sets)
            dropped = len(set.union(*seen_sets) - common)
            if dropped:
                print(f"[band filter]   {dropped} member(s) absent from at least one "
                      "campaign's pool; excluded from the intersection")
            keep_keys &= common
        else:
            keep_keys = self_keep

        for ippe in ppe_idx:
            sc, gid = ippe['sim_config'], ippe['global_id']
            band_keep_by_member[(sc, gid)] = cl.member_key(sc, gid, camp) in keep_keys
        print(f"[band filter] {sum(band_keep_by_member.values())}/{len(ppe_idx)} "
              f"members shown in band for '{camp}'")

    band_tag = cl.band_filter_tag(FILTER_VAR, FILTER_RATIO)
    if band_tag and join_camps:
        band_tag += '_isect-' + '-'.join(join_camps)

    def _panel_keep(train_keys):
        """Per-panel in-band mask, aligned with train_keys."""
        if band_edges is None:
            return np.ones(len(train_keys), dtype=bool)
        return np.array([band_keep_by_member.get(k, False) for k in train_keys],
                        dtype=bool)

    def _filter_title(train_keys):
        """Title suffix summarizing the filter outcome for this panel.

        In cross-campaign mode the count shown is the intersection, with this
        campaign's own in-band count alongside so the gap between them (the
        opposite-bias penalty) is visible.
        """
        n_keep, n_tot = int(_panel_keep(train_keys).sum()), len(train_keys)
        if join_camps:
            return (f"\n[r={FILTER_RATIO:g} isect {n_keep}/{n_tot}"
                    f" | {camp} {band_n_self}]")
        return f"\n[r={FILTER_RATIO:g}: {n_keep}/{n_tot} kept]"

    def _band_grid(na_train):
        """Dense Na grid spanning both the target cases and the PPE members."""
        ic_vals = band_edges[0]
        if ic_vals.size < 2:
            return ic_vals
        lo = min(ic_vals.min(), np.nanmin(na_train)) if len(na_train) else ic_vals.min()
        hi = max(ic_vals.max(), np.nanmax(na_train)) if len(na_train) else ic_vals.max()
        if lo <= 0:
            return np.linspace(lo, hi, 200)
        return np.logspace(np.log10(lo), np.log10(hi), 200)

    def _scatter_split(ax, x, y, colors, keep, label='Train PPE'):
        """Scatter PPE members, fading the ones the filter rejects.

        Rejected members keep their LWP-ratio color (alpha 0.25, no outline) so
        the ensemble-wide bias structure stays readable behind the survivors.
        """
        if band_edges is not None and not keep.all():
            ax.scatter(x[~keep], y[~keep], s=8, c=colors[~keep],
                       cmap=lwp_ratio_cmap, norm=lwp_ratio_norm,
                       alpha=0.25, linewidths=0)
            ax.scatter(x[keep], y[keep], label=f'{label} (in band)', s=8,
                       c=colors[keep], cmap=lwp_ratio_cmap, norm=lwp_ratio_norm,
                       edgecolors='black', linewidths=0.3)
        else:
            ax.scatter(x, y, label=label, s=8, c=colors,
                       cmap=lwp_ratio_cmap, norm=lwp_ratio_norm,
                       edgecolors='black', linewidths=0.3)

    # --- Per-member ratio to interpolated target LWP, used as the scatter color in
    # every panel below (both physical- and transformed-units plots). Interpolated
    # rather than matched against the nearest target Na case, since PPE members are
    # continuously sampled in Na and rarely land exactly on a target case.
    LWP_VAR = 'M3_dmpath_ss'
    tgt_lwp_mean = []
    for initcond_combo in itertools.product(*vars_strs):
        ic_str = "".join(initcond_combo)
        if l_pert:
            _ic_vals = [nc_dict[target_sim_config][target_mp][ic_str][ipert['global_id']][LWP_VAR]['value']
                       for ipert in target_pert_idx
                       if ipert['global_id'] in nc_dict[target_sim_config][target_mp][ic_str]]
            tgt_lwp_mean.append(np.mean(_ic_vals))
        else:
            tgt_lwp_mean.append(nc_dict[target_sim_config][target_mp][ic_str][LWP_VAR]['value'])
    tgt_lwp_mean = np.asarray(tgt_lwp_mean, dtype=float)
    _sort_idx = np.argsort(na)
    _na_sorted = na[_sort_idx]
    _tgt_lwp_sorted = tgt_lwp_mean[_sort_idx]

    # Keyed by (sim_config, global_id): global_id is only unique WITHIN a
    # sim_config (get_pert_idx sets it to int(member_dir)), so keying on the
    # bare gid silently collapses same-numbered members across sim_configs.
    lwp_ratio_by_member = {}
    for ippe in ppe_idx:
        sc = ippe['sim_config']
        gid = ippe['global_id']
        member_rec = nc_dict.get(sc, {}).get(train_mp, {}).get('cic', {}).get(gid)
        if member_rec is None or LWP_VAR not in member_rec:
            continue
        member_lwp = member_rec[LWP_VAR]['value']
        tgt_lwp_here = np.interp(member_rec['na'], _na_sorted, _tgt_lwp_sorted)
        lwp_ratio_by_member[(sc, gid)] = member_lwp / tgt_lwp_here if tgt_lwp_here != 0 else np.nan

    # Color by log2(ratio), not the raw ratio. On a raw-ratio scale a factor-2
    # overshoot (2.0) and a factor-2 undershoot (0.5) sit at distances 1.0 and
    # 0.5 from vcenter=1, so the low side gets compressed into a narrow band of
    # the colormap. In log2 space the two are symmetric about 0.
    lwp_log2ratio_by_member = {
        k: (np.log2(v) if (np.isfinite(v) and v > 0) else np.nan)
        for k, v in lwp_ratio_by_member.items()
    }

    _l2_vals = np.array([v for v in lwp_log2ratio_by_member.values() if np.isfinite(v)])
    if _l2_vals.size:
        # Symmetric limit clipping both tails equally, so the red/blue split
        # stays centered on the target regardless of ensemble-wide bias.
        _l2_lim = float(max(abs(np.percentile(_l2_vals, 2)),
                            abs(np.percentile(_l2_vals, 98))))
    else:
        _l2_lim = 1.0
    _l2_lim = max(_l2_lim, 0.1)
    lwp_ratio_norm = mcolors.Normalize(vmin=-_l2_lim, vmax=_l2_lim)
    lwp_ratio_cmap = plt.get_cmap('RdBu_r').copy()
    # Members with a missing or non-positive ratio map to the colormap's "bad"
    # color; default is transparent, which leaves them as bare black outlines.
    lwp_ratio_cmap.set_bad('0.75')
    lwp_ratio_label = f'PPE / target {LWP_VAR} (interpolated at member Na)'

    def _add_ratio_colorbar(fig, axs):
        """Colorbar drawn in log2 space but tick-labelled in ratio units."""
        sm = plt.cm.ScalarMappable(cmap=lwp_ratio_cmap, norm=lwp_ratio_norm)
        cbar = fig.colorbar(sm, ax=axs.tolist(), shrink=0.6,
                            label=lwp_ratio_label, extend='both')
        for _step in (1.0, 0.5, 0.25):
            _n = int(np.floor(_l2_lim / _step))
            _ticks = np.arange(-_n, _n + 1) * _step
            if _ticks.size >= 3:
                break

        def _lab(t):
            r = 2.0 ** t
            return f'{r:.3g}' if r >= 1 else f'1/{1.0 / r:.3g}'

        cbar.set_ticks(_ticks)
        cbar.set_ticklabels([_lab(t) for t in _ticks])
        return cbar

    tolerance = 3
    mask_post = None
    for var_interest_blk, block_name in zip(var_interest_blks, block_names):
        _nrow = int(np.ceil(len(var_interest_blk) / 4))
        fig, axs = plt.subplots(_nrow, 4, figsize=(12, 2.4 * _nrow), sharex=True)
        axs = axs.flatten()
        for ivar, var_name in enumerate(var_interest_blk):
            tgt_data = []
            train_data = []
            na_train = []
            for initcond_combo in itertools.product(*vars_strs):
                ic_str = "".join(initcond_combo)
                if l_pert:
                    ic_data = []
                    for ipert in target_pert_idx:
                        if ipert['global_id'] in nc_dict[target_sim_config][target_mp][ic_str]:
                            ic_data.append(nc_dict[target_sim_config][target_mp][ic_str][ipert['global_id']][var_name]['value'])
                    if ic_data:
                        tgt_data.append(ic_data)
                else:
                    tgt_data.append(nc_dict[target_sim_config][target_mp][ic_str][var_name]['value'])

            train_keys = []
            for ippe in ppe_idx:
                sc = ippe['sim_config']  # each member knows its own config
                if ippe['global_id'] in nc_dict[sc][train_mp]['cic']:
                    train_data.append(nc_dict[sc][train_mp]['cic'][ippe['global_id']][var_name]['value'])
                    na_train.append(nc_dict[sc][train_mp]['cic'][ippe['global_id']]['na'])
                    train_keys.append((sc, ippe['global_id']))
                else:
                    print(ippe['global_id'])

            tgt_data = np.array(tgt_data) # expected shape: (num_ic, num_pert)
            expected_ndim = 2 if l_pert else 1
            if tgt_data.ndim < expected_ndim:
                raise ValueError(
                    f"tgt_data for {var_name} has ndim={tgt_data.ndim}, "
                    f"expected {expected_ndim} (shape={tgt_data.shape})"
                )
            elif tgt_data.ndim > expected_ndim:
                extra_axes = tuple(range(expected_ndim, tgt_data.ndim))
                print(f"  [warn] {var_name}: averaging tgt_data over axes {extra_axes} "
                      f"(shape {tgt_data.shape} -> ndim {expected_ndim})")
                tgt_data = np.mean(tgt_data, axis=extra_axes)
            train_data = np.array(train_data)
            na_train = np.array(na_train)
            if l_pert:
                mean_tgt = np.mean(tgt_data, axis=1)
                min_tgt = np.min(tgt_data, axis=1)
                max_tgt = np.max(tgt_data, axis=1)
                axs[ivar].plot(na, mean_tgt, label=ic_str, linewidth=2, marker='o', alpha=0.5, color='tab:orange')
                axs[ivar].fill_between(na, min_tgt, max_tgt, alpha=0.3, color='tab:orange')
            else:
                axs[ivar].plot(na, tgt_data, label=ic_str, linewidth=2, marker='o', alpha=0.5, color='tab:orange')

            if len(train_data) > 0:
                train_colors = np.array([lwp_log2ratio_by_member.get(k, np.nan) for k in train_keys])
                keep = _panel_keep(train_keys)
                _scatter_split(axs[ivar], na_train, train_data, train_colors, keep)

                # axs[ivar].scatter(na_train[mask], train_data[mask], label='Train In Bounds', s=20, color='tab:pink', marker='*')

                # if l_pert and var_name == "M6_dmpath_ss":
                #     # Sort na and targets by na for interpolation
                #     na_flat = np.ravel(na)
                #     sort_idx = np.argsort(na_flat)
                #     na_sorted = na_flat[sort_idx]
                #     min_tgt_sorted = np.ravel(min_tgt)[sort_idx]
                #     max_tgt_sorted = np.ravel(max_tgt)[sort_idx]

                #     # Interpolate min_tgt and max_tgt at na_train points
                #     interp_min = np.interp(na_train, na_sorted, min_tgt_sorted)
                #     interp_max = np.interp(na_train, na_sorted, max_tgt_sorted)

                #     # Apply mask to highlight bounded train_data
                #     # mask = train_data > interp_max * tolerance
                #     mask = (train_data >= interp_min / tolerance) & (train_data <= interp_max * tolerance)
                #     axs[ivar].scatter(na_train[mask], train_data[mask], label='Train In Bounds', s=20, color='tab:pink', marker='*')

                    # # Accumulate mask_post using logical OR for the union
                    # if mask_post is None:
                    #     mask_post = mask.copy()
                    # else:
                    #     mask_post = mask_post & mask

            _title = cl.output_var_set[var_name]['longname']
            if band_edges is not None and var_name == FILTER_VAR:
                # The accept/reject bounds themselves, on the panel they act on.
                _grid = _band_grid(na_train)
                _lo, _hi = cl.band_bounds_at(_grid, band_edges)
                axs[ivar].plot(_grid, _lo, ls='--', lw=1.2, color='tab:red', zorder=5)
                axs[ivar].plot(_grid, _hi, ls='--', lw=1.2, color='tab:red', zorder=5,
                               label=f'filter band (r={FILTER_RATIO:g})')
                _title += _filter_title(train_keys)
            axs[ivar].set_title(_title, fontsize=9)
            axs[ivar].set_xscale('log')
            if 'onset' in var_name or 'frac' in var_name or 'cover' in var_name or 'persist' in var_name:
                axs[ivar].set_yscale('linear')
            else:
                axs[ivar].set_yscale('log')
            # if ivar == 4 or ivar == 5:
            #     axs[ivar].set_ylim([1e-8, 1e-1])

        plt.tight_layout()
        _add_ratio_colorbar(fig, axs)
        plt.savefig(f"{plot_dir}{sim_config_str}_{block_name}{band_tag}.png")

    # --- Summary plot in transformed space ---
    # Mirrors ppe_summary_cm1.py's eff0 logic: asinh(y/eff0)*eff0 then standard
    # scaler if eff0 is finite; plain standard scaler if eff0 is NaN. The
    # scaler is fit on the PPE training data only, then applied to tgt too.
    from sklearn.preprocessing import StandardScaler
    smooth_linlog = lambda y, e0: e0 * np.arcsinh(y / e0)

    def _eff0_for(ivar, ppe_pos_vals):
        """Match ppe_summary_cm1.py thresholds_eff0 logic."""
        if 'onset' in ivar or 'M3_' in ivar or 'cloud_thickness' in ivar:
            return np.nan
        if 'overshoot' in ivar or 'persist' in ivar:
            return 0
        if 'V_M' in ivar:
            return 0.1
        if 'prate' in ivar:
            return 1e-4
        if 'precip_frac' in ivar or 'cloud_cover' in ivar:
            return 0.01
        finite = ppe_pos_vals[np.isfinite(ppe_pos_vals)]
        return np.nanpercentile(finite, 10) if finite.size else 1.0

    _nrow = int(np.ceil(len(var_interest_blk1) / 4))
    fig, axs = plt.subplots(_nrow, 4, figsize=(12, 2.4 * _nrow), sharex=True)
    axs = axs.flatten()
    for ivar, var_name in enumerate(var_interest_blk1):
        tgt_data = []
        train_data = []
        na_train = []
        for initcond_combo in itertools.product(*vars_strs):
            ic_str = "".join(initcond_combo)
            if l_pert:
                ic_data = []
                for ipert in target_pert_idx:
                    if ipert['global_id'] in nc_dict[target_sim_config][target_mp][ic_str]:
                        ic_data.append(nc_dict[target_sim_config][target_mp][ic_str][ipert['global_id']][var_name]['value'])
                if ic_data:
                    tgt_data.append(ic_data)
            else:
                tgt_data.append(nc_dict[target_sim_config][target_mp][ic_str][var_name]['value'])
        train_keys = []
        for ippe in ppe_idx:
            sc = ippe['sim_config']
            if ippe['global_id'] in nc_dict[sc][train_mp]['cic']:
                train_data.append(nc_dict[sc][train_mp]['cic'][ippe['global_id']][var_name]['value'])
                na_train.append(nc_dict[sc][train_mp]['cic'][ippe['global_id']]['na'])
                train_keys.append((sc, ippe['global_id']))

        tgt_data = np.array(tgt_data)
        expected_ndim = 2 if l_pert else 1
        if tgt_data.ndim > expected_ndim:
            tgt_data = np.mean(tgt_data, axis=tuple(range(expected_ndim, tgt_data.ndim)))
        train_data = np.array(train_data, dtype=float)
        na_train = np.array(na_train)

        # Compute eff0 from POSITIVE PPE values (matches ppe_summary_cm1.py)
        ppe_pos = train_data[train_data > 0]
        eff0 = _eff0_for(var_name, ppe_pos)
        if np.isnan(eff0):
            transform_method = 'std'
        elif eff0==0:
            transform_method = 'log'
        else:
            transform_method = 'asinh'

        def transform(y):
            y = np.asarray(y, dtype=float)
            if transform_method == 'std':
                return y
            elif transform_method =='log':
                return np.log10(y)
            elif transform_method =='asinh':
                return smooth_linlog(y, eff0)

        # Fit standard scaler on PPE (transformed); apply to both
        ppe_t = transform(train_data).reshape(-1, 1)
        scaler = StandardScaler().fit(ppe_t)
        train_scaled = scaler.transform(ppe_t).ravel()
        tgt_scaled = scaler.transform(transform(tgt_data).reshape(-1, 1)).reshape(tgt_data.shape)

        if l_pert:
            mean_tgt = np.mean(tgt_scaled, axis=1)
            min_tgt = np.min(tgt_scaled, axis=1)
            max_tgt = np.max(tgt_scaled, axis=1)
            axs[ivar].plot(na, mean_tgt, label=ic_str, linewidth=2, marker='o', alpha=0.5, color='tab:orange')
            axs[ivar].fill_between(na, min_tgt, max_tgt, alpha=0.3, color='tab:orange')
        else:
            axs[ivar].plot(na, tgt_scaled, label=ic_str, linewidth=2, marker='o', alpha=0.5, color='tab:orange')

        if len(train_scaled) > 0:
            train_colors = np.array([lwp_log2ratio_by_member.get(k, np.nan) for k in train_keys])
            keep = _panel_keep(train_keys)
            _scatter_split(axs[ivar], na_train, train_scaled, train_colors, keep)

        if band_edges is not None and var_name == FILTER_VAR:
            # Push the bounds through the same transform + scaler as the data.
            _grid = _band_grid(na_train)
            _lo, _hi = cl.band_bounds_at(_grid, band_edges)
            _lo = scaler.transform(transform(_lo).reshape(-1, 1)).ravel()
            _hi = scaler.transform(transform(_hi).reshape(-1, 1)).ravel()
            axs[ivar].plot(_grid, _lo, ls='--', lw=1.2, color='tab:red', zorder=5)
            axs[ivar].plot(_grid, _hi, ls='--', lw=1.2, color='tab:red', zorder=5,
                           label=f'filter band (r={FILTER_RATIO:g})')

        if transform_method == 'std':
            label_t = 'standard only'
        elif transform_method == 'log':
            label_t = 'standard log'
        elif transform_method == 'asinh':
            label_t = f'asinh@eff0={eff0:.2e}'
        _title = f"{cl.output_var_set[var_name]['longname']}\n[{label_t}]"
        if band_edges is not None and var_name == FILTER_VAR:
            _title += _filter_title(train_keys)
        axs[ivar].set_title(_title, fontsize=9)
        axs[ivar].set_xscale('log')
        axs[ivar].set_yscale('linear')  # already standardized
        axs[ivar].axhline(0, color='k', lw=0.6, alpha=0.4)
        axs[ivar].set_ylabel('standardized')

    plt.tight_layout()
    _add_ratio_colorbar(fig, axs)
    plt.savefig(f"{plot_dir}{sim_config_str}_summary_transformed{band_tag}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('camp', nargs='?', default='dycoms',
                        help='Campaign name (e.g., dycoms, rico). Default: dycoms')
    args = parser.parse_args()
    main(camp=args.camp)


# # # analysis

# # In[ ]:


# import os
# import arviz as az
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Filter parameters using mask
# posterior_samples = params_ppe[mask, 4:]

# # Use column names from the original param_df if available, else generic names
# param_names = param_df['param_name'][4:].tolist()

# # Create DataFrame for seaborn pairplot
# posterior_df = pd.DataFrame(posterior_samples, columns=param_names)

# # Create ArviZ InferenceData and save to NetCDF
# # Add a chain dimension. arviz expects (chain, draw, *shape).
# posterior_dict = {name: np.expand_dims(posterior_samples[:, i], axis=0) for i, name in enumerate(param_names)}
# idata = az.from_dict(
#     posterior=posterior_dict,
#     dims={name: ['draw'] for name in param_names}
# )
# nc_path = f"MCMC_posterior/{sim_config}_posterior.nc"
# if os.path.exists(nc_path):
#     os.remove(nc_path)

# try:
#     idata.to_netcdf(nc_path, engine='netcdf4')
# except Exception:
#     idata.to_netcdf(nc_path, engine='h5netcdf')
# print(f"Saved posterior InferenceData to {nc_path}")

# # Create and save Seaborn pairplot
# sns.pairplot(posterior_df)
# plt.savefig(f"{plot_dir}{sim_config}_posterior_pairplot.pdf")
# plt.show()


# # In[ ]:


# train_data = []
# for ippe in ppe_idx:
#     ippe = ippe['global_id']
#     train_data.append(nc_dict[sim_config]['SLC-BOSS']['cic'][ippe]['precip_frac_ss']['value'])
# train_data = np.array(train_data)
# na_train = np.array(na_train)

# idx_rain = np.where(train_data>5e-3)[0]

# # idx_rain = np.where(np.logical_and(train_data>1e-2, na_train<2e7))[0]


# # In[ ]:


# idx_rain = np.where(mask)[0]


# # In[ ]:


# i = 28
# _=plt.hist(params_ppe[:, i], bins=20)
# _=plt.hist(params_ppe[idx_rain, i], bins=20)

