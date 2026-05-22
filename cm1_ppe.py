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


# In[15]:


def main(camp='dycoms'):
    nikki = 'ppe'
    target_nikki = 'target'
    lwp_threshold = 0.02

    ppe_basename = ['NCE_dycoms_4param_advmom_arviz',
                    # 'fullmp_dycoms_test54_no_snapping_advect_state_finer_arviz',
                    ]
    sim_configs = [bn.replace('dycoms', camp) for bn in ppe_basename]
    # sim_config_str is used for joblib paths, plot dirs, and saved-file names
    # (mirrors ppe_summary_cm1.py line 142: sim_configs[-1])
    sim_config_str = sim_configs[-1]

    if 'NCE' in ppe_basename[0]:
        target_sim_config = f'NCE_{camp}_tgt'
        l_pert = False
    else:
        target_sim_config = f'fullmp_{camp}_tgt_pert'
        l_pert = True

    if camp == 'rico':
        steady_state_hrs = 5
    elif camp == 'dycoms':
        steady_state_hrs = 2

    plot_dir = f"plots/{nikki}/{sim_config_str}_lwpthres{round(lwp_threshold*1e3)}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    n_init = 1
    target_mp = 'BIN-TAU'
    train_mp = 'SLC-BOSS'
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
                         'prate_dm_ss', 'prate_ds_ss', 'KX_dm_ss', 'KY346_dm_ss',
                         'sfM0_dm_10m_ss', 'sfM3_dm_10m_ss', 'sfM4_dm_10m_ss', 'sfM6_dm_10m_ss',
                         'sfM0_dm_100m_ss', 'sfM3_dm_100m_ss', 'sfM4_dm_100m_ss', 'sfM6_dm_100m_ss',
                         'sfM0_dm_250m_ss', 'sfM3_dm_250m_ss', 'sfM4_dm_250m_ss', 'sfM6_dm_250m_ss',
                         'sfM0_dm_500m_ss', 'sfM3_dm_500m_ss', 'sfM4_dm_500m_ss', 'sfM6_dm_500m_ss',
                         'M0_dm_10m_ss', 'M3_dm_10m_ss', 'M4_dm_10m_ss', 'M6_dm_10m_ss',
                         'M0_dm_100m_ss', 'M3_dm_100m_ss', 'M4_dm_100m_ss', 'M6_dm_100m_ss',
                         'M0_dm_250m_ss', 'M3_dm_250m_ss', 'M4_dm_250m_ss', 'M6_dm_250m_ss',
                         'M0_dm_500m_ss', 'M3_dm_500m_ss', 'M4_dm_500m_ss', 'M6_dm_500m_ss',
                ]

    train_file_info = {'dir': cl.output_dir,
                       'date': nikki,
                       'vars_vn': vars_vn,
                       'l_pert': False,
                       'sim_config': sim_configs,  # list — get_pert_idx handles multi-config
                       'mp_config': train_mp,
                      }

    tgt_file_info = {'dir': cl.output_dir,
                     'date': target_nikki,
                     'vars_vn': vars_vn,
                     'l_pert': l_pert,
                     'sim_config': target_sim_config,
                     'mp_config': target_mp,
                    }

    nc_dict = {}


    # # load data

    # In[16]:


    importlib.reload(cl)
    # load BOSS data — train_file_info['sim_config'] is a list; get_pert_idx returns
    # members with per-member 'sim_config' fields when given a list
    ppe_idx = cl.get_pert_idx(train_file_info)
    # Joblib path is keyed by sim_config_str (last/canonical config, mirrors ppe_summary_cm1.py)
    train_jl_path = f"{cl.output_dir}/{nikki}/joblibs/{sim_config_str}_lwpthres{round(lwp_threshold*1e3)}.joblib"
    vars_to_load = var_interest
    if os.path.exists(train_jl_path):
        print(f"Loading train data from {train_jl_path}")
        # Joblib stores {sc: nc_dict[sc]} for all configs; merge back into nc_dict
        saved = joblib.load(train_jl_path)
        if isinstance(saved, dict) and any(k in sim_configs for k in saved):
            # new multi-config format: {sc: {...}, ...}
            cl.deep_merge(nc_dict, saved)
        else:
            # legacy single-config format: bare inner dict under sim_config_str
            nc_dict[sim_config_str] = saved
        # Load global attributes from the first valid member
        if ppe_idx:
            print(f"Loading global attributes for {ppe_idx[0]['sim_config']}")
            finfo_attr = train_file_info.copy()
            cl.load_cm1_attrs(finfo_attr, nc_dict=nc_dict, ipert=ppe_idx[0], continuous_ic=True)
        # Check if all variables exist (use first member's config/id)
        try:
            ic_str = 'cic'
            first_ppe = ppe_idx[0]
            first_sc = first_ppe['sim_config']
            global_id = first_ppe['global_id'] if isinstance(first_ppe, dict) else int(first_ppe)
            existing_vars = nc_dict[first_sc][train_mp][ic_str][global_id].keys()
            vars_to_load = [v for v in var_interest if v not in existing_vars]
        except (KeyError, IndexError):
            vars_to_load = var_interest

    if vars_to_load:
        if vars_to_load != var_interest:
            print(f"Missing variables in train data: {vars_to_load}. Loading missing ones...")
        else:
            print(f"Train data not found or being fully reloaded at {train_jl_path}")
        dask_scratch = os.path.join(os.environ.get('PSCRATCH', '~/tmp'), 'dask-scratch-space')
        client = Client(n_workers=32, threads_per_worker=1, processes=True, local_directory=dask_scratch)
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print(f"Using 32 Processes. Scratch: {dask_scratch}")
        tasks = []
        for ippe in ppe_idx:
            # ippe carries 'sim_config' for the correct per-member directory;
            # pass train_file_info with the full list — load_cm1 uses ipert['sim_config']
            task = dask.delayed(cl.load_cm1)(
                train_file_info, vars_to_load, steady_state_hrs, nc_dict=None, continuous_ic=True,
                ipert=ippe, lwp_threshold=lwp_threshold
            )
            tasks.append(task)

        print("Computing PPE data in parallel...")
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)

        for r in tqdm(results, desc='merging PPE results'):
            cl.deep_merge(nc_dict, r)

        # Save all configs as a single dict keyed by sim_config_str (multi-config format)
        to_save = {sc: nc_dict[sc] for sc in sim_configs if sc in nc_dict}
        joblib.dump(to_save, train_jl_path)
        print(f"Dictionary saved to {train_jl_path}")
        # Shutdown client
        client.close()
    else:
        print("All variables of interest already exist in train data.")


    # In[17]:


    params_ppe = []
    for ippe, ppe in enumerate(tqdm(ppe_idx, desc='loading params')):
        sc = ppe['sim_config']   # per-member config (handles multi-config correctly)
        member = ppe['member']
        param_df = pd.read_csv(f"{cl.output_dir}{nikki}/{sc}/{train_mp}/{member}/params.csv")
        params_ppe.append(param_df.values[:,1])
    params_ppe = np.array(params_ppe)


    # In[18]:


    importlib.reload(cl)

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
        dask_scratch = os.path.join(os.environ.get('PSCRATCH', '~/tmp'), 'dask-scratch-space')
        client = Client(n_workers=32, threads_per_worker=1, processes=True, local_directory=dask_scratch)
        print(f"Dask dashboard available at: {client.dashboard_link}")
        print(f"Using 32 Processes. Scratch: {dask_scratch}")
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
    var_interest_blk1 = var_interest[:16]
    var_interest_blk2 = var_interest[16:32]
    var_interest_blk3 = var_interest[32:]
    # var_interest_blks = [var_interest_blk1]
    var_interest_blks = [var_interest_blk1, var_interest_blk2, var_interest_blk3]
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
    tolerance = 3
    mask_post = None
    for var_interest_blk, block_name in zip(var_interest_blks, block_names):
        fig, axs = plt.subplots(4, 4, figsize=(12, 12), sharex=True)
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

            for ippe in ppe_idx:
                sc = ippe['sim_config']  # each member knows its own config
                if ippe['global_id'] in nc_dict[sc][train_mp]['cic']:
                    train_data.append(nc_dict[sc][train_mp]['cic'][ippe['global_id']][var_name]['value'])
                    na_train.append(nc_dict[sc][train_mp]['cic'][ippe['global_id']]['na'])
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
                axs[ivar].scatter(na_train, train_data, label='Train PPE', s=5, color='tab:blue')

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

            axs[ivar].set_title(cl.output_var_set[var_name]['longname'])
            axs[ivar].set_xscale('log')
            axs[ivar].set_yscale('log')
            # if ivar == 4 or ivar == 5:
            #     axs[ivar].set_ylim([1e-8, 1e-1])

        plt.savefig(f"{plot_dir}{sim_config_str}_{block_name}.pdf")


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

