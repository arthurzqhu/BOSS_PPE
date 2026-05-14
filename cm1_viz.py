#!/usr/bin/env python
# coding: utf-8

# # setup

# In[1]:


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
from dask.distributed import Client, progress
import joblib


# In[2]:


import importlib
importlib.reload(cl)


# In[ ]:


nikki = '2026-04-08'
target_nikki = 'target'
camp = 'rico'
sim_config = 'fullmp_rico_test_resol'
sim_config = sim_config.replace('dycoms', camp)
# sim_configs = [simr2_config]
l_pert = True
lwp_threshold = 0
ss_hr = 2
target_sim_config = 'fullmp_{camp}_tgt_pert'
# sim2cat_config = 'fullmp_dycoms_2cat'
# target_sim_config = 'fullmp_rico_tgt_10min'
# sim_config = 'boss_dycoms_2hr_NCE_satadj_ratio=1'
# target_sim_config = 'tau_dycoms_2hr_NCE'

sim_configs = [sim_config]

plot_dir = f"plots/{nikki}/{sim_config}/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

n_init = 1
target_mp = 'BIN-TAU'
train_mp = 'SLC-BOSS'
ref_mps = ['2CAT-BOSS']
# ref_mps = ['2CAT-BOSS', '2CAT-KK2000', '2CAT-SB2001']
mconfigs = os.listdir(cl.output_dir + nikki)
vars_strs, vars_vn = lp.get_dics(cl.output_dir, nikki, sim_config, n_init)
var_interest = ['M0_dmpath', 'M3_dmpath', 'M4_dmpath', 'M6_dmpath', 'prate_dm',
                'M0_dmprof', 'M3_dmprof', 'M4_dmprof', 'M6_dmprof',
                'M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss',
                'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss',
                'M6_99th_ss', 'meanD_dm_03_ss', 'v_precip_onset', 'precip_frac_ss',
                'prate_dm_ss', 'prate_ds_ss', 'prate_90th_ss',
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
                  }

tgt_file_info = {'dir': cl.output_dir,
                 'date': target_nikki,
                 'vars_vn': vars_vn,
                 'l_pert': l_pert,
                 'sim_config': target_sim_config,
                 'mp_config': target_mp,
                }

if 'nc_dict' not in globals():
    nc_dict = {}


# In[5]:


# load BOSS data

for initcond_combo in tqdm(itertools.product(*[vars_strs[0]])):
    train_file_info.update({'vars_str': initcond_combo})
    for sim_config in sim_configs:
        train_file_info.update({'sim_config': sim_config})
        cl.load_cm1(train_file_info, var_interest, ss_hr, nc_dict, False, lwp_threshold=lwp_threshold)
    # for ref_mp in ref_mps:
    #     train_file_info_2cat = train_file_info.copy()
    #     train_file_info_2cat.update({'sim_config': sim2cat_config, 'mp_config': ref_mp})
    #     cl.load_cm1(train_file_info_2cat, var_interest, ss_hr, nc_dict, False, lwp_threshold=lwp_threshold)


# In[6]:


tgt_jl_path = f"{cl.output_dir}/{target_nikki}/joblibs/{target_sim_config}.joblib"
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
                    finfo_target, vars_to_load, ss_hr, nc_dict=None, continuous_ic=False,
                    ipert=ipert, lwp_threshold=lwp_threshold
                )
                tasks.append(task)
        else:
            task = dask.delayed(cl.load_cm1)(
                finfo_target, vars_to_load, ss_hr, nc_dict=None, continuous_ic=False,
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


time = nc_dict[sim_config]['time']/3600

color_order = ['tab:blue', 'tab:orange']
mp_markers = ['o', '*']
idx_to_plot = [0, 1]
mp_labels = ['SLC-BOSS', 'BIN-TAU']
line_styles = ['-', '-']
line_widths = [2.5, 2.5]
all_sim_configs = [sim_config, target_sim_config]

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

x = nc_dict[sim_config]['x']
z = nc_dict[sim_config]['z']
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
                    axs[iax].fill_between(time, min_tgt, max_tgt, label=mp_labels[i], linewidth=3, alpha=0.3, color=color_order[i])
                else:
                    axs[iax].plot(time, nc_dict[sc][mp][case][var]['value'],
                                label=mp_labels[idx],
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
        # if 'M6' in var:
        # if 'M6' in var or 'M4' in var:
        axs[iax].set_yscale('log')
        # else:
        #     axs[iax].set_yscale('linear')
    # After plotting all lines, add a single, consolidated legend for the entire figure
    # Collect handles and labels from the first axis (can be any axis as labels are consistent)
    # Place the legend in the empty subplot location (bottom right, i.e., axs[-1])
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-2].legend(handles, labels, loc='center', fontsize=14, frameon=True, edgecolor='black', fancybox=True)
    axs[-2].axis('off')  # Hide the axes in the empty subplot

    # # Remove the unused subplot if present
    # if len(varsplot) < len(axs):
    #     for j in range(len(varsplot), len(axs)):
    #         fig.delaxes(axs[j])
    #     axs = axs[:len(varsplot)]

    plt.tight_layout()
    # Use bbox_inches='tight' to ensure the legend at the top is fully captured in the saved figure
    plt.savefig(f"{plot_dir}{case}_dm_path_r1.pdf", bbox_inches='tight')


# ## profs

# In[9]:


varsplot = ['M0_dmprof', 'M3_dmprof', 'M4_dmprof', 'M6_dmprof']
# varsplot = ['M0_dmprof', 'M3_dmprof', 'M4_dmprof', 'M6_dmprof', 'meanD_03_dm']

for case in vars_strs[0]:
    fig, axs = plt.subplots(len(color_order), len(varsplot), figsize=(12, 5), sharex=True, sharey=True)
    for i, var in enumerate(varsplot):
        j = 0
        units = cl.output_var_set[var]['var_unit']
        longname = cl.output_var_set[var]['longname']
        shortname = longname.replace("Domain-Mean", "DM")
        im_thiscol = [None] * len(color_order)
        for idx in idx_to_plot:
            sc = all_sim_configs[idx]
            for mp in nc_dict[sc].keys():
                if any(x in mp for x in ['time', 'x', 'y', 'z']):
                    continue
                if 'BIN' in mp and l_pert:
                    data = nc_dict[sc][mp][case][1][var]['value'].T.copy()
                else:
                    data = nc_dict[sc][mp][case][var]['value'].T.copy()
                # Ensure SLC-BOSS uses BIN-TAU's colorbar limits and set colorbar to log scale
                if 'TAU' not in mp:
                    # Find the corresponding BIN-TAU data for colorbar normalization
                    bin_data = nc_dict[target_sim_config][target_mp][case][1][var]['value'].T
                    vmin = np.percentile(bin_data[bin_data > 0], 1)
                    vmax = np.nanmax(bin_data)
                else:
                    # For BIN-TAU, also use log scale
                    data_pos = data[data > 0]
                    if data_pos.size > 0:
                        vmin = np.percentile(data_pos, 1)
                        vmax = np.nanmax(data)
                data[data <= vmin] = np.nan
                im = axs[j, i].pcolormesh(time, z, data, 
                                          norm=LogNorm(vmin=vmin, vmax=vmax), 
                                          cmap='jet', 
                                          rasterized=True)
                # Annotate the top right corner of each subplot with mp
                axs[j, i].annotate(mp_labels[j], xy=(0.97, 0.85), xycoords='axes fraction', ha='right', va='bottom', fontsize=12, fontweight='bold')
                im_thiscol[j] = im
                if j == 0:
                    axs[j, i].set_title(longname, fontsize=14)
                if i == 0:
                    axs[j, i].set_ylabel('Height [m]', fontsize=12)
                if j == len(color_order) - 1:
                    axs[j, i].set_xlabel('Time [hr]', fontsize=12)
                j += 1

        # Make colorbar its own row beneath both subplot rows for this column,
        # but *significantly* increase the gap so the colorbar is far below xlabels and there's no overlap.
        # This workaround uses fig.add_axes & manual positioning rather than tight_layout.
        # You may need to tune cb_pad for more/less vertical space.

        # Get bounding boxes for column i for both rows
        bb_row0 = axs[0, i].get_position(fig)
        bb_row1 = axs[1, i].get_position(fig)
        # The colorbar should span the union of the two widths, and be below the bottom axis
        cb_left = min(bb_row0.xmin, bb_row1.xmin)
        cb_right = max(bb_row0.xmax, bb_row1.xmax)
        cb_width = cb_right - cb_left
        cb_height = 0.025

        # Place colorbar well below the xlabels (adjust cb_pad as needed)
        cb_pad = 0.15  # Large enough gap for safety
        cb_bottom = bb_row1.ymin - cb_pad - cb_height

        cb_ax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
        cbar = fig.colorbar(im_thiscol[-1], cax=cb_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(f"{shortname} [{units}]", fontsize=12)
    # plt.tight_layout()
    # plt.savefig(f"{sim_config}_dm_prof.pdf")
    plt.savefig(f"{plot_dir}{sim_config}_{case}_dm_prof.pdf", bbox_inches='tight')


# # comparison between cases

# In[17]:


time = nc_dict[target_sim_config]['time']/3600
color_order = ['tab:blue', 'tab:orange']
mps = ['SLC-BOSS', 'BIN-TAU']
idx_to_plot = [0, 1]
mp_labels = ['SLC-BOSS MAP', 'TAU']
mp_markers = ['o', '*']
x = nc_dict[target_sim_config]['x']
z = nc_dict[target_sim_config]['z']*1e3
plt.rc('font', size=16)
cases = vars_strs[0]
var_na_sens = ['M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss',
               'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss',
                 'prate_dm_ss', 'prate_ds_ss', 'v_precip_onset', 'precip_frac_ss',
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
            axs[ivar].set_yscale('log')
            axs[ivar].set_xscale('log')
            axs[ivar].tick_params(axis='both', which='both', labelsize=12)
            i += 1

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, mp_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), fontsize=16, ncol=3)
fig.suptitle("Sensitivity of Domain-Mean Variables on $n_{aero}$", fontsize=20)
plt.tight_layout()

plt.savefig(f"{plot_dir}{sim_config}_na_sensitivity.pdf", bbox_inches='tight')
