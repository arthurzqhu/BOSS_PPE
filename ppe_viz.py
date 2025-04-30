#!.venv/bin/python
import load_ppe_fun as lp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from time import sleep
import pickle
import warnings
import re
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm


vnum='0001'
nikki = '2025-04-29'
sim_config = 'condcoll'
l_cic = True
# sim_config = 'fullmic_Psed_r1_HMC'
target_simconfig = 'condcoll'

plot_dir = 'plots/' + nikki + '/' + sim_config + '/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# obtaining case related variables
mconfigs = os.listdir(lp.output_dir + nikki)
var1_strs, var2_strs = lp.get_dics(lp.output_dir, 'target', target_simconfig)
mps, nmp = lp.get_mps(lp.output_dir, nikki, sim_config, l_cic, var1_strs[0], var2_strs[0])
mps = lp.sort_strings_by_number(mps)

# var_interest = [106, 107] # see lp.indvar_ename_set
var_interest = [95, 107, 121, 122] # see lp.indvar_ename_set
# var_interest = [2,3,4] # see lp.indvar_ename_set
indvar_names = [lp.indvar_name_set[idx] for idx in var_interest]
indvar_enames = [lp.indvar_ename_set[idx] for idx in var_interest]
indvar_units = [lp.indvar_units_set[idx] for idx in var_interest]
# var_interest = [5, 10, 19, 48, 49, 64, 65, 82, 83, 93, 94, 96, 97]
file_info = {'dir': lp.output_dir, 
             'date': nikki, 
             'version_number': vnum}

var1_vn = re.search(r'^[A-Z]*[a-z]*', var1_strs[0])[0]
var2_vn = re.search(r'^[A-Z]*[a-z]*', var2_strs[0])[0]

nc_dict = {}
data_range = {}

# load PPE data from BOSS
nc_summary_pkl_fn = lp.output_dir + nikki + '/' + sim_config + '_ncs_' + str(var_interest) + '.pkl'
# data_range_pkl_fn = lp.output_dir + nikki + '/' + sim_config + '_dr_' + str(var_interest) + '.pkl'
if os.path.isfile(nc_summary_pkl_fn):
    with open(nc_summary_pkl_fn, 'rb') as file:
        nc_dict = pickle.load(file)
    # with open(data_range_pkl_fn, 'rb') as file:
    #     data_range = pickle.load(file)
else:
    # for var1_str in var1_strs:
    #     for var2_str in var2_strs:
    # send in the var_strs anyway to get the perturbed IC from global attributes
    file_info.update({'sim_config': sim_config, 
                      'var1_str': var1_strs[0], 
                      'var2_str': var2_strs[0]})

    # for imp, mp in enumerate(mps):
    for imp, mp in enumerate(tqdm(mps, desc='loading PPEs')):
        file_info['mp_config'] = mp
        nc_dict = lp.load_KiD(file_info, var_interest, nc_dict, data_range, continuous_ic=l_cic)[0]

    with open(nc_summary_pkl_fn, 'wb') as file:
        pickle.dump(nc_dict, file)

# load BIN_TAU
for var1_str in var1_strs:
    for var2_str in var2_strs:
        ic_str = var1_str + var2_str
        file_info.update({'sim_config': target_simconfig, 
                          'var1_str': var1_str, 
                          'var2_str': var2_str, 
                          'date': 'target',
                          'mp_config': 'BIN_TAU'})
        nc_dict, lin_or_log, data_range = \
                lp.load_KiD(file_info, var_interest, nc_dict, data_range, False)

# for imp, mp in enumerate(mps):
#     print(nc_dict['cic'][mp]['mean rain rate'])
# for var1_str in var1_strs:
#     for var2_str in var2_strs:
#         ic_str = var1_str + var2_str
#         print(nc_dict[ic_str]['BIN_TAU']['mean rain rate'])

# plotting
# for var1_str in var1_strs:
#     for var2_str in var2_strs:
# file_info.update({'var1_str': var1_str, 'var2_str': var2_str})
# ic_str = var1_str + var2_str
ic1_boss = np.zeros(nmp)
ic2_boss = np.zeros(nmp)
var_boss = np.zeros(nmp)
for var_ename, var_units in zip(indvar_enames, indvar_units):
    var_is_scalar = isinstance(nc_dict[ic_str]['BIN_TAU'][var_ename], np.float64)
    var_is_arr = isinstance(nc_dict[ic_str]['BIN_TAU'][var_ename], np.ma.core.MaskedArray)
    if var_is_scalar:
        plt.figure(figsize=(8, 4))

    # load BOSS PPEs
    if l_cic:
        ic_str = 'cic'
        for imp, mp in enumerate(mps):
            ic1_boss[imp] = nc_dict[ic_str][mp][var1_vn]
            ic2_boss[imp] = nc_dict[ic_str][mp][var2_vn]
            var_boss[imp] = nc_dict[ic_str][mp][var_ename]

    if var_is_arr and not l_cic:
        for var1_str in var1_strs:
            for var2_str in var2_strs:
                if var_is_arr:
                    plt.figure(figsize=(8, 4))
                # print(var1_str, var2_str)
                ic_str = var1_str + var2_str
                for mp in mps:
                    # print(nc_dict[ic_str].keys())
                    plt.plot(nc_dict['time'], nc_dict[ic_str][mp][var_ename], label=mp, color='tab:blue')
                plt.plot(nc_dict['time'], nc_dict[ic_str]['BIN_TAU'][var_ename], label='BIN_TAU', color='tab:orange')
                plt.legend()
                plt.ylabel(var_ename + var_units)
                plt.xlabel('Time [s]')
                plt.savefig(plot_dir + var_ename + ic_str + '.pdf')

    # if var_is_scalar:
    #     plt.scatter(ic1_boss, ic2_boss, c=var_boss, s=5)
    #     # plt.plot(nc_dict['time'], nc_dict[ic_str][mp][var_ename], 
    #     #          color='tab:blue', alpha=np.sqrt(1/nmp))
    # else:
    #     # create a video of for the time series
    #     warnings.warn('1D time series variable not implemented, will produce an animation')
    #     # fig, ax = plt.subplots(figsize=(8, 4))
    #     # im1 = ax.imshow(nc_dict[ic_str][mp][var_ename],aspect='auto',
    #     #                 extent=[0, 3600, 0, 6000],interpolation='none',
    #     #                 origin='lower')
    #     # cbar = fig.colorbar(im1, ax=ax, orientation='vertical', pad=0.02)
    #     # plt.xlabel('Time [s]')
    #     # plt.ylabel('Altitude [m]')
    #     # # plt.colorbar()
    #     # plt.savefig(plot_dir + var_ename + '_' + ic_str + \
    #     #         '_' + mp + '.pdf')
    # # #     plt.matshow(nc_dict[ic_str][mp][var_ename])

    mp = 'BIN_TAU'
    ic1_tau = np.array([])
    ic2_tau = np.array([])
    var_tau = np.array([])
    for var1_str in var1_strs:
        for var2_str in var2_strs:
            ic_str = var1_str + var2_str
            ic1_tau = np.append(ic1_tau, nc_dict[ic_str][mp][var1_vn])
            ic2_tau = np.append(ic2_tau, nc_dict[ic_str][mp][var2_vn])
            var_tau = np.append(var_tau, nc_dict[ic_str][mp][var_ename])

    vmax = np.max(np.concatenate((var_boss, var_tau)))
    vmin = np.min(np.concatenate((var_boss, var_tau)))
    # vmin = np.min(var_tau)
    # vmax = np.max(var_tau)
    if vmin > 0:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # if var_is_arr:
    #     plt.plot(nc_dict['time'], nc_dict[ic_str][mp][var_ename], label=mp, color='tab:orange')
    #     plt.legend()
    #     plt.ylabel(var_ename + var_units)
    #     plt.xlabel('Time [s]')
    #     plt.savefig(plot_dir + var_ename + '.pdf')

    if var_is_scalar:
        plt.scatter(ic1_boss, ic2_boss, c=var_boss, s=5, norm=norm)
        plt.scatter(ic1_tau, ic2_tau, c=var_tau, s=40, edgecolors=[1, 0, 0], norm=norm)
        plt.colorbar(label=var_ename + var_units)
#         # plt.plot(nc_dict['time'], nc_dict[ic_str][mp][var_ename], 
#         #          label=mp, color='tab:orange')
#         # plt.legend()
#         # plt.ylabel(var_ename + var_units)
#         # plt.xlabel('Time [s]')
#         # plt.title(sim_config)
#         # plt.yscale(lin_or_log[var_ename])
#         # if data_range[ic_str][var_ename] is not None:
#         #     plt.ylim(data_range[ic_str][var_ename])
        plt.savefig(plot_dir + var_ename + '.pdf')
#     else:
#         warnings.warn('2D variable not implemented')
#         # fig, ax = plt.subplots(figsize=(8, 4))
#         # im2 = ax.imshow(nc_dict[ic_str][mp][var_ename],aspect='auto',
#         #                 extent=[0, 3600, 0, 6000],interpolation='none',
#         #                 origin='lower')
#         # # plt.colorbar()
#         # cbar = fig.colorbar(im2, ax=ax, orientation='vertical', pad=0.02)
#         # plt.xlabel('Time [s]')
#         # plt.ylabel('Altitude [m]')
#         # plt.savefig(plot_dir + var_ename + '_' + ic_str + \
#         #         '_' + mp + '.pdf')
