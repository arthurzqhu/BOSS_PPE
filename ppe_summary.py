#!.venv/bin/python

import load_ppe_fun as lp
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
import re

vnum = '0001'
nikki = '2025-04-29'
sim_config = 'condcoll'
target_simconfig = 'condcoll'
csv_dir = 'PPE csv/'
l_cic = True

var1_strs, var2_strs = lp.get_dics(lp.output_dir, 'target', target_simconfig)
mps, nmp = lp.get_mps(lp.output_dir, nikki, sim_config, l_cic)
mps = lp.sort_strings_by_number(mps)
# condevp
snapshot_var_idx = [115, 116, 117, 118]
summary_var_idx = [95, 107, 121, 122]
# snapshot_var_idx = [115, 116]
# summary_var_idx = [95, 107]
# condcoll:
# snapshot_var_idx = [0, 1]
# summary_var_idx = [106, 109]
# fullmic:
# snapshot_var_idx = [0, 1, 9]
# summary_var_idx = [106, 107, 108, 109]
# sed:
# snapshot_var_idx = [21, 97, 98, 99]
# snapshot_var_idx = [0, 1, 9, 21]
# summary_var_idx = [106, 109]

var_interest = snapshot_var_idx + summary_var_idx

file_info = {'dir': lp.output_dir, 
             'date': nikki, 
             'version_number': vnum}

diag_dt = 60
diag_dz = 100

nc_dict = {}
data_range = {}
ppe_var = {}
target_var = {}

var1_str = var1_strs[0]
var2_str = var2_strs[0]

# load PPE data from BOSS
nc_summary_pkl_fn = lp.output_dir + nikki + '/' + sim_config + '_ncs_' + str(var_interest) + '.pkl'
if os.path.isfile(nc_summary_pkl_fn):
    with open(nc_summary_pkl_fn, 'rb') as file:
        nc_dict = pickle.load(file)
else:
    ic_str = 'cic'
    file_info.update({'sim_config': sim_config, 
                      'var1_str': var1_str, 
                      'var2_str': var2_str})
    for imp, mp in enumerate(tqdm(mps, desc='loading PPEs')):
        file_info['mp_config'] = mp
        nc_dict = lp.load_KiD(file_info, var_interest, nc_dict, data_range, 
                              continuous_ic=True, set_OOB_as_NaN=False, set_NaN_to_0=True)[0]

    # load BIN_TAU
    for var1_str in var1_strs:
        for var2_str in var2_strs:
            mp = 'BIN_TAU'
            file_info.update({'sim_config': target_simconfig, 
                              'date': 'target',
                              'mp_config': mp,
                              'var1_str': var1_str, 
                              'var2_str': var2_str})
            nc_dict = lp.load_KiD(file_info, var_interest, nc_dict, data_range, 
                                  continuous_ic=False, set_OOB_as_NaN=False, set_NaN_to_0=True)[0]

    with open(nc_summary_pkl_fn, 'wb') as file:
        pickle.dump(nc_dict, file)

# move from nc_dict into ppe_var/target_var
target_dfs = []
dt = nc_dict['time'][1] - nc_dict['time'][0]
dz = nc_dict['z'][1] - nc_dict['z'][0]
nt = len(nc_dict['time'])
nz = len(nc_dict['z'])

diag_t = [x for x in nc_dict['time'][1:] if x % diag_dt == 0]
diag_z = [z for z in nc_dict['z'][1:] if z % diag_dz == 0]

ic_str = 'cic'
for imp, mp in enumerate(tqdm(mps, desc='processing PPEs')):
    ppe_var.setdefault(mp, {})
    for ivar in summary_var_idx:
        var_ename = lp.indvar_ename_set[ivar]
        ppe_var[mp][var_ename] = nc_dict[ic_str][mp][var_ename]

    for ivar in snapshot_var_idx:
        var_ename = lp.indvar_ename_set[ivar]
        for idgt in diag_t:
            it = int(idgt / dt)
            if nc_dict[ic_str][mp][var_ename].shape[0] == nz:
                for idgz in diag_z:
                    iz = int(idgz / dz) - 1
                    ppe_var[mp][var_ename +'_'+ str(int(idgt)) +'_'+ str(int(idgz))] = \
                                        nc_dict[ic_str][mp][var_ename][iz, it]
                    # print(nc_dict[ic_str][mp][var_ename][iz, it])
                    # input('a')
            else:
                ppe_var[mp][var_ename +'_'+ str(int(idgt))] = \
                                    nc_dict[ic_str][mp][var_ename][it]


mp = 'BIN_TAU'
target_var.setdefault(mp, {})
var1_vn = re.search(r'^[A-Z]*[a-z]*', var1_str)[0]
var2_vn = re.search(r'^[A-Z]*[a-z]*', var2_str)[0]

for var1_str in var1_strs:
    for var2_str in var2_strs:
        ic_str = var1_str + var2_str
        target_var[mp][var1_vn] = nc_dict[ic_str][mp][var1_vn]
        target_var[mp][var2_vn] = nc_dict[ic_str][mp][var2_vn]
        for ivar in summary_var_idx:
            var_ename = lp.indvar_ename_set[ivar]
            target_var[mp][var_ename] = nc_dict[ic_str][mp][var_ename]

        for ivar in snapshot_var_idx:
            var_ename = lp.indvar_ename_set[ivar]
            for idgt in diag_t:
                if nc_dict[ic_str][mp][var_ename].shape[0] == nz:
                    for idgz in diag_z:
                        iz = int(idgz / dz) - 1
                        it = int(idgt / dt)
                        target_var[mp][var_ename +'_'+ str(int(idgt)) +'_'+ \
                                str(int(idgz))] = nc_dict[ic_str][mp][var_ename][iz, it]
                else:
                    target_var[mp][var_ename +'_'+ str(int(idgt))] = \
                            nc_dict[ic_str][mp][var_ename][it]
        target_dfs.append(pd.DataFrame(target_var).T)


ppe_df = pd.DataFrame(ppe_var).T
target_df = pd.concat(target_dfs)
ppe_df.to_csv(csv_dir + sim_config + "_LWP1234_ppe_var.csv")
target_df.to_csv(csv_dir + target_simconfig + "_LWP1234_target_var.csv")

# read the parameters
# param_dfs = []
param_files = []
var1_val = []
var2_val = []


for imp, mp in enumerate(mps):
    param_files.append(file_info['dir'] + nikki +'/'+ sim_config +'/'+ \
            mp +'/'+ 'params.csv')
    var1_val.append(nc_dict['cic'][mp][var1_vn])
    var2_val.append(nc_dict['cic'][mp][var2_vn])
param_df = pd.concat((pd.read_csv(f) for f in param_files))
param_df.insert(loc=0, column=var2_vn, value=var2_val)
param_df.insert(loc=0, column=var1_vn, value=var1_val)
param_df.index = mps

param_df.to_csv(csv_dir + sim_config + "_ppe_params.csv")
