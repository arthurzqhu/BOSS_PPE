import os
import re
import numpy as np
import netCDF4 as nc
from glob import glob
import platform
import socket

# constants: {{{
split_bins = [15, 14]
col = np.log(2)/3.
M3toQ = np.pi/6*1e3
QtoM3 = 1/M3toQ
ampORbin = ['amp', 'bin']
bintype = ['tau', 'sbm']
if 'macOS' in platform.platform():
    output_dir = '/Volumes/ESSD/research/KiD_output/'
    bossppe_dir = '/Users/arthurhu/github/BOSS_PPE/'
    nc_dir = '/Users/arthurhu/github/BOSS_PPE/summary_ncs/'
elif 'Linux' in platform.platform():
    hostname = socket.gethostname()
    if hostname == "simurgh":
        output_dir = '/data1/arthurhu/KiD_output/'
        nc_dir = '/home/arthurhu/BOSS_PPE/summary_ncs/'
        bossppe_dir = '/home/arthurhu/BOSS_PPE/'
    else:
        output_dir = '/pscratch/sd/a/arthurhu/KiD_output/'
        nc_dir = '/pscratch/sd/a/arthurhu/BOSS_PPE/summary_ncs/'
        bossppe_dir = '/pscratch/sd/a/arthurhu/BOSS_PPE/'

initvarSet = ['Na','w','dm','rh','sp','mc','cm','dmr','pmomx','pmomy','spc','spr',
   'pmomxy','dz','Na','spcr','sprc'];
fullnameSet = ['Aerosol concentration', 'Maximum vertical velocity',
   'Mean mass diameter', 'Relative humidity', r'Shape parameter (\nu)',
   'Initial mass content','Mc','Mean mass diameter (rain)',
   'Predicted Moment X', 'Predicted Moment Y','Shape parameter (L1)',
   'Shape parameter (L2)','Predicted Moments M_x-M_y','Cloud thickness','Aerosol Concentration',
   r'Assumed Shape Parameter \nu_1-\nu_2',r'Shape parameter \nu_2-\nu_1'];
symbolSet = ['N_a', 'w_{max}', 'D_m', 'RH', r'\nu', 'm_i', 'm_c', 'D_mr','M^p_x',
   'M^p_y',r'\nu_c',r'\nu_r','M^p_{xy}',r'\Deltaz cloud','N_a','nu_1-nu_2','nu_2-nu_1'];
unitSet = [' [/mg]', ' [m/s]', r' [\mum]', ' [%]', '', ' [g/kg]', ' [g/kg]', 
   r' [\mum]', '', '','','','',' [m]',' [/mg]','',''];
initVarName_dict = dict(zip(initvarSet, fullnameSet))
initVarSymb_dict = dict(zip(initvarSet, symbolSet))
initVarUnit_dict = dict(zip(initvarSet, unitSet))

# thresholds to be considered as clouds 
cloud_mr_th = [1e-7, np.inf]; # kg/kg, threshold for mixing ratio (kg/kg)
rain_mr_th = [1e-7, np.inf];
lwc_mr_th = [1e-7, np.inf];
cloud_n_th = [1e-1, np.inf]; # #/cc, threshold for droplet number concentration
rain_n_th = [1e2, np.inf]; # #/m2
cwp_th = [1e-5, np.inf]; # kg/m2 cloud water path threshold
rwp_th = [1e-4, np.inf]; # kg/m2 rain water path threshold
meanD_th = [0, np.inf];
sppt_th = [0.1, np.inf]; # mm/hr surface precipitation
mean_rwp_th = [0.1, np.inf];
mean_rn_th = [50, np.inf];

indvar_name_set = ['diagM3_cloud','diagM3_rain',
                   'cloud_M1_path','rain_M1_path',['cloud_M1_path','rain_M1_path'],
                   'diagM0_cloud','diagM0_rain',
                   'albedo','opt_dep','mean_surface_ppt','RH',
                   'gs_deltac','gs_sknsc',
                   'gs_deltar','gs_sknsr',
                   ['cloud_M1_path', 'rain_M1_path', 'time'],
                   'Dm_c','Dm_r','Dm_w',
                   'dm_cloud_coll','dm_rain_coll','dm_sed',
                   'dm_cloud_ce','dm_rain_ce', 'dm_ce',
                   'dn_cloud_ce','dn_rain_ce', 'dn_ce',
                   'dqv_adv','cloud_M1_adv','rain_M1_adv',
                   'dtheta_mphys','dqv_mphys','cloud_M1_mphys','rain_M1_mphys',
                   'cloud_M1_mphys','rain_M1_mphys',
                   'reldisp','std_DSD',
                   'oflagc','oflagr',
                   'ss_w','ss_wpremphys','RH_premphys',
                   'dm_nuc',
                   'Dn_c', 'Dn_r', 
                   'cloud_M1', 'cloud_M2', 'cloud_M3', 'cloud_M4', 
                   'cloud_M1_mphys', 'cloud_M2_mphys', 'cloud_M3_mphys', 'cloud_M4_mphys',
                   'cloud_M1_adv', 'cloud_M2_adv', 'cloud_M3_adv', 'cloud_M4_adv', 
                   'cloud_M1_force', 'cloud_M2_force', 'cloud_M3_force', 'cloud_M4_force',
                   'rain_M1', 'rain_M2', 'rain_M3', 'rain_M4',
                   'rain_M1_mphys', 'rain_M2_mphys', 'rain_M3_mphys', 'rain_M4_mphys',
                   'rain_M1_adv', 'rain_M2_adv', 'rain_M3_adv', 'rain_M4_adv',
                   'rain_M1_force', 'rain_M2_force', 'rain_M3_force', 'rain_M4_force',
                   'nu_c', 'nu_r', 
                   ['diagM0_cloud', 'diagM0_rain'], ['diagM3_cloud', 'diagM3_rain'],'w','vapour',
                   'theta',['cloud_M2_path', 'rain_M2_path'],'dn_liq_coll',
                   'dmx_liq_coll', 'dmy_liq_coll',
                   'dn_liq_nuc', 'dn_liq_evap',
                   'mean_surface_ppt', ['mean_surface_ppt', 'time'], 
                   ['cloud_M1_path', 'rain_M1_path', 'time'], ['cloud_M1_path', 'rain_M1_path'],
                   'mean_D', 'dn_sed', 'dm6_sed', 'dm9_sed',
                   'boss_dm_sed','boss_dn_sed','boss_dmx_sed','boss_dmy_sed',
                   'diagM6_liq','diagM9_liq','rain_M1_path', ['cloud_M2_path', 'rain_M2_path'], 
                   'mean_surface_ppt', 'cloud_M1_path', 'mass_dist', 'num_dist',
                   'temperature', 'pressure', 'reff',
                   ['cloud_M1', 'rain_M1'], ['cloud_M2', 'rain_M2'],
                   ['cloud_M3', 'rain_M3'], ['cloud_M4', 'rain_M4'],
                   ['cloud_M3_path', 'rain_M3_path'], ['cloud_M4_path', 'rain_M4_path'],
                   ['cloud_M3_path', 'rain_M3_path'], ['cloud_M4_path', 'rain_M4_path'],
                   'dn_sed', 'dm_sed', 'dm6_sed', 'dm9_sed', 
                   ['cloud_M2', 'rain_M2'], ['cloud_M1', 'rain_M1'],
                   ['cloud_M3', 'rain_M3'], ['cloud_M4', 'rain_M4'],
                   'V_M0', 'V_M3', 'V_Mx', 'V_My',
                   'V_M0', 'V_M3', 'V_Mx', 'V_My',
                   'dn_liq_coll', 'dmx_liq_coll', 'dmy_liq_coll',
                   ['cloud_M2', 'rain_M2', 'cloud_M1', 'rain_M1']
                   ]

indvar_ename_set = ['CWC','RWC', #1
                    'CWP','RWP','LWP', #4
                    'Nc','Nr', #6
                    'albedo','optical depth','surface pcpt. rate','RH', #10
                    'GS delta (c)','GS skewness (c)', #13
                    'GS delta (r)','GS skewness (r)', #14
                    'cloud half-life', #15
                    'D_{m,c}','D_{m,r}','D_{m,w}', #18
                    'dm cloud by coll','dm_{r,coll}','dm by sed', #21
                    'dm cloud by CE','dm rain by CE', 'dm liq by CE', #24
                    'dn cloud by CE','dn rain by CE', 'dn liq by CE', #27
                    'dqv adv','Mc adv','Mr adv', #30
                    'dtheta mphys','dqv mphys','Mc mphys','Mr mphys', #34
                    'cloud M1 mphys','rain M1 mphys', #36
                    'relative dispersion','standard deviation', #38
                    'flag (cloud)','flag (rain)',  #40
                    'supersaturation', 'SS pre-mphys', 'RH pre-mphys', #43
                    'dm by nuc',  #44
                    'Dn_c', 'Dn_r',  #46
                    'Mc', 'Nc', 'cloud M3', 'cloud M4',  #50
                    'Mc mphys', 'Nc mphys', 'cloud M3 mphys', 'cloud M4 mphys',  #54
                    'Mc adv', 'Nc adv', 'cloud M3 adv', 'cloud M4 adv',  #58
                    'Mc force', 'Nc force', 'cloud M3 force', 'cloud M4 force',  #62
                    'Mr', 'Nr', 'rain M3', 'rain M4',  #66
                    'Mr mphys', 'Nr mphys', 'rain M3 mphys', 'rain M4 mphys',  #70
                    'Mr adv', 'Nr adv', 'rain M3 adv', 'rain M4 adv',  #74
                    'Mr force', 'Nr force', 'rain M3 force', 'rain M4 force',  #78
                    'nu_c', 'nu_r',  #80
                    'Nd', 'LWC', 'vertical velocity', 'vapour', #84
                    'theta','LNP', 'dn liq by coll', #87
                    'dmx liq by coll', 'dmy liq by coll', #89
                    'dn liq by nuc', 'dn liq by evap', #91
                    'peak rain rate', 'peak RR time', 'LWP half-life', 'mean_LWP', #95
                    'mean D_w', 'dn by sed', 'dmx by sed', 'dmy by sed', #99
                    'boss dm sed','boss dn sed','boss dmx sed','boss dmy sed', #103
                    'M6 liq','M9 liq', 'mean RWP', 'mean_LNP', #107
                    'mean rain rate', 'mean CWP', 'DSDm', 'DSDn', #111
                    'temperature', 'pressure', 'reff', #114
                    'M3', 'M0', #116
                    'Mx', 'My', #118
                    'Mx_path', 'My_path', #120
                    'mean_Mx_path', 'mean_My_path', #122
                    'mean_M0_sed', 'mean_M3_sed', 'mean_Mx_sed', 'mean_My_sed', #126
                    'mean_M0', 'mean_M3', 'mean_Mx', 'mean_My', #130
                    'mean_V_M0', 'mean_V_M3', 'mean_V_Mx', 'mean_V_My', #134
                    'V_M0', 'V_M3', 'V_Mx', 'V_My', #138
                    'mean_dm0_coal', 'mean_dmx_coal', 'mean_dmy_coal', #141
                    'mean_dm', #142
                    ]

indvar_units_set = [' [kg/kg]',' [kg/kg]',
                    ' [kg/$m^2$]',' [kg/$m^2$]',' [kg/$m^2$]',
                    ' [1/cc]',' [1/kg]',
                    '','',' [mm/hr]',' [%]',
                    '','',
                    '','',
                    ' [s]',
                    r' [\mum]',r' [\mum]',r' [\mum]',
                    ' [kg/kg/s]',' [kg/kg/s]',' [kg/kg/s]',
                    ' [kg/kg/s]',' [kg/kg/s]',' [kg/kg/s]',
                    ' [1/kg/s]',' [1/kg/s]',' [1/kg/s]',
                    ' [kg/kg/s]',' [kg/kg/s]',' [kg/kg/s]',
                    ' [kg/kg/s]',' [kg/kg/s]',' [kg/kg/s]',' [kg/kg/s]',
                    ' [kg/kg/s]',' [kg/kg/s]',
                    '',' [m]',
                    '','',
                    ' [%]',' [%]',' [%]',
                    ' [kg/kg/s]',
                    r' [\mum]', r' [\mum]', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '', '', '', '', 
                    '','',
                    ' [1/cc]', ' [kg/kg]', ' [m/s]',' [kg/kg]',
                    ' [K]',' [1/m^2]', ' [1/kg/s]', 
                    ' [$m^x$/kg/s]', ' [$m^y$/kg/s]', 
                    ' [1/kg/s]', ' [1/kg/s]', 
                    ' [mm/hr]', ' [s]', ' [s]', ' [kg/$m^2$]', 
                    r' [\mum]', ' [#/kg/s]', ' [$m^x$/kg/s]', ' [$m^y$/kg/s]', 
                    ' [kg/kg/s]', ' [#/kg/s]', ' [$m^x$/kg/s]', ' [$m^y$/kg/s]', 
                    ' [$m^6$/kg]', ' [$m^9$/kg]', ' [kg/$m^2$]', ' [1/$m^2$]', 
                    ' [mm/hr]', ' [kg/$m^2$]', ' [kg/kg/ln(r)]', ' [1/kg/ln(r)]',
                    ' [K]', ' [mb]', r' [\mum]',
                    ' [$m^3$/kg]', ' [1/kg]',
                    ' [$m^6$/kg]', ' [$m^9$/kg]',
                    ' [$m^6$/$m^2$]', ' [$m^9$/$m^2$]',
                    ' [$m^6$/$m^2$]', ' [$m^9$/$m^2$]',
                    ' [kg/kg/s]', ' [#/kg/s]', ' [$m^x$/kg/s]', ' [$m^y$/kg/s]',
                    ' [1/$m^2$]', ' [$m^3$/$m^2$]',
                    ' [$m^6$/$m^2$]', ' [$m^9$/$m^2$]',
                    ' [m/s]', ' [m/s]', ' [m/s]', ' [m/s]', 
                    ' [m/s]', ' [m/s]', ' [m/s]', ' [m/s]', 
                    ' [1/kg/s]', ' [$m^x$/kg/s]', ' [$m^y$/kg/s]', 
                    ' [m]',
                    ]

# }}}


def filter_DS_Store(file_list):
    return list(filter(lambda x: x != '.DS_Store', file_list))

def get_dics(output_dir, nikki, mconfig, n_init): 
    # get discrete initial conditions from BIN
    vars_strs = []
    mconfig_dir = output_dir + nikki + '/' + mconfig + '/'
    for i_init in range(n_init):
        var_strs = sort_strings_by_number(os.listdir(mconfig_dir))
        vars_strs.append(var_strs)
        mconfig_dir += var_strs[0]
    return vars_strs

def get_mps(output_dir, nikki, mconfig, l_cic, vars_strs):
    # get microphysics scheme name
    if l_cic:
        mps = filter_DS_Store(os.listdir(output_dir + nikki + '/' + mconfig))
    else:
        vars_dir = "/".join([istr[0] for istr in vars_strs])
        mps = filter_DS_Store(os.listdir(output_dir + nikki + '/' + mconfig + '/' + vars_dir))
    nmp = len(mps)
    return mps, nmp

def sort_strings_by_number(strings):
    """
    Sorts strings by their last numeric substring.
    e.g. ["a1", "a25", "a100"] becomes ["a1", "a25", "a100"] (sorted by 1, 25, 100).
    """
    float_re = re.compile(r'(\d+(?:\.\d+)?)$')  # match digits, optional .digits, at end
    
    def numeric_key(s):
        # Find the first sequence of digits in the string
        m = float_re.search(s)
        return float(m.group(1)) if m else 0.0
    return sorted(strings, key=numeric_key)

def var2phys(raw_data, var_name, var_ename, set_OOB_as_NaN, set_NaN_to_0):
    """convert raw_data into useful physical variables"""
    lin_or_log = 'log'
    threshold = 0.
    data_range = None

    # dn during collision is either nonpositive or nonnegative depending on user definition
    # threshold is set to negative inf here is temporary and only for PPE emulator training
    if var_name == "dn_liq_coll":
        threshold = -np.inf
        
    if type(var_name) == str:
        raw_data[var_name][~np.isfinite(raw_data[var_name])] = np.nan
        if 'mean' in var_ename:
            filtered_data = raw_data[var_name][raw_data[var_name]>=threshold]
            output_data = np.mean(filtered_data)
        else:
            output_data = raw_data[var_name]

    # laundry list of variables: {{{
    match var_ename:
        case 'D_{m,c}' | 'D_{m,r}' | 'D_{m,w}':
            output_data = raw_data[var_name]*1e6; # meter -> micron
            threshold = 1
            data_range = [1,3e3]
        case 'Nc':
            output_data = raw_data[var_name]/1e6;
            threshold = cloud_n_th[0]
            data_range = [1e-1, 3e3]
        case 'Nr':
            threshold = rain_n_th[0]
            data_range = [1e1, 1e5]
        case 'Nd':
            output_data = (raw_data['diagM0_rain'] + raw_data['diagM0_rain'])/1e6
            threshold = cloud_n_th[0]
            data_range = [1e-1, 3e3]
        case 'CWC':
            output_data = raw_data[var_name]*M3toQ
            threshold = cloud_mr_th[0]
            data_range = [1e-7, 1e-2]
        case 'RWC':
            output_data = raw_data[var_name]*M3toQ
            threshold = rain_mr_th[0]
            data_range = [1e-7, 1e-2]
        case 'LWC':
            output_data = (raw_data['diagM3_cloud'] + raw_data['diagM3_rain'])*M3toQ
            threshold = cloud_mr_th[0]
            data_range = [1e-7, 1e-2]
        case 'CWP':
            output_data = raw_data[var_name]*M3toQ
            threshold = cwp_th[0]
            lin_or_log = 'linear'
        case 'RWP':
            output_data = raw_data[var_name]*M3toQ
            threshold = rwp_th[0]
            lin_or_log = 'linear'
        case 'LWP':
            output_data = (raw_data['cloud_M1_path'] + raw_data['rain_M1_path'])*M3toQ
            threshold = cwp_th[0]
            lin_or_log = 'linear'
        case 'LNP':
            output_data = raw_data['cloud_M2_path'] + raw_data['rain_M2_path']
            lin_or_log = 'log'
        case 'surface pcpt. rate':
            output_data = raw_data[var_name]*3600
            threshold = sppt_th[0]
            lin_or_log = 'linear'
            data_range = [0, max(output_data)*2]
        case 'mean rain rate':
            output_data = np.mean(raw_data[var_name])*3600
        case 'vapour':
            data_range = [.002, .02];
            lin_or_log = 'linear';
        case 'RH':
            data_range = [30, 100];
            lin_or_log = 'linear';
        case 'peak rain rate':
            output_data = np.nanmax(raw_data[var_name])
        case 'peak RR time':
            idx = np.nanargmax(raw_data['mean_surface_ppt'])
            output_data = raw_data['time'][idx]
            if idx == 0:
                output_data = raw_data['time'][-1]
        case 'cloud half-life':
            idx = np.nanargmax(raw_data['cloud_M1_path']<raw_data['rain_M1_path'])
            output_data = raw_data['time'][idx]
            if idx == 0:
                dt = raw_data['time'][1] - raw_data['time'][0]
                t_didx_10min = int(600/dt)
                dCWPdt = (raw_data['cloud_M1_path'][-1-t_didx_10min] - \
                        raw_data['cloud_M1_path'][-1])/dt
                dCWP = raw_data['cloud_M1_path'][-1] - 0.5*max(raw_data['cloud_M1_path'])
                if dCWPdt == 0:
                    print(raw_data['cloud_M1_path'])
                    print(raw_data['rain_M1_path'])
                    input('a')
                    dCWPdt = 1e-5
                t_extra = dCWP/dCWPdt
                output_data = t_extra + raw_data['time'][-1]
                # output_data = raw_data['time'][-1]*2
        case 'LWP half-life':
            LWP = raw_data['cloud_M1_path'] + raw_data['rain_M1_path']
            LWP_max = np.nanmax(LWP)
            LWP_imax = np.nanargmax(LWP)
            idx_after_max = np.nanargmax(np.array(LWP[LWP_imax:]) <= LWP_max / 2)
            if idx_after_max == 0:
                output_data = raw_data['time'][-1]*2
            else:
                output_data = raw_data['time'][LWP_imax + idx_after_max] - raw_data['time'][LWP_imax]
        case 'mean_LWP':
            output_data = np.nanmean(raw_data['rain_M1_path'] + raw_data['cloud_M1_path'])
        case 'mean_LNP':
            output_data = np.nanmean(raw_data['rain_M2_path'] + raw_data['cloud_M2_path'])
        case 'M3' | 'M0' | 'Mx' | 'My' | 'Mx_path' | 'My_path' :
            output_data = raw_data[var_name[0]] + raw_data[var_name[1]]
        case 'mean_Mx_path' | 'mean_My_path':
            output_data = np.nanmean(raw_data[var_name[0]] + raw_data[var_name[1]])
        case 'mean_M0' | 'mean_M3' | 'mean_Mx' | 'mean_My':
            output_data = np.nanmean(raw_data[var_name[0]] + raw_data[var_name[1]])
        case 'mean_dm':
            output_data = np.nanmean(((raw_data['cloud_M1'] + raw_data['rain_M1'])/
                                     (raw_data['cloud_M2'] + raw_data['rain_M2']))**(1./3.))

    # }}}

    if set_OOB_as_NaN:
        try:
            output_data[output_data<threshold] = np.nan
        except:
            if output_data<threshold:
                output_data = np.nan

    if set_NaN_to_0:
        if isinstance(output_data, np.ma.core.MaskedArray):
            output_data[np.isnan(output_data)] = 0.

            # WARNING: temporary fix. can probably delete when you see this: {{{
            output_data.mask = np.ma.nomask
            output_data[output_data < -998] = 0.
            # }}}

        else:
            if np.isnan(output_data):
                output_data = 0.

    return output_data, lin_or_log, data_range

def load_KiD(file_info, var_interest, nc_dict, data_range, continuous_ic, 
             set_OOB_as_NaN=True, set_NaN_to_0=True):
    lin_or_log = {}
    mp = file_info['mp_config']

    # files are put below two extra layers of initial condition directory so need to check
    # before loading
    vars_vn = file_info['vars_vn']
    
    if continuous_ic:
        filedir = file_info['dir'] + file_info['date'] +'/'+ file_info['sim_config'] +'/'+ \
                mp +'/'+ 'KiD_m-*c-010*_v-' + file_info['version_number'] + '.nc'
        ic_str = 'cic' # = continuous initial condition
    else:
        ic_str = "".join(file_info['vars_str'])
        vars_dir = "/".join([istr for istr in file_info['vars_str']])
        filedir = file_info['dir'] + file_info['date'] +'/'+ file_info['sim_config'] +'/'+ \
                vars_dir +'/'+ mp +'/'+ 'KiD_m-*c-010*_v-' + file_info['version_number'] + '.nc'
        
    try:
        filedir = glob(filedir)[0]
    except IndexError:
        print('perhaps no such file under:' + filedir)

    dataset = nc.Dataset(filedir, mode='r')
    nc_dict['time'] = dataset.variables['time'][:]
    nc_dict['z'] = dataset.variables['z'][:]
    nc_dict.setdefault(ic_str, {})
    nc_dict[ic_str].setdefault(mp, {})
    data_range.setdefault(ic_str, {})

    # get initial conditions
    for vn in vars_vn:
        nc_dict[ic_str][mp][vn] = dataset.getncattr(vn)
        
    for ivar in var_interest:
        var_name = indvar_name_set[ivar]
        var_ename = indvar_ename_set[ivar]
        raw_data = {}
        if type(var_name) == list:
            for var_name_component in var_name:
                raw_data[var_name_component] = dataset.variables[var_name_component][:]
        else:
            raw_data[var_name] = dataset.variables[var_name][:]

        nc_dict[ic_str][mp][var_ename], lin_or_log[var_ename], data_range[ic_str][var_ename] = \
                    var2phys(raw_data, var_name, var_ename, set_OOB_as_NaN, set_NaN_to_0)
    return nc_dict, lin_or_log, data_range