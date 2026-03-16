#!/usr/bin/env python
# coding: utf-8

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# Hide JAX warnings if any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import socket
hostname = socket.gethostname()

import time
import multiprocessing
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import glob
import netCDF4 as nc
import pickle
import sys

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

import load_ppe_fun as lp
import tuning_fun_jax as tu_jax
import emulator_fun_jax as ef_jax
import MCMC_fun_jax as mf_jax

# Enable parallel CPU cores for MCMC chains
numpyro.set_host_device_count(4)

# GPU set up for JAX is automatic but you can check:
print("JAX Devices:", jax.devices())

l_multiple_cases = True

run1_fn = 'fullmp_dycoms_pccs_blcoal_lhs_momval_lwp0.05_N1000.nc'
run1_name = run1_fn.replace('.nc','')
run2_fn = run1_fn.replace('dycoms','rico')
run2_name = run2_fn.replace('.nc','')
var_select = None

# -----------------
# Pre-processing
# -----------------
params_train1 = ef_jax.get_params(lp.nc_dir, run1_fn)
transform_method = 'standard_scaler_asinh'
throw_away_ratio = 0

x1_train, x1_val, y1_train, y1_val, tgt1_data, _, tgt1_initvar_matrix, ppe1_info, scalers1 = \
    ef_jax.get_train_val_tgt_data(lp.nc_dir, run1_fn, params_train1, transform_method, 
                                  l_multi_output=False, set_nan_to_neg1001=True, var_select=var_select)

if l_multiple_cases:
    params_train2 = ef_jax.get_params(lp.nc_dir, run2_fn)
    x2_train, x2_val, y2_train, y2_val, tgt2_data, _, tgt2_initvar_matrix, ppe2_info, scalers2 = \
        ef_jax.get_train_val_tgt_data(lp.nc_dir, run2_fn, params_train2, transform_method, 
                                      l_multi_output=False, set_nan_to_neg1001=True, var_select=var_select)

nobs = ppe1_info['nobs']
nvar = ppe1_info['nvar']
npar = ppe1_info['npar']
ncases = ppe1_info['ncases']
n_init = ppe1_info['n_init']
npert  = ppe1_info['npert']
nparam_init = ppe1_info['nparam_init']
varcons1 = ppe1_info['var_constraints']

impact_factor = np.ones(nvar)
for i, var in enumerate(varcons1):
    if 'frac' in var: impact_factor[i] = 10.
    elif 'meanD' in var: impact_factor[i] = 2.
    elif any(s in var for s in ['M4', 'M6', 'prate']): impact_factor[i] = 10.
    elif any(s in var for s in ['M3', 'precip']): impact_factor[i] = 5.

# -----------------
# Emulator Training
# -----------------
proj_name1 = f'gp_{run1_name}'
if isinstance(transform_method, str): proj_name1 += f'_{transform_method}'
if throw_away_ratio > 0: proj_name1 += f'_throw_{throw_away_ratio}'
print('proj_name 1:', proj_name1)

models1 = {}
for ivar, varcon in enumerate(varcons1):
    # check if missing/masked targets exist and drop them or mask them in GP. 
    # For simplicity, filtering out -1001
    y_t = y1_train[varcon][:, :nobs[ivar]]
    # Flatten because we'll just train one scalar GP per feature or a multi-output if shaped right.
    # Assuming scalar output GP per variable constraint for illustration:
    # y_t is shaped (N, 1) usually.
    y_t_1d = y_t[:, 0]
    
    # only train on valid data
    valid_mask = y_t_1d > -999
    x_train_valid = x1_train[valid_mask]
    y_train_valid = y_t_1d[valid_mask]
    
    y_train_valid = y_t_1d[valid_mask]
    
    cache_dir = "models/gp_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{proj_name1}_{varcon}.pkl"
    
    if os.path.exists(cache_file):
        print(f"Loading cached GP for {varcon} Case 1 ...", flush=True)
        try:
            with open(cache_file, 'rb') as f:
                theta_opt = pickle.load(f)
            # Ensure it's a dict and convert NumPy values back to JAX if needed (TinyGP handles this)
        except Exception as e:
            print(f"Failed to load cache for {varcon}: {e}. Retraining...", flush=True)
            theta_opt = tu_jax.train_gp(x_train_valid, y_train_valid, n_epochs=1000, lr=0.05)
            with open(cache_file, 'wb') as f:
                pickle.dump(jax.device_get(theta_opt), f)
    else:
        print(f"Training GP for {varcon} Case 1 ...", flush=True)
        theta_opt = tu_jax.train_gp(x_train_valid, y_train_valid, n_epochs=1000, lr=0.05)
        with open(cache_file, 'wb') as f:
            pickle.dump(jax.device_get(theta_opt), f)
            
    models1[varcon] = theta_opt

if l_multiple_cases:
    proj_name2 = f'gp_{run2_name}'
    if isinstance(transform_method, str): proj_name2 += f'_{transform_method}'
    if throw_away_ratio > 0: proj_name2 += f'_throw_{throw_away_ratio}'
    print('proj_name 2:', proj_name2)
    
    varcons2 = ppe2_info['var_constraints']
    models2 = {}
    for ivar, varcon in enumerate(varcons2):
        y_t = y2_train[varcon][:, :nobs[ivar]]
        y_t_1d = y_t[:, 0]
        valid_mask = y_t_1d > -999
        x_train_valid = x2_train[valid_mask]
        y_train_valid = y_t_1d[valid_mask]
        
        cache_file = f"{cache_dir}/{proj_name2}_{varcon}.pkl"
        if os.path.exists(cache_file):
            print(f"Loading cached GP for {varcon} Case 2 ...", flush=True)
            try:
                with open(cache_file, 'rb') as f:
                    theta_opt = pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache for {varcon}: {e}. Retraining...", flush=True)
                theta_opt = tu_jax.train_gp(x_train_valid, y_train_valid, n_epochs=1000, lr=0.05)
                with open(cache_file, 'wb') as f:
                    pickle.dump(jax.device_get(theta_opt), f)
        else:
            print(f"Training GP for {varcon} Case 2 ...", flush=True)
            theta_opt = tu_jax.train_gp(x_train_valid, y_train_valid, n_epochs=1000, lr=0.05)
            with open(cache_file, 'wb') as f:
                pickle.dump(jax.device_get(theta_opt), f)
                
        models2[varcon] = theta_opt

# -----------------
# Validation
# -----------------
print('Validating GP Emulators...', flush=True)
ef_jax.plot_emulator_results_gp(x1_val, y1_val, models1, ppe1_info, transform_method, scalers1, x1_train, y1_train, l_plot_uncertainty=True, l_plot_scatter=True)
if l_multiple_cases:
    ef_jax.plot_emulator_results_gp(x2_val, y2_val, models2, ppe2_info, transform_method, scalers2, x2_train, y2_train, l_plot_uncertainty=True, l_plot_scatter=True)

# -----------------
# MCMC Setup
# -----------------
num_burnin_steps = 1000
num_samples = 2000
nchains = 4

param_interest_idx = params_train1['param_interest_idx']
orig_param_csv = f'{lp.param_dir}param_fixbmcoal.csv'
param_table = pd.read_csv(orig_param_csv)
param_names = param_table.iloc[param_interest_idx, 0].to_list()
param_mean = param_table.iloc[param_interest_idx, 1].to_numpy().astype(np.float32)
param_std = param_table.iloc[param_interest_idx, 3].to_numpy().astype(np.float32)

tgt1_mu_list = [np.mean(x, axis=1, keepdims=True) for x in tgt1_data]
tgt1_std_list = [np.std(x, axis=1, keepdims=True) for x in tgt1_data]
tgt1_mu = jnp.concatenate([jnp.array(x, dtype=jnp.float32) for x in tgt1_mu_list], axis=-1)
tgt1_std = jnp.concatenate([jnp.array(x, dtype=jnp.float32) for x in tgt1_std_list], axis=-1)

tgt1_sim_ics = np.concatenate(tgt1_initvar_matrix, axis=1)
n_tgt1_ics = tgt1_data[0].shape[0]
IC1_with_dummy = np.concatenate((tgt1_sim_ics, np.zeros([n_tgt1_ics, npar])), axis=1)
IC1_norm = jnp.array(scalers1['x'].transform(IC1_with_dummy)[:,:n_init], dtype=jnp.float32)

if l_multiple_cases:
    tgt2_mu_list = [np.mean(x, axis=1, keepdims=True) for x in tgt2_data]
    tgt2_std_list = [np.std(x, axis=1, keepdims=True) for x in tgt2_data]
    tgt2_mu = jnp.concatenate([jnp.array(x, dtype=jnp.float32) for x in tgt2_mu_list], axis=-1)
    tgt2_std = jnp.concatenate([jnp.array(x, dtype=jnp.float32) for x in tgt2_std_list], axis=-1)

    tgt2_sim_ics = np.concatenate(tgt2_initvar_matrix, axis=1)
    n_tgt2_ics = tgt2_data[0].shape[0]
    IC2_with_dummy = np.concatenate((tgt2_sim_ics, np.zeros([n_tgt2_ics, npar])), axis=1)
    IC2_norm = jnp.array(scalers2['x'].transform(IC2_with_dummy)[:,:n_init], dtype=jnp.float32)

deflate_factor = 0.3
W_rico = 1.0

# Precompute GP states outside of numpyro_model
print("Pre-conditioning GPs for Case 1...", flush=True)
gp_states1 = {}
for varcon in varcons1:
    gp_theta = models1[varcon]
    y_t_1d = y1_train[varcon][:, 0]
    valid_mask = y_t_1d > -999
    x_train_valid = x1_train[valid_mask]
    y_train_valid = y_t_1d[valid_mask]
    gp_states1[varcon] = tu_jax.GPState(gp_theta, jnp.asarray(x_train_valid), jnp.asarray(y_train_valid))

if l_multiple_cases:
    print("Pre-conditioning GPs for Case 2...", flush=True)
    gp_states2 = {}
    varcons2 = ppe2_info['var_constraints']
    for varcon in varcons2:
        gp_theta = models2[varcon]
        y_t_1d = y2_train[varcon][:, 0]
        valid_mask = y_t_1d > -999
        x_train_valid = x2_train[valid_mask]
        y_train_valid = y_t_1d[valid_mask]
        gp_states2[varcon] = tu_jax.GPState(gp_theta, jnp.asarray(x_train_valid), jnp.asarray(y_train_valid))


def numpyro_model():
    """
    Numpyro Model replacing TensorFlow Probability get_BOSSemu_lp.
    """
    # Prior for parameters
    theta = numpyro.sample("theta", dist.Uniform(jnp.zeros(npar), jnp.ones(npar)))
    
    # Replicate theta for each IC
    theta_rep = jnp.tile(theta[None, :], (n_tgt1_ics, 1))
    x_eval1 = jnp.concatenate([IC1_norm, theta_rep], axis=-1)
    
    # Emulate Case 1
    mu_list1, var_list1 = [], []
    for varcon in varcons1:
        # Vectorized prediction using precomputed Cholesky and manual formula
        mu, var = tu_jax.fast_predict(gp_states1[varcon], x_eval1)
        mu_list1.append(mu)
        var_list1.append(var)
        
    emu_mu1 = jnp.stack(mu_list1, axis=-1)
    emu_var1 = jnp.stack(var_list1, axis=-1)
    
    residual1 = tgt1_mu - emu_mu1
    mean_res1 = jnp.mean(jnp.abs(residual1), axis=0, keepdims=True)
    var_res1 = jnp.var(residual1, axis=0)
    
    sigma_eff1 = jnp.sqrt(var_res1 + jnp.square(mean_res1) + emu_var1 + jnp.square(tgt1_std))
    
    # Likelihood 1
    numpyro.factor("obs_lp1", deflate_factor * jnp.sum(dist.Normal(loc=tgt1_mu, scale=sigma_eff1).log_prob(emu_mu1) * impact_factor))
    
    if l_multiple_cases:
        theta_rep2 = jnp.tile(theta[None, :], (n_tgt2_ics, 1))
        x_eval2 = jnp.concatenate([IC2_norm, theta_rep2], axis=-1)
        
        mu_list2, var_list2 = [], []
        for varcon in varcons2:
            mu, var = tu_jax.fast_predict(gp_states2[varcon], x_eval2)
            mu_list2.append(mu)
            var_list2.append(var)
            
        emu_mu2 = jnp.stack(mu_list2, axis=-1)
        emu_var2 = jnp.stack(var_list2, axis=-1)
        
        residual2 = tgt2_mu - emu_mu2
        mean_res2 = jnp.mean(jnp.abs(residual2), axis=0, keepdims=True)
        var_res2 = jnp.var(residual2, axis=0)
        
        sigma_eff2 = jnp.sqrt(var_res2 + jnp.square(mean_res2) + emu_var2 + jnp.square(tgt2_std))
        
        # Likelihood 2
        numpyro.factor("obs_lp2", W_rico * deflate_factor * jnp.sum(dist.Normal(loc=tgt2_mu, scale=sigma_eff2).log_prob(emu_mu2) * impact_factor))

# -----------------
# Run MCMC
# -----------------
import sys 
print("Running MCMC...", flush=True)
sys.stdout.flush()

# Switching to HMC as requested, which can be faster than NUTS if step_size is set or adapted well
from numpyro.infer import HMC
hmc_kernel = HMC(numpyro_model)
mcmc = MCMC(hmc_kernel, num_warmup=num_burnin_steps, num_samples=num_samples, num_chains=nchains)
mcmc.run(jax.random.PRNGKey(0))
print("MCMC run complete.", flush=True)
sys.stdout.flush()

samples = mcmc.get_samples()

# Save as Arviz InferenceData as well
try:
    print('Saving posterior as ArviZ InferenceData...', flush=True)
    idata = az.from_numpyro(mcmc)
    idata.to_netcdf("models/mcmc_posterior_gp_jax.nc")
    print('ArviZ InferenceData saved to models/mcmc_posterior_gp_jax.nc', flush=True)
except Exception as e:
    print(f"Failed to save ArviZ data: {e}", flush=True)

posterior_theta = samples["theta"] # Shape (num_samples * nchains, npar)

# -----------------
# Post-Processing
# -----------------
print('Post-processing...', flush=True)
sys.stdout.flush()

try:
    print(f"DEBUG: posterior_theta type={type(posterior_theta)}", flush=True)
    sys.stdout.flush()
    
    samples_flat = np.array(posterior_theta)
    print(f"DEBUG: samples_flat converted to numpy, shape={samples_flat.shape}", flush=True)
    sys.stdout.flush()

    if samples_flat.ndim == 3:
        samples_flat = samples_flat.reshape(-1, npar)
        print(f"DEBUG: Reshaped samples_flat from 3D to 2D: {samples_flat.shape}", flush=True)
    elif samples_flat.ndim == 1:
        samples_flat = samples_flat.reshape(-1, npar)
        print(f"DEBUG: Reshaped samples_flat from 1D to 2D: {samples_flat.shape}", flush=True)
    sys.stdout.flush()

    # Inverse transform to original parameter space
    total_features = n_init + nparam_init
    print(f"DEBUG: n_init={n_init}, nparam_init={nparam_init}, total_features={total_features}, npar={npar}", flush=True)
    sys.stdout.flush()

    samples_full = np.zeros((samples_flat.shape[0], total_features))
    print(f"DEBUG: samples_full zeroes shape={samples_full.shape}", flush=True)
    sys.stdout.flush()

    ic_arr = np.array(IC1_norm[0, :n_init])
    print(f"DEBUG: ic_arr shape={ic_arr.shape}", flush=True)
    sys.stdout.flush()
    
    samples_full[:, :n_init] = ic_arr
    print("DEBUG: Filled ICs in samples_full", flush=True)
    sys.stdout.flush()

    print(f"DEBUG: param_interest_idx={param_interest_idx}", flush=True)
    sys.stdout.flush()
    for i, idx in enumerate(param_interest_idx):
        if i < samples_flat.shape[1]:
            samples_full[:, n_init + idx] = samples_flat[:, i]
    print("DEBUG: Filled parameters of interest in samples_full", flush=True)
    sys.stdout.flush()

    print(f"DEBUG: Final samples_full shape before inverse_transform: {samples_full.shape}", flush=True)
    sys.stdout.flush()
    
    scaler_x = scalers1['x']
    print(f"DEBUG: scaler type={type(scaler_x)}", flush=True)
    if hasattr(scaler_x, 'n_features_in_'):
        print(f"DEBUG: scaler expected features={scaler_x.n_features_in_}", flush=True)
    sys.stdout.flush()

    print("DEBUG: Calling inverse_transform...", flush=True)
    sys.stdout.flush()
    samples_raw_full = scalers1['x'].inverse_transform(samples_full)
    print(f"DEBUG: samples_raw_full shape={samples_raw_full.shape}", flush=True)
    sys.stdout.flush()

    samples_raw = samples_raw_full[:, n_init + param_interest_idx]
    print(f"DEBUG: samples_raw shape={samples_raw.shape}", flush=True)
    sys.stdout.flush()

except Exception as e:
    print("\n!!! POST-PROCESSING CRITICAL ERROR !!!", flush=True)
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    raise e

# Continue with plots if successful
posterior_reshaped = posterior_theta.reshape(nchains, num_samples, npar)
mf_jax.plot_traces(np.array(posterior_reshaped), param_names)

df_samples = pd.DataFrame(samples_raw, columns=param_names)
pairplot = sns.pairplot(df_samples, corner=True, kind="hist")
pairplot.fig.suptitle("MCMC Posterior Distributions (GP + Numpyro)", fontsize=16)
pairplot.fig.subplots_adjust(top=0.98)
pairplot.fig.savefig(f"{plot_dir}/corner_plot.png")
plt.show()

# Update param_table using MAP
updated_params = param_table.copy()
for iparam, param_name in enumerate(param_names):
    iparam_all = param_interest_idx[iparam]
    mu_samples_flat = samples_raw[:, iparam]
    x, density = az.kde(mu_samples_flat)
    map_index = np.argmax(density)
    map_estimate = x[map_index]
    updated_params.loc[iparam_all, 'mean'] = np.mean(mu_samples_flat) # was map/mean flip
    updated_params.loc[iparam_all, 'isd'] = np.std(mu_samples_flat)
    if 'map' in updated_params.columns:
        updated_params.loc[iparam_all, 'map'] = map_estimate
    
updated_params.to_csv(f'{lp.param_dir}/param_{run1_name}_gp.csv', index=False)
print("Finished GP MCMC Pipeline.")
