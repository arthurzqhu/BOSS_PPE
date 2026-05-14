#!/usr/bin/env python
# coding: utf-8

import os
import socket
hostname = socket.gethostname()
if hostname == 'simurgh':
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# Avoid JAX pre-allocating 90% of GPU memory (similar to TF memory growth)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Hide JAX warnings if any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpyro
# Set host device count BEFORE JAX initialization
# This ensures parallel chains work on CPU if GPU is not used
numpyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
from jax import config

# Precision Toggle
# IMPORTANT: float64 is highly recommended for stable GP training to avoid singular matrices.
L_FLOAT64 = True # Set to True for numerical stability, False for faster GPU performance
config.update("jax_enable_x64", L_FLOAT64)
dtype = jnp.float64 if L_FLOAT64 else jnp.float32
print(f"Using {dtype} precision.")

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

from numpyro.infer import MCMC, NUTS, HMC
import numpyro.distributions as dist

# Detect devices
devices = jax.devices()
print("JAX Devices:", devices)
# More robust check: platform 'gpu' covers CUDA and other GPUs
use_gpu = any(d.platform == 'gpu' for d in devices)
print(f"Using GPU acceleration: {use_gpu}")

import load_ppe_fun as lp
import tuning_fun_jax as tu_jax
import emulator_fun_jax as ef_jax
import MCMC_fun_jax as mf_jax

l_multiple_cases = True

run1_fn = 'fullmp_dycoms_pccs_smmtrans_select_momval_lwp0.05_N750.nc'
run1_name = run1_fn.replace('.nc','')
run2_fn = run1_fn.replace('dycoms','rico')
run2_name = run2_fn.replace('.nc','')
var_select = None

plot_dir = f'plots/ppe/{run1_name}'
os.makedirs(plot_dir, exist_ok=True)

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
hurdle_vars = ppe1_info.get('hurdle_vars', set())
transformed_zeros = ppe1_info.get('transformed_zeros', [0.0] * nvar)

impact_factor = np.ones(nvar, dtype=np.float64 if L_FLOAT64 else np.float32)
for i, var in enumerate(varcons1):
    if 'frac' in var: impact_factor[i] = 10.
    elif 'meanD' in var: impact_factor[i] = 2.
    elif 'sfM' in var: impact_factor[i] = 0.25
    elif any(s in var for s in ['M4', 'M6', 'prate']): impact_factor[i] = 10.
    elif any(s in var for s in ['M3', 'precip']): impact_factor[i] = 5.

# -----------------
# Emulator Training
# -----------------
proj_name1 = f'gp_{run1_name}'
if isinstance(transform_method, str): proj_name1 += f'_{transform_method}'
if throw_away_ratio > 0: proj_name1 += f'_throw_{throw_away_ratio}'
proj_name1 += f'_p{npar}'
print('proj_name 1:', proj_name1)

t0_train = time.time()
cache_dir = "models/gp_cache"
os.makedirs(cache_dir, exist_ok=True)

def load_or_train_gp(cache_file, x_train_valid, y_train_valid, label=""):
    """Load cached GP or train a new one. Returns theta_opt dict."""
    if os.path.exists(cache_file):
        print(f"Loading cached GP for {label} ...", flush=True)
        try:
            with open(cache_file, 'rb') as f:
                theta_opt = pickle.load(f)
            if np.array(theta_opt['log_scale']).shape[0] != x_train_valid.shape[1]:
                raise ValueError(f"Cached GP has {np.array(theta_opt['log_scale']).shape[0]} dims, expected {x_train_valid.shape[1]}")
            return theta_opt
        except Exception as e:
            print(f"  Retraining ({e})...", flush=True)
    else:
        print(f"Training GP for {label} ...", flush=True)
    theta_opt = tu_jax.train_gp(x_train_valid, y_train_valid, n_epochs=3000, lr=0.005)
    with open(cache_file, 'wb') as f:
        pickle.dump(jax.device_get(theta_opt), f)
    return theta_opt

models1 = {}
for ivar, varcon in enumerate(varcons1):
    y_t = y1_train[varcon][:, :nobs[ivar]]
    y_t_1d = y_t[:, 0]
    valid_mask = y_t_1d > -999
    x_train_valid = x1_train[valid_mask]
    y_train_valid = y_t_1d[valid_mask]

    if varcon in hurdle_vars:
        # Presence GP: train on binary 0/1 for all valid members
        presence_targets = y1_train[f'presence_{varcon}'][:, 0].astype(np.float64)
        presence_targets_valid = presence_targets[valid_mask]
        cache_pres = f"{cache_dir}/{proj_name1}_presence_{varcon}.pkl"
        models1[f'presence_{varcon}'] = load_or_train_gp(
            cache_pres, x_train_valid, presence_targets_valid, f"presence_{varcon} Case 1")

        # Value-given-present GP: train only on non-zero members
        present_mask = valid_mask & (presence_targets > 0.5)
        x_train_present = x1_train[present_mask]
        y_train_present = y_t_1d[present_mask]
        cache_val = f"{cache_dir}/{proj_name1}_value_{varcon}.pkl"
        models1[f'value_{varcon}'] = load_or_train_gp(
            cache_val, x_train_present, y_train_present, f"value_{varcon} Case 1")
    else:
        # Standard single GP
        cache_file = f"{cache_dir}/{proj_name1}_{varcon}.pkl"
        models1[varcon] = load_or_train_gp(
            cache_file, x_train_valid, y_train_valid, f"{varcon} Case 1")

if l_multiple_cases:
    proj_name2 = f'gp_{run2_name}'
    if isinstance(transform_method, str): proj_name2 += f'_{transform_method}'
    if throw_away_ratio > 0: proj_name2 += f'_throw_{throw_away_ratio}'
    proj_name2 += f'_p{npar}'
    print('proj_name 2:', proj_name2)

    hurdle_vars2 = ppe2_info.get('hurdle_vars', set())
    transformed_zeros2 = ppe2_info.get('transformed_zeros', [0.0] * nvar)
    varcons2 = ppe2_info['var_constraints']
    models2 = {}
    for ivar, varcon in enumerate(varcons2):
        y_t = y2_train[varcon][:, :nobs[ivar]]
        y_t_1d = y_t[:, 0]
        valid_mask = y_t_1d > -999
        x_train_valid = x2_train[valid_mask]
        y_train_valid = y_t_1d[valid_mask]

        if varcon in hurdle_vars2:
            presence_targets = y2_train[f'presence_{varcon}'][:, 0].astype(np.float64)
            presence_targets_valid = presence_targets[valid_mask]
            cache_pres = f"{cache_dir}/{proj_name2}_presence_{varcon}.pkl"
            models2[f'presence_{varcon}'] = load_or_train_gp(
                cache_pres, x_train_valid, presence_targets_valid, f"presence_{varcon} Case 2")

            present_mask = valid_mask & (presence_targets > 0.5)
            x_train_present = x2_train[present_mask]
            y_train_present = y_t_1d[present_mask]
            cache_val = f"{cache_dir}/{proj_name2}_value_{varcon}.pkl"
            models2[f'value_{varcon}'] = load_or_train_gp(
                cache_val, x_train_present, y_train_present, f"value_{varcon} Case 2")
        else:
            cache_file = f"{cache_dir}/{proj_name2}_{varcon}.pkl"
            models2[varcon] = load_or_train_gp(
                cache_file, x_train_valid, y_train_valid, f"{varcon} Case 2")

t_train = time.time() - t0_train
print(f"Emulator training (or loading) took {t_train:.2f} seconds.")

# -----------------
# Validation
# -----------------
print('Validating GP Emulators...', flush=True)
# Plot emulator results and save them
ef_jax.plot_emulator_results_gp(x1_val, y1_val, models1, ppe1_info, transform_method, scalers1, x1_train, y1_train, l_plot_uncertainty=True, l_plot_scatter=True, savefig=True, prefix=run1_name)
plt.close('all')
if l_multiple_cases:
    ef_jax.plot_emulator_results_gp(x2_val, y2_val, models2, ppe2_info, transform_method, scalers2, x2_train, y2_train, l_plot_uncertainty=True, l_plot_scatter=True, savefig=True, prefix=run2_name)
    plt.close('all')

# -----------------
# MCMC Setup
# -----------------
num_burnin_steps = 1000
num_samples = 2000
nchains = 4

param_interest_idx = params_train1['param_interest_idx']
orig_param_csv = f'{lp.param_dir}param_smmtrans.csv'
param_table = pd.read_csv(orig_param_csv)
param_names = param_table.iloc[param_interest_idx, 0].to_list()
param_mean = param_table.iloc[param_interest_idx, 2].to_numpy().astype(np.float64 if L_FLOAT64 else np.float32)
param_std = param_table.iloc[param_interest_idx, 3].to_numpy().astype(np.float64 if L_FLOAT64 else np.float32)

tgt1_mu_list = [np.mean(x, axis=1, keepdims=True) for x in tgt1_data]
tgt1_std_list = [np.std(x, axis=1, keepdims=True) for x in tgt1_data]
tgt1_mu = jnp.concatenate([jnp.array(x, dtype=dtype) for x in tgt1_mu_list], axis=-1)
tgt1_std = jnp.concatenate([jnp.array(x, dtype=dtype) for x in tgt1_std_list], axis=-1)

tgt1_sim_ics = np.concatenate(tgt1_initvar_matrix, axis=1)
n_tgt1_ics = tgt1_data[0].shape[0]
IC1_with_dummy = np.concatenate((tgt1_sim_ics, np.zeros([n_tgt1_ics, npar])), axis=1)
IC1_norm = jnp.array(scalers1['x'].transform(IC1_with_dummy)[:,:n_init], dtype=dtype)
n_tgt1_ics = IC1_norm.shape[0]
print(f"Case 1: {n_tgt1_ics} ICs, {x1_train.shape[0]} training points.", flush=True)

if l_multiple_cases:
    tgt2_mu_list = [np.mean(x, axis=1, keepdims=True) for x in tgt2_data]
    tgt2_std_list = [np.std(x, axis=1, keepdims=True) for x in tgt2_data]
    tgt2_mu = jnp.concatenate([jnp.array(x, dtype=dtype) for x in tgt2_mu_list], axis=-1)
    tgt2_std = jnp.concatenate([jnp.array(x, dtype=dtype) for x in tgt2_std_list], axis=-1)

    tgt2_sim_ics = np.concatenate(tgt2_initvar_matrix, axis=1)
    n_tgt2_ics = tgt2_data[0].shape[0]
    IC2_with_dummy = np.concatenate((tgt2_sim_ics, np.zeros([n_tgt2_ics, npar])), axis=1)
    IC2_norm = jnp.array(scalers2['x'].transform(IC2_with_dummy)[:,:n_init], dtype=dtype)
    n_tgt2_ics = IC2_norm.shape[0]
    print(f"Case 2: {n_tgt2_ics} ICs, {x2_train.shape[0]} training points.", flush=True)

deflate_factor = 0.3
W_rico = 1.0

def stack_gps_hurdle(varcons, models, y_train_dict, x_train, h_vars, case_label=""):
    """Stack all GPs (presence + value for hurdle vars, standard for others).
    Returns stacked dict and index arrays for combine_hurdle_predictions."""
    print(f"Stacking GPs for {case_label}...", flush=True)
    all_alphas, all_Ls, all_amps, all_scales, all_X = [], [], [], [], []
    gp_entries = []  # list of (varcon, gp_type) tuples

    # Build ordered list of GPs to stack
    for varcon in varcons:
        if varcon in h_vars:
            gp_entries.append((varcon, 'presence'))
            gp_entries.append((varcon, 'value'))
        else:
            gp_entries.append((varcon, 'standard'))

    # Find max training size across all GPs for padding
    max_n = 0
    for varcon, gp_type in gp_entries:
        if gp_type == 'presence':
            y_t_1d = y_train_dict[varcon][:, 0]
            valid_mask = y_t_1d > -999
            max_n = max(max_n, int(np.sum(valid_mask)))
        elif gp_type == 'value':
            y_t_1d = y_train_dict[varcon][:, 0]
            presence = y_train_dict[f'presence_{varcon}'][:, 0]
            valid_mask = (y_t_1d > -999) & (presence > 0.5)
            max_n = max(max_n, int(np.sum(valid_mask)))
        else:
            y_t_1d = y_train_dict[varcon][:, 0]
            valid_mask = y_t_1d > -999
            max_n = max(max_n, int(np.sum(valid_mask)))

    # Build index maps for combine_hurdle_predictions
    presence_idx = np.zeros(len(varcons), dtype=int)
    value_idx = np.zeros(len(varcons), dtype=int)
    standard_idx = np.zeros(len(varcons), dtype=int)
    is_hurdle = np.zeros(len(varcons), dtype=bool)

    gp_counter = 0
    for ivar, varcon in enumerate(varcons):
        if varcon in h_vars:
            is_hurdle[ivar] = True
            presence_idx[ivar] = gp_counter
            gp_counter += 1
            value_idx[ivar] = gp_counter
            gp_counter += 1
        else:
            standard_idx[ivar] = gp_counter
            gp_counter += 1

    # Stack GP states with padding
    for varcon, gp_type in gp_entries:
        if gp_type == 'presence':
            gp_theta = models[f'presence_{varcon}']
            y_t_1d = y_train_dict[varcon][:, 0]
            valid_mask = y_t_1d > -999
            presence_targets = y_train_dict[f'presence_{varcon}'][:, 0].astype(np.float64)
            x_v = jnp.asarray(x_train[valid_mask])
            y_v = jnp.asarray(presence_targets[valid_mask])
        elif gp_type == 'value':
            gp_theta = models[f'value_{varcon}']
            y_t_1d = y_train_dict[varcon][:, 0]
            presence = y_train_dict[f'presence_{varcon}'][:, 0]
            present_mask = (y_t_1d > -999) & (presence > 0.5)
            x_v = jnp.asarray(x_train[present_mask])
            y_v = jnp.asarray(y_t_1d[present_mask])
        else:
            gp_theta = models[varcon]
            y_t_1d = y_train_dict[varcon][:, 0]
            valid_mask = y_t_1d > -999
            x_v = jnp.asarray(x_train[valid_mask])
            y_v = jnp.asarray(y_t_1d[valid_mask])

        n_curr = x_v.shape[0]
        state = tu_jax.GPState(gp_theta, x_v, y_v)

        alpha_pad = jnp.zeros(max_n, dtype=state.alpha.dtype)
        alpha_pad = alpha_pad.at[:n_curr].set(state.alpha)
        L_pad = jnp.eye(max_n, dtype=state.L.dtype)
        L_pad = L_pad.at[:n_curr, :n_curr].set(state.L)
        X_pad = jnp.zeros((max_n, x_v.shape[1]), dtype=x_v.dtype)
        X_pad = X_pad.at[:n_curr, :].set(x_v)

        all_alphas.append(alpha_pad)
        all_Ls.append(L_pad)
        all_amps.append(state.amplitude)
        all_scales.append(gp_theta["log_scale"])
        all_X.append(X_pad)

    scales_stacked = jnp.stack(all_scales)
    stacked = {
        "alphas": jnp.stack(all_alphas),
        "Ls": jnp.stack(all_Ls),
        "amps": jnp.stack(all_amps),
        "scales": scales_stacked,
        "X_trains_scaled": jnp.stack(all_X) / jnp.exp(scales_stacked)[:, None, :]
    }
    hurdle_info = {
        "is_hurdle": jnp.array(is_hurdle),
        "presence_idx": jnp.array(presence_idx),
        "value_idx": jnp.array(value_idx),
        "standard_idx": jnp.array(standard_idx),
    }
    return stacked, hurdle_info

stacked1, hurdle_info1 = stack_gps_hurdle(varcons1, models1, y1_train, x1_train, hurdle_vars, "Case 1")
hurdle_info1["transformed_zeros"] = jnp.array(transformed_zeros, dtype=dtype)

if l_multiple_cases:
    stacked2, hurdle_info2 = stack_gps_hurdle(varcons2, models2, y2_train, x2_train, hurdle_vars2, "Case 2")
    hurdle_info2["transformed_zeros"] = jnp.array(transformed_zeros2, dtype=dtype)

# Convert impact factor to JAX
jax_impact = jnp.array(impact_factor, dtype=dtype)

def predict_and_combine(stacked, hurdle_info, x_eval):
    """Run vmap GP prediction and combine hurdle outputs back to (nvar, n_points)."""
    raw_mu, raw_var = tu_jax.fast_predict_vmap(
        stacked["alphas"], stacked["Ls"], stacked["amps"], stacked["scales"],
        stacked["X_trains_scaled"], x_eval
    )
    emu_mu, emu_var = tu_jax.combine_hurdle_predictions(
        raw_mu, raw_var, hurdle_info["is_hurdle"],
        hurdle_info["presence_idx"], hurdle_info["value_idx"],
        hurdle_info["standard_idx"], hurdle_info["transformed_zeros"]
    )
    # Transpose to (n_ics, nvar) and handle NaN/Inf
    emu_mu = jnp.where(jnp.isfinite(emu_mu), emu_mu, 0.0).T
    emu_var = jnp.where(jnp.isfinite(emu_var), emu_var, 1.0).T
    return emu_mu, emu_var

def compute_likelihood(emu_mu, emu_var, tgt_mu, tgt_std):
    """Compute residuals and effective sigma for Student-T likelihood."""
    residual = jnp.where(jnp.isfinite(tgt_mu - emu_mu), tgt_mu - emu_mu, 0.0)
    mean_res = jnp.mean(jnp.abs(residual), axis=0, keepdims=True)
    var_res = jnp.var(residual, axis=0)
    sigma_eff_sq = var_res + jnp.square(mean_res) + emu_var + jnp.square(tgt_std)
    sigma_eff = jnp.sqrt(jnp.clip(sigma_eff_sq, a_min=1e-12))
    sigma_eff = jnp.where(jnp.isfinite(sigma_eff), sigma_eff, 1.0)
    sigma_eff = jnp.clip(sigma_eff, a_min=1e-3)
    return jnp.sum(dist.StudentT(df=3.0, loc=emu_mu, scale=sigma_eff).log_prob(tgt_mu) * jax_impact)

def numpyro_model():
    """Numpyro Model using vectorized GP predictions with hurdle support."""
    samples = []
    for name in param_names:
        s = numpyro.sample(name, dist.Uniform(jnp.array(0.0, dtype=dtype), jnp.array(1.0, dtype=dtype)))
        samples.append(s)
    theta = jnp.stack(samples)

    # Case 1
    theta_rep1 = jnp.tile(theta[None, :], (n_tgt1_ics, 1))
    x_eval1 = jnp.concatenate([IC1_norm, theta_rep1], axis=-1)
    emu_mu1, emu_var1 = predict_and_combine(stacked1, hurdle_info1, x_eval1)
    numpyro.factor("obs_lp1", deflate_factor * compute_likelihood(emu_mu1, emu_var1, tgt1_mu, tgt1_std))

    if l_multiple_cases:
        theta_rep2 = jnp.tile(theta[None, :], (n_tgt2_ics, 1))
        x_eval2 = jnp.concatenate([IC2_norm, theta_rep2], axis=-1)
        emu_mu2, emu_var2 = predict_and_combine(stacked2, hurdle_info2, x_eval2)
        numpyro.factor("obs_lp2", W_rico * deflate_factor * compute_likelihood(emu_mu2, emu_var2, tgt2_mu, tgt2_std))

# -----------------
# Run MCMC
# -----------------
import sys 
from numpyro.infer.initialization import init_to_median

print("\n--- MCMC Diagnostics ---", flush=True)
print(f"Target 1 Mu Stats: min={jnp.min(tgt1_mu)}, max={jnp.max(tgt1_mu)}, nan={jnp.any(jnp.isnan(tgt1_mu))}", flush=True)
if l_multiple_cases:
    print(f"Target 2 Mu Stats: min={jnp.min(tgt2_mu)}, max={jnp.max(tgt2_mu)}, nan={jnp.any(jnp.isnan(tgt2_mu))}", flush=True)
    # Check which variables are high in Case 2
    for ivar, varcon in enumerate(varcons2):
        v_mu = tgt2_mu[:, ivar]
        if jnp.max(jnp.abs(v_mu)) > 10.0:
            print(f"  WARNING: {varcon} Case 2 has extreme target: min={jnp.min(v_mu):.2f}, max={jnp.max(v_mu):.2f}", flush=True)

print("Running MCMC...", flush=True)
sys.stdout.flush()

# Use init_to_median for better stability if Uniform(0,1) fails
nuts_kernel = NUTS(numpyro_model, init_strategy=init_to_median)
mcmc = MCMC(nuts_kernel, num_warmup=num_burnin_steps, num_samples=num_samples, num_chains=nchains, 
            chain_method='vectorized' if use_gpu else 'parallel', progress_bar=False)
t0_mcmc = time.time()
mcmc.run(jax.random.PRNGKey(0))
t_mcmc = time.time() - t0_mcmc
print(f"\nMCMC run complete. Time: {t_mcmc:.2f} seconds.", flush=True)
print(f"Total time (Training + MCMC): {t_train + t_mcmc:.2f} seconds.", flush=True)
sys.stdout.flush()

samples = mcmc.get_samples()

# Save as Arviz InferenceData
try:
    print('Saving posterior as ArviZ InferenceData...', flush=True)
    os.makedirs("MCMC_posterior", exist_ok=True)
    idata = az.from_numpyro(mcmc)
    filename = f"MCMC_posterior/gp_jax_{run1_name}.nc"
    idata.to_netcdf(filename)
    print(f'ArviZ InferenceData saved to {filename}', flush=True)
except Exception as e:
    print(f"Failed to save ArviZ data: {e}", flush=True)

# Reconstruct posterior_theta from named samples
samples_list = [samples[name] for name in param_names]
posterior_theta = jnp.stack(samples_list, axis=-1)  # Shape: (nchains * num_samples, npar)

# -----------------
# Post-Processing
# (Can also be re-run by loading from ArviZ file)
# -----------------
print('Post-processing...', flush=True)
sys.stdout.flush()

try:
    samples_flat = np.array(posterior_theta)
    if samples_flat.ndim == 3:
        samples_flat = samples_flat.reshape(-1, npar)
    elif samples_flat.ndim == 1:
        samples_flat = samples_flat.reshape(-1, npar)
    print(f"DEBUG: samples_flat shape={samples_flat.shape}", flush=True)

    # The scaler 'x' was fitted on param_train['vals'] which has shape (N, nparam_init).
    # Layout [IC_0, ..., IC_{n_init-1}, P_0, ..., P_{npar-1}]
    # We create a zeroed array of that shape, then fill in the MCMC-sampled parameters
    # sequentially after the IC columns.
    print(f"DEBUG: nparam_init={nparam_init}, n_init={n_init}, npar={npar}", flush=True)
    sys.stdout.flush()

    # Build the full feature matrix (nparam_init columns) with a representative IC value
    # in the IC column(s), zeros elsewhere, then fill the perturbed params.
    samples_full = np.zeros((samples_flat.shape[0], nparam_init))

    # Fill IC column(s) with the mean IC value from training data
    ic_placeholder = scalers1['x'].transform(
        np.mean(params_train1['vals'], axis=0, keepdims=True)
    )  # shape (1, nparam_init)
    samples_full[:, :n_init] = ic_placeholder[0, :n_init]

    # Fill perturbed parameter columns sequentially
    for i in range(npar):
        col = n_init + i
        if col < nparam_init:
            samples_full[:, col] = samples_flat[:, i]

    print(f"DEBUG: samples_full shape before inverse_transform: {samples_full.shape}", flush=True)
    sys.stdout.flush()

    # Inverse transform back to physical parameter space
    samples_raw_full = scalers1['x'].inverse_transform(samples_full)
    print(f"DEBUG: samples_raw_full shape={samples_raw_full.shape}", flush=True)

    # Extract only the perturbed parameters (skip IC)
    samples_raw = samples_raw_full[:, n_init:]
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
mf_jax.plot_traces(np.array(posterior_reshaped), param_names, savepath=f"{plot_dir}/trace_{run1_name}.pdf")

df_samples = pd.DataFrame(samples_raw, columns=param_names)
pairplot = sns.pairplot(df_samples, corner=True, kind="hist")
pairplot.fig.suptitle("MCMC Posterior Distributions (GP + Numpyro)", fontsize=16)
pairplot.fig.subplots_adjust(top=0.98)
pairplot.fig.savefig(f"{plot_dir}/corner_{run1_name}.pdf")
plt.close()

# Update param_table using posterior mean and std
updated_params = param_table.copy()
for iparam, param_name in enumerate(param_names):
    iparam_all = param_interest_idx[iparam]
    mu_samples_flat = samples_raw[:, iparam]
    x, density, _ = az.kde(mu_samples_flat)
    map_index = np.argmax(density)
    map_estimate = x[map_index]
    updated_params.loc[iparam_all, 'mean'] = np.mean(mu_samples_flat)
    updated_params.loc[iparam_all, 'isd'] = np.std(mu_samples_flat)
    if 'map' in updated_params.columns:
        updated_params.loc[iparam_all, 'map'] = map_estimate

updated_params.to_csv(f'{lp.param_dir}/param_{run1_name}_gp.csv', index=False)
print("Finished GP MCMC Pipeline.", flush=True)
