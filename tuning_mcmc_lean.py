#!/usr/bin/env python
# coding: utf-8
"""Emulator + MCMC calibration, restructured.

Same pipeline and same numerics as tuning_mcmc.py. What changed is structure:
the two hardcoded cases became a list, the dead branches are gone, and the
per-chain Python loop in the log-prob became one batched expression.

Cache keys, seeds and file names are unchanged, so existing models/ and
MCMC_posterior/ artifacts stay valid.

Run `python tuning_mcmc_lean.py --selfcheck` to verify the vectorized
likelihood against the original per-chain loop without touching any data.
"""

import os
import json
import hashlib
import sys

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import socket

# Toggle whether HP tuning (keras_tuner / TF model training) runs on GPU or CPU.
# Must be set before `import tensorflow` -- CUDA_VISIBLE_DEVICES is only read
# once at TF init. Also inherited by the multiprocessing 'spawn' workers below.
HP_TUNING_USE_GPU = False

if not HP_TUNING_USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif socket.gethostname() == 'simurgh':
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"

import multiprocessing
import parallel_tuning
import netCDF4 as nc  # must be imported before TF on perlmutter
import time
import glob
import keras_tuner as kt
import tensorflow as tf

for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError as e:
        print(e)
print(tf.config.list_physical_devices('GPU'))

import keras
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import sklearn.model_selection as mod_sec
import pandas as pd
import arviz as az
import seaborn as sns
from tqdm import tqdm

import load_ppe_fun as lp
import cm1_load_utils as cl
import tuning_fun as tu
import MCMC_fun as mf
import emulator_fun as ef
import coarse_fun as cf

tfd = tfp.distributions
tfb = tfp.bijectors


# =====================================================================
# configuration
# =====================================================================

L_MULTIPLE_CASES = True

RUN1_FN = 'fullmp_D64bound_r1_pm_dycoms_arviz_lwp20_filtM3_dmpath_ss_r0.67_isect_N363.nc'
TRANSFORM_METHOD = 'standard_scaler_asinh'
THROW_AWAY_RATIO = 0

# =====================================================================
# coarse mode
# =====================================================================
# COARSE_MODE trades resolution for reachability. In the fine (original) setting every
# constraint variable enters as a Gaussian value term, which requires the emulator to
# predict magnitude everywhere. That fails for the rain-rate family: prate_dm_ss is
# ~60-70% exact zeros with the rest spread over five decades, so a Gaussian head cannot
# represent it at any N, and the MCMC then parks the posterior at whichever design edge
# is least bad and reports it confidently.
#
# Coarse mode keeps the well-behaved moments as value terms and COARSENS the rest to a
# threshold crossing, entering the likelihood as a Bernoulli term instead:
#
#   values : M0, M3, M4, M6 (both dm and ds paths)   -> Gaussian, unchanged
#   gates  : prate_dm_ss -> Bernoulli on 1[y > T]
#
# Applied to BOTH sides. Emulation: gated variables are trained on the 0/1 indicator, so
# the head learns a probability (a decision boundary, learnable from ~30 members) rather
# than a five-decade magnitude (not learnable from ~10 rainers). MCMC: the gate term is
# log p, which is unbounded below, so a region that produces no rain is penalised without
# limit instead of being scored relative to the rest of the box. That is the one thing a
# Gaussian value term structurally cannot do -- report that nothing in here works.
COARSE_MODE = False

# Ordered values-then-gates, so the emulator's output columns split cleanly.
COARSE_VALUE_VARS = ['M0_dmpath_ss', 'M3_dmpath_ss', 'M4_dmpath_ss', 'M6_dmpath_ss', 
                     'M0_dspath_ss', 'M3_dspath_ss', 'M4_dspath_ss', 'M6_dspath_ss']

# var -> (kind, value) threshold spec, in RAW physical units.
#   'abs'      : fixed threshold, same at every Na.
#   'tgt_frac' : T(Na) = frac * median_over_npert(target(Na)), Na-matched against the
#                target curve and interpolated onto each member's own na_PPE.
#
# prate_dm_ss at 1e-4 mm/hr is deliberately NOT below every TAU target: the DYCOMS
# target crosses it partway along the sweep (ON for Na <= 3.19e7, straddling at 3.76e7,
# OFF above), so the gate encodes where precipitation shuts off. RICO shuts off much
# later (ON through 7.3e7, straddling above), and that ~2.3x difference in shutoff Na is
# the DYCOMS/RICO divergence as a single crisp number.
#
# M6 was gated at tgt/1.5 in the first coarse run and is now a VALUE term instead. Two
# reasons that gate did not earn its place: at the posterior both M6 gates sat saturated
# near 1 across every Na, so they constrained nothing; and tgt/1.5 is a LOWER bound, i.e.
# it tests whether M6 is too low, while the measured failure is M6 too HIGH among the
# raining members (RICO rainers ~255x above target, Na-matched). "High enough to rain, low
# enough to match TAU" is two-sided, which a one-sided gate cannot express and a value
# term can.
COARSE_GATE_VARS = {
    'prate_dm_ss':  ('abs', 1e-4),
}

# Binary midpoint the gate probability is read off at. The head is trained on literal
# 0/1, and p = P(y_emu > 0.5) = Phi((mu - 0.5)/sigma) uses the head's own scale, so a
# member the emulator places near the boundary gets p near 0.5 rather than a confident
# call. Reading mu directly as a probability would be wrong -- CRPS is minimised by a
# quantile-like functional, not the mean, so mu is a monotone but uncalibrated function
# of P(y=1).
GATE_MID = 0.5
# Width, in DECADES, of the soft ramp on the gate's TRAINING label:
#     f = sigmoid( log10(y / T) / GATE_SOFT_DEX )
# None restores the hard indicator 1[y > T].
#
# A hard step discards how far a member sits from the threshold, which is expensive when
# only ~10 members per case are on the ON side, and it asks a smooth network to fit a
# discontinuous function of the parameters -- which it answers by hedging, part of why the
# first run's M6 gates came out saturated. 0.5 dex means a member a half-decade past the
# threshold trains as 0.73 rather than a flat 1.0, and one a half-decade short as 0.27.
# y = T maps to exactly 0.5 = GATE_MID, so the label the head learns and the point the
# likelihood reads p at are the same place.
#
# This is a separate concern from the SMOOTHNESS OF THE LIKELIHOOD, which is already
# handled by GATE_EPS below -- that one governs sampler geometry, this one governs what
# the emulator is asked to learn.
GATE_SOFT_DEX = 0.5
# Symmetric flip probability on the gate: p = GATE_EPS + (1 - 2 GATE_EPS) * Phi(z).
# Caps the veto at |log(GATE_EPS)| ~ 6.9 nats per entry, so p -> 0 cannot send one entry
# to -inf and poison the whole chain's scalar log-prob. Unlike the tf.clip_by_value this
# replaced, it is SMOOTH -- a clip has zero gradient outside its range and a step
# discontinuity at the boundary, which leapfrog integration turns into divergences.
# ponytail: fixed value. The principled version samples it like sigma_struct, giving the
# gate an inferred discrepancy; do that if the gates turn out to be over-confident.
GATE_EPS = 1e-3
# Per-variable weight for the VALUE terms, keyed on the moment. Multiplied by
# DM_DS_RATIO for the domain-mean variables, giving:
#
#     M0 / M3 / M4     dmpath 10      dspath 5
#     M6               dmpath  2      dspath 1
#
# i.e. M0/M3/M4 pull 5x M6, and dm pulls 2x ds. Only ratios matter -- the likelihood is
# divided by care_factor.mean() through deflate_factor, so scaling every entry by a
# constant cancels exactly.
MOMENT_WEIGHT = {'M0': 2.0, 'M3': 10.0, 'M4': 3.0, 'M6': 10.0}

# Base weight for the gate terms, before the dm multiplier. prate_dm_ss matches DM_MATCH,
# so its final weight is GATE_WEIGHT * DM_DS_RATIO -- 5.0 here puts the rain gate at 10,
# level with the M0/M3/M4 dmpath value terms.
GATE_WEIGHT = 5.0

# Weight on EACH domain-mean variable relative to EACH domain-sampled one. Per variable,
# not a balance of group totals.
# None disables it. Coarse mode only -- fine mode keeps its hand-tuned care_factor.
DM_DS_RATIO = 2.0
# Substring that puts a variable in each group. '_dm' deliberately catches BOTH the
# *_dmpath_* path integrals and prate_dm_ss, so the rain gate is weighted with the
# domain-mean group rather than sitting outside the balance.
DM_MATCH = '_dm'
DS_MATCH = '_ds'

VAR_SELECT = COARSE_VALUE_VARS + list(COARSE_GATE_VARS) if COARSE_MODE else None

# ---- data-split seeds ----
# HOLDOUT_FRAC members are carved off first as a COMPLETELY WITHHELD test set, used
# only for the validation-section plots (never for HP tuning or training). TEST_SEED
# draws that partition; VAL_SEED draws the HP-tuning train/val split within the
# remainder. Per-ensemble-member early-stopping splits use their own seeds.
#
# ONE val seed for every case, deliberately. Previously run1 got 101 and run2 got 202,
# so RICO was split one way as run2 of a joint run and another way as run1 of a
# single-case run: same data, different partition, different emulator, different
# data_sig, forcing a retrain for no reason. A dataset's split should be a property of
# the dataset, not of the slot it occupies. It is also the better statistical choice
# here: DYCOMS and RICO carry a byte-identical params_PPE, so one seed holds out the
# SAME parameter vectors in both regimes, which is what makes a joint-calibration
# holdout coherent.
HOLDOUT_FRAC = 0.15
TEST_SEED = 0
VAL_SEED = 101

MAX_TRIALS = 40
N_ENS = 10
N_TUNE_WORKERS = 10
N_TRAIN_WORKERS = 10

# MCMC
NCHAINS = 8
NUM_BURNIN = 5000
NUM_SAMPLES = 10000
N_PILOT = 500

SAMPLER = 'hmc'          # 'hmc' | 'nuts'
STEPSIZE = 0.05
HMC_LEAPFROG = 3
MAX_TREE = 8             # nuts only

# Relative weight on each case's likelihood. A tempering choice, not something the data
# can determine: it says how much each regime is trusted given structural error. 1.0 for
# both is the right default now that sigma_struct is inferred per variable per case -- a
# regime BOSS fits badly earns a large sigma_struct and stops dragging the other with it.
CASE_WEIGHTS = [1.0, 1.0]

# sigma_eff^2 = sigma_struct^2 + sigma_emu^2 + sigma_tgt^2, all in transformed units.
#   sigma_emu   emulator predictive sd (softplus of the CRPS scale head)
#   sigma_tgt   spread across the TAU target perturbation members
#   sigma_struct  structural discrepancy between BOSS and TAU. INFERRED, not chosen: one
#               extra sampled dimension per constraint variable per case carrying
#               u = log(sigma_struct), prior u ~ Normal(log MED, SD). The prior sits
#               directly on u, so there is no Jacobian term to add.
# Prior in transformed (StandardScaler) units where 1.0 is one PPE sd. Median 0.3
# matches the old sqrt(0.1) floor; sd 1.0 in log space spans ~0.04 to 2.2 at +/-2 sigma,
# weak enough to let a badly fit variable inflate, tight enough to stop it running away.
SIGMA_STRUCT_PRIOR_MED = 0.3
SIGMA_STRUCT_PRIOR_SD = 1.0

# ---- prior support: restrict sampling to the PPE design manifold ----
# PPE members are drawn as theta = mean + L z, with L the Cholesky factor of a previous
# posterior's covariance (ppe_util.f90:689 draw_mvnormal). When that source posterior is
# degenerate the design is rank deficient -- the 2cat designs carry r~0.998 within every
# (b_X, log_a_X) pair, giving 18 parameters but only ~9 independent directions. An
# independent box prior over all npar coordinates then puts essentially every likelihood
# evaluation off the manifold the emulator was trained on, because a rank-k sheet has
# zero volume inside its own npar-dimensional bounding box.
L_PCA_PRIOR = True
# Drop directions holding less than this share of design variance. 1e-3 sits inside the
# spectral gap for both schemes: the 2cat N200 design has 10 directions at 4.7-14.5% then
# a cliff to 1.7e-4 (factor ~270), while SLC advnu_r2 N400 bottoms out at 6.7e-3 and is
# kept whole at 28/28. Check the printed gap ratio before trusting a cut on a new design.
PCA_VAR_TOL = 1e-3

RUNDEETS = 'allunc'

RR_KEY = 'prate_dm_ss'
RR_THR_RAW = 1e-4
SW_KW = dict(w_zero=0.05, w_pos=1.0, smooth_alpha=6.0)


# =====================================================================
# helpers
# =====================================================================

def monitor_tuning_progress(directory, project_name, max_trials, procs=None):
    """Poll the tuner directory for started trials and drive a progress bar.

    `procs` is the worker process list. Without it this loop only exits when the trial
    count reaches max_trials, so a worker that dies mid-search leaves the count short and
    the main process spins here forever -- before it ever reaches p.join(). Watching the
    workers turns that hang into a short search.
    """
    pbar = tqdm(total=max_trials, desc="Tuning Progress")
    last = 0
    while last < max_trials:
        count = len(glob.glob(os.path.join(directory, project_name, 'trial_*/trial.json')))
        if count > last:
            pbar.update(count - last)
            last = count
        if count >= max_trials:
            break
        if procs is not None and not any(p.is_alive() for p in procs):
            print(f'\n  all tuning workers exited with {count}/{max_trials} trials started')
            break
        time.sleep(5)
    pbar.close()


def free_port():
    """An unused localhost port for the tuner oracle to bind."""
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_for_port(port, proc, timeout=180):
    """Block until the chief's gRPC server accepts connections.

    Workers construct an OracleClient channel at Tuner.__init__ and immediately issue
    RPCs, so starting them before the chief is listening loses the search.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not proc.is_alive():
            raise RuntimeError('oracle chief died before it started serving; '
                               'check its traceback above')
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f'oracle chief did not bind port {port} within {timeout}s')


def make_proj_name(run_name, info):
    """Emulator cache key. Identical to the original construction."""
    name = f'crps_mp_{run_name}'
    name += f'_{TRANSFORM_METHOD}' if isinstance(TRANSFORM_METHOD, str) else '_mixed_transform'
    if THROW_AWAY_RATIO > 0:
        name += f'_throw_{THROW_AWAY_RATIO}'
    if VAR_SELECT is not None:
        name += "".join(v[0] for v in VAR_SELECT)
    # Coarse mode rewrites the gated variables' training targets to 0/1 AFTER the loader
    # has run, so info['data_sig'] below cannot see it -- it only records var_select and
    # the transform. Without an explicit tag a coarse ensemble and a fine one would share
    # a cache key and silently load each other. The gate spec is hashed in too, so
    # changing a threshold forces a retrain instead of reusing a stale head.
    if COARSE_MODE:
        gsig = (json.dumps(COARSE_GATE_VARS, sort_keys=True)
                + f'|{GATE_MID}|{GATE_SOFT_DEX}')
        name += '_coarse' + hashlib.md5(gsig.encode()).hexdigest()[:6]
    # Data fingerprint. The key above describes the run and the transform NAME but
    # nothing about how the data was actually prepared, and the cache below accepts any
    # directory holding N_ENS matching files. Empty for the legacy config, so
    # pre-existing caches stay valid.
    sig = info.get('data_sig', '')
    if sig:
        name += f'_d{sig}'
    return name


def cache_ok(pname, info, npaths):
    """Refuse a cached ensemble whose sidecar is missing or disagrees.

    Raises rather than silently retraining: an unannounced 40-trial HP search is as
    unwelcome a surprise as a stale load.
    """
    if npaths != N_ENS:
        return False
    meta = f'models/{pname}_meta.json'
    if not os.path.exists(meta):
        raise RuntimeError(
            f'{pname}: found {npaths} cached models but no {meta}. These predate the '
            f'scaler-fit fix and were trained under a different normalization. Move them '
            f'aside to retrain, or pass scaler_fit="all", x_scaler_fit="all" to reproduce '
            f'the old behaviour.')
    with open(meta) as fh:
        if json.load(fh) != json.loads(info['data_sig_full']):
            raise RuntimeError(
                f'{pname}: cached models were trained under a different data '
                f'configuration than the one just built. Move them aside or change the '
                f'run name.')
    return True


def write_cache_meta(pname, info):
    with open(f'models/{pname}_meta.json', 'w') as fh:
        json.dump(json.loads(info['data_sig_full']), fh, indent=1, sort_keys=True)


def spawn_all(ctx, target, arglist, max_alive):
    """Start one process per arg tuple, never more than max_alive at once, then join."""
    procs = []
    for args in arglist:
        while sum(p.is_alive() for p in procs) >= max_alive:
            time.sleep(1)
        p = ctx.Process(target=target, args=args)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def build_target(tgt_data, label):
    """Target mean, sd, and a finite-mask over the TAU perturbation members.

    - nanmean/nanstd: tgt_data keeps literal NaN for masked entries (it never goes
      through emulator_fun._clean, unlike the training targets). Plain np.mean then puts
      NaN into tgt_mu, and because the likelihood reduce_sums every (ic, var) into one
      scalar per chain, a single NaN poisons the whole chain's log-prob rather than just
      dropping that datum.
    - ddof=1: with npert = 5 members, ddof=0 is biased low by sqrt(4/5) ~ 0.89.
    - a mask, so entries with no valid members are excluded from the likelihood instead
      of contributing a fabricated number.
    """
    mu_l, sd_l, mk_l = [], [], []
    for x in tgt_data:
        n_ok = np.sum(np.isfinite(x), axis=1, keepdims=True)
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = (np.nanstd(x, axis=1, keepdims=True, ddof=1) if x.shape[1] > 1
                  else np.zeros_like(mu))
        ok = np.isfinite(mu) & (n_ok > 0)
        mu_l.append(np.where(ok, mu, 0.))
        sd_l.append(np.where(np.isfinite(sd), sd, 0.))
        mk_l.append(ok.astype(np.float32))

    def cat(vals):
        return tf.concat([tf.cast(v, tf.float32) for v in vals], axis=-1)

    mu_t, sd_t, mk_t = cat(mu_l), cat(sd_l), cat(mk_l)
    ndrop = int(mk_t.shape[0] * mk_t.shape[1] - np.sum(mk_t.numpy()))
    if ndrop:
        print(f'  [{label}] {ndrop} target entries masked out (no finite members)')
    nzero = int(np.sum((sd_t.numpy() == 0) & (mk_t.numpy() > 0)))
    if nzero:
        print(f'  [{label}] {nzero} target entries have zero spread across the '
              f'{tgt_data[0].shape[1]} TAU members; sigma_struct carries their error')
    return mu_t, sd_t, mk_t


def apply_model_mixture(models, x_input, varcons, nobs, nchains, n_ic):
    """Combine the CRPS ensemble members as an equal-weight Gaussian mixture.

    ef.apply_model averages the packed [mu | raw_sigma] head across the members and the
    caller then softplus-es the AVERAGED raw scale. Two problems with that:

      - var_i(mu_i), the spread of the members' means, is discarded. That is the
        emulator's epistemic uncertainty, the term that grows where the design is sparse
        and the only thing that lets the likelihood widen automatically when the sampler
        leaves the training manifold.
      - softplus is convex, so softplus(mean(raw_i)) <= mean(softplus(raw_i)); even the
        aleatoric part came out biased low.

    Correct combination for an equal-weight Gaussian mixture:
        mu      = mean_i(mu_i)
        sigma^2 = mean_i(sigma_i^2) + var_i(mu_i)
    """
    if not isinstance(models, list):
        models = [models]
    outs = [m(x_input) for m in models]
    mu_l, sig_l = [], []
    for i, varcon in enumerate(varcons):
        mus, sgs = [], []
        for o in outs:
            full = tf.reshape(tf.cast(o[varcon], tf.float32), [nchains, n_ic, 2 * nobs[i]])
            mus.append(full[..., :nobs[i]])
            sgs.append(tf.nn.softplus(full[..., nobs[i]:]))
        mu_s, sig_s = tf.stack(mus), tf.stack(sgs)
        var_mix = (tf.reduce_mean(tf.square(sig_s), axis=0)
                   + tf.math.reduce_variance(mu_s, axis=0))
        mu_l.append(tf.reduce_mean(mu_s, axis=0))
        sig_l.append(tf.sqrt(var_mix))
    return tf.concat(mu_l, axis=-1), tf.concat(sig_l, axis=-1)


def case_loglik(emu_mu, emu_sigma, sig_struct, tgt_mu, tgt_sd, tgt_mask, care_factor,
                val_cols=None, gate_cols=None, tgt_gate=None, gate_mask=None,
                gate_care=None):
    """Per-chain log-likelihood for one case, batched over chains.

    emu_mu, emu_sigma : [nchains, n_ic, nvar_all]
    sig_struct          : [nchains, nvar_value]
    tgt_*             : [n_ic, nvar_value]     (already restricted to the value columns
                                                in coarse mode)
    tgt_gate/gate_mask: [n_ic, nvar_gate]
    returns           : [nchains]

    The mask is 0 where the target had no finite TAU member. Multiplying the per-point
    log-prob by it drops exactly those points instead of letting one NaN poison the whole
    chain's scalar log-prob.

    With gate_cols None this is the original expression, unchanged. In coarse mode the
    emulator's output columns split: val_cols keep the Gaussian value term, gate_cols get
    a Bernoulli term on the threshold crossing (see coarse_fun.gate_logprob).
    """
    def gauss(mu, sigma, t_mu, t_sd, mask, care, sd):
        sigma_eff = tf.sqrt(tf.square(sd)[:, None, :]
                            + tf.square(sigma)
                            + tf.square(t_sd)[None, ...])
        lp = tfd.Normal(loc=t_mu[None, ...], scale=sigma_eff).log_prob(mu)
        return tf.reduce_sum(lp * care * mask[None, ...], axis=[1, 2])

    # val_cols None means fine mode: every column is a value column, no gather, the
    # original expression verbatim.
    if val_cols is None:
        return gauss(emu_mu, emu_sigma, tgt_mu, tgt_sd, tgt_mask, care_factor, sig_struct)

    lp = gauss(tf.gather(emu_mu, val_cols, axis=-1),
               tf.gather(emu_sigma, val_cols, axis=-1),
               tgt_mu, tgt_sd, tgt_mask, care_factor, sig_struct)

    # Coarse mode with no gates at all: value terms only, on the selected subset. The
    # gather above still matters, so this cannot fall back to the fine-mode branch.
    if not gate_cols:
        return lp

    return lp + cf.gate_logprob(
        tf.gather(emu_mu, gate_cols, axis=-1),
        tf.gather(emu_sigma, gate_cols, axis=-1),
        tgt_gate, gate_mask, gate_care, GATE_MID, GATE_EPS)


def map_kde(flat_samples):
    """Marginal MAP of a 1-D sample vector via ArviZ's KDE."""
    x, density = az.kde(flat_samples)
    return x[np.argmax(density)]


def chain_draw(x, nchains):
    """TFP traces (draw, chain, ...); ArviZ wants (chain, draw, ...).

    A step size shared across chains comes back as (draw,) only, so broadcast it rather
    than letting ArviZ mis-read the draw axis as the chain axis.
    """
    a = np.asarray(x)
    if a.ndim == 1:
        a = np.repeat(a[:, None], nchains, axis=1)
    return np.swapaxes(a, 0, 1)


def build_sample_stats(dual_results, inner_results, nchains, ndraw, default_n_steps):
    """Collect sampler diagnostics into an ArviZ sample_stats dict.

    Field names follow ArviZ conventions (lp, diverging, step_size, acceptance_rate,
    n_steps, tree_depth, energy) so az.summary / az.plot_energy work without remapping.

    Every field is optional because the kernel stack differs by sampler: HMC reports no
    divergence flag AT ALL -- has_divergence, reach_max_depth, leapfrogs_taken and energy
    are NUTS-only. That is why an HMC run has no divergence count to save, and why
    n_steps falls back to the fixed leapfrog count.
    """
    stats = {}
    ir, dual = inner_results, dual_results

    def put(name, holder, attr, fn=None):
        # TFP's MetropolisHastings results nest the integrator's own fields one level down
        # in accepted_results -- target_log_prob in particular is NOT on the outer object,
        # so searching only the top level silently drops lp.
        for h in (holder, getattr(holder, 'accepted_results', None)):
            if h is not None and hasattr(h, attr):
                v = getattr(h, attr)
                v = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
                stats[name] = chain_draw(fn(v) if fn else v, nchains)
                return

    put('lp', ir, 'target_log_prob')
    put('acceptance_rate', ir, 'log_accept_ratio', lambda v: np.exp(np.minimum(0., v)))
    put('is_accepted', ir, 'is_accepted', lambda v: v.astype(bool))
    put('step_size', dual, 'new_step_size')
    put('diverging', ir, 'has_divergence', lambda v: v.astype(bool))
    put('reach_max_depth', ir, 'reach_max_depth', lambda v: v.astype(bool))
    put('tree_depth', ir, 'tree_depth')
    put('energy', ir, 'energy')
    put('n_steps', ir, 'leapfrogs_taken')
    if 'n_steps' not in stats:
        stats['n_steps'] = np.full((nchains, ndraw), int(default_n_steps), np.int32)

    for k, v in stats.items():
        if v.shape[:2] != (nchains, ndraw):
            raise RuntimeError(
                f'sample_stats[{k!r}] has shape {v.shape}, expected leading '
                f'({nchains}, {ndraw}); the draw/chain axes are probably transposed')
    return stats


# =====================================================================
# emulator
# =====================================================================

def load_or_train_emulator(icase, run_name, info, scalers, x_train, x_val,
                           y_train, y_val, pvar):
    """Load the cached CRPS ensemble for one case, or run HP search + train it."""
    proj_name = make_proj_name(run_name, info)
    print('proj_name:', proj_name)

    paths = glob.glob(f'models/{proj_name}_*.keras')
    if cache_ok(proj_name, info, len(paths)):
        loaded = []
        bad = []
        for p in paths:
            print(f"Loading existing model from {p}...")
            m = keras.models.load_model(p, compile=False)
            if any(not np.all(np.isfinite(w)) for w in m.get_weights()):
                bad.append(p)
            loaded.append(m)
        if bad:
            # Predates the finite-weights guard in parallel_tuning.run_training_worker,
            # or was written before that guard existed. Raise instead of silently
            # training on NaN predictions (see plot_emulator_results crash history).
            raise RuntimeError(
                f'{proj_name}: {len(bad)}/{len(paths)} cached models have non-finite '
                f'weights (diverged during a past training run): {bad}. Move these '
                f'aside to force a retrain.')
        return loaded

    varcons = info['var_constraints']
    nobs, nparam_init = info['nobs'], info['nparam_init']
    y_tr = {k: v for k, v in y_train.items() if 'presence_' not in k}
    y_va = {k: v for k, v in y_val.items() if 'presence_' not in k}

    def weights(yd):
        if pvar is None:
            return ()
        return (ef.make_weights_dict(yd, RR_KEY, RR_THR_RAW, info['eff0s'][pvar],
                                     scalers['y'][pvar].mean_, scalers['y'][pvar].scale_,
                                     **SW_KW),)

    sw = weights(y_tr) + weights(y_va)

    ctx = multiprocessing.get_context('spawn')

    # keras_tuner's native distributed protocol: one chief owns the oracle and hands out
    # trials over gRPC, workers proxy to it. Replaces the previous arrangement of
    # N independent RandomSearch instances sharing a directory, which raced on trial
    # numbering and corrupted trial_NN/checkpoint.weights.h5 ("file signature not found").
    port = free_port()
    chief = ctx.Process(
        target=parallel_tuning.run_tuning_chief,
        args=(port, nparam_init, varcons, nobs, MAX_TRIALS, proj_name, 'hp_tuning/crps'))
    chief.start()
    wait_for_port(port, chief)

    print(f"Launching {N_TUNE_WORKERS} parallel tuning workers for case {icase + 1} "
          f"(oracle chief on port {port})...")
    tune_args = [(i, x_train, y_tr, x_val, y_va, nparam_init, varcons, nobs,
                  MAX_TRIALS, proj_name, 'hp_tuning/crps', port) + sw
                 for i in range(N_TUNE_WORKERS)]
    procs = [ctx.Process(target=parallel_tuning.run_tuning_worker_dist, args=a)
             for a in tune_args]
    for p in procs:
        p.start()
    monitor_tuning_progress('hp_tuning/crps', proj_name, MAX_TRIALS, procs)
    for p in procs:
        p.join()

    # start_server returns on its own once the last worker disconnects; terminate only if
    # it is wedged, so a stuck chief cannot hang the run either.
    chief.join(timeout=120)
    if chief.is_alive():
        print('  oracle chief still alive after workers finished; terminating')
        chief.terminate()
        chief.join()

    tuner = kt.RandomSearch(
        lambda hp: tu.build_reg_crps_model(hp, nparam_init, varcons, nobs),
        objective="val_loss", max_trials=MAX_TRIALS,
        directory='hp_tuning/crps', project_name=proj_name)
    best_hps = tuner.get_best_hyperparameters(num_trials=N_ENS)

    print(f"Launching {N_TRAIN_WORKERS} parallel training workers for case "
          f"{icase + 1} ensemble...")
    train_args = []
    for i in range(N_ENS):
        y_tr_nn, y_va_nn = {}, {}
        for varcon in varcons:
            # per-member early-stopping split: seed 1000*(icase+1)+i keeps each ensemble
            # member's val distinct from the others and from the HP/test seeds.
            x_tr_nn, x_va_nn, y_tr_nn[varcon], y_va_nn[varcon] = mod_sec.train_test_split(
                x_train, y_tr[varcon], test_size=0.2, random_state=1000 * (icase + 1) + i)
        sw_nn = weights(y_tr_nn) + weights(y_va_nn)
        train_args.append((i, best_hps[i], x_tr_nn, y_tr_nn, x_va_nn, y_va_nn,
                           nparam_init, varcons, nobs, 1000, proj_name,
                           f'models/{proj_name}_{i}.keras') + sw_nn)
    spawn_all(ctx, parallel_tuning.run_training_worker, train_args, N_TRAIN_WORKERS)

    models = []
    for i in range(N_ENS):
        path = f'models/{proj_name}_{i}.keras'
        models.append(keras.models.load_model(path, compile=False))
        print(f"Loaded ensemble model {i} from {path}")
    write_cache_meta(proj_name, info)
    return models


# =====================================================================
# main
# =====================================================================

def main():
    # -----------------
    # data
    # -----------------
    run1_name = RUN1_FN.replace('.nc', '')
    fns = [RUN1_FN] + ([RUN1_FN.replace('dycoms', 'rico')] if L_MULTIPLE_CASES else [])
    run_names = [f.replace('.nc', '') for f in fns]
    run_name = run1_name.replace('dycoms', 'dycoms_rico') if L_MULTIPLE_CASES else run1_name
    ncase = len(fns)

    cases = []
    for fn in fns:
        params_train = ef.get_params(lp.nc_dir, fn)
        x_tr, x_va, y_tr, y_va, tgt_data, _, tgt_initvar, info, scalers = \
            ef.get_train_val_tgt_data(
                lp.nc_dir, fn, params_train, TRANSFORM_METHOD,
                l_multi_output=True, set_nan_to_neg1001=True, var_select=VAR_SELECT,
                holdout_frac=HOLDOUT_FRAC, test_seed=TEST_SEED, random_state=VAL_SEED,
                scaler_fit='train', x_scaler_fit='all')
        cases.append(dict(fn=fn, name=fn.replace('.nc', ''), params_train=params_train,
                          x_train=x_tr, x_val=x_va, y_train=y_tr, y_val=y_va,
                          tgt_data=tgt_data, tgt_initvar=tgt_initvar,
                          info=info, scalers=scalers))

    c0 = cases[0]
    info0 = c0['info']
    for c in cases[1:]:
        assert np.all(info0['var_constraints'] == c['info']['var_constraints'])

    varcons = info0['var_constraints']
    nobs = info0['nobs']
    nvar, npar, n_init = info0['nvar'], info0['npar'], info0['n_init']
    nparam_init = info0['nparam_init']

    # emulator output-column layout: apply_model_mixture concatenates nobs[i] columns per
    # variable in varcons order, so this maps a variable name to its column indices.
    col_start = np.cumsum([0] + list(nobs))[:-1]

    def cols_of(v):
        i = varcons.index(v)
        return list(range(int(col_start[i]), int(col_start[i]) + int(nobs[i])))

    val_cols = gate_cols = gate_care = None
    if COARSE_MODE:
        missing = [v for v in VAR_SELECT if v not in list(varcons)]
        if missing:
            raise KeyError(f'coarse-mode variables absent from {RUN1_FN}: {missing}. '
                           f'available: {list(varcons)}')
        if n_init != 1:
            raise NotImplementedError(
                f'coarse-mode gates Na-match on a single init variable; n_init={n_init}')
        bad_nobs = {v: nobs[varcons.index(v)] for v in COARSE_GATE_VARS
                    if nobs[varcons.index(v)] != 1}
        if bad_nobs:
            raise NotImplementedError(f'gated variables must have nobs == 1, got {bad_nobs}')
        # care_factor is indexed by output COLUMN below, and the existing likelihood
        # broadcasts a length-nvar care_factor against a [.., .., ncol] tensor, so the
        # whole file already assumes one column per variable. Check it rather than let a
        # multi-obs variable fail as an opaque broadcast error.
        if int(np.sum(nobs)) != nvar:
            raise NotImplementedError(
                f'coarse mode assumes one output column per variable; '
                f'sum(nobs)={int(np.sum(nobs))} but nvar={nvar}')
        val_cols = [c for v in COARSE_VALUE_VARS for c in cols_of(v)]
        gate_cols = [c for v in COARSE_GATE_VARS for c in cols_of(v)]
        value_vars = list(COARSE_VALUE_VARS)
        if not COARSE_VALUE_VARS:
            raise ValueError('COARSE_VALUE_VARS is empty: nothing would constrain the '
                             'likelihood. Gates alone cannot carry a calibration.')
        print(f'\nCOARSE_MODE: {len(value_vars)} value vars {value_vars}')
        if gate_cols:
            print(f'             {len(COARSE_GATE_VARS)} gates {list(COARSE_GATE_VARS)}')
            print(f'             GATE_WEIGHT={GATE_WEIGHT}  DM_DS_RATIO={DM_DS_RATIO}')
        else:
            # Legitimate configuration: coarse mode is then just a variable SUBSET with
            # custom weights, no binarization and no Bernoulli term anywhere.
            print(f'             no gates -- value terms only, DM_DS_RATIO={DM_DS_RATIO}')
    else:
        value_vars = list(varcons)

    # care_factor spans ALL columns; in coarse mode it is split before it reaches
    # case_loglik, gates carrying GATE_WEIGHT instead.
    care_factor = np.ones(nvar, dtype=np.float32)
    if COARSE_MODE:
        for i, var in enumerate(varcons):
            if var in COARSE_GATE_VARS:
                care_factor[i] = GATE_WEIGHT
                continue
            # Keyed on the leading moment token, so a renamed path suffix cannot silently
            # fall through to a default weight.
            key = next((k for k in MOMENT_WEIGHT if var.startswith(k)), None)
            if key is None:
                raise KeyError(
                    f'no MOMENT_WEIGHT entry for value variable {var!r}; known moments '
                    f'{sorted(MOMENT_WEIGHT)}. Add it rather than letting it default.')
            care_factor[i] = MOMENT_WEIGHT[key]

        # Domain-mean / domain-sampled balance, PER VARIABLE: every dm variable gets
        # DM_DS_RATIO times the weight of every ds variable. Not a balance of group
        # totals, so the summed ratio comes out as DM_DS_RATIO * n_dm / n_ds and is only
        # reported, not targeted.
        #
        # DM_MATCH = '_dm' puts prate_dm_ss in the dm group alongside the *_dmpath_*
        # variables, so it carries the same weight as them (exactly equal while
        # GATE_WEIGHT is 1; scaled by GATE_WEIGHT otherwise, since that knob is what
        # separates gate pull from value pull).
        if DM_DS_RATIO is not None:
            dm = np.array([DM_MATCH in v for v in varcons])
            ds = np.array([DS_MATCH in v for v in varcons])
            if not dm.any() or not ds.any():
                raise ValueError(
                    f'DM_DS_RATIO needs both groups present; found {int(dm.sum())} '
                    f'{DM_MATCH!r} and {int(ds.sum())} {DS_MATCH!r} variables in '
                    f'{list(varcons)}')
            if np.any(dm & ds):
                raise ValueError(
                    f'DM_MATCH={DM_MATCH!r} and DS_MATCH={DS_MATCH!r} both match '
                    f'{[v for k, v in enumerate(varcons) if dm[k] and ds[k]]}')
            care_factor[dm] *= np.float32(DM_DS_RATIO)
            print(f'  per-variable weight: dm {np.unique(care_factor[dm])} vs '
                  f'ds {np.unique(care_factor[ds])} (ratio {DM_DS_RATIO})')
            print(f'  resulting group sums: dm {care_factor[dm].sum():.4g} '
                  f'({int(dm.sum())} vars) / ds {care_factor[ds].sum():.4g} '
                  f'({int(ds.sum())} vars) = '
                  f'{care_factor[dm].sum() / care_factor[ds].sum():.4g}')
            excl = [v for k, v in enumerate(varcons) if not dm[k] and not ds[k]]
            if excl:
                print(f'  outside both groups, weight left as-is: {excl}')
    else:
        for i, var in enumerate(varcons):
            if any(s in var for s in ('M4', 'M6')):
                care_factor[i] = 3.
            elif 'thickness' in var:
                care_factor[i] = 5.
            if any(s in var for s in ('M3',)):
                care_factor[i] = 15.
            elif any(s in var for s in ('precip', 'prate', 'Dtail')):
                care_factor[i] = 8.

    if COARSE_MODE:
        care_val = np.asarray(care_factor)[val_cols].astype(np.float32)
        gate_care = (np.asarray(care_factor)[gate_cols].astype(np.float32)
                     if gate_cols else None)
    else:
        care_val = care_factor

    # `is not None`, not truthiness: pvar is an index, and index 0 is a real variable.
    # Disabled whenever RR_KEY is GATED, because its target is then a literal 0/1 and the
    # rain-rate sample weighting -- which ramps on the TRANSFORMED continuous value around
    # a transformed threshold -- has nothing left to act on. With no gates configured the
    # rain rate is an ordinary continuous target again, so the weighting is meaningful and
    # stays enabled.
    if RR_KEY in COARSE_GATE_VARS and COARSE_MODE:
        pvar = None
    else:
        pvar = varcons.index(RR_KEY) if RR_KEY in varcons else None

    # -----------------
    # coarse-mode targets: binarize BEFORE any training happens
    # -----------------
    if COARSE_MODE and COARSE_GATE_VARS:
        for i, c in enumerate(cases):
            print(f'\nbuilding gate targets (soft_dex={GATE_SOFT_DEX}), '
                  f'case {i + 1} ({c["name"]}):')
            cf.make_gate_targets(c, varcons, COARSE_GATE_VARS, TRANSFORM_METHOD,
                                 soft_dex=GATE_SOFT_DEX, label=f'case {i + 1}')

    # -----------------
    # emulators
    # -----------------
    for icase, c in enumerate(cases):
        c['models'] = load_or_train_emulator(
            icase, c['name'], c['info'], c['scalers'],
            c['x_train'], c['x_val'], c['y_train'], c['y_val'], pvar)

    # -----------------
    # validation on the COMPLETELY WITHHELD test set
    # -----------------
    # Not x_val -- that was used for HP selection, so it gives an optimistic picture.
    # Falls back to x_val if no holdout was requested.
    for c in cases:
        x_test = c['info'].get('x_test')
        y_test = c['info'].get('y_test')
        ef.plot_emulator_results(
            c['x_val'] if x_test is None else x_test,
            c['y_val'] if y_test is None else y_test,
            c['models'], c['info'], TRANSFORM_METHOD, c['scalers'],
            l_plot_uncertainty=True, l_plot_scatter=True, savefig=True, prefix=c['name'])

    # -----------------
    # MCMC setup
    # -----------------
    # deflate the obs lp according to the effective sample size
    deflate_factor = 1 / care_factor.mean()

    param_interest_idx = c0['params_train']['param_interest_idx']
    if '2cat' in run1_name:
        orig_param_csv = f'{lp.param_dir}/2cat_BOSS_priors_old.csv'
    else:
        orig_param_csv = (f'{lp.param_dir}/m46_ne3.csv')

    param_table = pd.read_csv(orig_param_csv)
    # Index by NAME, not position: the two prior CSVs order their columns differently.
    #   2cat_BOSS_priors_old.csv                 -> ,map,mean,isd
    #   param_fullmp_advnu_r2_e1_..._wr0.25.csv  -> ,mean,map,isd
    # Positional .iloc[:, 1] therefore picked up 'map' for 2cat runs and 'mean' for SLC
    # runs, so the printed prior-mean table and the orange reference line on every 2cat
    # posterior plot were actually the old MAP.
    for col in ('mean', 'isd'):
        if col not in param_table.columns:
            raise KeyError(f"{orig_param_csv} has no '{col}' column; "
                           f"found {list(param_table.columns)}")
    all_param_names = param_table.iloc[:, 0].to_list()
    param_names = param_table.iloc[param_interest_idx, 0].to_list()
    param_mean = param_table['mean'].iloc[param_interest_idx].to_numpy().astype(np.float32)
    param_std = param_table['isd'].iloc[param_interest_idx].to_numpy().astype(np.float32)
    nparam = len(param_names)

    print(pd.DataFrame({'param names': param_names,
                        'prior mean': param_mean,
                        'prior std': param_std}))

    # per-case target tensors and emulator inputs
    for i, c in enumerate(cases):
        c['tgt_mu'], c['tgt_sd'], c['tgt_mask'] = build_target(c['tgt_data'], f'case {i + 1}')
        c['tgt_gate'] = c['gate_mask'] = None
        if COARSE_MODE:
            if COARSE_GATE_VARS:
                # tgt_data is left CONTINUOUS for the gates -- build_gate_target needs the
                # raw TAU values to count how many members clear the threshold. Only the
                # training targets were binarized.
                c['tgt_gate'], c['gate_mask'] = cf.build_gate_target(
                    c, varcons, COARSE_GATE_VARS, TRANSFORM_METHOD,
                    soft_dex=GATE_SOFT_DEX, label=f'case {i + 1}')
            keep = tf.constant(val_cols, tf.int32)
            c['tgt_mu'] = tf.gather(c['tgt_mu'], keep, axis=-1)
            c['tgt_sd'] = tf.gather(c['tgt_sd'], keep, axis=-1)
            c['tgt_mask'] = tf.gather(c['tgt_mask'], keep, axis=-1)
        c['sim_ics'] = np.concatenate(c['tgt_initvar'], axis=1)
        c['n_ic'] = c['tgt_data'][0].shape[0]
        ic_with_dummy = np.concatenate(
            (c['sim_ics'], np.zeros([c['n_ic'], npar])), axis=1)
        c['IC_norm'] = c['scalers']['x'].transform(ic_with_dummy)[:, :n_init].astype('float32')
        c['IC_norm_3d'] = tf.tile(c['IC_norm'][None, :, :], [NCHAINS, 1, 1])
        c['weight'] = float(CASE_WEIGHTS[i])
    for i, c in enumerate(cases):
        print(f'case {i + 1} likelihood weight = {c["weight"]:.4g}')

    # -----------------
    # prior support
    # -----------------
    design_norm = c0['scalers']['x'].transform(
        c0['params_train']['vals'])[:, n_init:].astype('float32')   # [nppe, npar] in [0,1]

    if L_PCA_PRIOR:
        pca_mu = design_norm.mean(axis=0)
        _, S_d, Vt_d = np.linalg.svd(design_norm - pca_mu, full_matrices=False)
        evr = S_d ** 2 / np.sum(S_d ** 2)
        k_pca = int(np.sum(evr > PCA_VAR_TOL))
        Vk = Vt_d[:k_pca]                                   # [k, npar], orthonormal
        Zd = (design_norm - pca_mu) @ Vk.T                  # [nppe, k]
        zlo, zhi = Zd.min(axis=0), Zd.max(axis=0)

        print(f'PCA prior: keeping {k_pca}/{npar} directions, '
              f'{evr[:k_pca].sum() * 100:.4f}% of design variance')
        print('  explained variance ratio:', np.array2string(evr, precision=6))
        if 0 < k_pca < npar:
            gap = evr[k_pca - 1] / evr[k_pca]
            print(f'  spectral gap at the cut: {gap:.1f}x '
                  f'(last kept {evr[k_pca - 1]:.3e}, first dropped {evr[k_pca]:.3e})')
            if gap < 10.:
                print('  WARNING: cut is not in a clean gap -- inspect the spectrum and '
                      'set PCA_VAR_TOL by hand')
        print(f'  design participation ratio: '
              f'{evr.sum() ** 2 / np.sum(evr ** 2):.2f} / {npar}')

        pca_mu_tf = tf.constant(pca_mu, tf.float32)
        Vk_tf = tf.constant(Vk, tf.float32)
        zlo_tf = tf.constant(zlo, tf.float32)
        zhi_tf = tf.constant(zhi, tf.float32)
        ndim_s = k_pca

        def lift_to_theta(z01):
            """[..., k_pca] in (0,1) -> [..., npar] normalized parameter space.

            Affine with a fixed orthonormal Vk, so its Jacobian is constant and
            contributes nothing to the log posterior; only the sigmoid Jacobian is
            needed. The z box is the bounding box of the projected design, so its
            corners can sit slightly outside the design's convex hull (and outside
            [0,1]); that is a much smaller excursion than the full npar box and is left
            unclipped to keep gradients clean.
            """
            return pca_mu_tf + tf.tensordot(
                zlo_tf + (zhi_tf - zlo_tf) * z01, Vk_tf, axes=[[-1], [0]])
    else:
        ndim_s = npar

        def lift_to_theta(z01):
            return z01

    # extra sampled dimensions carrying log(sigma_struct): one per VALUE variable per case.
    # Gates get none -- sigma_struct scales a continuous residual and is meaningless on a
    # probability, and a sampled dimension the likelihood never reads would just return
    # its prior and clutter the traces.
    n_sig_var = len(value_vars)
    n_sig = n_sig_var * ncase
    sig_prior = tfd.Normal(loc=np.float32(np.log(SIGMA_STRUCT_PRIOR_MED)),
                           scale=np.float32(SIGMA_STRUCT_PRIOR_SD))

    def get_BOSSemu_lp(params_sigma, l_diag=False):
        # leading ndim_s columns are the parameter coordinates; the trailing n_sig
        # columns are u = log(sigma_struct), one per constraint variable per case
        params = params_sigma[:, :ndim_s]

        z01 = tfb.Sigmoid().forward(params)                    # [nchains, ndim_s]
        param_lp = tf.reduce_sum(tf.math.log(z01) + tf.math.log1p(-z01), axis=1)

        # sigma_struct block. The prior sits directly on u = log(sigma_struct), so no
        # change-of-variables term is needed here.
        u_sig = params_sigma[:, ndim_s:]                       # [nchains, n_sig]
        param_lp += tf.reduce_sum(sig_prior.log_prob(u_sig), axis=1)
        sig_struct = tf.exp(tf.reshape(u_sig, [NCHAINS, ncase, n_sig_var]))

        theta_2d = lift_to_theta(z01)                          # [nchains, npar]
        if l_diag:
            print('scaled parameters:', theta_2d[0, :].numpy())

        obs_lp = tf.zeros([NCHAINS], tf.float32)
        emu_mus = []
        for i, c in enumerate(cases):
            theta = tf.tile(theta_2d[:, None, :], [1, c['n_ic'], 1])
            x_in = tf.reshape(tf.concat([c['IC_norm_3d'], theta], axis=-1),
                              [NCHAINS * c['n_ic'], nparam_init])
            emu_mu, emu_sigma = apply_model_mixture(
                c['models'], x_in, varcons, nobs, NCHAINS, c['n_ic'])
            emu_mus.append(emu_mu)
            obs_lp += c['weight'] * case_loglik(
                emu_mu, emu_sigma, sig_struct[:, i, :],
                c['tgt_mu'], c['tgt_sd'], c['tgt_mask'], care_val,
                val_cols=val_cols, gate_cols=gate_cols,
                tgt_gate=c['tgt_gate'], gate_mask=c['gate_mask'], gate_care=gate_care)

        if l_diag:
            print('param_lp:', param_lp.numpy())
            print('obs_lp:', deflate_factor * obs_lp.numpy())
            return emu_mus

        # param_lp is a prior, not a per-case likelihood: counted once regardless of how
        # many cases contribute to obs_lp.
        return param_lp + obs_lp * deflate_factor

    # -----------------
    # initial state
    # -----------------
    tf.random.set_seed(2)
    # sampled dimension is ndim_s (= k_pca when the design manifold is restricted, else
    # npar); the lift to the npar-dimensional emulator input happens inside
    # get_BOSSemu_lp. n_sig further columns carry log(sigma_struct), started near the prior
    # median so the chains do not open with an absurd error model.
    initial_state = tf.concat(
        [tf.random.normal([NCHAINS, ndim_s], seed=3),
         np.float32(np.log(SIGMA_STRUCT_PRIOR_MED))
         + 0.1 * tf.random.normal([NCHAINS, n_sig], seed=4)], axis=1)
    print(f'sampling {ndim_s} parameter dims + {n_sig} log(sigma_struct) dims '
          f'= {ndim_s + n_sig} total')

    emu_mus = get_BOSSemu_lp(initial_state, l_diag=True)

    # sanity histogram: emulated vs target range for the rain-rate variable, in physical
    # units. Same range is good news; it does not guarantee a good fit either way.
    if pvar is not None:
        for c, emu_mu in zip(cases, emu_mus):
            eff0 = c['info']['eff0s'][pvar]
            sc = c['scalers']['y'][pvar]
            mu_phys = ef.inverse_transform_data(
                emu_mu.numpy()[:, :, pvar], TRANSFORM_METHOD, sc, eff0=eff0)
            tgt_phys = ef.inverse_transform_data(
                c['tgt_mu'].numpy()[:, pvar].reshape(-1, 1), TRANSFORM_METHOD, sc, eff0=eff0)
            plt.figure()
            sns.histplot(mu_phys.flatten(), bins=20, log_scale=True)
            sns.histplot(tgt_phys.flatten(), bins=20, log_scale=True)

    # -----------------
    # preconditioner
    # -----------------
    # The original pilot ran 100 HMC steps at a FIXED step_size of 5e-4 with no
    # adaptation, then took the variance pooled over draws AND chains. Total displacement
    # over those steps is order 0.25, so the chains barely left where they started, and
    # initial_state is N(0,1) across chains by construction: the variance it measured was
    # the spread of the STARTING distribution, not the posterior.
    #
    # Run a short step-size-adapted chain instead and measure its post-warmup draws.
    # Report within-chain and pooled variance separately: if they disagree the pilot has
    # not mixed and the preconditioner should not be trusted.
    pilot_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            get_BOSSemu_lp, step_size=STEPSIZE, num_leapfrog_steps=3),
        num_adaptation_steps=int(N_PILOT * 0.8), target_accept_prob=0.8)

    @tf.function
    def run_pilot_chain():
        return tfp.mcmc.sample_chain(
            num_results=N_PILOT, num_burnin_steps=N_PILOT,
            current_state=initial_state, kernel=pilot_kernel, trace_fn=None, seed=1)

    z_pilot = run_pilot_chain()                    # [n_pilot, nchains, ndim_total]
    var_within = tf.reduce_mean(tf.math.reduce_variance(z_pilot, axis=0), axis=0)
    var_pooled = tf.math.reduce_variance(z_pilot, axis=[0, 1])
    sigma = tf.sqrt(var_within + 1e-8)
    mix_ratio = float(tf.reduce_max(var_pooled / (var_within + 1e-12)))
    print('preconditioner sigma:', sigma.numpy())
    print(f'  max pooled/within variance ratio: {mix_ratio:.2f}')
    if mix_ratio > 4.:
        print('  WARNING: pilot chains have not mixed; preconditioner is unreliable. '
              'Raise N_PILOT or check the log-prob for pathologies.')

    # -----------------
    # sampling
    # -----------------
    if SAMPLER == 'nuts':
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            get_BOSSemu_lp, step_size=STEPSIZE, max_tree_depth=MAX_TREE)
    elif SAMPLER == 'hmc':
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            get_BOSSemu_lp, step_size=STEPSIZE, num_leapfrog_steps=HMC_LEAPFROG)
    else:
        raise ValueError(f'unknown SAMPLER: {SAMPLER}')

    kernel = tfp.mcmc.TransformedTransitionKernel(
        bijector=tfp.bijectors.Scale(sigma),
        inner_kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=int(NUM_BURNIN * 0.8),
            target_accept_prob=0.8))

    @tf.function
    def run_chain():
        return tfp.mcmc.sample_chain(
            num_results=NUM_SAMPLES, current_state=initial_state, kernel=kernel,
            num_burnin_steps=NUM_BURNIN,
            trace_fn=lambda cs, kr: kr, seed=0)

    start = time.time()
    samples, kr = run_chain()
    print(f'sampling took {time.time() - start:.1f} s')

    # kernel-results layout differs between NUTS and HMC, so reach for fields defensively
    _ir = kr.inner_results.inner_results
    print("acceptance rate:", float(np.mean(_ir.is_accepted.numpy())))
    print("step_size range:", kr.inner_results.new_step_size.numpy().min(),
          kr.inner_results.new_step_size.numpy().max())
    print("avg log_accept_ratio:", np.nanmean(_ir.log_accept_ratio))
    if hasattr(_ir, 'has_divergence'):
        div = _ir.has_divergence.numpy()
        print(f"divergences: {div.sum()}/{div.size} ({100. * div.mean():.2f}%)")
        if div.sum():
            print("  WARNING: divergences indicate the sampler cannot resolve the "
                  "posterior geometry; results are biased, not merely noisy.")
    if hasattr(_ir, 'reach_max_depth'):
        print(f"hit max_tree_depth: {int(_ir.reach_max_depth.numpy().sum())} draws "
              f"(raise MAX_TREE if this is large)")
    if hasattr(_ir, 'leapfrogs_taken'):
        print("mean leapfrogs/draw:", float(np.mean(_ir.leapfrogs_taken.numpy())))

    plot_dir = f"plots/posteriors/{run_name}/"
    os.makedirs(plot_dir, exist_ok=True)

    # samples live in the ndim_s-dimensional sampling space; lift back to the npar
    # normalized coordinates so everything downstream is unchanged
    transformed_samples = lift_to_theta(tfb.Sigmoid().forward(samples[:, :, :ndim_s]))
    # ylim=None, not the plot_traces default of (0, 1). The sigmoid output is in (0, 1)
    # but lift_to_theta is an affine map off that box: the z bounding box corners sit
    # outside the design's convex hull, so lifted coordinates routinely leave [0, 1].
    # The default box silently cropped those excursions, which is exactly the part of
    # the trace worth seeing (it says the chain is off the training manifold).
    mf.plot_traces(transformed_samples, param_names, ylim=None)
    plt.savefig(f'{plot_dir}/traces.pdf')

    # Convergence diagnostics belong in the sampling space, not the lifted space: after
    # an affine lift onto a rank-k sheet the npar coordinates are linearly dependent,
    # which makes per-coordinate ESS/R-hat misleading. cross_chain_dims=1 pools the
    # chains, giving one ESS per dimension; without it TFP returns shape [nchains, ndim].
    ess = tfp.mcmc.effective_sample_size(samples, cross_chain_dims=1).numpy()
    rhat = tfp.mcmc.potential_scale_reduction(samples).numpy()
    # value variables only -- gates carry no sigma_struct dimension
    sig_names = [f'sd_{n[:12]}_{v}' for n in run_names for v in value_vars]
    dim_names = [f'pc{j}' for j in range(ndim_s)] + [f'log{s}' for s in sig_names]
    print(f'\n{"dim":28s} {"ESS":>10s} {"Rhat":>8s}')
    for j in range(len(ess)):
        nm = dim_names[j] if j < len(dim_names) else f'dim{j}'
        flag = '  <-- low' if ess[j] < 400 or rhat[j] > 1.01 else ''
        print(f'{nm:28s} {ess[j]:10.1f} {rhat[j]:8.4f}{flag}')
    print('ESS  min/median:', float(np.nanmin(ess)), float(np.nanmedian(ess)))
    print('Rhat max/median:', float(np.nanmax(rhat)), float(np.nanmedian(rhat)))
    if np.nanmax(rhat) > 1.01:
        print('  WARNING: max R-hat > 1.01, chains have not converged')
    if np.nanmin(ess) < 400:
        print('  WARNING: min ESS < 400, posterior summaries are unreliable')

    # posterior sigma_struct, back on the natural scale
    sig_post = np.exp(samples[:, :, ndim_s:].numpy())      # [ndraw, nchain, n_sig]
    mf.plot_traces(sig_post, sig_names, ylim=None)         # unbounded, must autoscale
    plt.savefig(f'{plot_dir}/traces_sigma_struct.pdf')
    flat_sig = sig_post.reshape(-1, n_sig)
    print('\nposterior sigma_struct (transformed units, 1.0 = one PPE sd):')
    for j, nm in enumerate(sig_names):
        q = np.percentile(flat_sig[:, j], [5, 50, 95])
        print(f'  {nm:28s} median {q[1]:7.3f}   90% CI [{q[0]:7.3f}, {q[2]:7.3f}]')

    # -----------------
    # diagnostics file: sampler stats + the sigma_struct chains
    # -----------------
    # Deliberately a SEPARATE file, not a group added to the *_posterior_arviz.nc files
    # below. Those are read by ppe_util.F to build the next design, and ppe_util.F:496
    # perturbs every variable it finds in their `posterior` group -- a sigma_struct entry
    # there would silently become a fake model parameter. Keeping it out also means the
    # Fortran reader never sees an unfamiliar group.
    #
    # Named _diagnostics.nc rather than _..._arviz.nc so it cannot be picked up by
    # anything matching the posterior-file pattern.
    # sig_names is built as f'sd_{n[:12]}_{v}', and every run name in a joint run truncates
    # to the same 12 characters, so the names are NOT unique across cases. Positional plot
    # panels survive that, but a dict does not -- duplicate keys would silently drop one
    # case's 6 chains and keep the other's under an indistinguishable name. Key the
    # diagnostics file by case index instead, and assert uniqueness so this cannot recur.
    diag_sig = {f'sd_case{i + 1}_{v}': sig_post[:, :, i * len(value_vars) + j].T
                for i in range(ncase) for j, v in enumerate(value_vars)}
    if len(diag_sig) != n_sig:
        raise RuntimeError(f'sigma_struct name collision: {len(diag_sig)} unique names for '
                           f'{n_sig} chains')

    stats = build_sample_stats(kr.inner_results, _ir, NCHAINS, NUM_SAMPLES, HMC_LEAPFROG)
    diag_path = (f'MCMC_posterior/{run_name}_cf{care_factor.mean():.3g}_d{CASE_WEIGHTS[0]:.0f}r{CASE_WEIGHTS[1]:.0f}_diagnostics.nc')
    az.from_dict(posterior=diag_sig, sample_stats=stats).to_netcdf(
        diag_path, engine='netcdf4')
    print(f'  case index -> run: ' + ', '.join(f'{i + 1}={n}' for i, n in enumerate(run_names)))
    print(f'\nwrote {diag_path}')
    print(f'  sample_stats: {sorted(stats)}')
    print(f'  posterior group: {len(diag_sig)} sigma_struct chains')
    if 'diverging' in stats:
        d = stats['diverging']
        print(f'  divergences: {int(d.sum())}/{d.size} ({100. * d.mean():.2f}%), '
              f'per chain {[int(c) for c in d.sum(axis=1)]}')
    else:
        print(f'  no divergence flag: SAMPLER={SAMPLER!r} does not emit one '
              f'(NUTS-only field)')

    # -----------------
    # posterior in physical units
    # -----------------
    ic_col = tf.tile(c0['IC_norm'][None, 0, :n_init], [NUM_SAMPLES, 1])
    samples_with_ic = tf.concat(
        [tf.tile(ic_col[:, None, :], [1, NCHAINS, 1]), transformed_samples], axis=2)

    samples_origval = np.stack(
        [c0['scalers']['x'].inverse_transform(samples_with_ic[:, ic, :])[:, n_init:]
         for ic in range(NCHAINS)], axis=1).astype(np.float32)   # [ndraw, nchain, npar]

    # PPE design range. Previously used directly as set_xlim, which CLIPPED the marginal
    # wherever the posterior ran past the design -- and running past the design is the
    # single most important thing these panels can show, because it means the emulator is
    # extrapolating there. Now drawn as reference lines instead, with the axis sized to
    # hold the posterior, the design range and the prior mean all at once, so nothing is
    # cut off and the design edge stays visible.
    des_l = np.min(c0['params_train']['vals'][:, n_init:], axis=0)
    des_u = np.max(c0['params_train']['vals'][:, n_init:], axis=0)

    tag = f'{RUNDEETS}_cf{care_factor.mean():.3g}_d{CASE_WEIGHTS[0]:.0f}r{CASE_WEIGHTS[1]:.0f}'

    ncol_p = 4
    nrow_p = int(np.ceil(npar / ncol_p))
    fig = plt.figure(figsize=(14, 2.3 * nrow_p))
    gs = gridspec.GridSpec(nrow_p, ncol_p)
    for ipost in range(npar):
        ax = fig.add_subplot(gs[ipost])
        s = samples_origval[:, :, ipost]
        sns.kdeplot(s, fill=True, legend=False, ax=ax)
        # sns.kdeplot autoscales to the KDE support, which already extends past the
        # samples; widen further so the design bounds and the prior mean stay on-axis.
        kde_lo, kde_hi = ax.get_xlim()
        lo = min(s.min(), des_l[ipost], param_mean[ipost], kde_lo)
        hi = max(s.max(), des_u[ipost], param_mean[ipost], kde_hi)
        pad = 0.03 * (hi - lo) if hi > lo else 1.0
        ax.set_xlim(lo - pad, hi + pad)
        for edge in (des_l[ipost], des_u[ipost]):
            ax.axvline(edge, color='0.6', ls='--', lw=0.8)
        ax.axvline(param_mean[ipost], color='tab:orange')
        ax.set_title(param_names[ipost])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/params_{tag}.pdf')

    pairplot = sns.pairplot(
        pd.DataFrame(samples_origval.reshape(-1, npar), columns=param_names),
        corner=True, kind="hist", diag_kind="kde", plot_kws=dict(bins=25))
    # Rasterize only the off-diagonal 2D-hist panels (the bulk of the vector
    # bloat); leave the diagonal KDE curves and all text as vector.
    for i, row in enumerate(pairplot.axes):
        for j, ax in enumerate(row):
            if ax is None or i == j:
                continue
            for artist in ax.collections + ax.images:
                artist.set_rasterized(True)
    pairplot.fig.suptitle("MCMC Posterior Distributions (Seaborn)", fontsize=16)
    pairplot.fig.subplots_adjust(top=0.98)
    pairplot.fig.savefig(f"{plot_dir}/corner_plot_{tag}.pdf")

    # -----------------
    # posterior predictive checks
    # -----------------
    tsamples = transformed_samples.numpy().reshape(NUM_SAMPLES * NCHAINS, npar)
    nflat = tsamples.shape[0]
    theta_mean = tf.constant(tsamples.mean(axis=0)[None, :])
    theta_map = tf.constant(
        np.array([map_kde(tsamples[:, i]) for i in range(npar)],
                 dtype=np.float32)[None, :])

    rng = np.random.default_rng()
    nsample_ppc = 200
    ncol = min(4, nvar)
    nrow = int(np.ceil(nvar / 4))

    # median posterior sigma_struct per (case, value var), transformed units, same layout
    # as flat_sig's columns (i * n_sig_var + j). Constant across Na (sig_struct carries no
    # Na dimension), so the PPC band it drives is flat across the whole ics axis.
    sig_struct_med = np.median(flat_sig, axis=0)

    for icase, c in enumerate(cases):
        ic_norm, n_ic, models = c['IC_norm'], c['n_ic'], c['models']
        with_ic = tf.concat(
            [tf.tile(ic_norm[:, None, :], [1, nflat, 1]),
             tf.tile(tsamples[None, ...], [n_ic, 1, 1])], axis=2)
        y_ppc = ef.apply_model(models, np.reshape(with_ic, [n_ic * nflat, nparam_init]))
        y_mean = ef.apply_model(models, tf.concat(
            [ic_norm, tf.tile(theta_mean, [n_ic, 1])], axis=1))
        y_map = ef.apply_model(models, tf.concat(
            [ic_norm, tf.tile(theta_map, [n_ic, 1])], axis=1))

        # 0.9in of extra height is the legend strip reserved by the tight_layout rect
        # below; without it a two-row framed legend either overlaps the bottom panels or
        # falls outside the page.
        fig, axs = plt.subplots(nrow, ncol, figsize=(12, nrow * 2 + 0.9))
        axs = np.atleast_1d(axs).flatten()
        # nvar rarely fills the nrow x ncol grid (13 vars -> 16 panels). Drop the
        # remainder rather than leaving empty boxes.
        for ax in axs[nvar:]:
            ax.remove()
        axs = axs[:nvar]
        gate_pos = {v: k for k, v in enumerate(COARSE_GATE_VARS)} if COARSE_MODE else {}
        for ivar, varcon in tqdm(enumerate(varcons), total=nvar):
            is_gate = varcon in gate_pos
            nb = int(nobs[ivar])

            def curve(arr, _g=is_gate, _nb=nb):
                """Panel y-values for one packed [mu | raw_sigma] head output.

                For a GATE, plot the probability the likelihood actually consumes rather
                than the raw mu. mu comes off a Dense layer with no activation, so it is
                unbounded and overshoots past 1 wherever the training labels are pinned
                near 1 -- which looks like the model predicting a probability above 1.
                Mapping through gate_prob puts it in [eps, 1-eps], directly comparable to
                the orange TAU ON-fraction, and matches what case_loglik scores.
                """
                if not _g:
                    return arr[:, 0]
                return cf.gate_prob(tf.constant(np.asarray(arr[:, 0]), tf.float32),
                                    tf.nn.softplus(tf.constant(np.asarray(arr[:, _nb]),
                                                               tf.float32)),
                                    GATE_MID, GATE_EPS).numpy()

            post = np.reshape(curve(y_ppc[varcon]), [n_ic, nflat])
            sampled = rng.choice(post, size=nsample_ppc, replace=False, axis=1)
            ics = c['sim_ics']
            axs[ivar].plot(ics, sampled, color='tab:cyan',
                           alpha=nsample_ppc ** -0.5, label='PPC')
            if varcon in gate_pos:
                # This head predicts an ON LABEL, so tgt_data (still continuous) is not
                # what it is being compared against. Plot gate_prob against the TAU
                # ON-fraction; no inverse transform, which would be meaningless here.
                tgt_f = c['tgt_gate'].numpy()[:, gate_pos[varcon]]
                axs[ivar].plot(ics, tgt_f, color='tab:orange', alpha=0.7, label='target')
                axs[ivar].axhline(GATE_MID, color='0.6', ls='--', lw=0.8)
                axs[ivar].set_ylim(-0.1, 1.1)
            else:
                tgt = c['tgt_data'][ivar]
                # nan-aware to match build_target, which uses nanmean/nanstd because
                # tgt_data keeps literal NaN for masked entries. Plain min/max/mean would
                # blank the band and the line at any Na with one masked TAU member, i.e.
                # show less than the likelihood actually sees.
                with np.errstate(invalid='ignore'):
                    tgt_mean = np.nanmean(tgt, axis=1)
                    tgt_lo = np.nanmin(tgt, axis=1)
                    tgt_hi = np.nanmax(tgt, axis=1)
                # Internal variability first, then the structural strips STACKED outside
                # it (above the top, below the bottom) rather than centred on the mean, so
                # the two contributions never overlap and each is read off separately.
                axs[ivar].fill_between(ics.squeeze(), tgt_lo, tgt_hi, alpha=0.3,
                                       color='tab:orange', lw=0, zorder=2)
                if varcon in value_vars:
                    j = value_vars.index(varcon)
                    sigma_struct = sig_struct_med[icase * len(value_vars) + j]
                    axs[ivar].fill_between(ics.squeeze(), tgt_hi, tgt_hi + sigma_struct,
                                           alpha=0.12, color='tab:orange', lw=0,
                                           zorder=1, label='structural unc.')
                    axs[ivar].fill_between(ics.squeeze(), tgt_lo - sigma_struct, tgt_lo,
                                           alpha=0.12, color='tab:orange', lw=0,
                                           zorder=1)
                axs[ivar].plot(ics, tgt_mean, color='tab:orange',
                               alpha=0.9, zorder=3, label='target')
            axs[ivar].plot(ics, curve(y_mean[varcon]), color='tab:blue', ls='--',
                           label='mean')
            axs[ivar].plot(ics, curve(y_map[varcon]), color='tab:red', ls=':', label='MAP')
            # fall back to the raw key for any constraint not in output_var_set (derived
            # names carry summary-step suffixes that are not dict keys)
            long_name = cl.output_var_set.get(varcon, {}).get('longname', varcon)
            axs[ivar].set_title(f'{long_name} [gate]' if varcon in gate_pos else long_name,
                                fontsize=9)
            axs[ivar].set_xscale('log')

        # Explicit proxy handles rather than harvesting the axes: the PPC lines are drawn
        # at alpha = nsample_ppc**-0.5 (~0.07), which is invisible in a legend swatch, and
        # the two orange bands are distinguishable only by an alpha the harvested handles
        # would carry over. Fixed order, one entry per artist, all legible.
        legend_items = [
            Line2D([], [], color='tab:cyan', lw=1.5, label='PPC draws'),
            Line2D([], [], color='tab:orange', lw=1.5, label='TAU target (member mean)'),
            Patch(facecolor='tab:orange', alpha=0.3,
                  label='TAU internal variability (member min-max)'),
            Patch(facecolor='tab:orange', alpha=0.12,
                  label=r'learned structural unc. ($\sigma_{\rm struct}$, stacked outside)'),
            Line2D([], [], color='tab:blue', ls='--', label='posterior mean'),
            Line2D([], [], color='tab:red', ls=':', label='MAP'),
        ]
        legend_frac = 0.9 / (nrow * 2 + 0.9)
        fig.tight_layout(rect=[0, legend_frac, 1, 1])
        fig.legend(handles=legend_items, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=11,
                   frameon=True, edgecolor='0.5')
        plt.savefig(f'{plot_dir}/ppc{icase + 1}_{tag}.pdf')

    # -----------------
    # update params csv with MAP, and save arviz posteriors
    # -----------------
    with nc.Dataset(f"{lp.nc_dir}{RUN1_FN}") as ds:
        param_idx_group, perturbed_pgroup = ef.get_param_interest_idx(
            ds, return_perturbed_groupname=True)

    posterior_samples = np.transpose(samples_origval, (1, 0, 2))   # [chain, draw, npar]
    flat = samples_origval.reshape(-1, npar)
    sample_mean, sample_std = flat.mean(axis=0), flat.std(axis=0)

    updated_params = pd.read_csv(orig_param_csv)
    # The original also did `updated_params.iloc[:, 2] = updated_params.iloc[:, 1]` as
    # setup for a rename keyed on a column named 'sd'. Neither prior CSV has that column
    # (both use 'isd'), so the rename never fired and all the line did was overwrite
    # column 2 with column 1 for EVERY row, silently corrupting the carried-forward
    # entries for parameters outside the current MCMC run.
    for iparam, param_name in enumerate(param_names):
        iparam_all = param_interest_idx[iparam]
        map_estimate = (sample_mean[iparam] if 'select' in run1_name
                        else map_kde(flat[:, iparam]))
        updated_params.loc[iparam_all, 'map'] = map_estimate
        updated_params.loc[iparam_all, 'mean'] = sample_mean[iparam]
        updated_params.loc[iparam_all, 'isd'] = sample_std[iparam]
    updated_params[['Unnamed: 0', 'mean', 'map', 'isd']].to_csv(
        f'{lp.param_dir}/param_{run_name}_cf{care_factor.mean():.3g}_d{CASE_WEIGHTS[0]:.0f}r{CASE_WEIGHTS[1]:.0f}.csv', index=False)

    # By name, not position: param_idx_group flattened and param_names agree today, but
    # the name lookup is correct either way.
    def group_dict(param_idx):
        return {all_param_names[pidx]:
                posterior_samples[:, :, param_names.index(all_param_names[pidx])]
                for pidx in param_idx}

    for param_idx, pgname in zip(param_idx_group, perturbed_pgroup):
        az.from_dict(posterior=group_dict(param_idx)).to_netcdf(
            f'MCMC_posterior/{run_name}_{pgname}_cf{care_factor.mean():.3g}_d{CASE_WEIGHTS[0]:.0f}r{CASE_WEIGHTS[1]:.0f}_posterior_arviz.nc', engine="netcdf4")

    posterior_dict = {k: v for pi in param_idx_group for k, v in group_dict(pi).items()}
    outdir = f'MCMC_posterior/{run_name}_cf{care_factor.mean():.3g}_d{CASE_WEIGHTS[0]:.0f}r{CASE_WEIGHTS[1]:.0f}_all_posterior_arviz.nc'
    az.from_dict(posterior=posterior_dict).to_netcdf(outdir, engine="netcdf4")
    print(f"Wrote combined posterior with {len(posterior_dict)} params to {outdir}")


# =====================================================================
# self-check
# =====================================================================

def selfcheck():
    """Assert the batched likelihood equals the original per-chain loop."""
    rng = np.random.default_rng(0)
    nchains, n_ic, nvar = 5, 7, 4
    f32 = lambda *s: rng.normal(size=s).astype(np.float32)
    emu_mu, tgt_mu = f32(nchains, n_ic, nvar), f32(n_ic, nvar)
    emu_sigma = np.abs(f32(nchains, n_ic, nvar)) + 0.1
    tgt_sd = np.abs(f32(n_ic, nvar))
    sig_struct = np.exp(f32(nchains, nvar))
    mask = (rng.random((n_ic, nvar)) > 0.2).astype(np.float32)
    impact = np.array([1., 3., 5., 10.], dtype=np.float32)

    # original: one scalar per chain, built in a Python loop
    ref = []
    for i in range(nchains):
        s = np.sqrt(sig_struct[i][None, :] ** 2 + emu_sigma[i] ** 2 + tgt_sd ** 2)
        lp = tfd.Normal(loc=tgt_mu, scale=s).log_prob(emu_mu[i])
        ref.append(float(tf.reduce_sum(lp * impact * mask)))

    got = case_loglik(tf.constant(emu_mu), tf.constant(emu_sigma), tf.constant(sig_struct),
                      tf.constant(tgt_mu), tf.constant(tgt_sd), tf.constant(mask),
                      impact).numpy()
    np.testing.assert_allclose(got, np.array(ref), rtol=1e-5)

    # mask really drops points: zeroing the mask must zero the log-lik
    zero = case_loglik(tf.constant(emu_mu), tf.constant(emu_sigma), tf.constant(sig_struct),
                       tf.constant(tgt_mu), tf.constant(tgt_sd),
                       tf.zeros_like(mask), impact).numpy()
    assert np.allclose(zero, 0.)
    print('selfcheck ok:', np.max(np.abs(got - np.array(ref))))

    # ---- coarse branch: value part must equal the fine call on the value columns, and
    # the gate part must equal coarse_fun's own term. Guards the column split, which is
    # the one place a silent off-by-one would corrupt the likelihood without erroring.
    vcols, gcols = [0, 2], [1, 3]
    sd_v = np.exp(f32(nchains, len(vcols)))
    tg = np.clip(rng.random((n_ic, len(gcols))), 0., 1.).astype(np.float32)
    gmask = (rng.random((n_ic, len(gcols))) > 0.1).astype(np.float32)
    gcare = np.array([1., 2.], dtype=np.float32)

    ref_val = case_loglik(
        tf.constant(emu_mu[..., vcols]), tf.constant(emu_sigma[..., vcols]),
        tf.constant(sd_v), tf.constant(tgt_mu[:, vcols]), tf.constant(tgt_sd[:, vcols]),
        tf.constant(mask[:, vcols]), impact[vcols]).numpy()
    ref_gate = cf.gate_logprob(
        tf.constant(emu_mu[..., gcols]), tf.constant(emu_sigma[..., gcols]),
        tf.constant(tg), tf.constant(gmask), gcare, GATE_MID, GATE_EPS).numpy()

    got_c = case_loglik(
        tf.constant(emu_mu), tf.constant(emu_sigma), tf.constant(sd_v),
        tf.constant(tgt_mu[:, vcols]), tf.constant(tgt_sd[:, vcols]),
        tf.constant(mask[:, vcols]), impact[vcols],
        val_cols=vcols, gate_cols=gcols, tgt_gate=tf.constant(tg),
        gate_mask=tf.constant(gmask), gate_care=gcare).numpy()
    np.testing.assert_allclose(got_c, ref_val + ref_gate, rtol=1e-5)

    # coarse mode with NO gates: value terms on the selected subset only. Must still
    # gather val_cols (it is not the identity in general), so it cannot silently fall
    # through to the fine-mode branch.
    for empty in ([], None):
        got_ng = case_loglik(
            tf.constant(emu_mu), tf.constant(emu_sigma), tf.constant(sd_v),
            tf.constant(tgt_mu[:, vcols]), tf.constant(tgt_sd[:, vcols]),
            tf.constant(mask[:, vcols]), impact[vcols],
            val_cols=vcols, gate_cols=empty).numpy()
        np.testing.assert_allclose(got_ng, ref_val, rtol=1e-5)
    # and it must NOT equal the ungathered all-column answer, which is what a fall-through
    # to fine mode would return
    assert not np.allclose(ref_val, case_loglik(
        tf.constant(emu_mu), tf.constant(emu_sigma), tf.constant(sig_struct),
        tf.constant(tgt_mu), tf.constant(tgt_sd), tf.constant(mask), impact).numpy())

    # the gate must be able to veto: a chain the emulator places far below the boundary
    # while the target says ON pays the clipped penalty on every unmasked entry
    veto = case_loglik(
        tf.constant(np.full_like(emu_mu, -50.)), tf.constant(emu_sigma),
        tf.constant(sd_v), tf.constant(tgt_mu[:, vcols]), tf.constant(tgt_sd[:, vcols]),
        tf.zeros_like(tf.constant(mask[:, vcols])), impact[vcols],
        val_cols=vcols, gate_cols=gcols, tgt_gate=tf.constant(np.ones_like(tg)),
        gate_mask=tf.constant(np.ones_like(gmask)), gate_care=gcare).numpy()
    expect = float(np.log(GATE_EPS) * n_ic * gcare.sum())
    np.testing.assert_allclose(veto, np.full(nchains, expect), rtol=1e-5)
    print('coarse selfcheck ok: value+gate split and gate veto both exact')

    # ---- sample_stats assembly. Fake kernel results in both shapes, because the field
    # set differs by sampler and the draw/chain axes are easy to transpose silently.
    class NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    nch, nd = 4, 7
    tr = lambda *s: rng.normal(size=(nd,) + s).astype(np.float32)   # TFP order: draw first
    dual = NS(new_step_size=rng.random(nd).astype(np.float32))      # shared across chains

    # target_log_prob lives in accepted_results, NOT on the outer MH results object --
    # reading only the top level is what silently dropped lp from the first written file.
    hmc_ir = NS(log_accept_ratio=tr(nch), is_accepted=(rng.random((nd, nch)) > 0.2),
                accepted_results=NS(target_log_prob=tr(nch)))
    s_hmc = build_sample_stats(dual, hmc_ir, nch, nd, 3)
    assert 'lp' in s_hmc, 'lp must be found inside accepted_results'
    assert np.allclose(s_hmc['lp'], hmc_ir.accepted_results.target_log_prob.T)
    assert 'diverging' not in s_hmc, 'HMC must not report a divergence flag'
    assert np.all(s_hmc['n_steps'] == 3), s_hmc['n_steps']
    # step_size was (draw,) only and must broadcast, not transpose
    assert s_hmc['step_size'].shape == (nch, nd)
    assert np.allclose(s_hmc['step_size'][0], dual.new_step_size)
    assert np.allclose(s_hmc['step_size'][1], dual.new_step_size)
    # chain/draw axes really are swapped, not reshaped

    # acceptance_rate is a probability
    assert s_hmc['acceptance_rate'].min() >= 0. and s_hmc['acceptance_rate'].max() <= 1.

    nuts_ir = NS(target_log_prob=tr(nch), log_accept_ratio=tr(nch),
                 is_accepted=(rng.random((nd, nch)) > 0.2),
                 has_divergence=(rng.random((nd, nch)) > 0.9),
                 reach_max_depth=(rng.random((nd, nch)) > 0.95),
                 leapfrogs_taken=rng.integers(1, 64, (nd, nch)),
                 energy=tr(nch))
    s_nuts = build_sample_stats(dual, nuts_ir, nch, nd, 3)
    for f in ('lp', 'diverging', 'energy', 'n_steps', 'step_size', 'acceptance_rate'):
        assert f in s_nuts, f
    assert s_nuts['diverging'].dtype == bool
    assert int(s_nuts['diverging'].sum()) == int(nuts_ir.has_divergence.sum())
    assert np.array_equal(s_nuts['n_steps'], nuts_ir.leapfrogs_taken.T)

    # must survive a real ArviZ round-trip, since that is what actually gets written
    import tempfile
    post = {f'sd_v{i}': rng.normal(size=(nch, nd)) for i in range(3)}
    idata = az.from_dict(posterior=post, sample_stats=s_nuts)
    with tempfile.TemporaryDirectory(dir=os.path.expanduser('~/tmp')) as td:
        p = os.path.join(td, 'diag.nc')
        idata.to_netcdf(p, engine='netcdf4')
        back = az.from_netcdf(p)
        assert 'sample_stats' in back, back
        assert int(back.sample_stats['diverging'].values.sum()) == \
            int(nuts_ir.has_divergence.sum())
        assert dict(back.posterior.sizes)['chain'] == nch
        assert dict(back.posterior.sizes)['draw'] == nd
    print('sample_stats selfcheck ok: HMC/NUTS field sets, axis order, ArviZ round-trip')


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
    else:
        main()
