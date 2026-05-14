import jax
import jax.numpy as jnp
import numpy as np
import optax
from tinygp import kernels, GaussianProcess, transforms
from tinygp.helpers import JAXArray

# Typical kernel for Gaussian Processes
def build_gp(theta: dict, X: JAXArray, diag: float = 1e-5) -> GaussianProcess:
    """Builds a TinyGP Gaussian Process with Matern52 ARD kernel."""
    amp = jnp.exp(jnp.clip(theta["log_amp"], -12, 12))
    scale = jnp.exp(jnp.clip(theta["log_scale"], -12, 12))
    kernel = amp * transforms.Linear(jnp.diag(1.0 / (scale + 1e-8)), kernels.Matern52())
    noise = jnp.exp(jnp.clip(theta["log_noise"], -12, 5)) + diag
    return GaussianProcess(kernel, X, diag=noise)

@jax.jit
def nll_loss(theta: dict, X: JAXArray, y: JAXArray, diag: float = 0.05) -> float:
    """Negative log likelihood for the GP."""
    gp = build_gp(theta, X, diag=diag)
    return -gp.log_probability(y)

def _train_gp_core(X_train: JAXArray, y_train: JAXArray, n_epochs: int = 3000, lr: float = 0.001, seed: int = 0, diag: float = 1e-5):
    n_dim = X_train.shape[1]
    y_var = jnp.var(y_train)

    theta_init = {
        "log_amp": jnp.log(y_var + 1e-4),
        "log_scale": jnp.zeros(n_dim) + 0.5,
        "log_noise": jnp.log(jnp.clip(y_var * 0.1, 1e-4, None))  # 10% of signal variance
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(lr)
    )
    opt_state = optimizer.init(theta_init)

    @jax.jit
    def step(theta, opt_state):
        loss, grads = jax.value_and_grad(nll_loss)(theta, X_train, y_train, diag=diag)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(theta, updates), opt_state, loss

    theta = theta_init
    for i in range(n_epochs):
        theta, opt_state, loss = step(theta, opt_state)
        # Check for loss being finite - hard fail if not
        if not np.isfinite(float(loss)):
             print(f"\nCRITICAL: NaN/Inf Loss at Epoch {i}!")
             print(f"Theta: {theta}")
             raise RuntimeError(f"Optimizer hit NaN/Inf Loss (Loss: {loss})")
             
        if i % 400 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return theta

def train_gp(X_train: JAXArray, y_train: JAXArray, n_epochs: int = 3000, lr: float = 0.005, seed: int = 0):
    """Wrapper with retry on NaN loss — escalates jitter diag only."""
    max_retries = 3
    current_lr = lr
    current_diag = 1e-5

    for attempt in range(max_retries + 1):
        try:
            return _train_gp_core(X_train, y_train, n_epochs, current_lr, seed, diag=current_diag)
        except RuntimeError as e:
            if attempt == max_retries:
                print(f"GP Training failed after {max_retries} retries. Final error: {e}", flush=True)
                raise e

            current_diag *= 10.0  # 1e-5 → 1e-4 → 1e-3 → 1e-2
            print(f"NaN loss (attempt {attempt+1}/{max_retries}). Retrying with diag={current_diag:.1e}...", flush=True)

class GPState:
    def __init__(self, theta, X_train, y_train):
        gp = build_gp(theta, X_train)
        self.X_train = X_train
        self.kernel = gp.kernel
        self.alpha = gp.solver.solve_triangular(y_train - gp.loc)
        self.L = gp.solver.scale_tril
        self.noise = jnp.exp(theta["log_noise"])
        self.amplitude = jnp.exp(theta["log_amp"])

def fast_predict_single(alpha, L, amplitude, log_scale, X_train_scaled, X_test):
    """
    Manual GP prediction for a single variable, optimized for pre-scaled X_train.
    """
    scale = jnp.exp(log_scale)
    # Only scale X_test here; X_train_scaled should be precomputed
    X_test_scaled = X_test / (scale + 1e-6)
    
    # Reconstruct kernel with pre-scaled inputs
    kernel = amplitude * kernels.Matern52()
    
    K_star_t = kernel(X_test_scaled, X_train_scaled)
    mu = K_star_t @ alpha
    
    # Optimized variance calculation (only compute diagonal)
    v = jax.scipy.linalg.solve_triangular(L, K_star_t.T, lower=True)
    var = amplitude - jnp.sum(jnp.square(v), axis=0)
    
    return mu, jnp.clip(var, a_min=1e-12)

def fast_predict(state: GPState, X_test: JAXArray):
    """Predictive mean and variance for a single GPState."""
    K_star_t = state.kernel(X_test, state.X_train)
    mu = K_star_t @ state.alpha
    v = jax.scipy.linalg.solve_triangular(state.L, K_star_t.T, lower=True)
    var = state.amplitude - jnp.sum(jnp.square(v), axis=0)
    return mu, jnp.clip(var, a_min=1e-10)

# Vectorized version over variables
# in_axes: (0: alphas, 0: Ls, 0: amplitudes, 0: log_scales, 0: X_train_scaled, None: X_test)
fast_predict_vmap = jax.vmap(fast_predict_single, in_axes=(0, 0, 0, 0, 0, None))

def apply_gp(theta: dict, X_train: JAXArray, y_train: JAXArray, X_test: JAXArray):
    """
    Given a trained GP, calculate the predictive mean and variance at X_test.
    """
    state = GPState(theta, X_train, y_train)
    return fast_predict(state, X_test)


def combine_hurdle_predictions(raw_mu, raw_var, is_hurdle, presence_idx, value_idx,
                               standard_idx, transformed_zeros):
    """
    Combine presence + value GP predictions for hurdle variables.

    For hurdle vars: E[Y] = P * V + (1-P) * transformed_zero
    For standard vars: pass through directly.

    Args:
        raw_mu: (n_total_gps, n_points) - raw GP means
        raw_var: (n_total_gps, n_points) - raw GP variances
        is_hurdle: (nvar,) bool array
        presence_idx: (nvar,) int - index into raw arrays for presence GP
        value_idx: (nvar,) int - index into raw arrays for value GP
        standard_idx: (nvar,) int - index into raw arrays for standard GP
        transformed_zeros: (nvar,) float - transformed zero value per variable

    Returns:
        emu_mu: (nvar, n_points)
        emu_var: (nvar, n_points)
    """
    p_mu = jnp.clip(raw_mu[presence_idx], 0.01, 0.99)
    v_mu = raw_mu[value_idx]
    v_var = raw_var[value_idx]
    p_var = jnp.clip(raw_var[presence_idx], 1e-12, None)
    s_mu = raw_mu[standard_idx]
    s_var = raw_var[standard_idx]

    tz = transformed_zeros[:, None]  # (nvar, 1)

    # Hurdle: E[Y] = P * V + (1-P) * tz
    hurdle_mu = p_mu * v_mu + (1.0 - p_mu) * tz
    # Var[Y] = P^2 * Var[V] + (V - tz)^2 * Var[P] + Var[P] * Var[V]
    hurdle_var = p_mu**2 * v_var + (v_mu - tz)**2 * p_var + p_var * v_var

    emu_mu = jnp.where(is_hurdle[:, None], hurdle_mu, s_mu)
    emu_var = jnp.where(is_hurdle[:, None], hurdle_var, s_var)
    return emu_mu, emu_var
