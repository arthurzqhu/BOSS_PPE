import jax
import jax.numpy as jnp
import numpy as np
import optax
from tinygp import kernels, GaussianProcess
from tinygp.helpers import JAXArray
import MCMC_fun_jax as mjax
import emulator_fun_jax as ejax

# Typical kernel for Gaussian Processes
def build_gp(theta: dict, X: JAXArray, diag: float = 1e-4) -> GaussianProcess:
    """
    Builds a TinyGP Gaussian Process given hyperparameters.
    theta contains:
        - log_amp: log amplitude of the kernel
        - log_scale: log length scale for each dimension
        - log_noise: log observational noise
    """
    from tinygp import transforms
    
    amp = jnp.exp(theta["log_amp"])
    scale = jnp.exp(theta["log_scale"])
    
    # Correct TinyGP Linear transform syntax: Linear(matrix, base_kernel)
    kernel = amp * transforms.Linear(jnp.diag(1.0 / (scale + 1e-6)), kernels.Matern52())
    
    # Jitter + noise
    noise = jnp.exp(theta["log_noise"]) + diag
    return GaussianProcess(kernel, X, diag=noise)

@jax.jit
def nll_loss(theta: dict, X: JAXArray, y: JAXArray) -> float:
    """Negative log likelihood for the GP."""
    gp = build_gp(theta, X)
    return -gp.log_probability(y)

def train_gp(X_train: np.ndarray, y_train: np.ndarray, n_epochs: int = 1000, lr: float = 0.01, seed: int = 42):
    """
    Train a Gaussian Process model using Optax.
    """
    # ensure JAX arrays
    X = jnp.asarray(X_train)
    y = jnp.asarray(y_train)

    n_dim = X.shape[1]
    
    # Intialize hyperparameters
    theta_init = {
        "log_amp": jnp.log(jnp.var(y) + 1e-6),
        "log_scale": jnp.zeros(n_dim), # Vector scale (ARD)
        "log_noise": jnp.log(jnp.var(y) + 1e-6) - 4.0
    }

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta_init)

    @jax.jit
    def step(theta, opt_state):
        loss, grads = jax.value_and_grad(nll_loss)(theta, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    best_loss = jnp.inf
    patience = 50
    patience_counter = 0
    tol = 1e-4

    for i in range(n_epochs):
        theta_init, opt_state, loss = step(theta_init, opt_state)
        
        if loss < best_loss - tol:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            if i % 200 != 0: # Ensure we log the final epoch if not already logged
                print(f"Epoch {i}, Loss: {loss}, early stopping...", flush=True)
            break

        if i % 200 == 0:
            print(f"Epoch {i}, Loss: {loss}", flush=True)

    return theta_init

class GPState:
    def __init__(self, theta, X_train, y_train):
        gp = build_gp(theta, X_train)
        self.X_train = X_train
        self.kernel = gp.kernel
        self.alpha = gp.solver.solve_triangular(y_train - gp.loc)
        # For variance, we need the Cholesky of the grain's covariance
        self.L = gp.solver.scale_tril
        self.noise = jnp.exp(theta["log_noise"])

def fast_predict(state: GPState, X_test: JAXArray):
    """
    Manual GP prediction using precomputed alpha and L.
    mu = K(X*, X) @ alpha
    """
    K_star_t = state.kernel(X_test, state.X_train)
    mu = K_star_t @ state.alpha
    
    # Variance calculation (optional, but needed for our MCMC)
    # var = K(X*, X*) - K(X*, X) K(X, X)^-1 K(X, X*)
    # using L: v = L^-1 K(X, X*)
    v = jax.scipy.linalg.solve_triangular(state.L, K_star_t.T, lower=True)
    var = state.kernel(X_test, X_test) - jnp.sum(jnp.square(v), axis=0)
    
    # Ensure non-negative variance
    var = jnp.clip(var, a_min=1e-10)
    return mu, var

def apply_gp(theta: dict, X_train: JAXArray, y_train: JAXArray, X_test: JAXArray):
    """
    Given a trained GP, calculate the predictive mean and variance at X_test.
    """
    state = GPState(theta, X_train, y_train)
    return fast_predict(state, X_test)
