import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def plot_traces(posterior, pnames):
    its, chains, dims = posterior.shape
    fig, axes = plt.subplots(dims, 1, figsize=(8, 2.5*dims), sharex=True)
    for d in range(dims):
        for c in range(chains):
            axes[d].plot(posterior[:,c,d], alpha=0.6, label=f'chain {c+1}')
        axes[d].set_ylabel(pnames[d])
        axes[d].legend(loc='upper right')
    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()

def gaussian_crps_per_point(y_true, mu, sigma, eps=1e-6):
    """All tensors broadcastable to same shape."""
    sigma = tf.nn.softplus(sigma) + eps
    z     = (y_true - mu) / sigma
    SQRT2   = tf.sqrt(tf.constant(2.0, tf.float32))
    SQRT2PI = tf.sqrt(tf.constant(2.0 * np.pi, tf.float32))
    inv_sqrt_pi = 1.0 / tf.sqrt(tf.constant(np.pi, tf.float32))

    phi = tf.exp(-0.5 * tf.square(z)) / SQRT2PI
    Phi = 0.5 * (1.0 + tf.math.erf(z / SQRT2))
    return sigma * ( z * (2.0 * Phi - 1.0) + 2.0 * phi - inv_sqrt_pi )