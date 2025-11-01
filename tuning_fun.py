import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
import numpy as np
from keras import ops as K # for some reason, this is needed for the StochasticLatent layer

SQRT2   = tf.sqrt(tf.constant(2., tf.float32))
SQRT_PI = tf.sqrt(tf.constant(np.pi, tf.float32))
SQRT_2PI = tf.sqrt(tf.constant(2. * np.pi, tf.float32))

@keras.saving.register_keras_serializable(package="CustomLosses", name="MaskedMSE")
class MaskedMSE(Loss):
    def __init__(self, threshold=0.0, **kwargs):
        super(MaskedMSE, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        mask = K.cast(K.greater(y_true, self.threshold), K.floatx())
        squared_error = K.square(y_true - y_pred)
        masked_error = squared_error * mask
        return K.sum(masked_error) / (K.sum(mask) + K.epsilon())

    def get_config(self):
        config = super(MaskedMSE, self).get_config()
        config.update({"threshold": self.threshold})
        return config

def gaussian_nll(y_true, y_pred):
    """
    Custom negative log-likelihood loss for Gaussian outputs, masking y_true <= -999.
    y_pred: concatenated [mean, logvar] for each output
    """
    n = tf.shape(y_pred)[-1] // 2
    mean = y_pred[..., :n]
    logvar = y_pred[..., n:]
    # Mask: only include y_true > -999
    mask = tf.cast(tf.greater(y_true, -999.0), tf.float32)
    # clip logvar to avoid too small/large values and ensure minimum variance
    logvar = tf.clip_by_value(logvar, -4.0, 10.0)  # exp(-4) ≈ 0.018, exp(10) ≈ 22026
    nll = logvar + tf.square(y_true - mean) / (tf.exp(logvar) + 1e-6)
    masked_nll = nll * mask
    return 0.5 * tf.reduce_sum(masked_nll) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

def gaussian_nll_loss_factory(n_obs, mask_min=-999.0, min_sigma=1e-6):
    """
    Expects:
      y_true: [B, n_obs]
      y_pred: [B, 2*n_obs]  (first n_obs = mu, last n_obs = raw scale)
    Returns mean NLL over unmasked entries per batch.
    """
    LOG_SQRT_2PI = 0.5 * tf.math.log(tf.constant(2.0 * np.pi, tf.float32))

    def loss(y_true, y_pred):
        # be robust to mixed precision
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mu    = y_pred[:, :n_obs]
        raw_s = y_pred[:, n_obs:]
        sigma = tf.nn.softplus(raw_s) + tf.constant(min_sigma, tf.float32)

        # mask invalid targets
        mask   = tf.greater(y_true, tf.constant(mask_min, y_true.dtype))
        mask_f = tf.cast(mask, tf.float32)

        # NLL for N(mu, sigma): 0.5*((y-mu)/sigma)^2 + log(sigma) + 0.5*log(2π)
        z    = (y_true - mu) / sigma
        nll  = 0.5 * tf.square(z) + tf.math.log(sigma) + LOG_SQRT_2PI

        # average over observed entries, then over batch
        num = tf.reduce_sum(nll * mask_f, axis=1)                   # [B]
        den = tf.reduce_sum(mask_f, axis=1) + 1e-9                  # [B]
        return tf.reduce_mean(num / den)                            # scalar
    return loss

# A simple interpretable metric: masked MAE on μ (like your MSE era)
def masked_mae_from_mu_factory(n_obs, mask_min=-999.0):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        mu     = tf.cast(y_pred[:, :n_obs], tf.float32)
        mask_f = tf.cast(tf.greater(y_true, tf.constant(mask_min, y_true.dtype)), tf.float32)
        num = tf.reduce_sum(tf.abs(mu - y_true) * mask_f, axis=1)
        den = tf.reduce_sum(mask_f, axis=1) + 1e-9
        return tf.reduce_mean(num / den)
    metric.__name__ = "masked_mae_mu"
    return metric

def build_classreg_unc_model(hp, npar, varcons, nobs, l_dropout=False):
    """
    Presence heads: sigmoid + BCE.
    Contraint variable (convar) heads: Dense(2*nobs[i]) -> [mu, raw_sigma]; loss = masked Gaussian NLL.
    """
    inputs = layers.Input(shape=(npar,))
    x = inputs

    # Shared trunk (same pattern as your original)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    if l_dropout:
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    for i in range(num_shared_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)
        if l_dropout:
            # Keep dropout active at inference if you want MC-dropout samples
            x = layers.Dropout(dropout_rate)(x, training=True)

    outputs, loss_dict, metrics = {}, {}, {}

    for i, varcon in enumerate(varcons):
        # Presence (classification) — unchanged
        pres_name = f'presence_{varcon}'
        outputs[pres_name] = layers.Dense(nobs[i], activation='sigmoid', name=pres_name)(x)
        loss_dict[pres_name] = 'binary_crossentropy'
        metrics[pres_name]   = ['accuracy']

        # Contraint variable (convar) (uncertainty-aware regression): [mu, raw_sigma]
        outputs[varcon] = layers.Dense(2 * nobs[i], name=varcon)(x)

        loss_dict[varcon] = gaussian_nll_loss_factory(nobs[i], mask_min=-999.0, min_sigma=1e-6)
        metrics[varcon]   = [masked_mae_from_mu_factory(nobs[i], mask_min=-999.0)]

    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics)
    return model

def gaussian_crps_loss_factory(n_obs, mask_min=-999.0, min_sigma=1e-6):
    """
    Returns a loss(y_true, y_pred) that:
      - expects y_true: [B, n_obs]
      - expects y_pred: [B, 2*n_obs]  -> first n_obs are μ, last n_obs are raw scales
      - applies mask where y_true <= mask_min
      - computes mean CRPS over unmasked entries and over batch
    """
    inv_sqrt_pi = 1.0 / SQRT_PI

    def loss(y_true, y_pred):
        # Split μ and raw scale, then map raw -> σ > 0
        mu   = y_pred[:, :n_obs]
        rsca = y_pred[:, n_obs:]
        sigma = tf.nn.softplus(rsca) + tf.constant(min_sigma, tf.float32)

        # Mask invalid/missing targets
        mask = K.greater(y_true, K.cast(mask_min, y_true.dtype))
        # If everything is masked for a batch item, avoid NaNs by giving zero weight
        mask_f = K.cast(mask, "float32")

        # z = (y - μ)/σ
        z   = (y_true - mu) / sigma
        # φ(z), Φ(z)
        phi = K.exp(-0.5 * K.square(z)) / SQRT_2PI
        Phi = 0.5 * (1.0 + K.erf(z / SQRT2))

        # CRPS for N(μ, σ): σ [ z(2Φ-1) + 2φ - 1/√π ]
        crps = sigma * ( z * (2.0 * Phi - 1.0) + 2.0 * phi - inv_sqrt_pi )

        # Apply mask and average over observed entries then over batch
        crps_sum   = K.sum(crps * mask_f, axis=1)
        count_obs  = K.sum(mask_f, axis=1) + 1e-9  # avoid div-by-zero
        crps_mean_per_item = crps_sum / count_obs
        return K.mean(crps_mean_per_item)          # scalar

    return loss

# Metric to keep your masked MAE behavior but computed on μ
def masked_mae_from_mu_factory(n_obs, mask_min=-999.0):
    def metric(y_true, y_pred):
        mu = y_pred[:, :n_obs]
        mask = tf.greater(y_true, tf.constant(mask_min, y_true.dtype))
        mask_f = tf.cast(mask, tf.float32)
        abs_err = tf.abs(mu - y_true) * mask_f
        sum_err = tf.reduce_sum(abs_err, axis=1)
        count   = tf.reduce_sum(mask_f, axis=1) + 1e-9
        return tf.reduce_mean(sum_err / count)
    metric.__name__ = "masked_mae_mu"
    return metric

def build_reg_crps_model(hp, npar, varcons, nobs, l_dropout=False):
    inputs = layers.Input(shape=(npar,))
    x = inputs

    # Shared trunk
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    if l_dropout:
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    for i in range(num_shared_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)
        if l_dropout:
            x = layers.Dropout(dropout_rate)(x, training=True)  # MC dropout at inference

    outputs = {}
    loss_dict = {}
    metrics = {}

    for i, varcon in enumerate(varcons):
        # --- Contraint variable (convar) (regression) branch -> distribution head with CRPS ---
        # Single output that packs [μ, raw_scale] for each obs -> shape [B, 2*nobs[i]]
        outputs[varcon] = layers.Dense(2 * nobs[i], name=varcon)(x)

        # Use CRPS loss (Gaussian closed form) with your masking convention
        loss_dict[varcon] = gaussian_crps_loss_factory(nobs[i], mask_min=-999.0, min_sigma=1e-6)
        # Metric: masked MAE of μ (for easy monitoring)
        metrics[varcon]   = [masked_mae_from_mu_factory(nobs[i], mask_min=-999.0)]

    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics)
    return model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.plot(hist['epoch'],0.1+0.*hist['epoch'],'k:')
    plt.plot(hist['epoch'],0.2+0.*hist['epoch'],'k:')
    plt.plot(hist['epoch'],0.3+0.*hist['epoch'],'k:')
    plt.plot(hist['epoch'],0.4+0.*hist['epoch'],'k:')
    #plt.ylim([0,1])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.plot(hist['epoch'],0.1+0.*hist['epoch'],'k:')
    plt.plot(hist['epoch'],0.2+0.*hist['epoch'],'k:')
    plt.plot(hist['epoch'],0.3+0.*hist['epoch'],'k:')
    plt.plot(hist['epoch'],0.4+0.*hist['epoch'],'k:')
    #plt.ylim([0,1])
    plt.legend()
    plt.show()


# ------------------------- variational autoencoder tests -------------------------

# ----------------------------
# New: latent space building blocks
# ----------------------------

class StochasticLatent(layers.Layer):
    """
    Samples z and adds a KL penalty via self.add_loss().
    Logs kl_mean via an internal Metric tracker (Keras 3-safe).
    """
    def __init__(self, kl_weight=1e-4, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.kl_weight = float(kl_weight)
        self.kl_mean_tracker = keras.metrics.Mean(name="kl_mean")

    def call(self, inputs):
        mean_z, logvar_z = inputs
        eps = keras.random.normal(shape=K.shape(mean_z), dtype=mean_z.dtype)
        z = mean_z + K.exp(0.5 * logvar_z) * eps

        kl = -0.5 * K.sum(1.0 + logvar_z - K.square(mean_z) - K.exp(logvar_z), axis=-1)
        kl_mean = K.mean(kl)

        # register penalty
        self.add_loss(self.kl_weight * kl_mean)
        # log metric via tracker
        self.kl_mean_tracker.update_state(kl_mean)
        return z

    # expose tracker so Model sees it
    @property
    def metrics(self):
        return [self.kl_mean_tracker]

    # help some build paths with shape inference
    def compute_output_shape(self, input_shape):
        return input_shape[0]

class ReconPenalty(layers.Layer):
    """
    Adds MSE(x, x_recon) * weight via self.add_loss().
    Logs recon_mse via a Metric tracker (no add_metric()).
    """
    def __init__(self, weight=1e-3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = float(weight)
        self.recon_mse_tracker = keras.metrics.Mean(name="recon_mse")

    def call(self, inputs):
        x, x_recon = inputs
        mse = K.mean(K.square(x - x_recon))
        self.add_loss(self.weight * mse)
        self.recon_mse_tracker.update_state(mse)
        return x_recon

    @property
    def metrics(self):
        return [self.recon_mse_tracker]

    def compute_output_shape(self, input_shape):
        # returns shape of x_recon
        return input_shape[1]

def build_encoder_block(x, units_list, activation='relu', dropout_rate=None, training=False):
    h = x
    for i, u in enumerate(units_list):
        h = layers.Dense(u, activation=activation, name=f"enc_dense_{i}")(h)
        if dropout_rate is not None:
            # Keep your Monte Carlo dropout behavior if desired
            h = layers.Dropout(dropout_rate)(h, training=training)
    return h

def build_decoder_block(z, npar, units_list=(128, 256), activation='relu'):
    h = z
    for i, u in enumerate(units_list):
        h = layers.Dense(u, activation=activation, name=f"dec_dense_{i}")(h)
    x_recon = layers.Dense(npar, name="x_recon")(h)
    return x_recon

def build_reg_crps_latent_model(
    hp, npar, varcons, nobs, l_dropout=False,
):
    inputs = layers.Input(shape=(npar,), name="x")
    x = inputs

    # shared trunk (same as before)
    num_shared_layers = hp.Int("num_shared_layers", 1, 5, step=1)
    if l_dropout:
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
    for i in range(num_shared_layers):
        units = hp.Int(f"units_{i}", 32, 256, step=32)
        x = layers.Dense(units, activation="relu", name=f"enc_dense_{i}")(x)
        if l_dropout:
            x = layers.Dropout(dropout_rate)(x, training=True)

    # latent
    latent_k = hp.Int("latent_k", 2, 16, step=2)
    use_stochastic_latent = hp.Boolean("use_stochastic_latent", default=True)
    if use_stochastic_latent:
        z_mu     = layers.Dense(latent_k, name="z_mu")(x)
        z_logvar = layers.Dense(latent_k, name="z_logvar")(x)
        kl_weight = hp.Float("lambda_kl", 1e-5, 1e-2, sampling="LOG")
        z = StochasticLatent(kl_weight=kl_weight, name="z_sampler")([z_mu, z_logvar])
    else:
        z = layers.Dense(latent_k, name="z")(x)

    # optional decoder regularizer (no model.add_loss)
    use_decoder = hp.Boolean("use_decoder", default=False)
    if use_decoder:
        dec_u0 = hp.Int("dec_u0", 64, 256, step=64)
        dec_u1 = hp.Int("dec_u1", 64, 256, step=64)
        h = layers.Dense(dec_u0, activation="relu", name="dec_dense_0")(z)
        h = layers.Dense(dec_u1, activation="relu", name="dec_dense_1")(h)
        x_recon_raw = layers.Dense(npar, name="x_recon")(h)
        recon_w = hp.Float("lambda_recon", 1e-4, 1e-1, sampling="LOG")
        # attaches loss internally and returns x_recon
        _ = ReconPenalty(weight=recon_w, name="recon_penalty")([inputs, x_recon_raw])

    outputs = {}
    loss_dict = {}
    metrics = {}
    for i, varcon in enumerate(varcons):
        head_units = hp.Int(f"{varcon}_head_units", 32, 256, step=32)
        h = layers.Dense(head_units, activation="relu", name=f"{varcon}_head_dense")(z)
        outputs[varcon] = layers.Dense(2 * nobs[i], name=varcon)(h)
        loss_dict[varcon] = gaussian_crps_loss_factory(nobs[i], mask_min=-999.0, min_sigma=1e-6)
        metrics[varcon]   = [masked_mae_from_mu_factory(nobs[i], mask_min=-999.0)]

    model = keras.Model(inputs=inputs, outputs=outputs, name="reg_crps_latent")

    lr = hp.Float("adam_lr", 1e-5, 1e-2, sampling="LOG")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_dict, metrics=metrics)
    return model



# ------------------------- below are not being used at the moment -------------------------
def build_classreg_model(hp, npar, varcons, nobs, l_dropout=False):
    inputs = layers.Input(shape=(npar,))
    x = inputs
    
    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    if l_dropout:
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)
        if l_dropout:
            x = layers.Dropout(dropout_rate)(x, training=True)  # Keep dropout active during inference

    outputs = {}
    loss_dict = {}
    metrics = {}
    for i, varcon in enumerate(varcons):
        # Classifier branch: predicts water presence (binary for each of the 7202 outputs)
        outputs[f'presence_{varcon}'] = layers.Dense(nobs[i], activation='sigmoid', name=f'presence_{varcon}')(x)
        metrics[f'presence_{varcon}'] = ['accuracy']
        loss_dict[f'presence_{varcon}'] = 'binary_crossentropy'
        # Regression branch: predicts water content (continuous for each output)
        outputs[varcon] = layers.Dense(nobs[i], name=varcon)(x)
        loss_dict[varcon] = MaskedMSE(threshold=-999)
        metrics[varcon] = [masked_mae(threshold=-999)]
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        metrics=metrics,
    )
    
    return model

def build_classreg_fallspeed_model(hp, npar, nvar, nobs):
    inputs = layers.Input(shape=(npar,))
    x = inputs
    
    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)

    presence_outputs = []
    water_outputs = []
    metrics = {}
    for i in range(nvar):
        # Classifier branch: predicts water presence (binary for each of the 7202 outputs)
        presence_outputs.append(layers.Dense(nobs[i], activation='sigmoid', name=f"presence_{i}")(x))
        # Regression branch: predicts water content (continuous for each output)
        water_outputs.append(layers.Dense(nobs[i], activation='sigmoid', name=f"water_{i}")(x))
        metrics[f"presence_{i}"] = ['accuracy']
        metrics[f"water_{i}"] = [masked_mae(threshold=-999)]
    
    model = keras.Model(inputs=inputs, outputs={'presence': presence_outputs, 'water': water_outputs})
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss={'presence': 'binary_crossentropy', 'water': MaskedMSE(threshold=-999)},
        metrics=metrics,
    )
    
    return model

def total_mono_loss(y_true_list, y_pred_list, lambdas, ngrps=2, nvar=8):
    L_data = 0.
    # for y_true, y_pred in zip(y_true_list, y_pred_list):
    #     L_data += tf.MaskedMSE(threshold=-999)(y_true, y_pred)
    for i in range(nvar):
        L_data += tf.reduce_mean(tf.square(y_true_list[i] - y_pred_list[i]))
    
    raw_vals = []
    for igrp in range(ngrps):
        raw_vals.append(tf.stack([y_pred_list[i] for i in range(igrp*4, (igrp+1)*4)], axis=-1))
        # raw2 = tf.stack([y_pred_list[i] for i in range(4,8)], axis=-1)  # shape=(batch,4)

    # Compute monotonic penalty for each group:
    def mono_penalty(raw_group):
        diffs = raw_group[..., :-1] - raw_group[..., 1:]      # (batch,3)
        viol  = tf.nn.relu(diffs)                          # (batch,3)
        return tf.reduce_mean(tf.square(viol))  # scalar

    L_monos = 0.
    for igrp in range(ngrps):
        L_monos += lambdas[igrp] * mono_penalty(raw_vals[igrp])
    # L_monos = np.sum([mono_penalty(raw) for raw in raw_vals])
    # L_mono2 = mono_penalty(raw2)
    # Total loss is the sum (or weighted sum) of everything:
    return L_data + L_monos

def build_multihead_mono_model(hp, npar, nvar, nobs):
    # npar = inputs
    # nvar = number of variables (e.g., 8)
    # nobs = number of observation for each variable (e.g., [1, 1, 1, 1, 720, 720, 720, 720])
    
    inputs = layers.Input(shape=(npar,))
    x = inputs

    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)
    
    outputs = []
    
    for i in range(nvar):
        p_i = layers.Dense(nobs[i], activation="sigmoid", name=f"presence_{i}")(x)
        r_i = layers.Dense(nobs[i], activation="linear", name=f"raw_{i}")(x)
        combined_i = layers.Multiply(name=f"output_scalar_{i}")([p_i, r_i])
        outputs.append(combined_i)

    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    ngrps = nvar // 4  # Assuming each group has 4 outputs bc it's 4M BOSS, adjust as necessary
    lmono = [hp.Float("lambda_mono", 1e-4, 10.0, sampling="LOG")] * ngrps  # Monotonic penalty weight

    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: total_mono_loss(y_true, y_pred, lmono, ngrps),
        metrics=["mae"] * nvar
    )

    return model

def build_multihead_model(hp, npar, nvar, nobs):
    # npar = inputs
    # nvar = number of variables (e.g., 8)
    # nobs = number of observation for each variable (e.g., [1, 1, 1, 1, 720, 720, 720, 720])
    
    inputs = layers.Input(shape=(npar,))
    x = inputs

    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation=tf.nn.leaky_relu, name=f"L{i+1}")(x)
    
    outputs = []
    
    for i in range(nvar):
        p_i = layers.Dense(nobs[i], activation="sigmoid", name=f"presence_{i}")(x)
        r_i = layers.Dense(nobs[i], activation="linear", name=f"raw_{i}")(x)
        combined_i = layers.Multiply(name=f"water_{i}")([p_i, r_i])
        outputs.append(combined_i)

    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=MaskedMSE(threshold=-999),
        metrics=[masked_mae(threshold=-999)] * nvar
    )

    return model

def masked_mae(threshold=-999):
    """
    Returns a MAE metric that ignores (i.e. weights=0) any y_true <= threshold.
    """
    def _masked_mae(y_true, y_pred):
        # y_true, y_pred: arbitrary shape
        mask = tf.cast(tf.greater(y_true, threshold), tf.float32)
        abs_err = tf.abs(y_true - y_pred) * mask
        # sum over all elements, then divide by total valid count
        return tf.reduce_sum(abs_err) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
    return _masked_mae
    

def ensemble_predict(models, x):
    """
    Get predictions from ensemble and compute uncertainty.
    
    Args:
        models: List of trained models
        x: Input data
    
    Returns:
        mean_pred: Mean prediction across ensemble
        std_pred: Standard deviation (epistemic uncertainty)
    """
    predictions = []
    
    for model in models:
        pred = model(x)
        predictions.append(pred)
    
    # Stack predictions
    stacked_preds = tf.stack(predictions, axis=0)
    
    # Calculate mean and std across ensemble
    mean_pred = tf.reduce_mean(stacked_preds, axis=0)
    std_pred = tf.math.reduce_std(stacked_preds, axis=0)
    
    return mean_pred, std_pred


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def build_mc_dropout_model(hp, npar, nvar, nobs):
    """
    Model with Monte Carlo Dropout for epistemic uncertainty estimation.
    This is more appropriate for MCMC as it captures model uncertainty.
    """
    inputs = layers.Input(shape=(npar,))
    x = inputs

    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    
    for i in range(num_shared_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation=tf.nn.leaky_relu)(x)
        x = layers.Dropout(dropout_rate)(x, training=True)  # Keep dropout active during inference

    outputs = {}
    metrics = {}
    loss_dict = {}

    for i in range(nvar):
        # Presence output
        outputs[f'presence_{i}'] = layers.Dense(nobs[i], activation='sigmoid', name=f'presence_{i}')(x)
        metrics[f'presence_{i}'] = ['accuracy']
        loss_dict[f'presence_{i}'] = 'binary_crossentropy'

        # Water output (mean only - uncertainty comes from MC sampling)
        outputs[f'water_{i}'] = layers.Dense(nobs[i], name=f'water_{i}')(x)
        loss_dict[f'water_{i}'] = MaskedMSE(threshold=-999)
        metrics[f'water_{i}'] = [masked_mae(threshold=-999)]

    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        metrics=metrics,
    )

    return model

def mc_dropout_predict(model, x, n_samples=100):
    """
    Monte Carlo Dropout prediction to estimate epistemic uncertainty.
    
    Args:
        model: Trained model with dropout layers
        x: Input data
        n_samples: Number of MC samples
    
    Returns:
        mean_pred: Mean prediction
        std_pred: Standard deviation (epistemic uncertainty)
    """
    predictions = []
    
    for _ in range(n_samples):
        pred = model(x, training=True)  # Keep dropout active
        predictions.append(pred)
    
    # Stack predictions
    stacked_preds = tf.stack(predictions, axis=0)
    
    # Calculate mean and std across samples
    mean_pred = tf.reduce_mean(stacked_preds, axis=0)
    std_pred = tf.math.reduce_std(stacked_preds, axis=0)
    
    return mean_pred, std_pred

def build_ensemble_model(hp, npar, nvar, nobs, n_models=5):
    """
    Build an ensemble of models for uncertainty estimation.
    Each model in the ensemble provides a different prediction,
    and the spread gives us epistemic uncertainty.
    """
    models = []
    
    for i in range(n_models):
        # Each model gets different random initialization
        tf.random.set_seed(hp.Int(f"seed_{i}", 0, 10000))
        
        inputs = layers.Input(shape=(npar,))
        x = inputs

        num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
        for j in range(num_shared_layers):
            units = hp.Int(f"units_{j}", min_value=32, max_value=256, step=32)
            x = layers.Dense(units, activation='relu')(x)

        outputs = {}
        for k in range(nvar):
            outputs[f'water_{k}'] = layers.Dense(nobs[k], name=f'water_{k}_model_{i}')(x)
            if hp.Boolean("use_presence"):
                outputs[f'presence_{k}'] = layers.Dense(nobs[k], activation='sigmoid', name=f'presence_{k}_model_{i}')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        
        # Use different loss functions for different models to increase diversity
        if i % 2 == 0:
            loss = MaskedMSE(threshold=-999)
        else:
            loss = 'mse'
            
        model.compile(optimizer=optimizer, loss=loss)
        models.append(model)
    
    return models