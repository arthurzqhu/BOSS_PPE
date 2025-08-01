import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
import numpy as np

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


def build_crps_network(
        nvar,
        npar,
        nobs,
        l_has_presence=False,
        neurons=100,
        output_activation='linear',
        model_name="crps_model",
        learn_rate=0.001,
        compile_model=True,
        clear_session=True):
    ###
    # Build a simple neural network that creates an ensemble of 
    # predictions using the CRPS loss.
    ###
    # Inputs:
    # input_shape (int): number of features for training
    # n_members (int): number of ensemble members to estimate target.
    # neurons (int): number of nodes in each hidden layer
    # model_name (str): name of model
    # learn_rate (float): learning rate of model
    # compile_model (bool): compile model before returning?
    # clear_session (bool): clear the tensorflow session?
    #
    # Outputs:
    # tensorflow.keras.Model
    #
    if clear_session:
        tf.keras.backend.clear_session()

    # create features
    inputs = layers.Input(shape=(npar,), name="Input")

    # create three hidden layers
    x = layers.Dense(neurons, activation=tf.nn.leaky_relu, name="L1")(inputs)
    x = layers.Dense(neurons, activation=tf.nn.leaky_relu, name="L2")(x)
    x = layers.Dense(neurons, activation=tf.nn.leaky_relu, name="L3")(x)

    outputs = {}

    # create ensembles
    loss_dict = {}
    for i in range(nvar):
        outputs[f'water_{i}'] = layers.Dense(nobs[i], activation=output_activation, name=f'water_{i}')(x)
        loss_dict[f'water_{i}'] = loss_crps_sample_score
        if l_has_presence:
            outputs[f'presence_{i}'] = layers.Dense(nobs[i], activation='sigmoid', name=f'presence_{i}')(x)
    
    # create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    
    # Compile model, if desired
    if compile_model:
        opt = keras.optimizers.Adam(learning_rate=learn_rate)
        model.compile(loss=loss_dict, optimizer=opt)
    return model

def loss_crps_sample_score(y_true, y_pred):
    """Calculates the Continuous Ranked Probability Score (CRPS)
    for finite ensemble members and a single target.
    
    This implementation is based on the identity:
        CRPS(F, x) = E_F|y_pred - y_true| - 1/2 * E_F|y_pred - y_pred'|
    where y_pred and y_pred' denote independent random variables drawn from
    the predicted distribution F, and E_F denotes the expectation
    value under F.

    Following the approach by Steven Brey at 
    TheClimateCorporation (formerly ClimateLLC)
    https://github.com/TheClimateCorporation/properscoring
    
    Adapted from David Blei's lab at Columbia University
    http://www.cs.columbia.edu/~blei/ and
    https://github.com/blei-lab/edward/pull/922/files

    
    References
    ---------
    Tilmann Gneiting and Adrian E. Raftery (2005).
        Strictly proper scoring rules, prediction, and estimation.
        University of Washington Department of Statistics Technical
        Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    
    H. Hersbach (2000).
        Decomposition of the Continuous Ranked Probability Score
        for Ensemble Prediction Systems.
        https://doi.org/10.1175/1520-0434(2000)015%3C0559:DOTCRP%3E2.0.CO;2
    """

    # Variable names below reference equation terms in docstring above
    term_one = tf.reduce_mean(tf.abs(
        tf.subtract(y_pred, y_true)), axis=-1)
    term_two = tf.reduce_mean(
        tf.abs(
            tf.subtract(tf.expand_dims(y_pred, -1),
                        tf.expand_dims(y_pred, -2))),
        axis=(-2, -1))
    half = tf.constant(-0.5, dtype=term_two.dtype)
    score = tf.add(term_one, tf.multiply(half, term_two))
    score = tf.reduce_mean(score)
    return score

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


def build_classreg_unc_model(hp, npar, nvar, nobs):
    """
    Model outputs:
      - presence_{i}: sigmoid, for each variable
      - water_{i}: concatenated [mean, logvar] for each variable
    Loss:
      - presence: binary_crossentropy
      - water: gaussian_nll (negative log-likelihood)
    """
    inputs = layers.Input(shape=(npar,))
    x = inputs

    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    for i in range(num_shared_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)

    outputs = {}
    metrics = {}
    loss_dict = {}

    for i in range(nvar):
        # Presence output
        outputs[f'presence_{i}'] = layers.Dense(nobs[i], activation='sigmoid', name=f'presence_{i}')(x)
        metrics[f'presence_{i}'] = ['accuracy']
        loss_dict[f'presence_{i}'] = 'binary_crossentropy'

        # Water mean and logvar concatenated output
        water_mean = layers.Dense(nobs[i], name=f'water_mean_{i}')(x)
        water_logvar = layers.Dense(nobs[i], 
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=tf.keras.initializers.Constant(0.0),  # Start with log(1) = 0
                                   name=f'water_logvar_{i}')(x)
        # Add a minimum uncertainty floor for MCMC
        water_logvar = tf.maximum(water_logvar, tf.constant(-1.0))  # Minimum std of exp(-1) ≈ 0.37
        water_out = layers.Concatenate(name=f'water_{i}')([water_mean, water_logvar])
        outputs[f'water_{i}'] = water_out
        loss_dict[f'water_{i}'] = gaussian_nll
        
        def mae_on_mean(y_true, y_pred):
            n = tf.shape(y_pred)[-1] // 2
            mean = y_pred[..., :n]
            mask = tf.cast(tf.greater(y_true, -999.0), tf.float32)
            abs_err = tf.abs(y_true - mean) * mask
            return tf.reduce_sum(abs_err) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())
        metrics[f'water_{i}'] = [mae_on_mean]

    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-5, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        metrics=metrics,
    )

    return model

def build_classreg_model(hp, npar, nvar, nobs, l_dropout=False):
    inputs = layers.Input(shape=(npar,))
    x = inputs
    
    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    if l_dropout:
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='swish')(x)
        if l_dropout:
            x = layers.Dropout(dropout_rate)(x, training=True)  # Keep dropout active during inference

    outputs = {}
    loss_dict = {}
    metrics = {}
    for i in range(nvar):
        # Classifier branch: predicts water presence (binary for each of the 7202 outputs)
        outputs[f'presence_{i}'] = layers.Dense(nobs[i], activation='sigmoid', name=f'presence_{i}')(x)
        metrics[f'presence_{i}'] = ['accuracy']
        loss_dict[f'presence_{i}'] = 'binary_crossentropy'
        # Regression branch: predicts water content (continuous for each output)
        outputs[f'water_{i}'] = layers.Dense(nobs[i], name=f"water_{i}")(x)
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
    
class CombinedEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, loss_patience=10, acc_patience=10):
        super().__init__()
        self.loss_patience = loss_patience
        self.acc_patience = acc_patience
        self.wait_loss = 0
        self.wait_acc = 0
        self.best_loss = float('inf')
        self.best_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get the current validation loss for the regression output and the validation accuracy for the classification output.
        current_loss = logs.get('val_water_loss')
        current_acc = logs.get('val_presence_accuracy')

        # If the metrics are missing, we do nothing.
        if current_loss is None or current_acc is None:
            return

        # Update the wait counter for loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait_loss = 0
        else:
            self.wait_loss += 1

        # Update the wait counter for accuracy
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.wait_acc = 0
        else:
            self.wait_acc += 1

        # Only stop training if both metrics have not improved for their patience periods.
        if self.wait_loss >= self.loss_patience and self.wait_acc >= self.acc_patience:
            print(f"\nEpoch {epoch + 1}: combined early stopping triggered (loss wait: {self.wait_loss}, acc wait: {self.wait_acc}).")
            self.model.stop_training = True

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