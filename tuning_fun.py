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

def build_regressor_model(hp, npar, nobs):
    model = keras.Sequential()
    # Tune the number of layers.

    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'leaky_relu'])

    for i in range(hp.Int('num_layers', 1, 4)):
        # Tune the number of units for each layer.
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        if i == 0:
            model.add(layers.Input(shape=(npar,)))
        else:
            model.add(layers.Dense(units, activation=hp_activation))
        
        # Optionally, tune dropout.
        if hp.Boolean(f'dropout_{i}'):
            dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0, max_value=0.5, step=0.1)
            model.add(layers.Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(layers.Dense(nobs))
    
    # Choose the optimizer as a hyperparameter
    optimizer_choice = hp.Choice('optimizer', values=['rmsprop'])
    # optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    
    if optimizer_choice == 'adam':
        lr = hp.Float('adam_lr', 1e-5, 1e-2, sampling='LOG')
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == 'sgd':
        lr = hp.Float('sgd_lr', 1e-5, 1e-2, sampling='LOG')
        momentum = hp.Float('sgd_momentum', 0.0, 0.9, step=0.1)
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:  # rmsprop
        lr = hp.Float('rmsprop_lr', 1e-5, 1e-2, sampling='LOG')
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    # Tune learning rate.
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

@keras.saving.register_keras_serializable(package="CustomLosses", name="MaskedMSE")
class MaskedMSE(Loss):
    def __init__(self, threshold=0.0, **kwargs):
        super(MaskedMSE, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        mask = K.cast(K.greater(y_true, self.threshold), K.floatx())
        squared_error = K.square(y_true - y_pred)
        masked_error = squared_error * mask
        return K.sum(masked_error) / (K.sum(mask) + K.epsilon())

    def get_config(self):
        config = super(MaskedMSE, self).get_config()
        config.update({"threshold": self.threshold})
        return config

# def base_reg_loss(y_true, y_pred, threshold=-999):
#     mask = K.cast(K.greater(y_true, threshold), K.floatx())
#     squared_error = K.square(y_true - y_pred)
#     masked_error = squared_error * mask
#     return K.sum(masked_error) / (K.sum(mask) + K.epsilon())

def build_classreg_model(hp, npar, nobs):
    inputs = layers.Input(shape=(npar,))
    x = inputs
    
    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)
    
    # Classifier branch: predicts water presence (binary for each of the 7202 outputs)
    presence_output = layers.Dense(nobs, activation='sigmoid', name='presence')(x)
    # Regression branch: predicts water content (continuous for each output)
    water_output = layers.Dense(nobs, name='water')(x)
    
    model = keras.Model(inputs=inputs, outputs=[presence_output, water_output])
    lr = hp.Float('adam_lr', 1e-6, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss={'presence': 'binary_crossentropy', 'water': MaskedMSE(threshold=-999)},
        metrics={'presence': 'accuracy', 'water': 'mae'}
    )
    
    return model

def total_loss(y_true_list, y_pred_list, lambdas, ngrps=2, nvar=8):
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
        x = layers.Dense(units, activation='relu')(x)
    
    outputs = []
    
    for i in range(nvar):
        p_i = layers.Dense(nobs[i], activation="sigmoid", name=f"presence_{i}")(x)
        r_i = layers.Dense(nobs[i], activation="linear", name=f"raw_{i}")(x)
        combined_i = layers.Multiply(name=f"output_scalar_{i}")([p_i, r_i])
        outputs.append(combined_i)

    model = keras.Model(inputs=inputs, outputs=outputs)
    lr = hp.Float('adam_lr', 1e-6, 1e-1, sampling='LOG')
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    ngrps = nvar // 4  # Assuming each group has 4 outputs bc it's 4M BOSS, adjust as necessary
    lmono = [hp.Float("lambda_mono", 1e-4, 10.0, sampling="LOG")] * ngrps  # Monotonic penalty weight

    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: total_loss(y_true, y_pred, lmono, ngrps),
        metrics=["mae"] * nvar
    )

    return model

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