import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss

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

# def masked_mse_loss(threshold=-999):
#     def loss(y_true, y_pred):
#         # Create a mask: 1 when y_true > threshold, 0 otherwise
#         mask = K.cast(K.greater(y_true, threshold), K.floatx())
#         # Compute squared error and multiply by the mask
#         squared_error = K.square(y_true - y_pred)
#         masked_error = squared_error * mask
#         # Return the mean error over the active positions
#         # (adding a small epsilon to avoid division by zero)
#         return K.sum(masked_error) / (K.sum(mask) + K.epsilon())
#     return loss

def build_classifier_model(hp, npar, nobs):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 1, 4)):
        # Tune the number of units for each layer.
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        if i == 0:
            model.add(layers.Input(shape=(npar,)))
        else:
            model.add(layers.Dense(units, activation='relu'))

        # # tune dropout.
        # if hp.Boolean(f'dropout_{i}'):
        #     dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0, max_value=0.5, step=0.1)
        #     model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(nobs, activation='sigmoid'))
    
    model.compile(optimizer=keras.optimizers.Adam(
                      learning_rate=hp.Float('lr', 1e-5, 1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_classreg_model(hp, npar, nobs):
    inputs = layers.Input(shape=(npar,))
    x = inputs
    
    # Tune the number of shared layers (e.g., between 1 and 5 layers)
    num_shared_layers = hp.Int("num_shared_layers", min_value=1, max_value=5, step=1)
    for i in range(num_shared_layers):
        # Tune number of units for each layer
        units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
        x = layers.Dense(units, activation='relu')(x)
        # # Optionally add dropout
        # if hp.Boolean(f"dropout_{i}"):
        #     dropout_rate = hp.Float(f"dropout_rate_{i}", min_value=0.0, max_value=0.5, step=0.1)
        #     x = layers.Dropout(rate=dropout_rate)(x)
    
    # Classifier branch: predicts water presence (binary for each of the 7202 outputs)
    presence_output = layers.Dense(nobs, activation='sigmoid', name='presence')(x)
    # Regression branch: predicts water content (continuous for each output)
    water_output = layers.Dense(nobs, name='water')(x)
    
    model = keras.Model(inputs=inputs, outputs=[presence_output, water_output])
    
    # Choose the optimizer as a hyperparameter
    # optimizer_choice = hp.Choice('optimizer', values=['rmsprop'])
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    
    if optimizer_choice == 'adam':
        lr = hp.Float('adam_lr', 1e-6, 1e-2, sampling='LOG')
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == 'sgd':
        lr = hp.Float('sgd_lr', 1e-6, 1e-2, sampling='LOG')
        momentum = hp.Float('sgd_momentum', 0.0, 0.9, step=0.1)
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:  # rmsprop
        lr = hp.Float('rmsprop_lr', 1e-6, 1e-2, sampling='LOG')
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss={'presence': 'binary_crossentropy', 'water': MaskedMSE(threshold=-999)},
        metrics={'presence': 'accuracy', 'water': 'mae'}
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