#!/home/arthurhu/BOSS_PPE/.venv/bin/python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tuning_fun as tu
import emulator_fun as ef
import numpy as np
from tqdm.keras import TqdmCallback
import pandas as pd
import time

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # in MB
#         )
#     except RuntimeError as e:
#         print(e)

# time.sleep(600)

case_name = 'condcoll'
base_path = '/home/arthurhu/BOSS_PPE/PPE csv/'
ppe_params_fn = case_name + '_r0_nopartition_ppe_params.csv'
ppe_sim_fn = case_name + '_r0_nopartition_LWP1234_ppe_var.csv'
target_var_fn = case_name + '_LWP1234_target_var.csv'

param_all_idx = np.arange(40)
# REVIEW: check every time if this is true ... not sure how to implement this programatically
# param_interest_idx = np.arange(16,28)
# param_interest_idx = np.arange(40)
param_interest_idx = np.arange(16,28)
# param_interest_idx = np.arange(12)
# param_interest_idx = np.concatenate((np.arange(0,4), np.arange(25,35)))
param_not_int_idx = [i for i in param_all_idx if i not in param_interest_idx]

param_train = ef.get_params(base_path, ppe_params_fn, param_interest_idx)
# param_valid = ef.get_params(base_path, ppe_params_valid_fn, param_interest_idx)

# slim down the output sets
# output_slim = np.concatenate(([0],np.arange(1,62,4)), dtype=int)
output_slim = []
ppe_sim_train = ef.get_vars(base_path, ppe_sim_fn, slim_down=output_slim)
ppe_sim_train['vals'][np.isinf(ppe_sim_train['log_vals'])] = np.nan
# ppe_sim_train['log_vals'][np.isinf(ppe_sim_train['log_vals'])] = np.nan
# ppe_sim_valid = ef.get_vars(base_path, ppe_sim_valid_fn, slim_down=output_slim)
tgt_sim = ef.get_vars(base_path, target_var_fn)

nobs = ppe_sim_train['vals'].shape[1]
npar = param_train['vals'].shape[1]

eff0 = np.array([1e-12, 100, 1e-27, 1e-42])

y_thresholds = np.zeros(ppe_sim_train['vals'].shape[1])
y_thresholds[0] = eff0[0]
y_thresholds[1] = eff0[1]
# y_thresholds[2] = 1e-5
y_thresholds[2] = eff0[2]
y_thresholds[3] = eff0[3]
# y_thresholds[4:64] = 1e-5
y_thresholds[4:3604] = eff0[0]
y_thresholds[3604:7204] = eff0[1]
y_thresholds[7204:10804] = eff0[2]
y_thresholds[10804:] = eff0[3]

y_lin = ppe_sim_train['vals']
y_lin[ppe_sim_train['vals'] < y_thresholds] = np.nan
y_lin[~np.isfinite(y_lin)] = np.nan

from sklearn import preprocessing
import sklearn.model_selection as mod_sec

minmaxscale = preprocessing.MinMaxScaler().fit(param_train['vals'])
x_all = minmaxscale.transform(param_train['vals'])
# x_val = minmaxscale.transform(param_valid['vals'])

power = 1
# y_threshold = 1e-9
y_thresholds = np.zeros(ppe_sim_train['vals'].shape[1])
# y_thresholds[0]

y_thresholds[0] = eff0[0]
y_thresholds[1] = eff0[1]
# y_thresholds[2] = 1e-5
y_thresholds[2] = eff0[2]
y_thresholds[3] = eff0[3]
# y_thresholds[4:64] = 1e-5
y_thresholds[4:3604] = eff0[0]
y_thresholds[3604:7204] = eff0[1]
y_thresholds[7204:10804] = eff0[2]
y_thresholds[10804:] = eff0[3]

# standscale = preprocessing.StandardScaler().fit(ppe_sim_train['vals'])
y_all_wpresence = (ppe_sim_train['vals'] > y_thresholds).astype('float32')

# y_log = ppe_sim_train['log_vals']
# y_log[~np.isfinite(y_log)] = np.nan
# y_log[ppe_sim_train['vals'] < y_thresholds] = np.nan
# standscale = preprocessing.StandardScaler().fit(y_log)
# y_all = standscale.transform(y_log)

# y_lin = ppe_sim_train['vals']
# y_lin[~np.isfinite(y_lin)] = np.nan
# y_lin[ppe_sim_train['vals'] < y_thresholds] = np.nan
# standscale = preprocessing.StandardScaler().fit(y_lin)
# y_all = standscale.transform(y_lin)
# y_val = np.ma.masked_invalid(standscale.transform(ppe_sim_valid['log_vals']))

y_lin = ppe_sim_train['vals']
y_lin[ppe_sim_train['vals'] < y_thresholds] = np.nan
y_lin[~np.isfinite(y_lin)] = np.nan
y_asinh = ef.smooth_linlog(y_lin, y_thresholds)
standscale = preprocessing.StandardScaler().fit(y_asinh)
y_all = standscale.transform(y_asinh)

tgt_lin = tgt_sim['vals'][:,2:]
tgt_lin[tgt_sim['vals'][:,2:] < y_thresholds] = np.nan
tgt_lin[~np.isfinite(tgt_lin)] = np.nan
tgt_asinh = ef.smooth_linlog(tgt_lin, y_thresholds)
tgt_all = standscale.transform(tgt_asinh)


x_train, x_val, y_train_wpresence, y_val_wpresence = mod_sec.train_test_split(x_all, y_all_wpresence, test_size=0.2, random_state=1)
_, _, y_train, y_val = mod_sec.train_test_split(x_all, y_all, test_size=0.2, random_state=1)

y_train = np.nan_to_num(y_train, nan=-1001)
y_val = np.nan_to_num(y_val, nan=-1001)

proj_name = 'try29_condcoll_fullmom_just_adam_asinh_shared'

# tuner = kt.Hyperband(
#     lambda hp: tu.build_classreg_model(hp, npar, nobs),
#     objective="val_loss",
#     max_epochs=100,
#     hyperband_iterations=50,
#     distribution_strategy=tf.distribute.MirroredStrategy(),
#     directory='hp_tuning_withclass/multi-output',
#     project_name=proj_name,
# )



tuner = kt.RandomSearch(
    lambda hp: tu.build_classreg_model(hp, npar, nobs),
    objective="val_loss",
    max_trials=100,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory='hp_tuning_withclass/multi-output',
    project_name=proj_name,
)


# combined_early_stop = tu.CombinedEarlyStopping(loss_patience=10, acc_patience=10)

# class DeepcopyableTqdmCallback(TqdmCallback):
#     def __deepcopy__(self, memo):
#         # Create a new instance with the same configuration.
#         return DeepcopyableTqdmCallback(verbose=self.verbose)

# # Now use this subclass in tuner.search:
# tqdm_callback = DeepcopyableTqdmCallback(verbose=1)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(
    x_train,
    {'presence': y_train_wpresence, 'water': y_train},
    epochs=50,
    validation_data=(x_val, {'presence': y_val_wpresence, 'water': y_val}),
    callbacks=([stop_early])
)

# # Retrieve the best hyperparameters and build the best model:
# best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
# best_model = tuner.hypermodel.build(best_hp)


# best_model.summary()

# # stop_early = keras.callbacks.EarlyStopping(monitor='val_presence_accuracy', mode='max', patience=10)

# class PrintDot(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs):
#         if epoch % 100 == 0: print('')
#         print('.', end='')

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, {'presence': y_train_wpresence, 'water': y_train}))
# train_dataset = (train_dataset
#                  .shuffle(buffer_size=len(x_train))
#                  .cache()
#                  .batch(64)
#                  .prefetch(tf.data.AUTOTUNE))

# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, {'presence': y_val_wpresence, 'water': y_val}))
# val_dataset = (val_dataset
#                .cache()
#                .batch(64)
#                .prefetch(tf.data.AUTOTUNE))


# history = best_model.fit(
#     train_dataset,
#     epochs=20000,
#     verbose=0,
#     validation_data=val_dataset,
#     callbacks=[TqdmCallback(verbose=1)]
#     # callbacks=[PrintDot()]
# )

# plt.figure()
# plt.plot(history.epoch, history.history['presence_accuracy'], label='Training')
# plt.plot(history.epoch, history.history['val_presence_accuracy'], label='Validation')
# plt.ylabel('Water Presence Accuracy')
# plt.legend()
# plt.savefig('plots/condcoll_model_WPA_' + proj_name + '.pdf')

# plt.figure()
# plt.plot(history.epoch, history.history['water_loss'], label='Training')
# plt.plot(history.epoch, history.history['val_water_loss'], label='Validation')
# plt.ylabel('Water Content Loss')
# plt.legend()
# plt.savefig('plots/condcoll_model_WCL_' + proj_name + '.pdf')

# # save model:
# best_model.save('models/multioutput_' + proj_name + '.keras')
