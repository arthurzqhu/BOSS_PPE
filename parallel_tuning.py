import os

def run_tuning_worker(worker_id, x_train, y_train, x_val, y_val, sw_train, sw_val, nparam_init, varcons, nobs, max_trials, proj_name, directory):
    """
    Worker function to run a single keras_tuner search process.
    """
    # Set environment variables BEFORE importing TensorFlow
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Imports must be inside the worker function to avoid triggering TF init in parent
    import tensorflow as tf
    import keras_tuner as kt
    import tuning_fun as tu
    import util_fun as uf
    from keras.callbacks import TerminateOnNaN
    
    # Set GPU memory growth inside the process
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            pass
    
    # Instantiate tuner with same directory and project_name to coordinate between workers
    tuner = tu.SilentRandomSearch(
        lambda hp: tu.build_reg_crps_model(hp, nparam_init, varcons, nobs),
        objective="val_loss",
        max_trials=max_trials,
        directory=directory,
        project_name=proj_name,
    )
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    print(f"Worker {worker_id} starting search...")
    tuner.search(
        x_train,
        y_train,
        epochs=25,
        sample_weight=sw_train,
        validation_data=(x_val, y_val, sw_val),
        callbacks=[stop_early, TerminateOnNaN(), uf.MemoryCleanupCallback()],
        verbose=0
    )
    print(f"Worker {worker_id} finished search.")

def run_training_worker(worker_id, hp, x_train, y_train, x_val, y_val, sw_train, sw_val, nparam_init, varcons, nobs, epochs, proj_name, save_path):
    """
    Worker function to train a single model in parallel.
    """
    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b5356651-0d8e-5cd1-bdf3-ccbb8b221031"
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    import tensorflow as tf
    import keras
    import tuning_fun as tu
    import util_fun as uf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
            
    model = tu.build_reg_crps_model(hp, nparam_init, varcons, nobs)
    stop_early_train = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50)
    
    print(f"Training worker {worker_id} starting...")
    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        verbose=0,
        sample_weight=sw_train,
        validation_data=(x_val, y_val, sw_val),
        callbacks=[stop_early_train, uf.MemoryCleanupCallback()]
    )
    model.save(save_path)
    print(f"Training worker {worker_id} finished and saved to {save_path}")
