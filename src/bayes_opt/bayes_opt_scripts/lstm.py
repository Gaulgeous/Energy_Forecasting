import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose

import time
import os
import pandas as pd
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt

from bayes_opt_scripts.performance_analysis import *
    


def lstm_kt_model(hp):

    X = np.load("X_train_3d.npy")

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_reg = hp.Float("reg", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_dropout = hp.Float("dropout", min_value=1e-3, max_value=0.5, sampling="linear")
    hp_neuron_pct = hp.Float('NeuronPct', min_value=1e-3, max_value=1.0, sampling='linear')
    hp_neuron_shrink = hp.Float('NeuronShrink', min_value=1e-3, max_value=1.0, sampling='linear')
    
    hp_l_layer_1 = hp.Int('l_layer_1', min_value=1, max_value=100, step=10)
    hp_max_neurons = hp.Int('neurons', min_value=10, max_value=5000, step=10)

    neuron_count = int(hp_neuron_pct * hp_max_neurons)
    layers = 0

    model = Sequential()
    model.add(InputLayer((X.shape[1], X.shape[2])))
    model.add(LSTM(hp_l_layer_1, return_sequences=True, activity_regularizer=regularizers.l1(hp_reg)))
    model.add(Dropout(hp_dropout))
    model.add(Flatten())

    while neuron_count > 20 and layers < 20:

        model.add(Dense(units=neuron_count, activation=hp_activation))
        model.add(Dropout(hp_dropout))
        layers += 1
        neuron_count = int(neuron_count * hp_neuron_shrink)

    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=hp_learning_rate), 
                metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

    return model


def lstm_save_plots(history, graphs_directory, set_name, future):

    graph_names = {"Loss": "loss", "MAE": "mean_absolute_error", 
                   "MSE": "mean_squared_error", "MAPE": "mean_absolute_percentage_error"}
    
    for name, value in graph_names.items():
        graph_loc = graphs_directory + "/" + set_name + "_lstm_" + str(future) + "_" + name + ".png"
        if os.path.exists(graph_loc):
            os.remove(graph_loc)

        val_name = "val_" + value
        plt.plot(history.history[value])
        plt.plot(history.history[val_name])
        plt.title('LSTM {0} for {1} {2}'.format(name, set_name, future))
        plt.ylabel(name)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(graphs_directory + "/" + set_name + "_lstm_" + str(future) + "_" + name + ".png")
    

def lstm_train_model(future, batch_size, epochs,
                model_directory, set_name, X_train, y_train, y_scaler, epd):

    tuner = kt.Hyperband(lstm_kt_model, objective='mean_absolute_percentage_error', max_epochs=epochs, factor=3, 
                        directory=model_directory + "/" + set_name + "_kt_dir", project_name='kt_model_' + str(future), 
                        overwrite=True)

    monitor = EarlyStopping(monitor='mean_absolute_percentage_error', min_delta=1, patience=5, verbose=0, mode='auto', 
                    restore_best_weights=True)

    tuner.search(X_train, y_train, verbose=0, epochs=epochs, validation_split=0.2, batch_size=batch_size,
                callbacks=[monitor])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # Split on a 3 monthly basis
    tss = TimeSeriesSplit(n_splits=10, test_size=epd*90, gap=0)
    fold = 0
    total_metrics = {}

    for train_idx, val_idx in tss.split(X_train, y_train):

        fold_name = "Fold_" + str(fold)
        X_t = X_train[train_idx]
        X_v = X_train[val_idx]
        y_t = y_train[train_idx]
        y_v = y_train[val_idx]
        
        if fold == 9:
            history = model.fit(X_t, y_t, verbose=0, epochs=epochs, callbacks=[monitor],
                    batch_size=batch_size, validation_data=(X_v, y_v))
            graphs_directory = os.getcwd() + "/graphs"
            lstm_save_plots(history, graphs_directory, set_name, future)
            model.save(model_directory + "/" + set_name + "_lstm_" + str(future))
        
        model.fit(X_t, y_t, verbose=0, epochs=epochs, callbacks=[monitor],
                    batch_size=batch_size)
        preds = model.predict(X_v, verbose=0)
        preds = y_scaler.inverse_transform(preds)
        metrics = get_metrics(preds, y_v, 1, "lstm")
        total_metrics[fold_name] = metrics

        fold += 1

    cross_val_metrics(total_metrics, set_name, future, "lstm")



def lstm_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = load_model(model_directory + "/" + set_name + "_lstm_" + str(future))
    predictions = model.predict(X_test)
    predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, "lstm")

    print("Finished running lstm prediction on future window {0}".format(future))

    metric_outputs = get_metrics(predictions, y_test, 0, "lstm")
    return metric_outputs


def lstm_evaluate(future, set_name, X_train, y_train, epochs, batch_size, y_scaler, epd):

    time_start = time.time()

    model_directory = r"/home/david/Git/Energy-Forecasting/models"

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(30)

    lstm_train_model(future, batch_size, epochs,
            model_directory, set_name, X_train, y_train, y_scaler, epd)
    
    print("Finished evaluating lstm for future {0}".format(future))

    time_end = time.time()

    return time_end - time_start