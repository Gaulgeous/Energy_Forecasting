import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose

import time
import os
import pandas as pd
import numpy as np
import keras_tuner as kt

from bayes_opt_scripts.performance_analysis import *


def build_baseline_model():

    model = Sequential()
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), 
                metrics=['mean_squared_error'])

    return model


def build_simple_model():

    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), 
                metrics=['mean_squared_error'])

    return model


# Testing
def train_simple_model(model, X_frame, y_frame, split, data_epochs, batch_size, y_scaler):

    length = X_frame.shape[0]
    X_train = X_frame[:int(length*split),:]
    y_train = y_frame[:int(length*split)]

    X_test = X_frame[int(length*split):,:]
    y_test = y_frame[int(length*split):]

    model.fit(X_train, y_train, verbose=0, epochs=data_epochs,
                    batch_size=batch_size, validation_split=0.2)
    preds = model.predict(X_test, verbose=0)
    # preds = y_scaler.inverse_transform(preds)
    # y_test = y_scaler.inverse_transform(y_test)

    return mse(y_test, preds, squared=True)


def simple_evaluate(future, set_name, X_train, y_train, epochs, batch_size, X_opt_2d, y_opt):

    model_directory = os.getcwd() + "/models"

    time_start = time.time()

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(30)

    model = build_baseline_model()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    model.save(model_directory + "/" + set_name + "_baseline_" + str(future))
    
    print("Finished evaluating baseline for future {0}".format(future))

    time_end = time.time()

    return time_end - time_start


def simple_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    model_directory = r"/home/david/Git/Energy-Forecasting/models"
    csv_directory = r"/home/david/Git/Energy-Forecasting/data"

    model = load_model(model_directory + "/" + set_name + "_baseline_" + str(future))
    predictions = model.predict(X_test)
    # predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    # y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, "Baseline")

    print("Finished running baseline prediction on future window {0}".format(future))

    metric_outputs = get_metrics(predictions, y_test, 0, "Baseline")
    return metric_outputs

