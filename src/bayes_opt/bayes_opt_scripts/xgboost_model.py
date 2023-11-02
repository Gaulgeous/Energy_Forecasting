import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt import dump, load
from statsmodels.tsa.seasonal import seasonal_decompose

import time
import os
import pandas as pd
import numpy as np
import xgboost as xgb
    
from bayes_opt_scripts.performance_analysis import *

def xgb_train_model(future, epochs, model_directory, set_name, X_train, y_train, epd, X_opt, y_opt):
    
    split = 0.9

    length = X_train.shape[0]
    X_train_temp = X_opt[:int(length * split), :]
    y_train_temp = y_opt[:int(length * split)]
    X_val = X_opt[int(length * split):, :]
    y_val = y_opt[int(length * split):]

    # tss = TimeSeriesSplit(n_splits=1, test_size=epd*90, gap=0)
    estimator = xgb.XGBRegressor(booster='gbtree',    
            objective='reg:squarederror',
            verbosity=0)

    search_space = {
        "learning_rate": (0.01, 1.0, "log-uniform"),
        "min_child_weight": (0, 10),
        "max_depth": (1, 50),
        "subsample": (0.01, 1.0, "uniform"),
        "colsample_bytree": (0.01, 1.0, "log-uniform"),
        "reg_lambda": (1e-9, 1.0, "log-uniform"),
        "reg_alpha": (1e-9, 1.0, "log-uniform"),
        "gamma": (1e-9, 0.5, "log-uniform"),
        "n_estimators": (5, 5000),
    }

    model = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        n_iter=epochs,
        verbose=0,
        refit=True,
    )

    model = model.fit(X_train_temp, y_train_temp, eval_set=[(X_val, y_val)], verbose=False)
    dump(model, model_directory + "/" + set_name + "_xgb_" + str(future) + ".pkl")

    model = load(model_directory + "/" + set_name + "_xgb_" + str(future) + ".pkl")
    model.fit(X_train, y_train)
    dump(model, model_directory + "/" + set_name + "_xgb_" + str(future) + ".pkl")


def xgb_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    model_directory = r"/home/david/Git/Energy-Forecasting/models"
    csv_directory = r"/home/david/Git/Energy-Forecasting/data"

    model = load(model_directory + "/" + set_name + "_xgb_" + str(future) + ".pkl")
    predictions = model.predict(X_test).reshape(-1, 1)
    # predictions = y_scaler.inverse_transform(predictions)
    # y_test = y_scaler.inverse_transform(y_test)

    make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, "xgb")

    print("Finished running xgb prediction on future window {0}".format(future))

    metric_outputs = get_metrics(predictions, y_test, 0, "xgb")
    return metric_outputs


def xgb_evaluate(future, set_name, X_train, y_train, epochs, epd, X_opt_2d, y_opt):

    time_start = time.time()

    model_directory = r"/home/david/Git/Energy-Forecasting/models"

    absl.logging.set_verbosity(absl.logging.ERROR)

    xgb_train_model(future, epochs,
            model_directory, set_name, X_train, y_train, epd, X_opt_2d, y_opt)
    
    print("Finished evaluating xgb for future {0}".format(future))

    time_end = time.time()
    return time_end - time_start
    
