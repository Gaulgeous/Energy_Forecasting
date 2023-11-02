import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical
from skopt import dump, load
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import pandas as pd
import numpy as np

import time

from bayes_opt_scripts.performance_analysis import *
    

def rf_train_model(future, epochs, model_directory, set_name, X_train, y_train, epd, X_opt, y_opt):

    # tss = TimeSeriesSplit(n_splits=1, test_size=epd*90, gap=0)
    estimator = RandomForestRegressor()

    search_space = {
        "max_depth": (10, 1200),
        "min_samples_leaf": (0.001, 0.5, "uniform"),
        "min_samples_split": (0.001, 1.0, "uniform"),
        "n_estimators": (5, 5000),
        "criterion": Categorical(["squared_error"]),
        "max_features": Categorical(['sqrt', 'log2', None]),
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

    model = model.fit(X_opt, y_opt)
    dump(model, model_directory + "/" + set_name + "_rf_" + str(future) + ".pkl")

    model = load(model_directory + "/" + set_name + "_rf_" + str(future) + ".pkl")
    model.fit(X_train, y_train)
    dump(model, model_directory + "/" + set_name + "_rf_" + str(future) + ".pkl")


def rf_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    model_directory = r"/home/david/Git/Energy-Forecasting/models"
    csv_directory = r"/home/david/Git/Energy-Forecasting/data"

    model = load(model_directory + "/" + set_name + "_rf_" + str(future) + ".pkl")
    predictions = model.predict(X_test).reshape(-1, 1)
    # predictions = y_scaler.inverse_transform(predictions)
    # y_test = y_scaler.inverse_transform(y_test)

    make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, "rf")

    print(f"Finished running rf prediction on future window {0}", future)

    metric_outputs = get_metrics(predictions, y_test, 0, "rf")
    return metric_outputs

def rf_evaluate(future, set_name, X_train, y_train, epochs, epd, X_opt, y_opt):

    time_start = time.time()

    model_directory = r"/home/david/Git/Energy-Forecasting/models"

    absl.logging.set_verbosity(absl.logging.ERROR)

    rf_train_model(future, epochs,
            model_directory, set_name, X_train, y_train, epd, X_opt, y_opt)
    
    print("Finished evaluating rf for future {0}".format(future))

    time_end = time.time()

    return time_end - time_start


    
