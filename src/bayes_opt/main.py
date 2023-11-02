import math
from bayes_opt_scripts.datacleaner import *
from bayes_opt_scripts.basic_nn import *
from bayes_opt_scripts.lstm import *
from bayes_opt_scripts.cnn import *
from bayes_opt_scripts.xgboost_model import *
from bayes_opt_scripts.rf_model import *
from bayes_opt_scripts.performance_analysis import normalise_metrics, make_metrics_csvs

from copy import deepcopy


if __name__=="__main__":

    csv_directory = r"/home/david/Git/Energy-Forecasting/data"

    # These are the values that need changing per different dataset
    file_path = csv_directory + "/aemo_nsw_target_only.csv"
    set_name = "AEMO"
    target = "TOTALDEMAND"
    trend_type = "Additive"
    epd = 288
    future = 288

    partition = 5000
    data_epochs = 10

    cleaning_parameters = {
        'pca_dimensions': [None, math.inf, -math.inf],
        'scalers': ['standard', 'minmax']
    }

    window = 10
    split = 0.8
    epochs = 1
    batch_size = 32

    data, outputs = feature_adder(csv_directory, file_path, target, trend_type, future, epd,  set_name)
    best_results = data_cleaning_pipeline(data[:partition], outputs[:partition], cleaning_parameters, target, split, data_epochs, batch_size, csv_directory)

    print("finished cleaning")
    X_frame, y_data, pred_dates, y_scaler = finalise_data(data, outputs, target, best_results)

    length = X_frame.shape[0]

    pred_dates_test = pred_dates[int(length*split) + window:]

    X_2d = create_dataset_2d(X_frame, window)
    X_3d = create_dataset_3d(X_frame, window)

    y_test = y_data[int(length*split) + window:]
    X_test_2d = X_2d[int(length*split):]
    X_test_3d = X_3d[int(length*split):]

    y_train = np.ravel(y_data[window:int(length*split) + window])
    X_train_2d = X_2d[:int(length * split)]
    X_train_3d = X_3d[:int(length * split)]

    np.save("X_train_3d.npy", X_train_3d)

    if X_train_2d.shape[0] > 10000:
        X_opt_2d = X_train_2d[:10000,:]
        X_opt_3d = X_train_3d[:10000,:,:]

        y_opt = y_train[:10000]

    # RF 
    run_time = rf_evaluate(future, set_name, X_train_2d, y_train, epochs, epd, X_opt_2d, y_opt)
    metrics = rf_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)
    
    # XGBoost
    run_time = xgb_evaluate(future, set_name, X_train_2d, y_train, epochs, epd, X_opt_2d, y_opt)
    metrics = xgb_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)
    
    # Base model
    # run_time = simple_evaluate(future, set_name, X_train_2d, y_train, epochs, batch_size, X_opt_2d, y_opt)
    # metrics = simple_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)

    metrics['TIME'] = run_time
    metrics = {"model": metrics}
    # metrics = normalise_metrics(metrics, training)

    make_metrics_csvs(csv_directory, metrics, set_name, future, 1)

    if os.path.exists("X_train_3d.npy"):
        os.remove("X_train_3d.npy")
