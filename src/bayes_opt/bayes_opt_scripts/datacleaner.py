from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import numpy as np
import os
import math

from bayes_opt_scripts.simple_regression import *

def collapse_columns(data):
    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.to_series().apply(lambda x: "__".join(x))
    return data
    
def create_dataset_2d(input, win_size):
    
    np_data = np.array(input.copy())

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    X = np.array(X)
    X = X.reshape(X.shape[0], -1)

    return X
    

def create_dataset_3d(input, win_size):
    
    np_data = np.array(input.copy())

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    return np.array(X)


def load_datasets(csv_directory, set_name, future):

    data_name = csv_directory + "/" + set_name + "_data_" + str(future) + ".csv"
    output_name = csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv"

    data = pd.read_csv(data_name).set_index("Date")
    outputs = pd.read_csv(output_name).set_index("Date")

    return data, outputs


def finalise_data(data, outputs, target, best_results):

    pred_dates = outputs.index

    pca_dim = best_results.get("pca_dimensions")
    y_scaler = None

    outputs = outputs[target]
    
    if best_results.get("scaler") == "minmax":
        X_scaler = MinMaxScaler(feature_range=(0,1))
        y_scaler = MinMaxScaler(feature_range=(0,1))
        data = X_scaler.fit_transform(data)
        # outputs = y_scaler.fit_transform(outputs[[target]])

    elif best_results.get("scaler") == "standard":
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        data = X_scaler.fit_transform(data)
        # outputs = y_scaler.fit_transform(outputs[[target]])

    if pca_dim == "None":
        pca = PCA()
        data = pca.fit_transform(data)
    elif pca_dim == "mle":
        pca = PCA(n_components="mle")
        data = pca.fit_transform(data)
    elif pca_dim != "NO_PCA":
        pca = PCA(n_components=pca_dim)
        data = pca.fit_transform(data)

    X_frame = np.array(data)
    y_data = np.array(outputs)

    return X_frame, y_data, pred_dates, y_scaler


def data_cleaning_pipeline(data_in, outputs_in, cleaning_parameters, target, split, data_epochs, batch_size, csv_directory):

    best_results = {"MSE": [math.inf], "scaler": [None], "pca_dimensions": [None]}

    for scale_type in cleaning_parameters.get('scalers'):
            for pca_dim in cleaning_parameters.get('pca_dimensions'):

                data = data_in.copy()
                outputs = outputs_in.copy()
                y_frame = np.array(outputs[target])

                if scale_type == 'minmax':
                    X_scaler = MinMaxScaler(feature_range=(0,1))
                    y_scaler = MinMaxScaler(feature_range=(0,1))
                    data = X_scaler.fit_transform(data)
                    # outputs = y_scaler.fit_transform(outputs[[target]])

                elif scale_type == 'standard':
                    X_scaler = StandardScaler()
                    y_scaler = StandardScaler()
                    data = X_scaler.fit_transform(data)
                    # outputs = y_scaler.fit_transform(outputs[[target]])

                if pca_dim == None:
                    pca = PCA()
                    data = pca.fit_transform(data)
                elif pca_dim == -math.inf:
                    pca = PCA(n_components="mle")
                    data = pca.fit_transform(data)
                elif pca_dim != math.inf:
                    pca = PCA(n_components=pca_dim)
                    data = pca.fit_transform(data)

                X_frame = np.array(data)
                # y_frame = np.array(outputs)[:,1].astype('float32')
                
                model = build_simple_model()
                mse = train_simple_model(model, X_frame, y_frame, split, data_epochs, batch_size, y_scaler)
                print("Trained scale:{0} dim:{1}".format(scale_type, pca_dim))
                if mse < best_results.get("MSE"):
                    if pca_dim == None:
                        pca_dim = "None"
                    elif pca_dim == math.inf:
                        pca_dim = "NO_PCA"
                    elif pca_dim == -math.inf:
                        pca_dim = "mle"
                    best_results["MSE"][0] = mse
                    best_results["pca_dimensions"][0] = pca_dim
                    best_results["scaler"][0] = scale_type

    results_data = pd.DataFrame.from_dict(best_results)
    results_data.to_csv(csv_directory + "/best_data_parameters.csv", index=False)

    best_results = {"MSE": best_results.get("MSE"), "scaler": best_results.get("scaler")[0], "pca_dimensions": best_results.get("pca_dimensions")[0]}
    return best_results


def feature_adder(csv_directory, file_path, target, trend_type, future, epd,  set_name):

    data = pd.read_csv(file_path).set_index("Dates")
    data = collapse_columns(data)

    data['PrevDaySameHour'] = data[target].copy().shift(epd)
    data['PrevWeekSameHour'] = data[target].copy().shift(epd*7)
    data['Prev24HourAveLoad'] = data[target].copy().rolling(window=epd*7, min_periods=1).mean()

    try:
        data['Weekday'] = data.index.dayofweek
        if 'Holiday' in data.columns.values:
            data.loc[(data['Weekday'] < 5) & (data['Holiday'] == 0), 'IsWorkingDay'] = 1
            data.loc[(data['Weekday'] > 4) | (data['Holiday'] == 1), 'IsWorkingDay'] = 0
        else:
            data.loc[data['Weekday'] < 5, 'IsWorkingDay'] = 1
            data.loc[data['Weekday'] > 4, 'IsWorkingDay'] = 0
    except AttributeError:
        pass

    dec_daily = seasonal_decompose(data[target], model=trend_type, period=epd)
    data['IntraDayTrend'] = dec_daily.trend
    data['IntraDaySeasonal'] = dec_daily.seasonal
    data['IntraDayTrend'] = data['IntraDayTrend'].shift(epd)
    data['IntraDaySeasonal'] = data['IntraDaySeasonal'].shift(epd)

    dec_weekly = seasonal_decompose(data[target], model=trend_type, period=epd*7)
    data['IntraWeekTrend'] = dec_weekly.trend
    data['IntraWeekSeasonal'] = dec_weekly.seasonal
    data['IntraWeekTrend'] = data['IntraWeekTrend'].shift(epd*7)
    data['IntraWeekSeasonal'] = data['IntraWeekSeasonal'].shift(epd*7)

    data[target] = y = data[target].shift(-future)
    data = data.dropna(how='any', axis='rows')
    y = data[target].reset_index(drop=True)

    future_dates = pd.Series(data.index[future:])
    outputs = pd.DataFrame({"Date": future_dates, "{0}".format(target): y})

    
    data = data.drop("{0}".format(target), axis=1)

    data_name = csv_directory + "/" + set_name + "_data_" + str(future) + ".csv"
    output_name = csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv"

    data.to_csv(data_name)
    outputs.to_csv(output_name, index=False)

    print("Saved future window {0} to csvs".format(future))

    return data, outputs
    

if __name__=="__main__":

    try:
        
        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"
        
        data = pd.read_excel(csv_directory + r'\ausdata.xlsx').set_index("Date")
        holidays = pd.read_excel(csv_directory + r'\Holidays2.xls')

        data['Holiday'] = data.index.isin(holidays['Date']).astype(int)

        file_name = csv_directory + "/matlab_temp.xlsx"
        data.to_excel(file_name)
    
    except FileNotFoundError:

        print("Ausdata and Holidays2 xl files are not present in \"csvs\" directory.")
        print("Ensure they are before continuing")
