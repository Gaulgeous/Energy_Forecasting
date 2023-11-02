from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype
import pandas as pd
import numpy as np
import platform

if platform.system() == "Windows":
    from genetic_algorithm.genetic_algorithm_scripts.constants import *
elif platform.system() == "Linux":
    from genetic_algorithm_scripts.constants import *


def moving_average(a, n) :
    test = np.cumsum(a, dtype=float)
    test[n:] = test[n:] - test[:-n]
    return test / n


def collapse_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses 3-dimensional dataframes into 2d ones

    @param data: Dataframe to check and collapse

    @return data: Dataframe after it has been collapsed
    """

    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.to_series().apply(lambda x: "__".join(x))
    return data

    
def create_dataset_2d(np_data: np.array, win_size: int) -> np.array:

    """
    Creates the X dataset. This is the 2d version, used by most sklearn regression models

    @param input: Dataset to transform
    @param win_size: Size of the window

    @return X: Input transformed into the correct 2d shape
    """
    
    # np_data = np.array(input.copy())

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    X = np.array(X)
    X = X.reshape(X.shape[0], -1)

    return X
    

def create_dataset_3d(input: pd.DataFrame, win_size: int) -> np.array:

    """
    Creates the X dataset. This is the 3d version, used by many TF models

    @param input: Dataset to transform
    @param win_size: Size of the window

    @return X: Input transformed into the correct 3d shape
    """
    
    np_data = np.array(input.copy())

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    return np.array(X)


def feature_cleaning(genome: list, 
                     data: pd.DataFrame, 
                     target: str, 
                     epd: int, 
                     trend_type: str, 
                     future: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    """
    Applies all feature cleaning filters. This is the entire data cleaning pipeline

    @param genome: List of components to apply to datacleaning
    @param data: dataset to train against
    @param target: Name of the dataset column to work towards
    @param epd: Entries per day in the dataset
    @param trend_type: Type of trend for seasonal decomposition
    @param future: Number of days into the future to predict for

    @return data: The filtered and cleaned dataset
    @return outputs: A dataframe with the target values and their datetime
    @return y: Target values
    """

    imputer = genome[genome_layout.index("imputer")]
    outlier = genome[genome_layout.index("outlier")]

    date_times = 0

    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]) or column == "Dates" or column == "Date":
            col = column
            date_times += 1

        elif is_integer_dtype(data[column]):
            data[column] = pd.to_numeric(data[column], downcast='integer')

        elif is_float_dtype(data[column]):
            data[column] = pd.to_numeric(data[column], downcast='float')

    if 0 < date_times < 2:
        data.dropna(subset=[col, target], inplace=True)
        data = data.set_index(col)
    else:
        raise ValueError("Ensure that there is one and only one datetime column present within the dataset")    
            
    if outlier:
        outlier_mask = pd.Series(False, index=data.index)
        for column in data.columns:
            if data[column].nunique() > MIN_VALUES_THRESHOLD:
                z_scores = stats.zscore(data[column])
                column_outlier_mask = (z_scores > 2.5) | (z_scores < -2.5)
                outlier_mask |= column_outlier_mask

        data = data[~outlier_mask]

    
    if imputer == "mean":
        for column in data.columns.values:
            if data[column].nunique() > MIN_VALUES_THRESHOLD:
                if is_integer_dtype(data[column]) or is_float_dtype(data[column]):
                    mean_value=data[column].mean()
                    data[column].fillna(value=mean_value, inplace=True)

    elif imputer == "median":
        for column in data.columns.values:
            if data[column].nunique() > MIN_VALUES_THRESHOLD:
                if is_integer_dtype(data[column]) or is_float_dtype(data[column]):
                    median_value=data[column].median()
                    data[column].fillna(value=median_value, inplace=True)

    elif imputer == "mode":
        for column in data.columns.values:
            if data[column].nunique() > MIN_VALUES_THRESHOLD:
                if is_integer_dtype(data[column]) or is_float_dtype(data[column]):
                    mode_value=data[column].mode()
                    data[column].fillna(value=mode_value, inplace=True)


    # Select columns with integer, float, or datetime data types
    selected_columns = data.select_dtypes(include=['int', 'float', 'datetime']).columns

    # Drop columns that don't have the selected data types
    data = data[selected_columns]

    try:
        data['Weekday'] = data.index.dayofweek
    except AttributeError:
        pass

    data['PrevDaySameHour'] = data[target].copy().shift(epd)
    data['PrevWeekSameHour'] = data[target].copy().shift(epd*7)
    data['Prev24HourAveLoad'] = data[target].copy().rolling(window=epd*7, min_periods=epd*7).mean()
    data["Prev_step"] = data[target].copy().shift(1) 

    if trend_type != "None":
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

    data[target] = data[target].shift(-future)
    data.dropna(how='any', axis='rows', inplace=True)

    y = data[target].reset_index(drop=True)

    future_dates = pd.Series(data.index[future:])
    outputs = pd.DataFrame({"Date": future_dates, "Actual": y})

    return data, outputs, y


def create_dataframes(data: pd.DataFrame, 
                     target: str, 
                     epd: int, 
                     trend_type: str, 
                     future: int, 
                     save_path: str):
    """
    Datacleaning that we apply is:

    imputing -> None, mean, median, mode
    outlier_removal -> none, 2 standard deviations
    categorical_encoding -> none, present

    -Reduce size of dtypes to minimum
    -use inplace=True wherever possible

    Then return a big ol array filled with all the dataframes
    """

    date_times = 0
    target_pos = 0

    for pos, column in enumerate(data.columns.values):
        if pd.api.types.is_datetime64_any_dtype(data[column]) or column == "Dates" or column == "Date":
            col = column
            date_times += 1

        if column == target:
            target_pos = pos

    if 0 < date_times < 2:
        data.dropna(subset=[col, target], inplace=True)
        data = data.set_index(col)
    else:
        raise ValueError("Ensure that there is one and only one datetime column present within the dataset")  
    
    try:
        data['Weekday'] = data.index.dayofweek
    except AttributeError:
        pass

    # Select columns with integer, float, or datetime new_data types
    selected_columns = data.select_dtypes(include=['int', 'float', 'datetime']).columns

    # Drop columns that don't have the selected new_data types
    new_data = data[selected_columns]
    X = np.array(new_data)
    del new_data, data

    original_cols = X.shape[1]

    X = np.append(X, np.roll(X[:, target_pos - 1], epd).reshape(-1, 1), axis=1)
    X = np.append(X, np.roll(X[:, target_pos - 1], epd*7).reshape(-1, 1), axis=1)
    X = np.append(X, np.roll(X[:, target_pos - 1], 1).reshape(-1, 1), axis=1)
    X = np.append(X, moving_average(X[:, target_pos-1], epd*7).reshape(-1, 1), axis=1)

    if trend_type != "None":
        dec_daily = seasonal_decompose(X[:, target_pos-1], model=trend_type, period=epd, extrapolate_trend="freq")
        X = np.append(X, np.roll(dec_daily.trend, epd).reshape(-1, 1), axis=1)
        X = np.append(X, np.roll(dec_daily.seasonal, epd).reshape(-1, 1), axis=1)

        dec_weekly = seasonal_decompose(X[:, target_pos-1], model=trend_type, period=epd*7, extrapolate_trend="freq")
        X = np.append(X, np.roll(dec_weekly.trend, epd*7).reshape(-1, 1), axis=1)
        X = np.append(X, np.roll(dec_weekly.seasonal, epd*7).reshape(-1, 1), axis=1)

    X = X[epd*7:, :]

    for imputer_type in imputers:
        for outlier in outliers:

            X_new = X.copy()

            if outlier:
                for pos, col in enumerate(X_new.T):
                    uniques = len(np.unique(col))
                    if uniques > MIN_VALUES_THRESHOLD:
                        mean = np.mean(col)
                        standard_deviation = np.std(col)
                        distance_from_mean = abs(col - mean)
                        max_deviations = 2.5
                        not_outlier = distance_from_mean > max_deviations * standard_deviation
                        col[not_outlier] = np.nan
                        X_new[:, pos] = col

            if imputer_type == "mean":
                imputer = SimpleImputer(strategy="mean")
                for col in range(original_cols):
                    X_new[:,col] = imputer.fit_transform(X_new[:, col].reshape(-1, 1)).flatten()

                del imputer

            elif imputer_type == "median":
                imputer = SimpleImputer(strategy="median")
                for col in range(original_cols):
                    X_new[:,col] = imputer.fit_transform(X_new[:, col].reshape(-1, 1)).flatten()

                del imputer

            elif imputer_type == "mode":
                imputer = SimpleImputer(strategy="most_frequent")
                for col in range(original_cols):
                    X_new[:,col] = imputer.fit_transform(X_new[:, col].reshape(-1, 1)).flatten()

                del imputer

            X_new[:, target_pos-1] = np.roll(X_new[:, target_pos - 1], -future)
            if future > 0:
                X_new = X_new[:-future, :]
            X_new = X_new[~np.isnan(X_new).any(axis=1), :]
            y = X_new[:, target_pos-1].copy()
            X_new = np.delete(X_new, obj=target_pos-1, axis=1)

            save_str = save_path + f"/{outlier}_{imputer_type}"

            np.save(save_str + "_X.npy", X_new)
            np.save(save_str + "_y.npy", y)

            del X_new, y

    del X

