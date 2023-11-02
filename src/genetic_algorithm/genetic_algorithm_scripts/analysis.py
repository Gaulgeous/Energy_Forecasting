from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
import os
import pandas as pd
import numpy as np
import platform

if platform.system() == "Windows":
    from genetic_algorithm.genetic_algorithm_scripts.datacleaner import create_dataset_2d, feature_cleaning
    from genetic_algorithm.genetic_algorithm_scripts.constants import *
if platform.system() == "Linux":
    from genetic_algorithm_scripts.datacleaner import create_dataset_2d, feature_cleaning
    from genetic_algorithm_scripts.constants import *


def get_metrics(predictions: np.array, actual: np.array) -> dict:

    """
    Returns the performance metrics (R2, MSE etc.) of the model's predictions

    @param predictions: model's predictions
    @param actual: The actual values it was trying to predict

    @return metrics: dictionary of all the performance metrics
    """

    MSE = mse(actual, predictions, squared=False)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)
    
    metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}

    return metrics


def final_model_prediction(model: dict, 
                           data: pd.DataFrame, 
                           target: str, 
                           epd: int, 
                           trend_type: str, 
                           future: int) -> Tuple[pd.DataFrame, dict]:
    
    """
    Creates the predictions for the finalised model

    @param model: Model dictionary including the name, genome and loadout
    @param data: dataset to train against
    @param target: Name of the dataset column to work towards
    @param epd: Entries per day in the dataset
    @param trend_type: Type of trend for seasonal decomposition
    @param future: Number of days into the future to predict for

    @return outputs: Dataframe of the outputted predictions against the actual values
    @return metrics: Dictionary of the performance metrics of the model (R2, etc.)
    """
    
    genome = model["genome"]
    loadout = model["loadout"]

    scaler = loadout["scaler"]["type"]
    dimension = loadout["dimension"]["type"]

    data, outputs, target = feature_cleaning(model["genome"], data, target, epd, trend_type, future)

    X = np.array(data)
    del data

    if scaler is not None and scaler != 0:
        index = genome_layout.index("scaler")
        X = genome[index].fit_transform(X)

    if dimension is not None:
        index = genome_layout.index("dimension")
        X = genome[index].fit_transform(X)

    try:
        index = genome_layout.index("window")
        window = genome[index].get("window")
    except AttributeError:
        window = genome[index]

    X = create_dataset_2d(X, window)

    X_train = X[:int(X.shape[0]*TEST_SPLIT)]
    y_train = target[window:int(X.shape[0]*TEST_SPLIT)+window]

    X_test = X[int(X.shape[0]*TEST_SPLIT):]
    y_test = target[int(X.shape[0]*TEST_SPLIT)+window:]

    index = genome_layout.index("model")
    model = genome[index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    outputs = outputs.iloc[target.shape[0]-predictions.shape[0]:]
    outputs["prediction"] = predictions
    outputs["Actual"] = y_test

    metrics = get_metrics(predictions, y_test)

    return outputs, metrics
    
   
def make_cv_sets(X: np.array, 
                 y: np.array, 
                 window: int, 
                 cv: int) -> Tuple[list, list, list, list]:
    
    """
    Creates the cross-validation sets for training on

    @param X: X values
    @param y: target values
    @param window: Size of the window
    @param cv: Amount of cross-validation folds

    @return X_train: List of k-fold X_trains
    @return y_train: List of k-fold y_trains
    @return X_test: List of k-fold X_tests
    @return y_test: List of k-fold y_tests
    """

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    if cv > 1:

        partitions = [float(i/(cv+1)) for i in range(1, cv+1)]

        for partition in partitions:

            X_train.append(X[:int(partition*X.shape[0])])
            X_test.append(X[int(partition*X.shape[0]):])
            y_train.append(y[window:int(partition*X.shape[0])+window])
            y_test.append(y[int(partition*X.shape[0])+window:])
    
    else:

        X_train.append(X[:int(0.75*X.shape[0])])
        X_test.append(X[int(0.75*X.shape[0]):])
        y_train.append(y[window:int(0.75*X.shape[0])+window])
        y_test.append(y[int(0.75*X.shape[0])+window:])

    return X_train, X_test, y_train, y_test

    
# @delayed
# @wrap_non_picklable_objects
def calc_fitness(genome_dict: dict, 
                 tuning_dataframes: list, 
                 target: pd.Series, 
                 cv: int, 
                 epd: int, 
                 trend_type: str, 
                 future: int, 
                 frame_path: str) -> int:
    
    """
    Evaluates each particular genome to see how fit it was for time series prediction

    @param genome_dict: Attributes to construct the model from
    @param data: dataset to train on
    @param cv: Number of cross-validation folds
    @param target: Name of the dataset column to work towards
    @param epd: Entries per day in the dataset
    @param trend_type: Type of trend for seasonal decomposition
    @param future: Number of days into the future to predict for

    @return loss: fitness of that particular genome
    """

    losses = []
    
    genome = genome_dict["genome"]
    loadout = genome_dict["loadout"]

    scaler = loadout["scaler"]["type"]
    dimension = loadout["dimension"]["type"]

    save_path = frame_path + "/{0}_{1}".format(loadout["outlier"]["type"], imputers[loadout["imputer"]["type"]])

    data = np.load(save_path + "_X.npy")
    y = np.load(save_path + "_y.npy")

    try:

        if scaler is not None and scaler != 0:
            index = genome_layout.index("scaler")
            data = genome[index].fit_transform(data)

        if dimension is not None:
            index = genome_layout.index("dimension")
            data = genome[index].fit_transform(data)

        try:
            index = genome_layout.index("window")
            window = genome[index].get("window")
        except AttributeError:
            window = genome[index]

        X = create_dataset_2d(data, window)

        X_train, X_test, y_train, y_test = make_cv_sets(X, y, window, cv)
        index = genome_layout.index("model")
        model = genome[index]

        for i in range(len(X_train)):

            model.fit(X_train[i], y_train[i])
            prediction = model.predict(X_test[i])
            losses.append(r2_score(y_test[i], prediction))

        loss = sum(losses) / len(losses)

        del data, X_train, X_test, y_train, y_test, X, y

        gc.collect()

        return loss
    
    except InvalidParameterError:
        return -math.inf
    
    except AttributeError:
        return -math.inf


def get_dict_key(dictionary: dict, n: int = 0) -> str:
    """
    Returns the name of the dictionary key at the nth index

    @param dictionary: Dictionary to sort through
    @param n: Index position of the key to find

    @return key: The name of the key at that position
    """

    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


def population_fitness(population: Population) -> int:
    """
    Returns the average fitness of the entire population

    @param population: The population to calculate over

    @return loss: Average loss over the population
    """

    counting = [genome["fitness"] for genome in population if genome["fitness"] != -math.inf]

    return sum(counting) / len(counting)