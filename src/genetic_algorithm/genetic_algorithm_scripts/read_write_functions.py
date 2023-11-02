import platform
import regex as re

if platform.system() == "Windows":
    from genetic_algorithm.genetic_algorithm_scripts.constants import *
    from genetic_algorithm.genetic_algorithm_scripts.generation_and_evolution import *
elif platform.system() == "Linux":
    from genetic_algorithm_scripts.constants import *
    from genetic_algorithm_scripts.generation_and_evolution import *

def feature_printing(writer: object, feature_dict: dict):

    """
    Writes the feature attributes for any given object into a new file

    @param writer: File pointer used for writing into
    @param feature_dict: Dictionary containing all the features that must be written
    """

    values = len(feature_dict)
    counter = 0
    for key, value in feature_dict.items():
        if key != "type":
            if counter > 0 and counter < values - 1:
                if type(value) == str:
                    writer.write(f"{key}=\"{value}\", ")
                else:
                    writer.write(f"{key}={value}, ")
            elif counter == values - 1:
                if type(value) == str:
                    writer.write(f"{key}=\"{value}\")\n")
                else:
                    writer.write(f"{key}={value})\n")
        counter += 1
        

def save_population(population: Population, 
                    generation: int, 
                    file_path: str, 
                    mode: str, 
                    model_directory: str):
    
    """
    Save the population at given generation intervals
    For use with warm starts

    @param population: Population to save
    @param generation: Number of generation undertaken
    @param file_path: Location of the dataset being trained on
    @param mode: Search mode being engaged by the genetic algorithm
    @param model_dictory: Location to save the model file into
    """

    set_name = os.path.basename(file_path).split('/')[-1].split(".")[0]

    file_name = "/{0}_{1}_{2}.txt".format(set_name, mode, generation)
    dump_path = model_directory + file_name
    out_file = open(dump_path, "w")

    for genome in population:
        
        genome = str(genome["loadout"])
        out_file.write(genome)
        out_file.write("\n")

    out_file.close()


def file_finder(file_path: str, model_directory: str, mode: str) -> object:

    """
    Finds a previous save of for the given dataset that a warm start can run off

    @param file_path: The location of the dataset being trained on
    @param model_directory: Directory to search through to find previous checkpoints
    @param mode: Mode of search being used by the regressor

    @return reader: Pointer to the checkpoint that has the highest number of generations
    """

    set_name = os.path.basename(file_path).split('/')[-1].split(".")[0]
    dir_name = model_directory

    relevant_models = []

    for file in os.listdir(dir_name):
        name = file.split(".")[0]
        search_flag = re.findall("{0}_{1}_[0-9]+".format(set_name, mode), name)
        if len(search_flag) > 0:
            relevant_models.append(search_flag[0])

    if len(relevant_models) > 0:

        highest_gen = 0

        for file in relevant_models:
            generations = re.findall("[0-9]+", file)
            for generation in generations:
                generation = int(generation)
                if generation > highest_gen:
                    highest_gen = generation

        file_name = "/{0}_{1}_{2}.txt".format(set_name, mode, highest_gen)
        path = dir_name + file_name
        reader = open(path, "r")

        return reader, True
    
    else:
        logging.warning("No previous models for this dataset and search mode exist. \
                        Creating new population")
        
        return None, False
    

def extract_model(model_str) -> dict:

    """
    Extracts the model attributes from the saved file

    @param model_str: String format of the saved model attributes

    @return model_dict: Model attributes converted into a dictionary format
    """
    
    model_str = re.sub("\'", "\"", model_str)
    model_str = re.sub("None", "\"None\"", model_str)
    model_str = re.sub("False", "\"False\"", model_str)
    model_str = re.sub("True", "\"True\"", model_str)

    model_name = re.findall("\"model_name\": \"[a-zA-Z_]+\"", model_str)[0]
    model_name = model_name.split(" ")[1][1:-1]

    params = re.findall(r'"([^"]+)": ({((?:[^{}]+|(?2))*)})', model_str)

    model_dict = {"model_name": model_name}

    for value in params:
        checkpoint = 0
        model_dict[value[0]] = json.loads(value[1])

        for key, _ in model_dict[value[0]].items():
            if model_dict[value[0]][key] == "False":
                model_dict[value[0]][key] == False
            elif model_dict[value[0]][key] == "True":
                model_dict[value[0]][key] == True
            elif model_dict[value[0]][key] == "None":
                model_dict[value[0]][key] == None
    
    return model_dict


def load_population(file_path: str, model_directory: str, mode: str, models: dict, population: dict, population_size: int) -> Population:
        
        """
        Checks to see if a checkpoint exists, and loads the population from the given checkpoint

        @param file_path: Path to the dataset to train on
        @param model_directory: Directory to search through to find previous checkpoints
        @param mode: Mode of search being used by the regressor

        @return population: Returns the discovered save population, or generates a new one if none exists
        """
        
        reader, flag = file_finder(file_path, model_directory, mode)

        if flag:
            population = []
            for model_str in reader.readlines():
                model_dict = extract_model(model_str)
                imputer = model_dict["imputer"]["type"]
                outlier = model_dict["outlier"]["type"]
                window = model_dict["window"]
                scaler = generate_scaler(model_dict["scaler"])
                dimension = generate_dimension(model_dict["dimension"])
                model = generate_component(model_dict["model"])

                genome = [window, imputer, outlier, scaler, dimension, model]
                population.append({"fitness": -math.inf, "genome": genome, "loadout": model_dict})

            reader.close()

            return population

        else:
            return generate_population(models, population, population_size)
        


def save_final_model(loadout: dict, 
                     file_path: str, 
                     mode: str, 
                     model_directory: str, 
                     target: str, 
                     epd: int, 
                     trend_type: str, 
                     future: int):
    
    """
    Creates a python file with that automatically creates the most successful genome, and trains
    it on the dataset provided

    @param loadout: Loadout attributes of the genome
    @param file_path: Location of the dataset to train on
    @param mode: Search mode being used by the regressor
    @param model_directory: Location of model directory to save the .py file in
    @param target: Name of the dataset column to work towards
    @param epd: Entries per day in the dataset
    @param trend_type: Type of trend for seasonal decomposition
    @param future: Number of days into the future to predict for
    """

    set_name = os.path.basename(file_path).split('/')[-1].split(".")[0]
    csv_path = model_directory + "/{0}_{1}.py".format(set_name, mode)
    with open(csv_path, 'w', encoding='utf-8') as file:
        file.write("import pandas as pd\n")
        file.write("import numpy as np\n")
        file.write("import time\n")
        file.write("import math\n")
        file.write("import os\n")
        file.write("import logging\n")
        file.write("import json\n")

        file.write("from typing import List, Optional, Callable, Tuple\n")
        file.write("from random import randint, randrange, random, choices\n")
        file.write("from sklearn.linear_model import RidgeCV, LassoLarsCV, ElasticNetCV, Ridge, LassoLars\n")
        file.write("from sklearn.impute import SimpleImputer\n")
        file.write("from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler, LabelEncoder\n")
        file.write("from sklearn.cluster import FeatureAgglomeration\n")
        file.write("from sklearn.neighbors import KNeighborsRegressor\n")
        file.write("from sklearn.tree import DecisionTreeRegressor\n")
        file.write("from sklearn.svm import LinearSVR\n")
        file.write("from sklearn.ensemble import RandomForestRegressor\n")
        file.write("from xgboost import XGBRegressor\n")
        file.write("from sklearn.decomposition import PCA\n")
        file.write("from sklearn.metrics import mean_squared_error as mse\n")
        file.write("from statsmodels.tsa.seasonal import seasonal_decompose\n")
        file.write("from scipy import stats\n\n\n")

        file.write("genome_layout = [\"window\", \"imputer\", \"outlier\", \"scaler\", \"dimension\", \"model\"]\n")
        file.write("TEST_SPLIT = 0.8\n")
        file.write("COLUMN_REMOVE = 0.3\n")
        file.write("MIN_VALUES_THRESHOLD = 100\n\n\n")

        file.write("def collapse_columns(data):\n\n")
        file.write("    data = data.copy()\n")
        file.write("    if isinstance(data.columns, pd.MultiIndex):\n")
        file.write("        data.columns = data.columns.to_series().apply(lambda x: \"__\".join(x))\n")
        file.write("    return data\n\n\n")

        file.write("def create_dataset_2d(input, win_size):\n\n")
    
        file.write("    np_data = np.array(input.copy())\n")

        file.write("    X = []\n\n")

        file.write("    for i in range(len(np_data)-win_size):\n")
        file.write("        row = [r for r in np_data[i:i+win_size]]\n")
        file.write("        X.append(row)\n\n")

        file.write("    X = np.array(X)\n")
        file.write("    X = X.reshape(X.shape[0], -1)\n\n")

        file.write("    return X\n\n\n")


        file.write("def feature_cleaning(genome, data, target, epd, trend_type, future):\n\n")

        file.write("    if epd <= 0:\n")
        file.write("        raise ValueError(\"Please input a positive integer for the entries per day\")\n")

        file.write("    data = collapse_columns(data)\n\n")

        file.write("    col = None\n")
        file.write("    date_times = 0\n\n")

        file.write("    duplicate_columns = data.columns[data.columns.duplicated()]\n")

        file.write("    data.drop(columns=duplicate_columns, inplace=True)\n")

        file.write("    missing_percentage = (data.isna().sum() / len(data)) * 100\n")

        file.write("    columns_with_missing = missing_percentage[missing_percentage >= COLUMN_REMOVE].index\n")

        file.write("    data.drop(columns=columns_with_missing, inplace=True)\n\n")

        file.write("    for column in data.columns:\n")
        file.write("        if pd.api.types.is_datetime64_any_dtype(data[column]):\n")
        file.write("            col = column\n")
        file.write("            date_times += 1\n\n")

        file.write("    if 0 < date_times < 2:\n")
        file.write("        data.dropna(subset=[target, col], inplace=True)\n")
        file.write("        data = data.set_index(col)\n\n")
        file.write("    else:\n")
        file.write("        raise ValueError(\"Ensure that there is one and only one datetime column present within the dataset\")\n\n")
            
        file.write("    imputer_val = genome[genome_layout.index(\"imputer\")]\n")
        file.write("    outlier = genome[genome_layout.index(\"outlier\")]\n\n")

        file.write("    if imputer_val > 0:\n")
        file.write("        if imputer_val == 1:\n")
        file.write("            imputer = SimpleImputer(strategy=\"mean\")\n")
        file.write("        elif imputer_val == 2:\n")
        file.write("            imputer = SimpleImputer(strategy=\"median\")\n")
        file.write("        if imputer_val == 3:\n")
        file.write("            imputer = SimpleImputer(strategy=\"most_frequent\")\n")
        file.write("        ind = data.index\n")
        file.write("        data = pd.DataFrame(imputer.fit_transform(data), columns = data.columns).set_index(ind)\n\n")

        file.write("    if outlier:\n")
        file.write("        outlier_mask = pd.Series(False, index=data.index)\n")
        file.write("        for column in data.columns:\n")
        file.write("            if data[column].nunique() > MIN_VALUES_THRESHOLD:\n")
        file.write("                z_scores = stats.zscore(data[column])\n")
        file.write("                column_outlier_mask = (z_scores > 2.5) | (z_scores < -2.5)\n")
        file.write("                outlier_mask |= column_outlier_mask\n\n")

        file.write("        data = data[~outlier_mask]\n\n")

        file.write("    data['Weekday'] = data.index.dayofweek\n")

        file.write("    data['PrevDaySameHour'] = data[target].copy().shift(epd)\n")
        file.write("    data['PrevWeekSameHour'] = data[target].copy().shift(epd*7)\n")
        file.write("    data['Prev24HourAveLoad'] = data[target].copy().rolling(window=epd*7, min_periods=epd*7).mean()\n\n")
            
        file.write("    if 'Holiday' in data.columns.values:\n")
        file.write("        data.loc[(data['Weekday'] < 5) & (data['Holiday'] == 0), 'IsWorkingDay'] = 1\n")
        file.write("        data.loc[(data['Weekday'] > 4) | (data['Holiday'] == 1), 'IsWorkingDay'] = 0\n")
        file.write("    else:\n")
        file.write("        data.loc[data['Weekday'] < 5, 'IsWorkingDay'] = 1\n")
        file.write("        data.loc[data['Weekday'] > 4, 'IsWorkingDay'] = 0\n\n")

        file.write("    if trend_type != \"None\":\n")
        file.write("        dec_daily = seasonal_decompose(data[target], model=trend_type, period=epd)\n")
        file.write("        data['IntraDayTrend'] = dec_daily.trend\n")
        file.write("        data['IntraDaySeasonal'] = dec_daily.seasonal\n")
        file.write("        data['IntraDayTrend'] = data['IntraDayTrend'].shift(epd)\n")
        file.write("        data['IntraDaySeasonal'] = data['IntraDaySeasonal'].shift(epd)\n\n")

        file.write("        dec_weekly = seasonal_decompose(data[target], model=trend_type, period=epd*7)\n")
        file.write("        data['IntraWeekTrend'] = dec_weekly.trend\n")
        file.write("        data['IntraWeekSeasonal'] = dec_weekly.seasonal\n")
        file.write("        data['IntraWeekTrend'] = data['IntraWeekTrend'].shift(epd*7)\n")
        file.write("        data['IntraWeekSeasonal'] = data['IntraWeekSeasonal'].shift(epd*7)\n\n")

        file.write("    data[target] = data[target].shift(-epd*future)\n")
        file.write("    data = data.dropna(how='any', axis='rows')\n")
        file.write("    y = data[target].reset_index(drop=True)\n\n")

        file.write("    future_dates = pd.Series(data.index[future*epd:])\n")
        file.write("    outputs = pd.DataFrame({\"Date\": future_dates, \"{0}\".format(target): y})\n")

        file.write("    data = data.drop(\"{0}\".format(target), axis=1)\n\n")

        file.write("    return data, outputs, y\n\n\n")
    

        file.write("if __name__==\"__main__\":\n\n")

        _, filetype = os.path.splitext(file_path)

        if filetype == ".csv":
            file.write("    data = pd.read_csv(r\"{0}\")\n".format(file_path))
        elif filetype == ".xls" or filetype == ".xlsx":
            file.write("    data = pd.read_excel(r\"{0}\")\n".format(file_path))
        file.write("    target = \"{0}\"\n".format(target))
        file.write("    epd = {0}\n".format(epd))
        file.write("    trend_type = \"{0}\"\n".format(trend_type))
        file.write("    future = {0}\n".format(future))

        window = loadout["window"]["window"]
        imputer = loadout["imputer"]["type"]
        outlier = loadout["outlier"]["type"]
        file.write("    window = {0}\n\n".format(window))
        file.write("    imputer = {0}\n".format(imputer))
        file.write("    outlier = 0\n".format(outlier))
        file.write("    scaler = None\n")
        file.write("    dimension = None\n")
        file.write("    model = None\n\n")

        for key, _ in loadout.items():
            try:
                if key == "scaler":
                    if loadout[key]['type'] == "normalizer":
                        file.write(f"    scaler = Normalizer(")
                        feature_printing(file, loadout[key])
                    elif loadout[key]['type'] == "robust":
                        file.write("    scaler = RobustScaler()\n")
                    elif loadout[key]['type'] == "standard":
                        file.write("    scaler = StandardScaler()\n")
                elif key == "dimension":
                        if loadout[key]['type'] == "FeatureAgglomeration":
                            file.write(f"    dimension = FeatureAgglomeration(")
                            feature_printing(file, loadout[key])
                        elif loadout[key]["type"] == "PCA":
                            file.write(f"    dimension = PCA(")
                            feature_printing(file, loadout[key])
                elif key == "model":
                    if loadout[key]["type"] == "xgboost":
                        file.write(f"    model = XGBoostRegressor(")
                    elif loadout[key]["type"] == "random_forest":
                        file.write(f"    model = RandomForestRegressor(")
                    elif loadout[key]["type"] == "decision_tree":
                        file.write(f"    model = DecisionTreeRegressor(")
                    elif loadout[key]["type"] == "k_neighbours":
                        file.write(f"    model = KNeighborsRegressor(")
                    elif loadout[key]["type"] == "linear_svr":
                        file.write(f"    model = LinearSVR(")
                    elif loadout[key]["type"] == "elastic":
                        file.write(f"    model = ElasticNetCV(")
                    elif loadout[key]["type"] == "lasso":
                        file.write(f"    model = LassoLarsCV()")
                    elif loadout[key]["type"] == "ridge":
                        file.write(f"    model = RidgeCV()")
                    feature_printing(file, loadout[key])
                    
            except TypeError:
                continue

        file.write("\n")

        file.write("    genome = [window, imputer, outlier, scaler, dimension, model]\n")

        file.write("    data, _, y = feature_cleaning(genome, data, target, epd, trend_type, future)\n")
        file.write("    X = create_dataset_2d(data, window)\n")

        file.write("    X_train = X[:int(X.shape[0]*TEST_SPLIT)]\n")
        file.write("    y_train = target[:int(X.shape[0]*TEST_SPLIT)]\n")

        file.write("    X_test = X[int(X.shape[0]*TEST_SPLIT):]\n")
        file.write("    y_test = target[int(X.shape[0]*TEST_SPLIT):]\n\n")

        file.write("    model.fit(X_train, y_train)\n")
        file.write("    predictions = model.predict(X_test)\n")

        file.close()
