import pandas as pd
import numpy as np
import os
import re
import datetime

from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class Forecaster:

    def __init__(self):
        path = os.getcwd()
        model_path = path + "/models/matlab_basic_nn_0"
        self.model = load_model(model_path)

        file_path = path + "/csvs/matlab_temp_shortened.xlsx"
        target = "SYSLoad"
        trend_type = "additive"
        future = 0
        epd = 48
        self.data, self.targets = self.feature_adder(file_path, target, trend_type, future, epd)


    def return_data(self):
        return self.data[['DryBulb', 'WetBulb', 'Humidity', 'ElecPrice', 'DewPnt']]
    

    def feature_adder(self, file_path, target, trend_type, future, epd):

        data = pd.read_excel(file_path).set_index("Date")

        data['PrevDaySameHour'] = data[target].copy().shift(epd)
        data['PrevWeekSameHour'] = data[target].copy().shift(epd*7)
        data['Prev24HourAveLoad'] = data[target].copy().rolling(window=epd*7, min_periods=1).mean()
        data['Weekday'] = data.index.dayofweek

        if 'Holiday' in data.columns.values:
            data.loc[(data['Weekday'] < 5) & (data['Holiday'] == 0), 'IsWorkingDay'] = 1
            data.loc[(data['Weekday'] > 4) | (data['Holiday'] == 1), 'IsWorkingDay'] = 0
        else:
            data.loc[data['Weekday'] < 5, 'IsWorkingDay'] = 1
            data.loc[data['Weekday'] > 4, 'IsWorkingDay'] = 0

        dec_daily = seasonal_decompose(data[target], model=trend_type, period=epd)
        dec_weekly = seasonal_decompose(data[target], model=trend_type, period=epd*7)

        data['IntraDayTrend'] = dec_daily.trend
        data['IntraDaySeasonal'] = dec_daily.seasonal
        data['IntraWeekTrend'] = dec_weekly.trend
        data['IntraWeekSeasonal'] = dec_weekly.seasonal

        data = data.dropna(how='any', axis='rows')
        y = data[target].shift(-epd*future).reset_index(drop=True)
        y = y.dropna(how='any', axis='rows')

        # future > 10 needs addressing - it is not yet implemented
        if future > 10:
            data = data[['DryBulb', 'DewPnt', 'Prev5DayHighAve', 'Prev5DayLowAve', 'Hour', 'Weekday', 'IsWorkingDay']]
        else:
            data = data.drop("{0}".format(target), axis=1)

        return data, y
        

    def create_input(self, humidity, holiday, drybulb, dewpoint, wetbulb, window):

        # hour, drybulb, dewpoint, wetbuln, humidity, elecprice, holiday, prevdaysamehour, prevweeksamehour, 
        # prev24hourav, weekday, workingday, intradaytrend, intradayseasonal, intraweektrend, intraweekseasonal

        now = datetime.datetime.now()
        hour = now.hour
        minute = (now.minute // 30) * 0.5
        current_time = hour + minute

        tail = self.data.tail(window - 1)
        tail = np.array(tail)

        last_row = self.data.loc[self.data["Hour"] == current_time]
        last_row = last_row.tail(1)
        input = np.array([current_time, drybulb, dewpoint, wetbulb, humidity, last_row['ElecPrice'].iloc[0], holiday, 
                 last_row["PrevDaySameHour"].iloc[0], last_row["PrevWeekSameHour"].iloc[0], last_row['Prev24HourAveLoad'].iloc[0], 1, 1, 
                 last_row["IntraDayTrend"].iloc[0], last_row["IntraDaySeasonal"].iloc[0], last_row["IntraWeekTrend"].iloc[0], last_row["IntraWeekSeasonal"].iloc[0]])
        
        input = np.append(tail, input).reshape(1, -1)
        
        return input

    def predict(self, input):
        return self.model.predict(input)[0][0]
    
    def return_targets(self):
        return self.targets


class availablemodes:

    def __init__(self):
        self.datasets = {}
        self.find_datasets()


    def get_keys(self):
        return list(self.datasets.keys())


    def get_models(self, dataset, directory):

        file_path = directory + "/" + dataset
        frame = pd.read_csv(file_path)
        cols = frame.columns.values
        cols = [col for col in cols if col != "Date" and col != "Actual"]
        return cols
    

    def get_periods(self, dataset, directory):

        periods = []
        files = os.listdir(directory)
        files = " ".join(files)
        datasets = re.findall(" " + dataset + "_performances_[0-9]+.csv", files)

        for file in datasets:
            if dataset in file:
                periods.append(int(re.findall(r'\d+', file)[0]))

        return periods
                

    def find_datasets(self):

        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"
        files = os.listdir(csv_directory)
        files = " ".join(files)
        datasets = re.findall(" [a-zA-Z]+_performances_[0-9]+.csv", files)
        for dataset in datasets:
            set_name = str.split(dataset, '_')[0][1:]
            periods = self.get_periods(set_name, csv_directory)
            models = self.get_models(dataset[1:], csv_directory)
            self.datasets[set_name] = {"Periods": periods, "Models": models}


    def return_all_datasets(self):
        return self.datasets


    def return_dataset(self, dataset):
        return self.datasets.get(dataset)
    
    
    def return_periods(self, dataset):
        periods_models = self.return_dataset(dataset)
        periods = periods_models.get("Periods")
        return periods
    
    
    def return_models(self, dataset):
        periods_models = self.return_dataset(dataset)
        models = periods_models.get("Models")
        return models


class infokeeper:

    def __init__(self):
        self.model = None
        self.baseline = 0
        self.period = 0

    def set_model(self, model):
        self.model = model

    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_period(self, period):
        self.period = period

    def get_stats(self):
        return self.model, self.baseline, self.period


class datasets:

    def __init__(self):

        self.performance_0 = {}
        self.performance_1 = {}
        self.performance_7 = {}

        self.metrics_0 = {}
        self.metrics_1 = {}
        self.metrics_7 = {}

        self.get_datasets()



    def get_datasets(self):

        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"
        files = os.listdir(csv_directory)
        files = " ".join(files)
        datasets = re.findall(" [a-zA-Z]+_performances_[0-9]+.csv", files)
        for dataset in datasets:
            file_path = csv_directory + "/" + dataset[1:]
            characteristics = str.split(dataset, '_')
            set_name = characteristics[0][1:]
            period = int(str.split(characteristics[2], '.')[0])
            if period == 0:
                self.performance_0[set_name] = pd.read_csv(file_path)
            if period == 1:
                self.performance_1[set_name] = pd.read_csv(file_path)
            if period == 7:
                self.performance_7[set_name] = pd.read_csv(file_path)

        datasets = re.findall(" [a-zA-Z]+_metrics_[0-9]+.csv", files)
        for dataset in datasets:
            if '_cv_' not in dataset: 
                file_path = csv_directory + "/" + dataset[1:]
                characteristics = str.split(dataset, '_')
                set_name = characteristics[0][1:]
                period = int(str.split(characteristics[2], '.')[0])
                if period == 0:
                    self.metrics_0[set_name] = pd.read_csv(file_path)
                if period == 1:
                    self.metrics_1[set_name] = pd.read_csv(file_path)
                if period == 7:
                    self.metrics_7[set_name] = pd.read_csv(file_path)


    def get_performance_data(self, set_name, model, baseline, period):
        
        if period == 0:
            dataframe = self.performance_0.get(set_name)
        elif period == 1:
            dataframe = self.performance_1.get(set_name)
        elif period == 7:
            dataframe = self.performance_7.get(set_name)

        if baseline:
            dataframe = dataframe[["Date", "Actual", model, "Baseline"]]
        else:
            dataframe = dataframe[["Date", "Actual", model]]

        return dataframe
    
    
    def get_metrics_data(self, set_name, model, baseline, period):
        
        if period == 0:
            dataframe = self.metrics_0.get(set_name)
        elif period == 1:
            dataframe = self.metrics_1.get(set_name)
        elif period == 7:
            dataframe = self.metrics_7.get(set_name)

        if baseline:
            baseline_row = dataframe[dataframe['Model'] == "Baseline"]
            model_row = dataframe[dataframe['Model'] == model]
            rows = pd.concat([baseline_row, model_row], ignore_index=True)
        else:
            rows = dataframe[dataframe['Model'] == model]

        return rows
    

    def get_performance_0(self):
        return self.performance_0
    

    def get_performance_1(self):
        return self.performance_1
    

    def get_performance_7(self):
        return self.performance_7
    

    def get_metrics_0(self):
        return self.metrics_0
    

    def get_metrics_1(self):
        return self.metrics_1
    
    
    def get_metrics_7(self):
        return self.metrics_7