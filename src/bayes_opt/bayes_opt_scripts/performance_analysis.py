from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
import os
import pandas as pd
import numpy as np


def get_metrics(predictions, actual, cv, model_name):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)
    if cv:
        metrics = {model_name + '_RMSE': RMSE, model_name + '_R2': R2, model_name +'_MSE': MSE, 
                   model_name + '_MAE': MAE, model_name + '_MAPE': MAPE}
    else:
        metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics


def cross_val_metrics(total_metrics, set_name, future, model_name):

    csv_directory = os.getcwd() + "/csvs"
    df = pd.DataFrame(total_metrics)
    df.to_csv(csv_directory + "/" + set_name + "_" + model_name + "_cv_metrics_" + str(future) + ".csv", index=False)


def normalise_metrics(metrics, training):

    rmse = [key["RMSE"] for key in metrics]
    mse = [key["MSE"] for key in metrics]
    mae = [key["MAE"] for key in metrics]
    r2 = [key["R2"] for key in metrics]

    metrics_sets = {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2}

    if training:
        time = [key["TIME"] for key in metrics]
        metrics_sets = {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2, "TIME": time}

    for name, set in metrics_sets.items():
        top = max(set)
        counter = 0

        while top > 10:
            top /= 10
            set = [entry / 10 for entry in set]
            counter += 1
        
        i = 0

        for key in metrics:
            key[name] = set[i]
            i += 1

    return metrics

    
# This is going to have to be rejigged
def make_metrics_csvs(csv_directory, metrics, set_name, future, training):

    for model_name, metric_outputs in metrics.items():
 
        if not os.path.exists(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv"):
            metrics = pd.DataFrame({"Model": [], "Metric": [], "Value": []})
            metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "RMSE", "Value": metric_outputs.get("RMSE")}
            metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "MAE", "Value": metric_outputs.get("MAE")}
            metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "MAPE", "Value": metric_outputs.get("MAPE")}
            metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "R2", "Value": metric_outputs.get("R2")}
            if training:
                metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "TIME", "Value": metric_outputs.get("TIME")}

            metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
        else:

            metrics = pd.read_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv")

            if model_name in metrics['Model'].values:
                metrics.loc[(metrics['Model'] == model_name) & (metrics["Metric"] == "RMSE"), 'Value'] = metric_outputs.get("RMSE")
                metrics.loc[(metrics['Model'] == model_name) & (metrics["Metric"] == "MAE"), 'Value'] = metric_outputs.get("MAE")
                metrics.loc[(metrics['Model'] == model_name) & (metrics["Metric"] == "MAPE"), 'Value'] = metric_outputs.get("MAPE")
                metrics.loc[(metrics['Model'] == model_name) & (metrics["Metric"] == "R2"), 'Value'] = metric_outputs.get("R2")
                if training:
                    metrics.loc[(metrics['Model'] == model_name) & (metrics["Metric"] == "TIME"), 'Value'] = metric_outputs.get("TIME")
            else:
                metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "RMSE", "Value": metric_outputs.get("RMSE")}
                metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "MAE", "Value": metric_outputs.get("MAE")}
                metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "MAPE", "Value": metric_outputs.get("MAPE")}
                metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "R2", "Value": metric_outputs.get("R2")}
                if training:
                    metrics.loc[len(metrics)] = {"Model": model_name, "Metric": "TIME", "Value": metric_outputs.get("TIME")}
            metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)


def make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, model_name):

    if not os.path.exists(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test.flatten(), model_name: predictions.flatten()})
        performances = performances.iloc[-1000:,:]
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv")
        performances[model_name] = predictions[-1000:]
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)

   