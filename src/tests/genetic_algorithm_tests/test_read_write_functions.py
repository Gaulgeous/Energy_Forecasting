import pytest 

from genetic_algorithm.genetic_algorithm_scripts.read_write_functions import *

class TestClass:

    
    def test_save_final_model(self):
    
        file_path = r"C:\Users\gauld\OneDrive\Desktop\Git\Energy-Forecasting\data\matlab_temp.xlsx"
        model_directory = r"C:\Users\gauld\OneDrive\Desktop\Git\Energy-Forecasting\models"
        target = "SYSLoad"
        mode = "fast"
        epd = 48
        trend_type = "additive"
        future = 0

        iterations = 1

        models = {"decision_tree": total_models["decision_tree"],
                    "k_neighbours": total_models["k_neighbours"],
                    "random_forest": total_models["random_forest"],
                    "xgboost": total_models["xgboost"]}

        for i in range(iterations):
            model_tic = randint(0, len(models) - 1)

            _, _, genome_loadout = generate_genome(model_tic, models)
            
            save_final_model(loadout=genome_loadout, file_path=file_path, mode=mode, 
                             model_directory=model_directory, target=target, epd=epd, 
                             trend_type=trend_type, future=future)
            
            # exec(open(r"C:\Users\gauld\OneDrive\Desktop\Git\Energy-Forecasting\models\matlab_temp_fast.py").read())

            print(f"passed iteration {i}")

        mode = "normal"
        models = {"ridge": total_models["ridge"],
                    "lasso": total_models["lasso"],
                    "elastic": total_models["elastic"],
                    "linear_svr": total_models["linear_svr"]}
        

        for i in range(iterations):
            model_tic = randint(0, len(models) - 1)

            _, _, genome_loadout = generate_genome(model_tic, models)
            
            save_final_model(loadout=genome_loadout, file_path=file_path, mode=mode, 
                             model_directory=model_directory, target=target, epd=epd, 
                             trend_type=trend_type, future=future)
            
            # exec(open(r"C:\Users\gauld\OneDrive\Desktop\Git\Energy-Forecasting\models\matlab_temp_fast.py").read())

            print(f"passed iteration {i}")
        

    
    def test_save_load(self):

        file_path = r"C:\Users\gauld\OneDrive\Desktop\Git\Energy-Forecasting\data\matlab_temp.xlsx"
        model_directory = r"C:\Users\gauld\OneDrive\Desktop\Git\Energy-Forecasting\models"

        mode = "fast"
        for _ in range(100):
            
            populations = generate_population(mode, 20)
            for _, population in populations.items():
                save_population(population, 5, file_path, mode, model_directory)
                loaded_population = load_population(file_path, model_directory, mode)

        mode = "normal"
        for _ in range(100):

            populations = generate_population(mode, 20)
            for _, population in populations.items():
                save_population(population, 5, file_path, mode, model_directory)
                loaded_population = load_population(file_path, model_directory, mode)


if __name__=="__main__":
    tester = TestClass()
    # tester.test_save_load()
    tester.test_save_final_model()
            
