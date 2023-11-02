import platform
import shutil

if platform.system() == "Windows":
    from genetic_algorithm.genetic_algorithm_scripts.constants import *
    from genetic_algorithm.genetic_algorithm_scripts.datacleaner import *
    from genetic_algorithm.genetic_algorithm_scripts.generation_and_evolution import *
    from genetic_algorithm.genetic_algorithm_scripts.read_write_functions import *
    from genetic_algorithm.genetic_algorithm_scripts.analysis import *
elif platform.system() == "Linux":
    from genetic_algorithm_scripts.constants import *
    from genetic_algorithm_scripts.datacleaner import *
    from genetic_algorithm_scripts.generation_and_evolution import *
    from genetic_algorithm_scripts.read_write_functions import *
    from genetic_algorithm_scripts.analysis import *


class TimeSeriesRegressor:
    def __init__(self,
                directory: str = os.getcwd(),
                max_time_mins: int = math.inf,
                mode: str = "normal",
                crossover_rate: float = 0.1,
                n_successors: int = 2,
                checkpoint: bool = True,
                warmstart: bool = False,
                early_stop: float = 1,
                cv: int = 5,
                generations: int = 50,
                population: int = 20,
                n_jobs: int=1):
        
        """"
        Initialise all parameters and inherent functions

        @param directory: Sets the directory for ml models
        @param max_time_mins: Sets the maximum runtime for the regressor
        @param mode: Determines which models are used 
        @param crossover_rate: Rate at which successive genes are crossed over
        @param n_successors: Number of successful genomes that go into the next gen
        @param checkpoint: If enabled, saves the population every few iterations
        @param warmstart: If enabled, will resume from a previously saved checkpoint
        @param early_stop: Stops the process early if the MSE has been achieved
        @param cv: Number of cross-validations
        @param generations: Number of generations undertaken
        @param population: Size of each successive population
        """
        
        self.model_directory = directory
        self.max_time = max_time_mins*60
        self.mode = mode
        self.checkpoint = checkpoint
        self.early_stop = early_stop
        self.cv = cv
        self.generations = generations
        self.population_size = population
        self.n_successors = n_successors
        self.crossover_rate = crossover_rate
        self.warmstart = warmstart
        self.n_jobs = n_jobs


        if mode == "fast":
            self.models = {"ridge": total_models["ridge"],
                        "lasso": total_models["lasso"],
                        "elastic": total_models["elastic"],
                        "linear_svr": total_models["linear_svr"],
                        "decision_tree": total_models["decision_tree"],
                        "k_neighbours": total_models["k_neighbours"]}
            
            self.population: Population = {"linear_svr": [], "ridge": [], "lasso": [], "elastic": [], "decision_tree": [], "k_neighbours": []}
            
        elif mode == "normal":
            self.models = {
                        "random_forest": total_models["random_forest"],
                        "xgboost": total_models["xgboost"]}
            
            self.population: Population = {"random_forest": [], "xgboost": []}

        self.frame_path = self.model_directory + "/frames"
        if not os.path.exists(self.frame_path):
            os.makedirs(self.frame_path)

        self.populate_func = generate_population
        self.fitness_func = calc_fitness
        self.selection_func = selection_pair
        self.crossover_func = single_point_crossover
        self.mutation_func = mutation


    def fit(self, 
            file_path: str, 
            target: str,
            epd: int,
            trend_type: str = "None",
            future: int = 0
            ):
        
        """
        Fitting function to run evolution

        @param file_path: Path to the dataset
        @param target: Name of the data column to train for
        @param epd: Entries per day in the dataset
        @param trend_type: Trend type for seasonal decomposition
        @param future: Number of days to predict into the future
        """
        
        self.start_time = time.time()
        
        _, filetype = os.path.splitext(file_path)

        if filetype == ".csv":
            data = pd.read_csv(file_path)
        elif filetype == ".xls" or filetype == ".xlsx":
            data = pd.read_excel(file_path)
        else:
            raise ValueError("File type must be of type .csv, .xlsx or .xls")

        length = data.shape[0]
        if length < MIN_SAMPLES:
            print("At least {0} samples should be present within the dataset for accurate results".format(MIN_SAMPLES))
        
        if length > MAX_SAMPLES:
            print("Length of dataset quite large. Reducing the size to {0} samples".format(MAX_SAMPLES))
            tuning_data = data.iloc[:MAX_SAMPLES]
        
        else:
            tuning_data = data

        tuning_dataframes = create_dataframes(tuning_data, target, epd, trend_type, future, self.frame_path)
        gc.collect()

        model, _ = self.run_evolution(tuning_dataframes, file_path, target, epd, trend_type, future)

        """
        Develop a finalising function to save the model with its parameters
        """

        predictions, metrics = final_model_prediction(model, data, target, epd, trend_type, future)

        print()
        print()
        print("Finished final iteration. Final Values of:")
        print("MAE {0}".format(metrics["MAE"]))
        print("MAPE {0}".format(metrics["MAPE"]))
        print("MSE {0}".format(metrics["MSE"]))
        print("R2 {0}".format(metrics["R2"]))
        print("Total Time {0}".format(time.time() - self.start_time))

        predictions.to_csv(os.getcwd() + '/src/genetic_algorithm/predictions.csv', index=False)

        metric_frame = pd.DataFrame({"Metric": [], "Value": []})
        metric_frame.loc[len(metric_frame)] = {"Metric": "RMSE", "Value": metrics["MSE"]}
        metric_frame.loc[len(metric_frame)] = {"Metric": "MAE", "Value": metrics["MAE"]}
        metric_frame.loc[len(metric_frame)] = {"Metric": "MAPE", "Value": metrics["MAPE"]}
        metric_frame.loc[len(metric_frame)] = {"Metric": "R2", "Value": metrics["R2"]}
        metric_frame.loc[len(metric_frame)] = {"Metric": "TIME", "Value": (time.time() - self.start_time)/60}

        metric_frame.to_csv(os.getcwd() + '/src/genetic_algorithm/metrics.csv', index=False)

        shutil.rmtree(self.frame_path)

        save_final_model(model["loadout"], file_path, self.mode, self.model_directory, target, epd, trend_type, future)


    def propagate(self, 
                  population: Population, 
                  tuning_dataframes: list, 
                  target: str,
                  epd: int, 
                  trend_type: str, 
                  future: int, 
                  parallel: object) -> Tuple[list, bool, int]:
        
        """
        Runs each successive propagation step in the evolution

        @param population: Population of genome candidates
        @param data: Dataset to train on
        @param target: Name of the dataset column to work towards
        @param epd: Entries per day in the dataset
        @param trend_type: Type of trend for seasonal decomposition
        @param future: Number of days into the future to predict for

        @return next_generation: The population the makes up the next generation
        @return bool: Whether the early-stopping value was achieved
        @return best_loss: Current loss that has been achieved
        """

        # Sorts the population by how it ranks in the fitness function
        fitnesses = parallel(delayed(calc_fitness)(model, tuning_dataframes, target, self.cv, epd, trend_type, future, self.frame_path) for model in population)
        gc.collect()
        for count, fitness in enumerate(fitnesses):
            population[count]["fitness"] = fitness

        population = sorted(population, key=lambda x:x["fitness"], reverse=True)
        best_loss = population[0]["fitness"]

        # You found a good enough working solution
        if best_loss > self.early_stop:
            return population, True, best_loss

        # Take the two most successful items
        next_generation = population[0:min(2, self.n_successors)]
        for genome in next_generation:
            genome["fitness"] = -math.inf
        crossed = []

        counter = 0
        while counter < int(self.crossover_rate*self.population_size):

            parent_a, parent_b = self.selection_func(population)
            offspring_a, offspring_b = self.crossover_func(deepcopy(parent_a), deepcopy(parent_b))
            crossed.append(offspring_a)
            crossed.append(offspring_b)
            counter += 2

        next_generation = next_generation + crossed
        mutators = len(next_generation) - 1

        while (len(next_generation)) < self.population_size:
            index = randint(0, mutators)
            mutated = self.mutation_func(deepcopy(next_generation[index]))
            next_generation.append(mutated)

        for genome in next_generation:
            if genome["genome"][genome_layout.index("window")] == None:
                window = {'type': 'window', 'window': randint(1, MAX_WINDOWS)}
                genome["genome"][genome_layout.index("window")] = window
                genome["loadout"]["window"] = window

            elif genome["genome"][genome_layout.index("model")] == None:
                _, model, model_dict = generate_model(list(total_models.key()).index(genome["loadout"]["model"]["type"]), self.models)
                genome["genome"][genome_layout.index("model")] = model
                genome["loadout"]["model"] = model_dict

        # Rinse. Repeat
        return next_generation, False, best_loss
        

    def run_evolution(self, 
                    tuning_dataframes: list, 
                    file_path: str,
                    target: str,
                    epd: int, 
                    trend_type: str, 
                    future: int) -> Tuple[list, int]:
        
        """
        The main evolution function

        @param data: dataset to train on
        @param file_path: path to dataset
        @param target: Name of the dataset column to work towards
        @param epd: Entries per day in the dataset
        @param trend_type: Type of trend for seasonal decomposition
        @param future: Number of days into the future to predict for

        @return genome: most successful genome
        @return generation: generation at which this was achieved
        """
        
        stop = False

        best_losses = [-math.inf for _ in range(5)]
        
        if self.warmstart:
            population = load_population(file_path, self.model_directory, self.mode, self.models, self.population, self.population_size)
        else:
            population = self.populate_func(self.models, self.population, self.population_size)

        with Parallel(n_jobs=self.n_jobs, backend=PARALLEL_BACKEND) as parallel:
            for generation in range(self.generations):

                gen_time = time.time()

                """sort the genomes in order of performance"""

                if time.time() - self.start_time > self.max_time:
                    break

                if generation > GEN_MERGE:
                    population, stop, current_loss = self.propagate(population, tuning_dataframes, target, epd, trend_type, future, parallel)
                    gc.collect()

                    best_losses.append(current_loss)

                    if len(best_losses) > 5:
                        best_losses.pop(0)

                    max_loss = max(best_losses)
                    min_loss = min(best_losses)

                    # if max_loss - min_loss < 3*1e-3 and -math.inf not in best_losses:
                    #     break

                elif generation == GEN_MERGE:

                    # Sorts the population by how it ranks in the fitness function
                    for key, _ in population.items():
                        # Sorts the population by how it ranks in the fitness function
                        
                        fitnesses = parallel(delayed(calc_fitness)(model, tuning_dataframes, target, self.cv, epd, trend_type, future, self.frame_path) for model in population[key])
                        gc.collect()
                        
                        for count, fitness in enumerate(fitnesses):
                            population[key][count]["fitness"] = fitness
                        population[key] = sorted(population[key], key=lambda x:x["fitness"], reverse=True)

                    best_fitness = -math.inf
                    best_model = None

                    for model_name, model_type in population.items():
                        fitness = population_fitness(model_type)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_model = model_name

                    population = population[best_model]
                    current_loss = best_fitness

                else:
                    for model_name, model_type in population.items():

                        if time.time() - self.start_time > self.max_time:
                            break

                        population[model_name], stop, current_loss = self.propagate(model_type, tuning_dataframes, target, epd, trend_type, future, parallel)
                        gc.collect()
                        
                        print(f"finished {model_name}")

                if self.checkpoint and generation % GEN_SAVE == 0 and generation > GEN_MERGE:
                    save_population(population, generation, file_path, self.mode, self.model_directory)

                print()
                print("Completed generation {0}".format(generation))
                print("Time taken {0}".format(time.time() - gen_time))

                if generation >= GEN_MERGE:
                    print("Best model is: {0}".format(population[0]["loadout"]["model_name"]))
                    print("With a loss function value of {0}".format(current_loss))

                if stop:
                    break

        if type(population) == dict:
            
            best_fitness = -math.inf
            best_model = None

            for model_name, model_type in population.items():
                fitness = population_fitness(model_type)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_model = model_name

            population = population[best_model]

        return population[0], generation

    
if __name__=="__main__":

    # tracemalloc.start()
    model_directory = r"/home/david/Git/Energy-Forecasting/models"
    file_path = r"/home/david/Git/Energy-Forecasting/data/aemo_nsw_target_only.csv"

    regressor = TimeSeriesRegressor(directory=model_directory, population=20, generations=50, 
                                    mode="normal", max_time_mins=300, n_jobs=-1, cv=1, early_stop=0.99)
    regressor.fit(file_path=file_path, target="TOTALDEMAND", 
                  epd=288, trend_type="Additive", future=288)