import platform
from sklearn.utils._param_validation import InvalidParameterError

if platform.system() == "Windows":
    from genetic_algorithm.genetic_algorithm_scripts.constants import *
    from genetic_algorithm.genetic_algorithm_scripts.analysis import *
elif platform.system() == "Linux":
    from genetic_algorithm_scripts.constants import *
    from genetic_algorithm_scripts.analysis import *

    
def generate_component(loadout: dict) -> object:

    """
    Generates a machine learning model based on the loadout dictionary

    @param loadout: Dictionary detailing the loadout specifications of the model

    @return model: Generated machine learning model
    """

    model_name = loadout["type"]
    
    if model_name == "elastic":
        l1_ratio = loadout["l1_ratio"]
        tol = loadout["tol"]
        return ElasticNetCV(l1_ratio=l1_ratio, tol=tol)
    
    elif model_name == "decision_tree":
        
        max_depth = loadout["max_depth"]
        min_samples_split = loadout["min_samples_split"]
        min_samples_leaf = loadout["min_samples_leaf"]
        return DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf)
    
    elif model_name == "k_neighbours":
                
        n_neighbors = loadout["n_neighbors"]
        weights = loadout["weights"]
        p = loadout["p"]
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    
    elif model_name == "linear_svr":

        loss = loadout["loss"]
        tol = loadout["tol"]
        C = loadout["C"]
        epsilon = loadout["epsilon"]

        if loss == "epsilon_insensitive":
            dual = True

        return LinearSVR(loss=loss, tol=tol, C=C, epsilon=epsilon, dual="auto")
    
    elif model_name == "random_forest":

        n_estimators = loadout["n_estimators"]
        max_features = loadout["max_features"]
        min_samples_split = loadout["min_samples_split"]
        min_samples_leaf = loadout["min_samples_leaf"]
        bootstrap = loadout["bootstrap"]

        return RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, 
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     bootstrap=bootstrap)
    
    elif model_name == "xgboost":

        n_estimators = loadout["n_estimators"]
        max_depth = loadout["max_depth"]
        learning_rate = loadout["learning_rate"]
        subsample = loadout["subsample"]
        min_child_weight = loadout["min_child_weight"]

        return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            subsample=subsample, min_child_weight=min_child_weight)
    
    elif model_name == "ridge":
        return Ridge()
    
    elif model_name == "lasso":
        return LassoLars()
    
    else:
        raise ValueError("Incorrect input into generate_components")
    

def generate_window(window_dict: dict) -> int:
    """
    Extracts the window size from the loadout dictionary

    @param window_dict: Dictionary containing the window size

    @return win_size: Size of the window
    """
    return window_dict['window']


def generate_scaler(scaler_dict: dict) -> object:

    """
    Extracts the scaler from the loadout dictionary

    @param scaler_dict: Dictionary containing the scaler

    @return scaler: The type of scaler specified
    """

    scaler = scaler_dict['type']

    if scaler == 'normalizer':
        return Normalizer(norm=scaler_dict['norm'])
    elif scaler == 'standard':
        return StandardScaler()
    elif scaler == 'robust':
        return RobustScaler()
    elif scaler == "None": 
        return None


def generate_dimension(dimension_dict: dict) -> object:

    """
    Extracts the type of dimensionality reduction from the loadout dictionary

    @param dimension_dict: Dictionary containing the type of dimensionality reduction

    @return dimension: Type of dimensionality reducer
    """

    dimension = dimension_dict['type']

    if dimension == "PCA":
        if dimension_dict["n_components"] == "randomized":
            return PCA(svd_solver=dimension_dict["n_components"])
        else:
            return PCA(n_components=dimension_dict["n_components"])
    elif dimension == "FeatureAgglomeration":
        return FeatureAgglomeration(linkage=dimension_dict["linkage"], metric=dimension_dict["metric"])
    elif dimension == "None":
        return None
    

def generate_population(models: dict, population: dict, population_size: int) -> Population:

    """
    Generates the initial population depending upon the mode of operation

    @param mode: Determines whether the program is running in fast mode, etc.
    @param population_size: Size of the population to return

    @return population: The generated population
    """
        
    for j in range(len(models)):
        for _ in range(population_size):

            model_tic = j

            model_name, genome, genome_loadout = generate_genome(model_tic, models)
            genome_dict = {"fitness": -math.inf, "genome": genome, "loadout": genome_loadout}
            population[model_name].append(genome_dict)

    return population


def generate_genome(model_tic: int, models: list) -> Tuple[str, Genome, dict]:

    """
    Create each specific genome within the population

    @param model_tic: The model index to detail which sort of model will be created
    @param models: Dictionary containing all the possible ML models

    @return model_name: The name of the machine learning model
    @return genome: The genome generated
    @return genome_dict: Loadout specification of this particular genome
    """

    imputer = {"type": 0}
    outlier = {"type": 0}
    window = {"type": "window", "window": choices(range(1, 101), k=1)[0]}
    scaler = {"type": None}
    dimension = {"type": None}

    model_name, model, model_dict = generate_model(model_tic, models)

    genome = [window, 0, 0, 0, None, model]
    genome_dict = {"model_name": model_name, "window": window, "imputer": imputer, "outlier": outlier,
                   "scaler": scaler, "dimension": dimension, "model": model_dict}

    for i in range(1, len(genome) - 1):
        chance = randint(0, 3)
        if chance > 0:
            genome, genome_dict = mutate_genome(genome, i, genome_dict)

    return model_name, genome, genome_dict


def generate_model(model_tic: int, models: dict) -> Tuple[str, object, dict]:

    """
    Creates the machine learning model complete with random attributes

    @param model_tic: The model index to detail which sort of model will be created
    @param models: Dictionary containing all the possible ML models

    @return model_name: Type of machine learning model
    @return model: The ML model itself
    @return model_dict: Loadout attributes of the generated model
    """

    model_name = get_dict_key(models, model_tic)
    loadout = models[model_name]

    if model_name == "decision_tree":

        depth = randint(0, len(loadout.get("max_depth")) - 1)
        sample_split = randint(0, len(loadout.get("min_samples_split")) - 1)
        sample_leaf = randint(0, len(loadout.get("min_samples_leaf")) - 1)

        depth_val = loadout.get("max_depth")[depth]
        split_val = loadout.get("min_samples_split")[sample_split]
        leaf_val = loadout.get("min_samples_leaf")[sample_leaf]

        model = DecisionTreeRegressor(max_depth=depth_val, 
                                        min_samples_split=split_val,
                                        min_samples_leaf=leaf_val)
        
        model_dict = {"type": "decision_tree", "max_depth": depth_val, "min_samples_split": split_val, 
                        "min_samples_leaf": leaf_val}

    elif model_name == "k_neighbours":

        n_neighbours = randint(0, len(loadout.get("n_neighbors")) - 1)
        weights = randint(0, len(loadout.get("weights")) - 1)
        p = randint(0, len(loadout.get("p")) - 1)

        n_neighbours_val = loadout.get("n_neighbors")[n_neighbours]
        weights_val = loadout.get("weights")[weights]
        p_val = loadout.get("p")[p]

        model = KNeighborsRegressor(n_neighbors=n_neighbours_val, 
                                    weights=weights_val,
                                    p=p_val)
        
        model_dict = {"type": "k_neighbours", "n_neighbors": n_neighbours_val, "weights": weights, "p": p_val}

    elif model_name == "linear_svr":

        loss = randint(0, len(loadout.get("loss")) - 1)
        tol = randint(0, len(loadout.get("tol")) - 1)
        C = randint(0, len(loadout.get("C")) - 1)
        epsilon = randint(0, len(loadout.get("epsilon")) - 1)

        loss_val = loadout.get("loss")[loss]
        tol_val = loadout.get("tol")[tol]
        C_val = loadout.get("C")[C]
        epsilon_val = loadout.get("epsilon")[epsilon]

        model = LinearSVR(loss=loss_val, 
                                tol=tol_val, 
                                C=C_val, 
                                epsilon=epsilon_val,
                                dual="auto")
        
        model_dict = {"type": "linear_svr", "loss": loss_val, "tol": tol_val, 
                        "C": C_val, "epsilon": epsilon_val}

    elif model_name == "random_forest":

        n_estimators = randint(0, len(loadout.get("n_estimators")) - 1)
        max_features = randint(0, len(loadout.get("max_features")) - 1)
        min_samples_split = randint(0, len(loadout.get("min_samples_split")) - 1)
        min_samples_leaf = randint(0, len(loadout.get("min_samples_leaf")) - 1)
        bootstrap = randint(0, len(loadout.get("bootstrap")) - 1)

        n_estimators_val = loadout.get("n_estimators")[n_estimators]
        max_features_val = loadout.get("max_features")[max_features]
        min_samples_split_val = loadout.get("min_samples_split")[min_samples_split]
        min_samples_leaf_val = loadout.get("min_samples_leaf")[min_samples_leaf]
        bootstrap_val = loadout.get("bootstrap")[bootstrap]

        model = RandomForestRegressor(n_estimators=n_estimators_val, 
                                    max_features=max_features_val,
                                    min_samples_split=min_samples_split_val, 
                                    min_samples_leaf=min_samples_leaf_val, 
                                    bootstrap=bootstrap_val)
        
        model_dict = {"type": "random_forest", "n_estimators": n_estimators_val, "max_features": max_features_val, 
                        "min_samples_split": min_samples_split_val, "min_samples_leaf": min_samples_leaf_val,
                        "bootstrap": bootstrap_val}

    elif model_name == "xgboost":

        n_estimators = randint(0, len(loadout.get("n_estimators")) - 1)
        max_depth = randint(0, len(loadout.get("max_depth")) - 1)
        learning_rate = randint(0, len(loadout.get("learning_rate")) - 1)
        subsample = randint(0, len(loadout.get("subsample")) - 1)
        min_child_weight = randint(0, len(loadout.get("min_child_weight")) - 1)

        n_estimators_val = loadout.get("n_estimators")[n_estimators]
        max_depth_val = loadout.get("max_depth")[max_depth]
        learning_rate_val = loadout.get("learning_rate")[learning_rate]
        subsample_val = loadout.get("subsample")[subsample]
        min_child_weight_val = loadout.get("min_child_weight")[min_child_weight]


        model = XGBRegressor(n_estimators=n_estimators_val, 
                            max_depth=max_depth_val, 
                            learning_rate=learning_rate_val, 
                            subsample=subsample_val, 
                            min_child_weight=min_child_weight_val)
        
        model_dict = {"type": "xgboost", "n_estimators": n_estimators_val, "max_depth": max_depth_val, 
                        "learning_rate": learning_rate_val, "subsample": subsample_val, 
                        "min_child_weight": min_child_weight_val}
        
    elif model_name == "ridge":
        model = Ridge()
        model_dict = {"type": "ridge"}

    elif model_name == "lasso":

        model = LassoLars()
        model_dict = {"type": "lasso"}

    elif model_name == "elastic":

        l1_ratio = randint(0, len(loadout.get("l1_ratio")) - 1)
        tol = randint(0, len(loadout.get("tol")) - 1)

        l1_ratio_val = loadout.get("l1_ratio")[l1_ratio]
        tol_val = loadout.get("tol")[tol]

        model = ElasticNetCV(l1_ratio=l1_ratio_val, 
                            tol=tol_val)
        
        model_dict = {"type": "elastic", "l1_ratio": l1_ratio_val, "tol": tol_val}
 
    else:
        raise ValueError("Incorrect model name")

    return model_name, model, model_dict
    



def mutation(genome_dict: dict, 
             num_mutations: int = 3, 
             type_probability: float = 0.5, 
             value_probability: float = 0.5) -> dict:
    
    """
    Performs random genetic mutations over the given genome

    @param genome_dict: Loadout attributes of the given genome
    @param num_mutations: Number of mutations to perform on the genome
    @param type_probability: Probability of mutating the type of objects (e.g. swapping imputers)
    @param value_probability: Probability of changing a numerical value (e.g. Number of leaves)

    @return genome: The mutated genome
    @return loadout: The associated genome dictionary
    """
    
    genome = genome_dict["genome"]
    loadout = genome_dict["loadout"]

    for _ in range(num_mutations):
        mutation_chance = random()
        
        if mutation_chance < type_probability:
            index = randint(0, genome_layout.index("dimension"))
            genome, loadout = mutate_genome(genome, index, loadout)

        elif mutation_chance < type_probability + value_probability:
            genome, loadout = mutate_model(genome, loadout)

    for key, value in loadout["model"].items():
        if type(value) != str and type(total_models[loadout["model"]["type"]][key][0]) == str:
            loadout["model"][key] = total_models[loadout["model"]["type"]][key][value]
            genome[genome_layout.index("model")] = generate_component(loadout["model"])

    genome_dict["genome"] = genome
    genome_dict["loadout"] = loadout

    return genome_dict


def single_point_crossover(a: dict, b: dict) -> Tuple[Genome, Genome]:

    """
    Returns two meshings of the two most successful genomes

    @param a: Genome a
    @param b: Genome b

    @return new_a: Crossed over genome a
    @return new_b: Crossed over genome b
    """

    genome_a = a["genome"]
    loadout_a = a["loadout"]

    genome_b = b["genome"]
    loadout_b = b["loadout"]

    # Error check to ensure they're the same length
    if len(genome_a) != len(genome_b):
        raise ValueError("Genomes a and b must be of same length")
    
    # If there exists only two items for the genome, just construct the two of them together
    length = len(genome_a)
    if length < 2:
        return a, b
    
    selection = [i % 2 for i in range(length)]

    new_genome_a = []
    new_genome_b = []

    new_loadout_a = {}
    new_loadout_b = {}

    new_loadout_a["model_name"] = loadout_a["model_name"]
    new_loadout_b["model_name"] = loadout_b["model_name"]

    for i in range(length):

        key_a = get_dict_key(loadout_a, i+1)
        value_a = loadout_a.get(key_a)
        key_b = get_dict_key(loadout_b, i+1)
        value_b = loadout_b.get(key_b)

        if selection[i]:
            new_genome_a.append(genome_a[i])
            new_genome_b.append(genome_b[i])

            new_loadout_a[key_a] = value_a
            new_loadout_b[key_b] = value_b
        else:
            new_genome_a.append(genome_b[i])
            new_genome_b.append(genome_a[i])

            new_loadout_a[key_a] = value_b
            new_loadout_b[key_b] = value_a

    new_a = {"fitness": -math.inf, "genome": new_genome_a, "loadout": new_loadout_a}
    new_b = {"fitness": -math.inf, "genome": new_genome_b, "loadout": new_loadout_b}

    return new_a, new_b


def mutate_model_component(model_loadout: dict) -> Tuple[object, dict]:

    model_name = model_loadout["type"]
    counter = 0

    index = randint(1, len(model_loadout) - 1)
    key = get_dict_key(model_loadout, index)
    current_value = model_loadout[key]

    num_features = len(total_models[model_name][key])
    selection_index = randint(0, num_features - 1)
    selection_value = total_models[model_name][key][selection_index]

    while num_features < 2 or current_value == selection_value:

        if counter > 9:
            model = generate_component(model_loadout)

            return model, model_loadout

        index = randint(1, len(model_loadout) - 1)
        key = get_dict_key(model_loadout, index)
        current_value = model_loadout[key]

        num_features = len(total_models[model_name][key])
        selection_index = randint(0, num_features - 1)
        selection_value = total_models[model_name][key][selection_index]

        counter += 1

    model_loadout[key] = selection_value
    model = generate_component(model_loadout)

    return model, model_loadout


def mutate_model(genome: Genome, loadout: dict) -> Tuple[Genome, dict]:

    """
    Swaps out a random feature within the genome (e.g. changing the type of imputer)

    @param genome: The genome to mutate
    @param loadout: The loadout attributes of the given genome

    @return genome: Mutated genome
    @return loadout: Mutated loadout
    """

    index = randint(genome_layout.index("scaler")+1, len(genome))
    mutating_feature = get_dict_key(loadout, index)
    features = len(loadout[mutating_feature]) - 1

    counter = 0

    while features < 1:

        if counter > 10:
            return genome, loadout

        index = randint(genome_layout.index("scaler"), len(genome))
        mutating_feature = get_dict_key(loadout, index)
        features = len(loadout[mutating_feature]) - 1

        counter += 1

    feature_index = randint(1, features)
    feature_key = get_dict_key(loadout[mutating_feature], feature_index)
    genome_index = genome_layout.index(mutating_feature)

    current_value = loadout[mutating_feature][feature_key]

    if genome[genome_index] is not None:
        temp = genome[genome_index]
        genome[genome_index] = None
        del temp

    if mutating_feature == "scaler":

        scale_type = loadout[mutating_feature]["type"]
        new_value = choices(scalers[scale_type][feature_key], k=1)[0]
        while new_value == current_value:
            new_value = choices(scalers[scale_type][feature_key], k=1)[0]
        loadout[mutating_feature][feature_key] = new_value
        genome[genome_index] = generate_scaler(loadout[mutating_feature])

    elif mutating_feature == "dimension":

        # affin = choices(dimensions[key].get("metric"), k=1)[0]
        # link = choices(dimensions[key].get("linkage"), k=1)[0]

        # if link == "ward":
        #     affin = "euclidean"

        dimension_type = loadout[mutating_feature]["type"]
        new_value = choices(dimensions[dimension_type][feature_key], k=1)[0]
        while new_value == current_value:
            new_value = choices(dimensions[dimension_type][feature_key], k=1)[0]
        loadout[mutating_feature][feature_key] = new_value
        if loadout["dimension"]["type"] == "FeatureAgglomeration" and loadout["dimension"]["linkage"] == "ward":
            loadout["dimension"]["metric"] = 'euclidean'
        genome[genome_index] = generate_dimension(loadout["dimension"])
        
    elif mutating_feature == "model":
        genome[genome_layout.index("model")], loadout["model"] = mutate_model_component(loadout["model"])

    # else:
    #     raise ValueError("Undefined component found in mutate_model")

    return genome, loadout
    

def mutate_genome(genome: Genome, index: int, genome_dict: dict) -> Tuple[Genome, dict]:

    """
    Changes a random numerical value within the genome (e.g. Number of leaves)

    @param genome: The genome to mutate
    @param loadout: The loadout attributes of the given genome

    @return genome: Mutated genome
    @return loadout: Mutated loadout
    """

    mutation = None
    if genome_layout[index] == "window":
        mutation = choices(range(1, 101), k=1)[0]
        genome_dict["window"] = {"type": "window", "window": mutation}
    elif genome_layout[index] == "imputer":
        mutation = randint(0, len(imputers) - 1)
        genome_dict["imputer"] = {"type": mutation}

    elif genome_layout[index] == "outlier":
        mutation = randint(0, 1)
        genome_dict["outlier"] = {"type": mutation}

    elif genome_layout[index] == "scaler":
        current_scaler = genome_dict["scaler"]["type"]
        key = None

        tick = randint(0, len(scalers))
        if tick > 0:
            key = get_dict_key(scalers, tick - 1)
        while current_scaler == key:
            tick = randint(0, len(scalers))
            if tick > 0:
                key = get_dict_key(scalers, tick - 1)

        if tick == 0:
            genome_dict["scaler"] = {"type": None}
        elif key == "standard":
            mutation = StandardScaler()
            genome_dict["scaler"] = {"type": "standard"}
        elif key == "robust":
            mutation = RobustScaler()
            genome_dict["scaler"] = {"type": "robust"}
        elif key == "normalizer":

            norm = randint(0, len(scalers[key].get("norm")) - 1)
            norm_val = scalers[key].get("norm")[norm]

            mutation = Normalizer(norm=norm_val)
            genome_dict["scaler"] = {"type": "normalizer", "norm": norm_val}
        else:
            raise ValueError("Incorrectly keyed command for mutating scaler")

    elif genome_layout[index] == "dimension":

        current_dimension = genome_dict["dimension"]["type"]
        key = None

        tick = randint(0, len(dimensions))
        if tick > 0:
            key = get_dict_key(dimensions, tick - 1)
        while current_dimension == key:
            tick = randint(0, len(dimensions))
            if tick > 0:
                key = get_dict_key(dimensions, tick - 1)

        if tick == 0:
            genome_dict["dimension"] = {"type": None}

        elif key == "PCA":

            solve = choices(dimensions[key].get("n_components"), k=1)[0]
            genome_dict["dimension"] = {"type": "PCA", "n_components": solve}

            try:

                if solve == "randomized":
                    mutation = PCA(svd_solver="randomized")
                else:
                    mutation = PCA(n_components=solve)

            except InvalidParameterError:
                mutation = PCA()

        elif key == "FeatureAgglomeration":

            affin = choices(dimensions[key].get("metric"), k=1)[0]
            link = choices(dimensions[key].get("linkage"), k=1)[0]

            if link == "ward":
                affin = "euclidean"

            mutation = FeatureAgglomeration(metric=affin, linkage=link)
            genome_dict["dimension"] = {"type": "FeatureAgglomeration", "metric": affin, 
                                        "linkage": link}
            
        else:
            raise ValueError("Incorrectly keyed command for dimension in mutate_genome")

    if genome[index] is not None:
        temp = genome[index]
        genome[index] = mutation
        del temp
    else:
        genome[index] = mutation
    return genome, genome_dict

        
def selection_pair(population: Population) -> Population:

    """
    Returns the two strongest genomes within the population to continue on with
    Note that it has a chance to throw in a wildcard. This helps with the evolution

    @param population: The population of genomes to select from

    @return population: A smaller subset containing the most successful genomes
    """

    pos_1 = choices(population=[i for i in range(len(population))], weights=[5*(20 - i) for i in range(len(population))], k=1)[0]
    pos_2 = choices(population=[i for i in range(len(population))], weights=[5*(20 - i) for i in range(len(population))], k=1)[0]

    while pos_1 == pos_2:
        pos_1 = choices(population=[i for i in range(len(population))], weights=[5*(20 - i) for i in range(len(population))], k=1)[0]

    return population[pos_1], population[pos_2]
       

