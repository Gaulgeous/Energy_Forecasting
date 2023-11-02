import pandas as pd
import numpy as np
import time
import math
import os
import logging
import json
import gc
import tracemalloc

from typing import List, Optional, Callable, Tuple
from random import randint, randrange, random, choices
from sklearn.linear_model import RidgeCV, LassoLarsCV, ElasticNetCV, Ridge, LassoLars
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, Normalizer, StandardScaler, LabelEncoder
from sklearn.cluster import FeatureAgglomeration
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils._param_validation import InvalidParameterError
from joblib import Parallel, delayed, wrap_non_picklable_objects
from copy import deepcopy
import joblib

SELECTION_EXPONENT = 1.00005
MAX_WINDOWS = 101
MIN_SAMPLES = 1500
MAX_SAMPLES = 10000
GEN_MERGE = 5
GEN_SAVE = 10
TEST_SPLIT = 0.8
COLUMN_REMOVE = 0.3
NAN_THRESHOLD = 0.3
MIN_VALUES_THRESHOLD = 100
PARALLEL_BACKEND = "threading"

Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

genome_layout = ["window", "imputer", "outlier", "scaler", "dimension", "model"]

imputers = ["None", "mean", "median", "mode"]

outliers = [0, 1]

scalers = {"standard": {},
           "robust": {},
           "normalizer": {'norm': ['l1', 'l2', 'max']}}

dimensions = {"PCA": {'n_components': [None, 'mle', 'randomized']},
            "FeatureAgglomeration": {'linkage': ['ward', 'complete', 'average'],
            'metric': ['euclidean', 'l1', 'l2', 'manhattan']}}

total_models = {"decision_tree": {'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 
                                    'min_samples_leaf': range(1, 21)}, 
                "k_neighbours": {'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 
                                    'p': [1, 2]}, 
                "linear_svr": {'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"], 
                                'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
                                'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 
                                'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]},
                "random_forest": {'n_estimators': [50, 75, 100], 'max_features': np.arange(0.05, 1.01, 0.05),
                                'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21), 
                                'bootstrap': [True, False]},
                "xgboost": {'n_estimators': [50, 75, 100], 'max_depth': range(1, 11), 
                            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.], 
                            'subsample': np.arange(0.05, 1.01, 0.05), 
                            'min_child_weight': range(1, 21)},
                "ridge": {},
                "lasso": {},
                "elastic": {"l1_ratio": np.arange(0.05, 1.0, 0.05), "tol": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}}

