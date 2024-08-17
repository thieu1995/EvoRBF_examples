#!/usr/bin/env python
# Created by "Thieu" at 19:56, 13/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import StringVar, IntegerVar, BoolVar
from mealpy import FloatVar
from pathlib import Path

BASE_PATH = Path.cwd().parent


class Config:

    # Design the boundary (parameters)
    PROBLEM_BOUNDS = [
        IntegerVar(lb=10, ub=30, name="size_hidden"),
        StringVar(valid_sets=("kmean", "random"), name="center_finder"),
        FloatVar(lb=(0.01,), ub=(10.0,), name="sigmas"),
        BoolVar(n_vars=1, name="regularization"),
        FloatVar(lb=(0.001,), ub=(0.99,), name="lamda"),
    ]

    PATH_READ = f"{BASE_PATH}/data/clean"
    PATH_SAVE = f"{BASE_PATH}/data/history"
    STATISTIC_FILE_NAME = "statistic-results.csv"
    FIGURE_SIZE = (10, 4.8)

    DATA_NAME_01 = "Dermatology"
    DATA_NAME_02 = "Heart"
    DATA_NAME_03 = "Hepatitis"
    DATA_NAME_04 = "ChronicKidney"
    DATA_NAME_05 = "IndianLiver"
    DATA_NAME_06 = "Parkinsons"
    DATA_NAME_07 = "Spect"
    DATA_NAME_08 = "Spectf"
    DATA_NAME_09 = "BCWD"
    DATA_NAME_10 = "BCWP"

    SEED_SPLIT_DATA = 42
    TEST_SIZE = 0.25
    SCORING_LOSS = "F1S"

    EPOCH = 300
    POP_SIZE = 20
    LIST_SEEDS = [7, 8, 11, 15, 20, 21, 22, 23, 24, 27, 28, 30, 32, 35, 37, 39, 40, 41, 42, 45]
    LIST_METRICS = ["PS", "RS", "NPV", "F1S", "F2S", "SS", "CKS", "GMS", "AUC", "LS", "AS"]

    LIST_MODELS = [
        {"model_name": "GA", "algorithm": "BaseGA", "para": {"name": "GA", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "JADE", "algorithm": "JADE", "para": {"name": "JADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"model_name": "SADE", "algorithm": "SADE", "para": {"name": "SADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "ARO", "algorithm": "OriginalARO", "para": {"name": "ARO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "AVOA", "algorithm": "OriginalAVOA", "para": {"name": "AVOA", "epoch": EPOCH, "pop_size": POP_SIZE}},

        {"model_name": "AGTO", "algorithm": "OriginalAGTO", "para": {"name": "AGTO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "FOX", "algorithm": "OriginalFOX", "para": {"name": "FOX", "epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"model_name": "HHO", "algorithm": "OriginalHHO", "para": {"name": "HHO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "PSO", "algorithm": "OriginalPSO", "para": {"name": "PSO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "WOA", "algorithm": "OriginalWOA", "para": {"name": "WOA", "epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"model_name": "ASO", "algorithm": "OriginalASO", "para": {"name": "ASO", "epoch": EPOCH, "pop_size": POP_SIZE}},

        {"model_name": "M-EO", "algorithm": "ModifiedEO", "para": {"name": "M-EO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"model_name": "HGSO", "algorithm": "OriginalHGSO", "para": {"name": "HGSO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"model_name": "MVO", "algorithm": "OriginalMVO", "para": {"name": "MVO", "epoch": EPOCH, "pop_size": POP_SIZE}},

        {"model_name": "RIME", "algorithm": "OriginalRIME", "para": {"name": "RIME", "epoch": EPOCH, "pop_size": POP_SIZE}},
        # {"model_name": "AEO", "algorithm": "OriginalAEO", "para": {"name": "AEO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "AAEO", "algorithm": "AugmentedAEO", "para": {"name": "AAEO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "GBO", "algorithm": "OriginalGBO", "para": {"name": "GBO", "epoch": EPOCH, "pop_size": POP_SIZE}},

        {"model_name": "INFO", "algorithm": "OriginalINFO", "para": {"name": "INFO", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "RUN", "algorithm": "OriginalRUN", "para": {"name": "RUN", "epoch": EPOCH, "pop_size": POP_SIZE}},
    ]

    # EPOCH = 10
    # POP_SIZE = 20
    # LIST_SEEDS = [7, 8]
    #
    # LIST_MODELS = [
    #     {"model_name": "GA", "algorithm": "BaseGA", "para": {"name": "GA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "JADE", "algorithm": "JADE", "para": {"name": "JADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    # ]
