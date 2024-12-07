#!/usr/bin/env python
# Created by "Thieu" at 19:56, 13/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import StringVar, IntegerVar, BoolVar
from mealpy import FloatVar
from pathlib import Path

# BASE_PATH = Path.cwd().parent
BASE_PATH = Path.cwd()

class Config:

    PATH_READ = f"{BASE_PATH}/data/clean"
    PATH_SAVE = f"{BASE_PATH}/data/history"
    STATISTIC_FILE_NAME = "statistic-results.csv"
    FIGURE_SIZE = (10, 4.8)

    VERBOSE = False
    SEED_SPLIT_DATA = 42
    TEST_SIZE = 0.25
    OBJ_CLS = "F1S"
    OBJ_REG = "MSE"

    EPOCH = 250
    POP_SIZE = 20
    # LIST_SEEDS = [7, 8, 11, 15, 20, 21, 22, 23, 24, 27, 28, 30, 32, 35, 37, 39, 40, 41, 42, 45]
    # LIST_METRICS = ["PS", "RS", "NPV", "F1S", "F2S", "SS", "CKS", "GMS", "AUC", "LS", "AS"]
    LIST_SEEDS = [10, 15, 21, 24, 27, 29, 30, 35, 40, 42]
    LIST_METRIC_CLS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
    LIST_METRIC_REG = ["df"]
    N_WORKERS = 10

    LIST_MODELS = [
        {"name": "BBO-RVFL", "class": "OriginalBBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SADE-RVFL", "class": "SADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SHADE-RVFL", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "LCO-RVFL", "class": "OriginalLCO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "INFO-RVFL", "class": "OriginalINFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "QLE-SCA-RVFL", "class": "QleSCA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SHIO-RVFL", "class": "OriginalSHIO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "EFO-RVFL", "class": "OriginalEFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "A-EO-RVFL", "class": "AdaptiveEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "RIME-RVFL", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "IM-ARO-RVFL", "class": "LARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "HHO-RVFL", "class": "OriginalHHO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "AIW-PSO-RVFL", "class": "AIW_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "CL-PSO-RVFL", "class": "CL_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    ]

    # LIST_MODELS = [
    #     {"model_name": "GA", "algorithm": "BaseGA", "para": {"name": "GA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "JADE", "algorithm": "JADE", "para": {"name": "JADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"model_name": "SADE", "algorithm": "SADE", "para": {"name": "SADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "ARO", "algorithm": "OriginalARO", "para": {"name": "ARO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "AVOA", "algorithm": "OriginalAVOA", "para": {"name": "AVOA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"model_name": "AGTO", "algorithm": "OriginalAGTO", "para": {"name": "AGTO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "FOX", "algorithm": "OriginalFOX", "para": {"name": "FOX", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"model_name": "HHO", "algorithm": "OriginalHHO", "para": {"name": "HHO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "PSO", "algorithm": "OriginalPSO", "para": {"name": "PSO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "WOA", "algorithm": "OriginalWOA", "para": {"name": "WOA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"model_name": "ASO", "algorithm": "OriginalASO", "para": {"name": "ASO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"model_name": "M-EO", "algorithm": "ModifiedEO", "para": {"name": "M-EO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"model_name": "HGSO", "algorithm": "OriginalHGSO", "para": {"name": "HGSO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"model_name": "MVO", "algorithm": "OriginalMVO", "para": {"name": "MVO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"model_name": "RIME", "algorithm": "OriginalRIME", "para": {"name": "RIME", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"model_name": "AEO", "algorithm": "OriginalAEO", "para": {"name": "AEO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "AAEO", "algorithm": "AugmentedAEO", "para": {"name": "AAEO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "GBO", "algorithm": "OriginalGBO", "para": {"name": "GBO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"model_name": "INFO", "algorithm": "OriginalINFO", "para": {"name": "INFO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"model_name": "RUN", "algorithm": "OriginalRUN", "para": {"name": "RUN", "epoch": EPOCH, "pop_size": POP_SIZE}},
    # ]

    EPOCH = 10
    POP_SIZE = 20
    LIST_SEEDS = [7, 8]

    LIST_MODELS = [
        {"model_name": "GA", "algorithm": "BaseGA", "para": {"name": "GA", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"model_name": "JADE", "algorithm": "JADE", "para": {"name": "JADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    ]
