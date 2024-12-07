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
    LIST_METRIC_REG =  ["MAE", "RMSE", "NNSE", "WI", "R", "KGE"]
    N_WORKERS = 10

    LIST_MODELS = [
        {"name": "BBO-RVFL", "class": "OriginalBBO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SADE-RVFL", "class": "SADE", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SHADE-RVFL", "class": "OriginalSHADE", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "LCO-RVFL", "class": "OriginalLCO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "INFO-RVFL", "class": "OriginalINFO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "QLE-SCA-RVFL", "class": "QleSCA", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "SHIO-RVFL", "class": "OriginalSHIO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "EFO-RVFL", "class": "OriginalEFO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "A-EO-RVFL", "class": "AdaptiveEO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "RIME-RVFL", "class": "OriginalRIME", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "IM-ARO-RVFL", "class": "LARO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "HHO-RVFL", "class": "OriginalHHO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "AIW-PSO-RVFL", "class": "AIW_PSO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "CL-PSO-RVFL", "class": "CL_PSO", "parass": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    ]

    # LIST_MODELS = [
    #     {"name": "GA", "class": "BaseGA", "paras": {"name": "GA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "JADE", "class": "JADE", "paras": {"name": "JADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"name": "SADE", "class": "SADE", "paras": {"name": "SADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "ARO", "class": "OriginalARO", "paras": {"name": "ARO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "AVOA", "class": "OriginalAVOA", "paras": {"name": "AVOA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"name": "AGTO", "class": "OriginalAGTO", "paras": {"name": "AGTO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "FOX", "class": "OriginalFOX", "paras": {"name": "FOX", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"name": "HHO", "class": "OriginalHHO", "paras": {"name": "HHO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "PSO", "class": "OriginalPSO", "paras": {"name": "PSO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "WOA", "class": "OriginalWOA", "paras": {"name": "WOA", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"name": "ASO", "class": "OriginalASO", "paras": {"name": "ASO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"name": "M-EO", "class": "ModifiedEO", "paras": {"name": "M-EO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"name": "HGSO", "class": "OriginalHGSO", "paras": {"name": "HGSO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"name": "MVO", "class": "OriginalMVO", "paras": {"name": "MVO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"name": "RIME", "class": "OriginalRIME", "paras": {"name": "RIME", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     # {"name": "AEO", "class": "OriginalAEO", "paras": {"name": "AEO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "AAEO", "class": "AugmentedAEO", "paras": {"name": "AAEO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "GBO", "class": "OriginalGBO", "paras": {"name": "GBO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #
    #     {"name": "INFO", "class": "OriginalINFO", "paras": {"name": "INFO", "epoch": EPOCH, "pop_size": POP_SIZE}},
    #     {"name": "RUN", "class": "OriginalRUN", "paras": {"name": "RUN", "epoch": EPOCH, "pop_size": POP_SIZE}},
    # ]

    EPOCH = 10
    POP_SIZE = 20
    LIST_SEEDS = [7, 8]

    LIST_MODELS = [
        {"name": "GA", "class": "BaseGA", "paras": {"name": "GA", "epoch": EPOCH, "pop_size": POP_SIZE}},
        {"name": "JADE", "class": "JADE", "paras": {"name": "JADE", "epoch": EPOCH, "pop_size": POP_SIZE}},
    ]
