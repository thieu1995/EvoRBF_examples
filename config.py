#!/usr/bin/env python
# Created by "Thieu" at 19:56, 13/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
# BASE_PATH = Path.cwd().parent
BASE_PATH = Path.cwd()

class Config:

    PATH_READ = f"{BASE_PATH}/data/clean"
    PATH_SAVE = f"{BASE_PATH}/data/history"
    STATISTIC_FILE_NAME = "statistic-results.csv"
    FILE_LOSS = "df_loss.csv"
    FILE_RESULT = "df_result.csv"
    FIGURE_SIZE = (10, 4.8)

    VERBOSE = False
    SEED_SPLIT_DATA = 42
    TEST_SIZE = 0.25
    OBJ_CLS = "F1S"
    OBJ_REG = "MSE"

    # 11162 samples, 17 features, 2 classes
    # (11162, 34) - after pre-processing
    DATA01 = {
        "name": "bank_marketing",
        "size_hidden": 50,
        "center_finder": "kmeans"
    }

    # 6819 samples, 96 features, 2 classes
    # (6819, 10) - after pre-processing
    DATA02 = {
        "name": "bankruptcy",
        "size_hidden": 21,
        "center_finder": "kmeans"
    }

    # 1727 samples, 6 features, 4 classes
    # (1737, 6) - after pre-processing
    DATA03 = {
        "name": "car",
        "size_hidden": 25,
        "center_finder": "kmeans"
    }

    # 20000 samples, 16 features, 26 classes
    # (20000, 16) - after pre-processing
    DATA04 = {
        "name": "letter",
        "size_hidden": 33,
        "center_finder": "kmeans"
    }

    # 8124 samples, 22 features, 2 classes
    # (8124, 22) - after pre-processing
    DATA05 = {
        "name": "mushroom",
        "size_hidden": 45,
        "center_finder": "kmeans"
    }

    # 3810 samples, 7 features, 2 classes
    # (3724, 7) - after pre-processing
    DATA06 = {
        "name": "rice",
        "size_hidden": 15,
        "center_finder": "kmeans"
    }

    # 21263 samples, 81 features
    # (21263, 81) - after pre-processing
    DATA07 = {
        "name": "superconductivity",
        "size_hidden": 40,
        "center_finder": "kmeans"
    }

    # 9568 samples, 4 features
    # (9568, 4) - after pre-processing
    DATA08 = {
        "name": "ccpp",
        "size_hidden": 9,
        "center_finder": "kmeans"
    }

    ## Load data object
    # 1030 samples, 8 features
    # (1030, 8) - after pre-processing
    DATA09 = {
        "name": "concrete",
        "size_hidden": 17,
        "center_finder": "kmeans"
    }

    ## Load data object
    # 9471 samples, 14 features
    # (8191, 12) - after pre-processing
    DATA10 = {
        "name": "air_quality",
        "size_hidden": 25,
        "center_finder": "kmeans"
    }


    EPOCH = 250
    POP_SIZE = 20
    # LIST_SEEDS = [7, 8, 11, 15, 20, 21, 22, 23, 24, 27, 28, 30, 32, 35, 37, 39, 40, 41, 42, 45]
    # LIST_METRICS = ["PS", "RS", "NPV", "F1S", "F2S", "SS", "CKS", "GMS", "AUC", "LS", "AS"]
    LIST_SEEDS = [10, 15, 21, 24, 27, 29, 30, 35, 40, 42]
    LIST_METRIC_CLS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
    LIST_METRIC_REG =  ["MAE", "RMSE", "NNSE", "WI", "R", "KGE"]
    N_WORKERS = 8

    # EPOCH = 50
    # POP_SIZE = 20
    # LIST_SEEDS = [7]

    LIST_MODELS = [
        {"name": "LDW-PSO-RBF", "class": "LDW_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 5.5 - 7.5
        {"name": "CL-PSO-RBF", "class": "CL_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 5.5 - 7.5
        # {"name": "AGTO-RBF", "class": "OriginalAGTO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 13
        # {"name": "AVOA-RBF", "class": "OriginalAVOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 5.5 - 7.5    But getting warning divide by 0
        {"name": "SMA-RBF", "class": "OriginalSMA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 5.5 - 7.5
        # {"name": "SOS-RBF", "class": "OriginalSOS", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 21 - 26
        {"name": "GBO-RBF", "class": "OriginalGBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},          # 7
        {"name": "PSS-RBF", "class": "OriginalPSS", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},           # 7
        # {"name": "E-AEO-RBF", "class": "EnhancedAEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 14 - 16
        # {"name": "AAEO-RBF", "class": "AugmentedAEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 14
        {"name": "SADE-RBF", "class": "SADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},                 # 7
        {"name": "CMA-ES-RBF", "class": "CMA_ES", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},             # 7
        {"name": "SHADE-RBF", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 4.5
        # {"name": "TLO-RBF", "class": "OriginalTLO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},     # 15
        # {"name": "QSA-RBF", "class": "OriginalQSA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 14
        {"name": "EFO-RBF", "class": "OriginalEFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 0.33
        # {"name": "M-EO-RBF", "class": "ModifiedEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},       # 13
        {"name": "RIME-RBF", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},     # 8
        # {"name": "MGTO-RBF", "class": "MGTO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},         # 15
        {"name": "HI-WOA-RBF", "class": "HI_WOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},     # 6
        # {"name": "SHO-RBF", "class": "OriginalSHO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},   # 19 - 36
        # {"name": "WMQI-MRFO-RBF", "class": "WMQIMRFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},    # Warning divide by 0,
        {"name": "WOA-FOA-RBF", "class": "WhaleFOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},      # 6
    ]


