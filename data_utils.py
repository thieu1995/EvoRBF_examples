#!/usr/bin/env python
# Created by "Thieu" at 15:26, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import Config


def get_bank_marketing(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/bank-marketing-eda-and-classification
    # adult = fetch_ucirepo(id=222)

    df = pd.read_csv(path)
    X = df.drop('deposit', axis=1).values
    y = df['deposit'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Bank Marketing")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_bankruptcy(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/bankruptcy-classification-ml-95-acc
    # taiwanese_bankruptcy_prediction = fetch_ucirepo(id=572)

    df = pd.read_csv(path)
    X = df.drop('Bankrupt?', axis=1).values
    y = df['Bankrupt?'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Bankruptcy")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_car_evaluate(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/decision-tree-classification-car-evaluate
    # car_evaluation = fetch_ucirepo(id=19)
    # encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

    df = pd.read_csv(path)
    X = df.drop('class', axis=1).values
    y = df['class'].values

    y = LabelEncoder().fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Car Evaluation")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_letter_recognition(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/adityaw2604/support-vector-machine-92/input
    # letter_recognition = fetch_ucirepo(id=59)

    df = pd.read_csv(path)
    X = df.drop('letter', axis=1).values
    y = df['letter'].values
    y = LabelEncoder().fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Letter Recognition")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_mushroom(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/mushroom-analysis-and-classification-accu-100
    # mushroom = fetch_ucirepo(id=73)

    df = pd.read_csv(path)
    X = df.drop('edibility', axis=1).values
    y = df['edibility'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Mushroom")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_rice(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/rice-classification-ml-90-accuracy
    # rice_cammeo_and_osmancik = fetch_ucirepo(id=545)
    ## Rice (Cammeo and Osmancik)

    df = pd.read_csv(path)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Rice")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_bike_sharing_demand(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/bike-sharing-demand-eda-regression-90-acc
    # https://www.kaggle.com/code/behradkarimi/check-some-regression-model-and-find-the-best-one
    # bike_sharing = fetch_ucirepo(id=275)

    df = pd.read_csv(path)
    X = df.drop('count', axis=1).values
    y = df['count'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Bike Sharing Demand")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_ccpp(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/combined-cycle-power-plant-forecast-ml-93-acc
    # combined_cycle_power_plant = fetch_ucirepo(id=294)

    df = pd.read_csv(path)
    X = df.drop('AT', axis=1).values
    y = df['AT'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Combined Cycle Power Plant")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_concrete(path, verbose=False):
    # (1030, 9)
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/concrete-compression-forecasting-ml-95-acc

    df = pd.read_csv(path)
    X = df.drop('concrete_compressive_strength', axis=1).values
    y = df['concrete_compressive_strength'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Concrete Compression Strength")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_air_quality(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/air-quality-with-ml-85-acc-dl-98-acc
    # air_quality = fetch_ucirepo(id=360)

    df = pd.read_csv(path)
    X = df.drop('AH', axis=1).values
    y = df['AH'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Concrete Compression Strength")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_superconductivty(path, verbose=False):
    # (21263, 81)
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/superconductivty-forecasting-ml-88-acc
    # superconductivty_data = fetch_ucirepo(id=464)

    df = pd.read_csv(path)
    X = df.drop('critical_temp', axis=1).values
    y = df['critical_temp'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Concrete Compression Strength")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_spambase(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/email-spam-classification-ml-model-97-accuracy
    # spambase = fetch_ucirepo(id=94)

    df = pd.read_csv(path)
    X = df.drop('spam', axis=1).values
    y = df['spam'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Concrete Compression Strength")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_titanic(path, verbose=False):
    ## Data pre-processing from here: https://www.kaggle.com/code/thieunv/titanic-classification-ml-90-accuracy

    df = pd.read_csv(path)
    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE,
                                                        random_state=Config.SEED_SPLIT_DATA)
    if verbose:
        print(f"\nData: Concrete Compression Strength")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"unique y_train: {np.unique(y_train)}, unique y_test: {np.unique(y_test)}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

