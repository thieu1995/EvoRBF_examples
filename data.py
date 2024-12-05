#!/usr/bin/env python
# Created by "Thieu" at 16:38, 05/12/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_wine, load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.2
RANDOM_STATE = 42

# Helper function to standardize datasets and encode categorical target if needed
def preprocess_data(X, y):
    # Convert X to a DataFrame if it's a numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Ensure y is a Series (in case it's passed as numpy array)
    y = pd.Series(y)

    # Combine X and y into a single DataFrame to drop NaN rows in both
    data = X.copy()
    data['target'] = y

    # Remove rows with NaN values
    data = data.dropna()

    # Separate X and y after dropping NaNs
    X = data.drop(columns=['target'])
    y = data['target']

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Define transformers for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    # Check if target is categorical and encode it if necessary
    y = y.values
    if y.dtype == 'object' or y.dtype.name == 'category':  # Object type, usually indicating non-numeric labels
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def load_california():
    from sklearn.datasets import fetch_california_housing

    df = fetch_california_housing(as_frame=True)
    return preprocess_data(df.data, df.target)


def load_ames():
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(name="house_prices", return_X_y=True)
    return X, y


# Helper function to load and preprocess each dataset
def load_and_preprocess(dataset_name):
    if dataset_name == 'diabetes':
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
    elif dataset_name == 'wine':
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
    elif dataset_name == 'iris':
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
    elif dataset_name == 'pima':
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        data = pd.read_csv(url, names=column_names)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'heart':
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart-disease.csv"
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        data = pd.read_csv(url, names=column_names)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'titanic':
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        data = pd.read_csv(url)
        data = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
        data = data.dropna()
        X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
        y = data['Survived']
    elif dataset_name == 'adult':
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult.csv"
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                        'marital-status', 'occupation', 'relationship', 'race', 'sex',
                        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        data = pd.read_csv(url, names=column_names, na_values=' ?')
        data = data.dropna()
        X = data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
        y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    elif dataset_name == 'mushroom':
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/mushroom.csv"
        column_names = [f'feature_{i}' for i in range(1, 23)] + ['class']
        data = pd.read_csv(url, names=column_names)
        X = pd.get_dummies(data.iloc[:, :-1])
        y = data['class'].apply(lambda x: 1 if x == 'e' else 0)
    elif dataset_name == 'concrete':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete-compressive-strength/concrete.csv"
        data = pd.read_csv(url)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'energy':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/energydata_complete.csv"
        data = pd.read_csv(url)
        data = data[['T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19']]
        X = data
        y = data['T6']  # Example regression task
    elif dataset_name == 'power':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/Combined_Cycle_Power_Plants.zip"
        # Unzip and load the dataset
        import zipfile
        with zipfile.ZipFile("Combined_Cycle_Power_Plants.zip", 'r') as zip_ref:
            zip_ref.extractall("power_dataset")
        data = pd.read_csv("power_dataset/CCPP/Folds5x2_pp.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'airfoil':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00219/airfoil_self_noise.dat"
        data = pd.read_csv(url, sep='\t', header=None)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'cover_type':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00372/covtype.data"
        column_names = [f'feature_{i}' for i in range(1, 55)] + ['label']
        data = pd.read_csv(url, names=column_names)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'car':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        data = pd.read_csv(url, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
        X = pd.get_dummies(data.iloc[:, :-1])
        y = data['class'].apply(lambda x: 1 if x == 'unacc' else 0)
    elif dataset_name == 'german':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        column_names = ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
                        'savings', 'employment', 'location', 'personal_status', 'other_parties', 'residence_since',
                        'property_magnitude', 'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents',
                        'own_telephone', 'foreign_worker', 'class']
        data = pd.read_csv(url, names=column_names, sep=' ')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    elif dataset_name == 'automobile':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/automobile/automobile.data"
        column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
                        'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
                        'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
                        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
        data = pd.read_csv(url, names=column_names, na_values='?')
        data = data.dropna()
        X = data[['symboling', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'horsepower',
                  'city-mpg', 'highway-mpg']]
        y = data['price']
    else:
        raise ValueError("Dataset not supported")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

# List of all datasets to load
datasets = ['boston', 'diabetes', 'wine', 'iris', 'pima', 'heart', 'titanic', 'adult', 'mushroom',
            'concrete', 'energy', 'power', 'airfoil', 'cover_type', 'car', 'german', 'automobile']
data_dict = {}

# Loading and preprocessing all datasets
for dataset in datasets:
    try:
        X_train, y_train, X_test, y_test = load_and_preprocess(dataset)
        data_dict[dataset] = (X_train, y_train, X_test, y_test)
        print(f"Dataset {dataset} loaded and preprocessed successfully.")
    except Exception as e:
        print(f"Error loading {dataset}: {e}")

