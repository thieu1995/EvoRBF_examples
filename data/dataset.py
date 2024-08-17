#!/usr/bin/env python
# Created by "Thieu" at 06:35, 30/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def get_fill_value_for_bool_based_counter(col):
    # Count the number of True and False values in the boolean column
    true_count = col.value_counts().get(True, 0)
    false_count = col.value_counts().get(False, 0)
    # Determine the value to fill based on the count
    fill_value = False if true_count > false_count else True
    return fill_value


def get_fill_value_for_binary_based_counter(col):
    # Count the number of True and False values in the boolean column
    true_count = col.value_counts().get(1, 0)
    false_count = col.value_counts().get(0, 0)
    # Determine the value to fill based on the count
    fill_value = 1 if true_count > false_count else 0
    return fill_value


def get_dermatology_data():
    # https://archive.ics.uci.edu/dataset/33/dermatology

    dermatology = fetch_ucirepo(id=33)

    # metadata
    # print(dermatology.metadata)
    # variable information
    # print(dermatology.variables)

    # data (as pandas dataframes)
    X = dermatology.data.features
    y = dermatology.data.targets

    # print(X.describe())
    print(X.isna().sum())
    print(y.isna().sum())

    age_average = int(round(X['age'].mean()))
    X.loc[:, "age"].fillna(age_average, inplace=True)
    # X["age"].fillna(age_average, inplace=True)
    print(X.isna().sum())
    return X.values, y.values


def get_heart_disease():
    # https://archive.ics.uci.edu/dataset/45/heart+disease

    heart_disease = fetch_ucirepo(id=45)

    # # metadata
    # print(heart_disease.metadata)
    # # variable information
    # print(heart_disease.variables)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # print(X.describe())
    print(X.isna().sum())
    print(y.isna().sum())

    print(X.info())
    print(y.info())

    X.loc[:, "ca"].fillna(X['ca'].mean(), inplace=True)
    X.loc[:, "thal"].fillna(X['thal'].mean(), inplace=True)
    print(X.info())

    return X.values, y.values


def get_hepatitis_data():
    # https://archive.ics.uci.edu/dataset/46/hepatitis

    hepatitis = fetch_ucirepo(id=46)

    # # metadata
    # print(hepatitis.metadata)
    # # variable information
    # print(hepatitis.variables)

    # data (as pandas dataframes)
    X = hepatitis.data.features
    y = hepatitis.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    X.fillna(X.mean(), inplace=True)
    print(X.info())
    return X.values, y.values


def get_chronic_kidney_disease_data():
    #     # https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

    chronic_kidney_disease = fetch_ucirepo(id=336)

    # # metadata
    # print(chronic_kidney_disease.metadata)
    # # variable information
    # print(chronic_kidney_disease.variables)

    # data (as pandas dataframes)
    X = chronic_kidney_disease.data.features
    y = chronic_kidney_disease.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    ## Handling bool columns
    X['htn'] = X['htn'].replace({'yes': 1, 'no': 0})
    X['htn'].fillna(get_fill_value_for_binary_based_counter(X['htn']), inplace=True)
    X['dm'] = X['dm'].replace({'yes': 1, 'no': 0})
    X['dm'].fillna(get_fill_value_for_binary_based_counter(X['dm']), inplace=True)
    X['cad'] = X['cad'].replace({'yes': 1, 'no': 0})
    X['cad'].fillna(get_fill_value_for_binary_based_counter(X['cad']), inplace=True)
    X['appet'] = X['appet'].replace({'good': 1, 'poor': 0})
    X['appet'].fillna(get_fill_value_for_binary_based_counter(X['appet']), inplace=True)
    X['pe'] = X['pe'].replace({'yes': 1, 'no': 0})
    X['pe'].fillna(get_fill_value_for_binary_based_counter(X['pe']), inplace=True)
    X['ane'] = X['ane'].replace({'yes': 1, 'no': 0})
    X['ane'].fillna(get_fill_value_for_binary_based_counter(X['ane']), inplace=True)

    # Binary Encoding
    # bool_columns = X.select_dtypes(include='bool').columns
    # X[bool_columns] = X[bool_columns].astype(int)

    ## Hanlding the float columns
    float_columns = X.select_dtypes(include='float').columns
    # Fill NaN values in float columns with their mean
    X[float_columns] = X[float_columns].fillna(X[float_columns].mean())

    ## Handling the categorical columns
    # For categorical columns, fill missing values with a specific category or 'Unknown'
    X['rbc'].fillna('Unknown', inplace=True)
    X['pc'].fillna('Unknown', inplace=True)

    # Fill NaN values in object columns with the mode
    X["pcc"] = X["pcc"].fillna(X["pcc"].mode().iloc[0])
    X["ba"] = X["ba"].fillna(X["ba"].mode().iloc[0])

    # Find object columns
    object_columns = X.select_dtypes(include='object').columns
    for col in object_columns:
        X[col] = X[col].astype('category').cat.codes
    # X[object_columns] = X[object_columns].astype('category')  # Convert to categorical type
    # X[object_columns] = X[object_columns].cat.codes  # Map categories to numerical values

    print(X.info())
    return X.values, y.values


def get_indian_liver_patient_data():
    #  https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset

    ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225)

    # # metadata
    # print(ilpd_indian_liver_patient_dataset.metadata)
    # # variable information
    # print(ilpd_indian_liver_patient_dataset.variables)

    # data (as pandas dataframes)
    X = ilpd_indian_liver_patient_dataset.data.features
    y = ilpd_indian_liver_patient_dataset.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    ## Handling bool columns
    X['Gender'] = X['Gender'].replace({'Male': 1, 'Female': 0})

    print(X.info())
    return X.values, y.values


def get_primary_tumor_data():
    # https://archive.ics.uci.edu/dataset/83/primary+tumor

    primary_tumor = fetch_ucirepo(id=83)

    # # metadata
    # print(primary_tumor.metadata)
    # # variable information
    # print(primary_tumor.variables)

    # data (as pandas dataframes)
    X = primary_tumor.data.features
    y = primary_tumor.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    # Calculate the frequencies of the values
    value_counts = X['sex'].value_counts()
    # Fill NaN values with the less frequent value
    X["sex"] = X["sex"].fillna(value_counts.idxmin())
    X["sex"] = X["sex"].astype(int)


    X['histologic-type'] = X['histologic-type'].interpolate(method='linear')
    X["histologic-type"] = X["histologic-type"].fillna(3)
    X['degree-of-diffe'] = X['degree-of-diffe'].interpolate(method='linear')

    value_counts = X['skin'].value_counts()
    # Fill NaN values with the less frequent value
    X["skin"] = X["skin"].fillna(value_counts.idxmin())
    X["skin"] = X["skin"].astype(int)

    value_counts = X['axillar'].value_counts()
    # Fill NaN values with the less frequent value
    X["axillar"] = X["axillar"].fillna(value_counts.idxmin())
    X["axillar"] = X["axillar"].astype(int)

    print(X.info())
    return X.values, y.values


def get_parkinsons_data():
    # https://archive.ics.uci.edu/dataset/174/parkinsons

    parkinsons = fetch_ucirepo(id=174)

    # # metadata
    # print(parkinsons.metadata)
    # # variable information
    # print(parkinsons.variables)

    # data (as pandas dataframes)
    X = parkinsons.data.features
    y = parkinsons.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    print(X.info())
    return X.values, y.values


def get_spect_heart_data():
    # https://archive.ics.uci.edu/dataset/95/spect+heart

    spect_heart = fetch_ucirepo(id=95)

    # # metadata
    # print(spect_heart.metadata)
    # # variable information
    # print(spect_heart.variables)

    # data (as pandas dataframes)
    X = spect_heart.data.features
    y = spect_heart.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    print(X.info())
    return X.values, y.values


def get_spectf_heart_data():
    # https://archive.ics.uci.edu/dataset/96/spectf+heart

    spectf_heart = fetch_ucirepo(id=96)

    # # metadata
    # print(spectf_heart.metadata)
    # # variable information
    # print(spectf_heart.variables)

    # data (as pandas dataframes)
    X = spectf_heart.data.features
    y = spectf_heart.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    print(X.info())
    return X.values, y.values


def get_vertebral_column_data():
    # https://archive.ics.uci.edu/dataset/212/vertebral+column

    vertebral_column = fetch_ucirepo(id=212)

    # # metadata
    # print(vertebral_column.metadata)
    # # variable information
    # print(vertebral_column.variables)

    # data (as pandas dataframes)
    X = vertebral_column.data.features
    y = vertebral_column.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    print(X.info())
    return X.values, y.values


def get_breast_cancer_wisconsin_diagnostic_data():
    # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # # metadata
    # print(breast_cancer_wisconsin_diagnostic.metadata)
    # # variable information
    # print(breast_cancer_wisconsin_diagnostic.variables)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    # Perform one-hot encoding and keep original column
    y = pd.get_dummies(y, columns=['Diagnosis'], drop_first=True)

    print(X.info())
    return X.values, y.values


def get_breast_cancer_wisconsin_prognostic_data():
    # https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic

    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16)

    # # metadata
    # print(breast_cancer_wisconsin_prognostic.metadata)
    # # variable information
    # print(breast_cancer_wisconsin_prognostic.variables)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_prognostic.data.features
    y = breast_cancer_wisconsin_prognostic.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    # Perform one-hot encoding and keep original column
    y = pd.get_dummies(y, columns=['Outcome'], drop_first=True)

    print(X.info())
    return X.values, y.values


def get_cirrhosis_patient_survival_data():
    # https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1

    cirrhosis_patient_survival_prediction = fetch_ucirepo(id=878)

    # # metadata
    # print(cirrhosis_patient_survival_prediction.metadata)
    # # variable information
    # print(cirrhosis_patient_survival_prediction.variables)

    # data (as pandas dataframes)
    X = cirrhosis_patient_survival_prediction.data.features
    y = cirrhosis_patient_survival_prediction.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    # Find rows in X that contain NaN values
    nan_rows = X[X.isna().any(axis=1)]
    # Remove corresponding rows from Y
    y = y[~y.index.isin(nan_rows.index)]
    X = X.dropna()

    # Get the list of categorical column names
    categorical_columns = ['Drug', 'Sex', 'Ascites', "Hepatomegaly", "Spiders", "Edema"]

    # Iterate over each categorical column
    for column in categorical_columns:
        # Perform label encoding
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(X[column])
        # Replace the original column with the encoded labels
        X[column] = encoded_labels

    categorical_columns = ['Cholesterol', 'Copper', 'Tryglicerides', "Platelets"]
    # Iterate over each categorical column
    for column in categorical_columns:
        # Perform label encoding
        X[column] = pd.to_numeric(X[column].replace('NaNN', np.nan), errors='coerce')
        X[column] = X[column].interpolate().astype(int)

    lb_encoder = LabelEncoder()
    y["Status"] = lb_encoder.fit_transform(y["Status"])

    print(X.info())
    print(y.info())
    return X.values, y.values


def get_heart_failure_clinical_data():
    # https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

    heart_failure_clinical_records = fetch_ucirepo(id=519)

    # # metadata
    # print(heart_failure_clinical_records.metadata)
    # # variable information
    # print(heart_failure_clinical_records.variables)

    # data (as pandas dataframes)
    X = heart_failure_clinical_records.data.features
    y = heart_failure_clinical_records.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    return X.values, y.values


def get_national_poll_on_healthy_aging_data():
    # https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)

    national_poll_on_healthy_aging_npha = fetch_ucirepo(id=936)

    # # metadata
    # print(national_poll_on_healthy_aging_npha.metadata)
    # # variable information
    # print(national_poll_on_healthy_aging_npha.variables)

    # data (as pandas dataframes)
    X = national_poll_on_healthy_aging_npha.data.features
    y = national_poll_on_healthy_aging_npha.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    return X.values, y.values


def get_hcv_data():
    # https://archive.ics.uci.edu/dataset/571/hcv+data

    hcv_data = fetch_ucirepo(id=571)

    # # metadata
    # print(hcv_data.metadata)
    # # variable information
    # print(hcv_data.variables)

    # data (as pandas dataframes)
    X = hcv_data.data.features
    y = hcv_data.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    lb_01 = LabelEncoder()
    X["Sex"] = lb_01.fit_transform(X["Sex"])

    categorical_columns = ['ALB', 'ALP', 'CHOL', "PROT", "ALT"]
    # Iterate over each categorical column
    for column in categorical_columns:
        X[column] = X[column].interpolate()

    lb_02 = LabelEncoder()
    y["Category"] = lb_02.fit_transform(y["Category"])

    print(X.info())
    print(y.info())
    return X.values, y.values


def get_aids_clinical_trials_group_study_data():
    # https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175

    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

    # # metadata
    # print(aids_clinical_trials_group_study_175.metadata)
    # # variable information
    # print(aids_clinical_trials_group_study_175.variables)

    # data (as pandas dataframes)
    X = aids_clinical_trials_group_study_175.data.features
    y = aids_clinical_trials_group_study_175.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    return X.values, y.values


def get_maternal_health_risk_data():
    # https://archive.ics.uci.edu/dataset/863/maternal+health+risk

    maternal_health_risk = fetch_ucirepo(id=863)

    # # metadata
    # print(maternal_health_risk.metadata)
    # # variable information
    # print(maternal_health_risk.variables)

    # data (as pandas dataframes)
    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets

    lb_02 = LabelEncoder()
    y["RiskLevel"] = lb_02.fit_transform(y["RiskLevel"])

    print(X.isna().sum())
    print(X.info())
    print(y.info())

    return X.values, y.values


def get_mammographic_mass_data():
    # https://archive.ics.uci.edu/dataset/161/mammographic+mass

    mammographic_mass = fetch_ucirepo(id=161)

    # # metadata
    # print(mammographic_mass.metadata)
    # # variable information
    # print(mammographic_mass.variables)

    # data (as pandas dataframes)
    X = mammographic_mass.data.features
    y = mammographic_mass.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())


    # Find rows in X that contain NaN values
    nan_rows = X[X.isna().any(axis=1)]
    # Remove corresponding rows from Y
    y = y[~y.index.isin(nan_rows.index)]
    X = X.dropna()

    print(X.info())
    print(y.info())

    return X.values, y.values


def get_mask_reminder_data():
    # https://archive.ics.uci.edu/dataset/161/mammographic+mass

    mammographic_mass = fetch_ucirepo(id=161)

    # # metadata
    # print(mammographic_mass.metadata)
    # # variable information
    # print(mammographic_mass.variables)

    # data (as pandas dataframes)
    X = mammographic_mass.data.features
    y = mammographic_mass.data.targets

    print(X.isna().sum())
    print(X.info())
    print(y.info())


    # Find rows in X that contain NaN values
    nan_rows = X[X.isna().any(axis=1)]
    # Remove corresponding rows from Y
    y = y[~y.index.isin(nan_rows.index)]
    X = X.dropna()

    print(X.info())
    print(y.info())

    return X.values, y.values
