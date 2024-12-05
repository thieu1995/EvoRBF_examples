#!/usr/bin/env python
# Created by "Thieu" at 20:07, 13/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from evorbf import Data, InaRbfTuner
import pandas as pd
from helper import *
from config import Config

## Load data object
DATA_NAME = Config.DATA_NAME_03
df = pd.read_csv(f"{Config.PATH_READ}/{DATA_NAME}.csv")
X, y = df.values[:,:-1], df.values[:,-1:]
print(np.unique(y))
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=Config.TEST_SIZE, random_state=Config.SEED_SPLIT_DATA)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))


results_gbest_fit = []   # Working with global best fitness - statistic test
results_curve = {}
results_gbest_box = []
for idx_model, my_model in enumerate(Config.LIST_MODELS):

    list_curve_trial = []
    for idx_trial, seed in enumerate(Config.LIST_SEEDS):
        print(f"Data: {DATA_NAME}, Model: {my_model['model_name']}, Trial: {idx_trial}")

        model = InaRbfTuner(problem_type="classification", bounds=Config.PROBLEM_BOUNDS, cv=4, scoring=Config.SCORING_LOSS,
                            optimizer=my_model["algorithm"], optimizer_paras=my_model["para"], verbose=Config.VERBOSE, seed=seed)
        model.fit(data.X_train, data.y_train)
        y_pred = model.predict(data.X_test)

        ## Generate the confusion matrix
        draw_confusion_matrix(data.y_test, y_pred, model_name=my_model['model_name'], data_name=DATA_NAME,
                              save_file=True, save_file_path=f"{Config.PATH_SAVE}/{DATA_NAME}/{my_model['model_name']}/trial{idx_trial}")
        model.save_model(f"{Config.PATH_SAVE}/{DATA_NAME}/{my_model['model_name']}/trial{idx_trial}")

        ## Save best results (Average, Best, SD)
        res1 = {
            'Model': my_model["model_name"],
            'Trial': idx_trial,
            "Seed": seed,
            'Global_Best_Fitness': model.optimizer.g_best.target.fitness
        }
        res2 = model.best_estimator.evaluate(data.y_test, y_pred, list_metrics=Config.LIST_METRICS)
        res = {**res1, **res2}
        results_gbest_fit.append(res)

        ## Save convergence curve
        list_curve_trial.append(model.loss_train)

        ## Save results for box plot
        res = {"Model": my_model["model_name"], **res2}
        results_gbest_box.append(res)

    ## Save convergence curve
    results_curve[my_model["model_name"]] = list_curve_trial

## Save results for statistic test
df = pd.DataFrame(data=results_gbest_fit)
df.to_csv(f"{Config.PATH_SAVE}/{DATA_NAME}/{Config.STATISTIC_FILE_NAME}", index=False)

## Plot each convergence of all models for each trial
for idx_trial, seed in enumerate(Config.LIST_SEEDS):
    draw_convergence_curve_for_each_trial(Config.LIST_MODELS, results_curve, DATA_NAME, idx_trial, fig_size=Config.FIGURE_SIZE,
                                          save_file=True, save_file_path=f"{Config.PATH_SAVE}/{DATA_NAME}")

# Plot average convergence of these models together for each function
draw_average_convergence_curve(Config.LIST_MODELS, results_curve, DATA_NAME, fig_size=Config.FIGURE_SIZE, save_file=True, save_file_path=f"{Config.PATH_SAVE}/{DATA_NAME}")

# Plot the stability chart using seaborn boxplot for each function with different colors for each model
df_gbest_box = pd.DataFrame(results_gbest_box)
for metric in Config.LIST_METRICS:
    df_box = df_gbest_box[["Model", metric]]
    draw_stability_chart(df_box, DATA_NAME, metric, fig_size=Config.FIGURE_SIZE, save_file=True, save_file_path=f"{Config.PATH_SAVE}/{DATA_NAME}")
