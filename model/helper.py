#!/usr/bin/env python
# Created by "Thieu" at 15:29, 01/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def draw_convergence_curve_for_each_trial(models, fit_results, data_name, trial_idx, fig_size=(10, 6),
                                          save_file=True, save_file_path="history/convergence", verbose=False):
    Path(save_file_path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=fig_size)
    for idx_model, my_model in enumerate(models):
        plt.plot(fit_results[my_model['model_name']][trial_idx], label=my_model['model_name'], ls="--")
    plt.title(f'Convergence curve of compared algorithms for {data_name} dataset - Trial {trial_idx + 1}')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.legend()
    if verbose:
        plt.show()
    if save_file:
        plt.savefig(f"{save_file_path}/{data_name}_data-trial_{trial_idx}-convergence.png", bbox_inches='tight')
    return None


def draw_average_convergence_curve(models, fit_results, data_name, fig_size=(10, 6),
                                   save_file=True, save_file_path="history/convergence", verbose=False):
    Path(save_file_path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=fig_size)
    for idx_model, my_model in enumerate(models):
        avg_fitness = np.mean(fit_results[my_model['model_name']], axis=0)
        plt.plot(avg_fitness, label=my_model['model_name'], ls="--")
    plt.title(f'Average convergence curve of compared algorithms for {data_name} dataset')
    plt.xlabel('Iterations')
    plt.ylabel('Average fitness Value')
    plt.legend()
    if verbose:
        plt.show()
    if save_file:
        plt.savefig(f"{save_file_path}/{data_name}_data-average-convergence.png", bbox_inches='tight')
    return None


def draw_stability_chart(fit_results, data_name, metric_name, fig_size=(10, 6),
                         save_file=True, save_file_path="history/convergence", verbose=False):
    Path(save_file_path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=fig_size)
    sns.boxplot(x='Model', y=metric_name, data=fit_results, palette='Set2', hue="Model", legend=False)
    plt.title(f'{metric_name} stability chart for {data_name} dataset')
    plt.xlabel('Algorithms')
    plt.ylabel('Global best fitness')
    if verbose:
        plt.show()
    if save_file:
        plt.savefig(f"{save_file_path}/{data_name}_data-{metric_name}-stability-chart.png", bbox_inches='tight')
    return None


def draw_confusion_matrix(y_test, y_pred, model_name, data_name,
                          save_file=True, save_file_path="history/convergence", verbose=False):
    Path(save_file_path).mkdir(parents=True, exist_ok=True)
    ## Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion matrix of {model_name} optimizer for {data_name} dataset")

    # Show the plot
    if verbose:
        plt.show()
    if save_file:
        plt.savefig(f"{save_file_path}/confusion_matrix.png", bbox_inches='tight')
    return None
