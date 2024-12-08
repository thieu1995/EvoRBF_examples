#!/usr/bin/env python
# Created by "Thieu" at 18:06, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import Config


def get_figure(data_name, metrics, path_read, path_save, exts=(".png", ), verbose=False):
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f"{path_read}/{data_name}/{Config.FILE_RESULT}")

    # # Define a list of color palettes for each metric
    # color_palettes = ["Set2", "coolwarm", "Spectral", "cubehelix", "viridis", "Accent"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="model_name", y=metric, palette="Spectral", hue="model_name")
        plt.title(f"Boxplot of {metric} metric across models on {data_name} dataset", fontsize=17)
        plt.xlabel("Models", fontsize=16)
        plt.ylabel(metric, fontsize=16)
        plt.xticks(rotation=45, fontsize=14, ha="right")
        plt.yticks(fontsize=14)

        for ext in exts:
            plt.savefig(f"{path_save}/{data_name}-{metric}{ext}", bbox_inches="tight")
        if verbose:
            plt.show()


path_save = f"{Config.PATH_SAVE}/visual/boxplot"
get_figure(data_name="bank_marketing", metrics=Config.LIST_METRIC_CLS, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="bankruptcy", metrics=Config.LIST_METRIC_CLS, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="car", metrics=Config.LIST_METRIC_CLS, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="letter", metrics=Config.LIST_METRIC_CLS, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="mushroom", metrics=Config.LIST_METRIC_CLS, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="rice", metrics=Config.LIST_METRIC_CLS, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="superconductivity", metrics=Config.LIST_METRIC_REG, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="ccpp", metrics=Config.LIST_METRIC_REG, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="concrete", metrics=Config.LIST_METRIC_REG, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
get_figure(data_name="air_quality", metrics=Config.LIST_METRIC_REG, path_read=f"{Config.PATH_SAVE}", path_save=path_save)
