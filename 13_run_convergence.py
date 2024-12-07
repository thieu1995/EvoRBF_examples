#!/usr/bin/env python
# Created by "Thieu" at 18:10, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import Config


def get_figure(data_name, path_read, path_save, exts=(".png", ), verbose=False):
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    # Calculate the average loss for each model over epochs
    df = pd.read_csv(f"{path_read}/{data_name}/df_loss.csv")

    # # Define a list of color palettes for each metric
    # color_palettes = ["Set2", "coolwarm", "Spectral", "cubehelix", "viridis", "Accent" ,"Dark2"]
    average_loss = df.groupby(["model_name", "epoch"])["loss"].mean().reset_index()

    # Plot the convergence chart
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=average_loss, x="epoch", y="loss", hue="model_name", linewidth=2, palette="tab20")
    plt.title(f"Convergence Chart of Average Fitness Value on {data_name} dataset.", fontsize=17)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Average Fitness Value", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(title="Model", fontsize=12, title_fontsize=14)
    plt.legend(title="Model Name", fontsize=12, title_fontsize=14, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(alpha=0.3)

    for ext in exts:
        plt.savefig(f"{path_save}/{data_name}-average{ext}", bbox_inches="tight")
    if verbose:
        plt.show()


path_save = f"{Config.PATH_SAVE}/visual/boxplot"
get_figure(data_name="bank_marketing", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="bankruptcy", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="car", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="letter", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="mushroom", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="rice", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="superconductivity", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="ccpp", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="concrete", path_read=Config.PATH_SAVE, path_save=path_save)
get_figure(data_name="air_quality", path_read=Config.PATH_SAVE, path_save=path_save)
