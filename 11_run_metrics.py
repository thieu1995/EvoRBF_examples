#!/usr/bin/env python
# Created by "Thieu" at 18:00, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from config import Config


def get_metrics(data_name, path_read):

    df = pd.read_csv(f"{path_read}/{data_name}/df_result.csv")

    # Group by 'model_name' and calculate the mean and standard deviation for each metric
    result_df = df.groupby("model_name").agg(["mean", "std"])

    # Save the results to a CSV file
    result_df.to_csv(f"{path_read}/{data_name}/df_metrics_summary.csv")


get_metrics(data_name="bank_marketing", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="bankruptcy", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="car", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="letter", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="mushroom", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="rice", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="superconductivity", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="ccpp", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="concrete", path_read=f"{Config.PATH_SAVE}")
get_metrics(data_name="air_quality", path_read=f"{Config.PATH_SAVE}")
