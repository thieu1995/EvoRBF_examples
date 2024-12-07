#!/usr/bin/env python
# Created by "Thieu" at 16:15, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from evorbf import NiaRbfClassifier, DataTransformer
from config import Config
from data_utils import get_letter_recognition


## Load data object
# 20000 samples, 16 features, 26 classes
# (20000, 16) - after pre-processing
X_train, X_test, y_train, y_test = get_letter_recognition(f"{Config.PATH_READ}/letter_recognition.csv", verbose=False)
# Specific parameters
DATA_NAME = "letter"
SIZE_HIDDEN = 33
CENTER_FINDER = "kmeans"
Path(f"{Config.PATH_SAVE}/{DATA_NAME}").mkdir(parents=True, exist_ok=True)

## Scaling dataset
dt = DataTransformer(scaling_methods=("minmax",))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)
data = (X_train_scaled, X_test_scaled, y_train, y_test)


# Function to train, test, and evaluate a model for a single seed
def run_trial(opt, seed, data):
    X_train, X_test, y_train, y_test = data

    # Initialize model
    model = NiaRbfClassifier(size_hidden=SIZE_HIDDEN, center_finder=CENTER_FINDER,
                             regularization=True, obj_name=Config.OBJ_CLS,
                             optim=opt["class"], optim_paras=opt["paras"], verbose=False, seed=seed)
    # Train the model
    model.fit(X=X_train, y=y_train)

    # Collect epoch-wise training loss
    res_epoch_loss = [{"model_name": opt["name"], "seed": seed, "epoch": epoch + 1, "loss": loss}
                      for epoch, loss in enumerate(model.loss_train)]

    # Predict and evaluate
    y_pred = model.predict(X_test)
    res = model.evaluate(y_test, y_pred, list_metrics=Config.LIST_METRIC_CLS)
    res_predict = {"model_name": opt["name"], "seed": seed, **res}

    return res_epoch_loss, res_predict


if __name__ == "__main__":
    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []

    with ProcessPoolExecutor(max_workers=Config.N_WORKERS) as executor:
        futures = []
        for opt in Config.LIST_MODELS:
            for seed in Config.LIST_SEEDS:
                futures.append(executor.submit(run_trial, opt, seed, data))

        # Collect results as they complete
        for future in futures:
            res_epoch_loss, res_predict = future.result()
            all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
            all_results.append(res_predict)  # Add evaluation result for this trial

    # Create DataFrames with headers
    df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed

    # Save DataFrames to CSV with headers
    df_loss.to_csv(f"{Config.PATH_SAVE}/{DATA_NAME}/df_loss.csv", index=False, header=True)
    df_result.to_csv(f"{Config.PATH_SAVE}/{DATA_NAME}/df_result.csv", index=False, header=True)

    print(f"Done with data: {DATA_NAME}.")
