#!/usr/bin/env python
# Created by "Thieu" at 15:54, 07/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from evorbf import NiaRbfClassifier, DataTransformer
from config import Config
from data_utils import get_bankruptcy


def run_trial(opt, seed, data, cf):
    # Function to train, test, and evaluate a model for a single seed
    X_train, X_test, y_train, y_test = data

    # Initialize model
    model = NiaRbfClassifier(size_hidden=cf.DATA02['size_hidden'], center_finder=cf.DATA02['center_finder'],
                             regularization=True, obj_name=cf.OBJ_CLS,
                             optim=opt["class"], optim_paras=opt["paras"], verbose=False, seed=seed)
    # Train the model
    model.fit(X=X_train, y=y_train)

    # Collect epoch-wise training loss
    res_epoch_loss = [{"model_name": opt["name"], "seed": seed, "epoch": epoch + 1, "loss": loss}
                      for epoch, loss in enumerate(model.loss_train)]

    # Predict and evaluate
    y_pred = model.predict(X_test)
    res = model.evaluate(y_test, y_pred, list_metrics=cf.LIST_METRIC_CLS)
    res_predict = {"model_name": opt["name"], "seed": seed, **res}

    return res_epoch_loss, res_predict


if __name__ == "__main__":
    ## Load data object
    # 6819 samples, 96 features, 2 classes
    # (6819, 10) - after pre-processing
    Path(f"{Config.PATH_SAVE}/{Config.DATA02['name']}").mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = get_bankruptcy(f"{Config.PATH_READ}/{Config.DATA02['name']}.csv", verbose=False)
    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)
    data = (X_train_scaled, X_test_scaled, y_train, y_test)

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []

    ## Run parallel ==============================================================
    with ProcessPoolExecutor(max_workers=Config.N_WORKERS) as executor:
        futures = []
        for opt in Config.LIST_MODELS:
            for seed in Config.LIST_SEEDS:
                futures.append(executor.submit(run_trial, opt, seed, data, Config))

        # Collect results as they complete
        for future in as_completed(futures):
            res_epoch_loss, res_predict = future.result()
            all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
            all_results.append(res_predict)  # Add evaluation result for this trial

    ## Run sequential =============================================================
    # for opt in Config.LIST_MODELS:
    #     for seed in Config.LIST_SEEDS:
    #         res_epoch_loss, res_predict = run_trial(opt, seed, data, Config)
    #         all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
    #         all_results.append(res_predict)  # Add evaluation result for this trial

    # Create DataFrames with headers
    df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed

    # Save DataFrames to CSV with headers
    df_loss.to_csv(f"{Config.PATH_SAVE}/{Config.DATA02['name']}/{Config.FILE_LOSS}", index=False, header=True)
    df_result.to_csv(f"{Config.PATH_SAVE}/{Config.DATA02['name']}/{Config.FILE_RESULT}", index=False, header=True)

    print(f"Done with data: {Config.DATA02['name']}")
