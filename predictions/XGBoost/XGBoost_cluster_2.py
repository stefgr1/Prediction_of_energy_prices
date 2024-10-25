import os
import yaml
import pandas as pd
import optuna
import shutil
import platform
import torch
from darts import TimeSeries
from darts.models import XGBModel
from darts.metrics import mape, rmse, mse, mae
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
import plotly.graph_objs as go
import sys
import importlib
from pytorch_lightning.callbacks import EarlyStopping

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamically import utils from the parent directory
utils = importlib.import_module('utils')

# Load functions from utils file
future_covariates_columns = utils.future_covariates_columns
check_cuda_availability = utils.check_cuda_availability
set_random_seed = utils.set_random_seed
create_early_stopping_callback = utils.create_early_stopping_callback
create_scalers = utils.create_scalers
load_and_prepare_data = utils.load_and_prepare_data
prepare_time_series = utils.prepare_time_series
scale_data = utils.scale_data
create_logger = utils.create_logger
save_results = utils.save_results
copy_results_to_home = utils.copy_results_to_home
plot_forecast = utils.plot_forecast


def load_config(config_path=None):
    # Default to using config.yaml in the current script's directory
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_xg_boost.yml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set n_jobs to the number of available CPU cores if not defined
    if config.get("n_jobs") is None or config["n_jobs"] == -1:
        config["n_jobs"] = os.cpu_count()

    return config


def main():
    # Load configuration parameters
    config = load_config()

    # Detect if on Mac or Linux and adjust base path accordingly
    if platform.system() == "Darwin":  # macOS
        base_path = os.path.expanduser(
            "~/Documents/Masterarbeit/Prediction_of_energy_prices/")
    else:  # Assuming Linux for the cluster
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'

    tmp_dir = os.path.join(os.getenv('TMPDIR'), 'predictions/XGBoost')
    home_results_dir = os.path.join(base_path, 'predictions/XGBoost/')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Load data
    train_df = utils.load_and_prepare_data(
        os.path.join(base_path, 'data/Final_data/train_df.csv'))
    test_df = utils.load_and_prepare_data(
        os.path.join(base_path, 'data/Final_data/test_df.csv'))

    # Prepare time series
    series_train, series_test, future_covariates_train, future_covariates_test = utils.prepare_time_series(
        train_df, test_df, future_covariates_columns, config["max_input_chunk_length"])

    # Scale data
    series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_test_scaled, scaler_series = utils.scale_data(
        series_train, series_test, future_covariates_train, future_covariates_test)

    # Your original code continues here
    def objective(trial):
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=os.path.join(tmp_dir, f"XGBoost_Optuna_{trial.number}")
        )

        # Suggest hyperparameters
        max_depth = trial.suggest_int('max_depth', 2, 8)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        input_chunk_length = trial.suggest_int('input_chunk_length', 7, 30)
        min_child_weight = trial.suggest_float('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.7, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0)
        gamma = trial.suggest_float('gamma', 0, 2)

        model = XGBModel(
            lags=input_chunk_length,
            output_chunk_length=1,
            lags_future_covariates=[0],
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=42,
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor="train_loss", patience=20, mode="min")
        model.fit(series_train_scaled, future_covariates=future_covariates_train_scaled,
                  verbose=False, callbacks=[early_stop])
        forecast_scaled = model.predict(
            n=len(series_test_scaled), future_covariates=future_covariates_test_scaled)
        forecast = scaler_series.inverse_transform(forecast_scaled)
        error = rmse(series_test, forecast)

        tb_logger.log_metrics({"rmse": float(error)}, step=trial.number)
        return error


    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(
        objective, n_trials=config["optuna_trials"], n_jobs=config["n_jobs"])

    best_params = study.best_params

    # Create a new logger for the final model training with best parameters
    final_tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=tmp_dir, name="XGBoost_Best_Model")

    # Train model with best hyperparameters and TensorBoard logging
    best_model = XGBModel(
        lags=best_params['input_chunk_length'],
        output_chunk_length=1,
        lags_future_covariates=[0],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        random_state=42,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    best_model.fit(series_train_scaled,
                   future_covariates=future_covariates_train_scaled)
    forecast_scaled = best_model.predict(
        n=len(series_test_scaled), future_covariates=future_covariates_test_scaled)
    forecast = scaler_series.inverse_transform(forecast_scaled)

    # Debugging: Print forecast values after scaling
    print("First few values of forecast_scaled:",
          forecast_scaled.values()[:5].squeeze())
    print("First few values of forecast after inverse transform:",
          forecast.values()[:5].squeeze())
    print("First few values of test series for comparison:",
          series_test.values()[:5].squeeze())

    # Plot forecast using the utility function
    fig = utils.plot_forecast(series_test, forecast,
                              title="XGBoost Model Forecast")

    # Save forecast plot and metrics
    forecast_plot_path, forecast_csv_path, metrics_csv_path = utils.save_results(
        forecast, series_test, scaler_series, fig, optuna_epochs=config["optuna_epochs"], model_name="XGBoost")

    # Copy results to the home directory, including TensorBoard logs
    final_tb_log_dir = final_tb_logger.log_dir
    tmp_files = [forecast_plot_path, forecast_csv_path, metrics_csv_path]
    if os.path.exists(final_tb_log_dir):
        tmp_files.append(final_tb_log_dir)

    utils.copy_results_to_home(tmp_files, home_results_dir)

    print(f"Results copied to {home_results_dir}")


if __name__ == "__main__":
    main()
