import shutil
import os
import random
import numpy as np
import traceback
import torch
import pandas as pd
import optuna
import platform
import sys
import importlib
import yaml
from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mape, mae, rmse, mse, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.callbacks import TFMProgressBar
import plotly.graph_objects as go
from pytorch_lightning import loggers as pl_loggers
import yaml

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamically import utils
utils = importlib.import_module('utils')

# Load functions from utils file
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
        config_path = os.path.join(
            os.path.dirname(__file__), "config_TFT.yml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set n_jobs to the number of available CPU cores if not defined
    if config.get("n_jobs") is None or config["n_jobs"] == -1:
        config["n_jobs"] = os.cpu_count()

    return config


def train_best_model(best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs, devices, output_path, lag_suffix, optuna_trials):
    tb_logger = create_logger(best_model=True)
    dropout = best_params.get('dropout', 0.0)

    best_model = TFTModel(
        input_chunk_length=best_params['input_chunk_length'],
        output_chunk_length=1,
        hidden_size=best_params['hidden_dim'],
        lstm_layers=best_params['n_layers'],
        dropout=dropout,
        batch_size=best_params['batch_size'],
        n_epochs=best_model_epochs,
        optimizer_kwargs={'lr': best_params['learning_rate']},
        random_state=42,
        pl_trainer_kwargs={
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'log_every_n_steps': 20,
        },
    )

    try:
        best_model.fit(series_train_scaled,
                       future_covariates=future_covariates_train_scaled, verbose=True)
        # Save the best model to the specified output path
        model_save_path = os.path.join(
            output_path, f'best_tft_model_epochs_{best_model_epochs}_{optuna_trials}{lag_suffix}.pth')
        best_model.save(model_save_path)
        print(f"Best model saved at: {model_save_path}")
    except RuntimeError as e:
        print(f"RuntimeError during training: {e}")
    except Exception as e:
        print(f"Exception during training: {e}")
        traceback.print_exc()

    return best_model


def objective(trial, series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices, early_stop_callback):
    # Define hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 50, 512)
    n_layers = trial.suggest_int('n_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    input_chunk_length = trial.suggest_int('input_chunk_length', 10, 150)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    tb_logger = create_logger(trial.number)

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=1,
        hidden_size=hidden_dim,
        lstm_layers=n_layers,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=optuna_epochs,
        optimizer_kwargs={'lr': learning_rate},
        random_state=42,
        pl_trainer_kwargs={
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
            'log_every_n_steps': 5,
        }
    )

    try:
        # Fit model
        model.fit(series_train_scaled, future_covariates=future_covariates_train_scaled, verbose=False)

        # Forecast and calculate error metrics
        n = len(series_test_scaled)
        forecast_val = model.predict(n=n, future_covariates=future_covariates_for_prediction_scaled)

        # Calculate error metrics
        rmse_val = rmse(series_test_scaled, forecast_val)
        mape_val = mape(series_test_scaled, forecast_val)
        mae_val = mae(series_test_scaled, forecast_val)
        mse_val = mse(series_test_scaled, forecast_val)
        smape_val = smape(series_test_scaled, forecast_val)

        # Save metrics in user attributes for later retrieval
        trial.set_user_attr("rmse", rmse_val)
        trial.set_user_attr("mape", mape_val)
        trial.set_user_attr("mae", mae_val)
        trial.set_user_attr("mse", mse_val)
        trial.set_user_attr("smape", smape_val)

        # Print metrics for each trial
        print(f"Trial {trial.number} - RMSE: {rmse_val}, MAPE: {mape_val}, MAE: {mae_val}, MSE: {mse_val}, SMAPE: {smape_val}")

        # If metrics have NaN or Inf, mark trial as failed
        if torch.isnan(torch.tensor(rmse_val)) or torch.isinf(torch.tensor(rmse_val)):
            print(f"NaN or Inf detected in RMSE during trial {trial.number}")
            return float('inf')

        return rmse_val  # Minimize RMSE

    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')



def run_optuna_optimization(series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs, devices, early_stop_callback):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, series_train_scaled, future_covariates_train_scaled,
                                           series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices, early_stop_callback), n_trials=optuna_trials)

    best_params = study.best_params
    return best_params, study


def inspect_best_trial(study):
    """
    Inspect and print the error metrics from the best trial in the Optuna study.
    """
    best_trial = study.best_trial
    print(f"Best Trial {best_trial.number}:")
    print(f"  RMSE: {best_trial.user_attrs.get('rmse', 'N/A')}")
    print(f"  MAPE: {best_trial.user_attrs.get('mape', 'N/A')}%")
    print(f"  MAE: {best_trial.user_attrs.get('mae', 'N/A')}")
    print(f"  MSE: {best_trial.user_attrs.get('mse', 'N/A')}")
    print(f"  SMAPE: {best_trial.user_attrs.get('smape', 'N/A')}")



# Main execution block
if __name__ == "__main__":
    # Load configuration parameters
    config = load_config()
    lag_suffix = "_with_lags" if config["lags"] else "_no_lags"

    # Detect if on Mac or Linux and adjust base path and temp path accordingly
    if platform.system() == "Darwin":  # macOS
        base_path = os.path.expanduser(
            "~/Documents/Masterarbeit/Prediction_of_energy_prices/")
        output_path = os.path.join(base_path, 'predictions/TFT/')
    else:  # Assuming Linux for the cluster
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'
        # Temporary save location
        output_path = os.path.join(
            os.getenv('TMPDIR', '/tmp'), 'predictions/TFT/')

    # Ensure output path is created in TMPDIR on Linux or base path on macOS
    os.makedirs(output_path, exist_ok=True)

    set_random_seed(42)

    early_stop_callback = EarlyStopping(
        monitor='train_loss', patience=60, verbose=True)

    # Use the correct columns depending on whether lags are used
    if config["lags"] == True:
        future_covariates_columns = utils.future_covariates_columns
        train_df = utils.load_and_prepare_data(
            os.path.join(base_path, 'data/Final_data/train_df.csv')
        )
        test_df = utils.load_and_prepare_data(
            os.path.join(base_path, 'data/Final_data/test_df.csv')
        )

    else:
        future_covariates_columns = utils.future_covariates_columns_2
        train_df = utils.load_and_prepare_data(
            os.path.join(base_path, 'data/Final_data/train_df_no_lags.csv')
        )
        test_df = utils.load_and_prepare_data(
            os.path.join(base_path, 'data/Final_data/test_df_no_lags.csv')
        )

    # Extract parameters from the configuration
    optuna_epochs = config["optuna_epochs"]
    optuna_trials = config["optuna_trials"]
    best_model_epochs = config["best_model_epochs"]
    max_input_chunk_length = config["max_input_chunk_length"]
    devices = config["devices"]

    # Prepare time series
    series_train, series_test, future_covariates_train, future_covariates_for_prediction = utils.prepare_time_series(
        train_df, test_df, future_covariates_columns, max_input_chunk_length)

    # Scale data
    series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series = utils.scale_data(
        series_train, series_test, future_covariates_train, future_covariates_for_prediction)

    # Run Optuna optimization and get the study and best parameters
    best_params, study = run_optuna_optimization(
        series_train_scaled, future_covariates_train_scaled, series_test_scaled,
        future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs, devices, early_stop_callback
    )

    # Handle TMPDIR and output paths
    if platform.system() != "Darwin":  # Linux cluster
        tmpdir_path = os.getenv('TMPDIR', '/tmp')
        if not os.access(tmpdir_path, os.W_OK):  # Check if TMPDIR is writable
            tmpdir_path = os.path.join(base_path, 'temp/')
        os.makedirs(tmpdir_path, exist_ok=True)
    else:
        tmpdir_path = os.path.join(base_path, 'temp/')
        os.makedirs(tmpdir_path, exist_ok=True)

    output_path = os.path.join(base_path, 'predictions/TFT/')
    os.makedirs(output_path, exist_ok=True)

    # Train the best model
    best_model = train_best_model(
        best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs, devices, output_path, lag_suffix, optuna_trials
    )

    # Make predictions
    n = len(series_test_scaled)
    forecast = best_model.predict(
        n=n, future_covariates=future_covariates_for_prediction_scaled
    )
    forecast = scaler_series.inverse_transform(forecast)

    inspect_best_trial(study)

    # Plot and save results
    fig = utils.plot_forecast(series_test, forecast,
                              title="TFT Model forecast")
    forecast_plot_path, forecast_csv_path, metrics_csv_path = utils.save_results(
        forecast, series_test_scaled, scaler_series, fig, optuna_epochs, optuna_trials, output_path, lag_suffix
    )

    # Copy results from TMPDIR to the home directory (only for non-macOS)
    if platform.system() != "Darwin":
        tmp_file_paths = [forecast_plot_path,
                          forecast_csv_path, metrics_csv_path]
        try:
            copy_results_to_home(tmp_file_paths, os.path.join(
                base_path, 'predictions/TFT/'))
            print(f"Results copied to {base_path}/predictions/TFT/")
        except Exception as e:
            print(
                f"Failed to copy results to {base_path}/predictions/TFT/: {e}")

    # Print best hyperparameters at the end
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')
