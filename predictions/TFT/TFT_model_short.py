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
from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mape, mae, rmse, mse
from darts.dataprocessing.transformers import Scaler
from darts.utils.callbacks import TFMProgressBar
import plotly.graph_objects as go
from pytorch_lightning import loggers as pl_loggers

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamically import utils
utils = importlib.import_module('utils')

# Now you can access functions from utils like this:
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

def train_best_model(best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs, devices):
    """
    Train the TFT model with the best hyperparameters found by Optuna.
    """
    best_training_length = max(500, best_params['input_chunk_length'])

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

    best_model.fit(series_train_scaled,
                   future_covariates=future_covariates_train_scaled, verbose=True)

    return best_model


def objective(trial, series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices):
    """
    Optuna objective function for TFT model hyperparameter tuning.
    """
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    n_layers = trial.suggest_int('n_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    input_chunk_length = trial.suggest_int('input_chunk_length', 10, 500)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    if series_train_scaled.pd_dataframe().isnull().values.any() or future_covariates_train_scaled.pd_dataframe().isnull().values.any():
        print(f"NaN values detected in the input data during trial {trial.number}")
        return float('inf')

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
            'log_every_n_steps': 20,
        }
    )

    try:
        model.fit(series_train_scaled, future_covariates=future_covariates_train_scaled, verbose=False)

        n = len(series_test_scaled)
        forecast_val = model.predict(n=n, future_covariates=future_covariates_for_prediction_scaled)
        error = rmse(series_test_scaled, forecast_val)

        if torch.isnan(torch.tensor(error)) or torch.isinf(torch.tensor(error)):
            print(f"NaN or Inf detected in RMSE during trial {trial.number}")
            return float('inf')

    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')

    return error


def run_optuna_optimization(series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs, devices):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, series_train_scaled, future_covariates_train_scaled,
                                           series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices), n_trials=optuna_trials)

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


# Main execution block
if __name__ == "__main__":
    # Detect if on Mac or Linux and adjust base path accordingly
    if platform.system() == "Darwin":  # macOS
        base_path = os.path.expanduser("~/Documents/Masterarbeit/Prediction_of_energy_prices/")
    else:  # Assuming Linux for the cluster
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'

    set_random_seed(42)

    early_stop_callback = EarlyStopping(monitor='train_loss', patience=50, verbose=True)

    # Load in the train and test data
    train_df = load_and_prepare_data(os.path.join(base_path, 'data/Final_data/train_df.csv'))
    test_df = load_and_prepare_data(os.path.join(base_path, 'data/Final_data/test_df.csv'))

    # Define future covariates
    future_covariates_columns = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)',
       'Temperature (°C)', 'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)',
       'Lignite (GWh)', 'Natural_gas (GWh)', 'Other (GWh)',
       'Pumped_storage_generation (GWh)', 'Solar_energy (GWh)',
       'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
       'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
       'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Lag_1_day',
       'Lag_2_days', 'Lag_3_days', 'Lag_4_days', 'Lag_5_days', 'Lag_6_days',
       'Lag_7_days', 'Day_of_week', 'Month', 'Rolling_mean_7']

     # Customizable parameters
    optuna_epochs = 51  # Define the number of epochs for Optuna trials
    optuna_trials = 500  # Define the number of trials for Optuna
    best_model_epochs = 51  # Define the number of epochs for the best model

    MAX_INPUT_CHUNK_LENGTH = 500
    DEVICES = 4

    # Prepare time series
    series_train, series_test, future_covariates_train, future_covariates_for_prediction = prepare_time_series(
        train_df, test_df, future_covariates_columns, MAX_INPUT_CHUNK_LENGTH)

    # Scale data
    series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series = scale_data(
        series_train, series_test, future_covariates_train, future_covariates_for_prediction)

    # Run Optuna optimization and get the study and best parameters
    best_params, study = run_optuna_optimization(
        series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs, DEVICES)

    # Train the best model
    best_model = train_best_model(
        best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs, DEVICES)

        # Ensure the directory exists before copying
    tmpdir_path = os.path.join(os.getenv('TMPDIR'), 'predictions/TFT/')
    if os.path.exists(tmpdir_path):
        shutil.copytree(tmpdir_path, home_results_dir, dirs_exist_ok=True)
        print(f"Results copied to {home_results_dir}")
    else:
        print(f"Directory {tmpdir_path} does not exist. Skipping copy operation.")

    os.makedirs(tmpdir_path, exist_ok=True)

    # Save the best model
    model_save_path = os.path.join(tmpdir_path, f'best_tft_model_epochs_{optuna_epochs}.pth')
    best_model.save(model_save_path)
    print(f"Best model saved at: {model_save_path}")

    # Make predictions
    n = len(series_test_scaled)
    forecast = best_model.predict(n=n, future_covariates=future_covariates_for_prediction_scaled)

    forecast = scaler_series.inverse_transform(forecast)

    inspect_best_trial(study)

    # Plot and save results
    fig = plot_forecast(series_test, forecast, title="TFT Model forecast")
    save_results(forecast, series_test_scaled, scaler_series, fig, optuna_epochs)

    # Copy results from TMPDIR to the home directory
    home_results_dir = os.path.join(base_path, 'predictions/TFT/')
    shutil.copytree(os.path.join(os.getenv('TMPDIR'), 'predictions/TFT/'), home_results_dir, dirs_exist_ok=True)
    print(f"Results copied to {home_results_dir}")

    # ** Print best hyperparameters at the end **
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')
