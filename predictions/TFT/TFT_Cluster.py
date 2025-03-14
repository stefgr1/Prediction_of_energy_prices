import shutil
import os
import random
import numpy as np
import traceback
import torch
import pandas as pd
import optuna
import platform
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from sklearn.preprocessing import MaxAbsScaler
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mape, mae, rmse, mse
from darts.dataprocessing.transformers import Scaler
from darts.utils.callbacks import TFMProgressBar
import plotly.graph_objects as go
from pytorch_lightning import loggers as pl_loggers

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_random_seed(42)

early_stop_callback = EarlyStopping(
    monitor='train_loss', patience=20, verbose=True
)

scaler_series = Scaler(MaxAbsScaler())
scaler_covariates = Scaler(MaxAbsScaler())

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_time_series(df_train, df_test, covariates_columns):
    series_train = TimeSeries.from_dataframe(
        df_train, 'Date', 'Day_ahead_price (€/MWh)').astype('float32')
    series_test = TimeSeries.from_dataframe(
        df_test, 'Date', 'Day_ahead_price (€/MWh)').astype('float32')
    future_covariates_train = TimeSeries.from_dataframe(
        df_train, 'Date', covariates_columns).astype('float32')

    max_input_chunk_length = 200
    required_covariate_start = series_test.start_time() - pd.DateOffset(days=(max_input_chunk_length - 1))
    required_covariate_end = series_test.end_time()

    future_covariates_full = TimeSeries.from_dataframe(
        pd.concat([df_train, df_test]), 'Date', covariates_columns, fill_missing_dates=True, freq="D"
    ).astype('float32')

    future_covariates_for_prediction = future_covariates_full.slice(
        required_covariate_start, required_covariate_end)

    return series_train, series_test, future_covariates_train, future_covariates_for_prediction

def scale_data(series_train, series_test, future_covariates_train, future_covariates_for_prediction):
    scaler_series = Scaler(MaxAbsScaler())
    scaler_covariates = Scaler(MaxAbsScaler())

    series_train_scaled = scaler_series.fit_transform(series_train)
    future_covariates_train_scaled = scaler_covariates.fit_transform(
        future_covariates_train)

    series_test_scaled = scaler_series.transform(series_test)
    future_covariates_for_prediction_scaled = scaler_covariates.transform(
        future_covariates_for_prediction)

    return series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series

def create_logger(trial_number=None, best_model=False):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base_log_dir = os.getenv('TMPDIR', '/tmp')

    if best_model:
        log_dir = f'{base_log_dir}/TFT_logs/best_model/{timestamp}_best_model'
    else:
        log_dir = f'{base_log_dir}/TFT_logs/{timestamp}_trial_{trial_number}'

    os.makedirs(log_dir, exist_ok=True)
    return pl_loggers.TensorBoardLogger(save_dir=log_dir, name='TFT', default_hp_metric=False)

def plot_forecast(test_series, forecast, plot_path):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test_series.time_index,
        y=test_series.values().squeeze(),
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=forecast.time_index,
        y=forecast.values().squeeze(),
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='TFT Model',
        xaxis_title='Date',
        yaxis_title='Day Ahead Price (€/MWh)',
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            bordercolor='black',
            borderwidth=1
        ),
        template='plotly'
    )

    fig.write_image(plot_path)

def train_best_model(best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs):
    """
    Train the TFT model with the best hyperparameters found by Optuna.
    A separate TensorBoard logger is created for this training.
    """
    best_training_length = max(200, best_params['input_chunk_length'])

    # Create a new logger specifically for the best model training
    tb_logger = create_logger(best_model=True)

    dropout = best_params.get('dropout', 0.0)

    best_model = TFTModel(
        input_chunk_length=best_params['input_chunk_length'],
        output_chunk_length=1,
        hidden_size=best_params['hidden_dim'],
        lstm_layers=best_params['n_layers'],
        dropout=dropout,
        batch_size=best_params['batch_size'],
        n_epochs=best_model_epochs,  # Use best model epochs from main
        optimizer_kwargs={'lr': best_params['learning_rate']},
        random_state=42,
        pl_trainer_kwargs={
            'accelerator': 'cpu',
            'devices': 1,
            'enable_progress_bar': True,
            'logger': tb_logger,  # Pass the separate logger here
            'enable_model_summary': False,
        }
    )

    # Train the best model
    best_model.fit(series_train_scaled,
                   future_covariates=future_covariates_train_scaled, verbose=True)

    return best_model


def objective(trial, series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs):
    """
    Optuna objective function for TFT model hyperparameter tuning.
    """
    hidden_dim = trial.suggest_int('hidden_dim', 16, 64)
    n_layers = trial.suggest_int('n_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    input_chunk_length = trial.suggest_int('input_chunk_length', 30, 200)

    # Check for NaN in data
    if series_train_scaled.pd_dataframe().isnull().values.any() or future_covariates_train_scaled.pd_dataframe().isnull().values.any():
        print(
            f"NaN values detected in the input data during trial {trial.number}")
        return float('inf')

    # Create a new logger for each trial
    tb_logger = create_logger(trial.number)

    # TFT Model initialization
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
            'accelerator': 'cpu',
            'devices': 1,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'gradient_clip_val': 0.5,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
        }
    )

    try:
        # Model fitting
        model.fit(series_train_scaled,
                  future_covariates=future_covariates_train_scaled, verbose=False)

        # Predict and calculate RMSE
        n = len(series_test_scaled)
        forecast_val = model.predict(
            n=n, future_covariates=future_covariates_for_prediction_scaled)

        # Calculate RMSE
        error = rmse(series_test_scaled, forecast_val)

        # Check for NaN/Inf in loss
        if torch.isnan(torch.tensor(error)) or torch.isinf(torch.tensor(error)):
            print(f"NaN or Inf detected in RMSE during trial {trial.number}")
            return float('inf')

    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')

    return error


def run_optuna_optimization(series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs):
    """
    Run Optuna optimization to find the best hyperparameters for TFT.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, series_train_scaled, future_covariates_train_scaled,
                                           series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs), n_trials=optuna_trials)

    best_params = study.best_params
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')

    # Return both the best parameters and the study itself
    return best_params, study

def save_results(forecast, test_series, scaler_series, plot_path, metrics_path):
    test_series = scaler_series.inverse_transform(test_series)

    error_metrics = pd.DataFrame({
        'MAE': [mae(test_series, forecast)],
        'MAPE': [mape(test_series, forecast)],
        'MSE': [mse(test_series, forecast)],
        'RMSE': [rmse(test_series, forecast)]
    })
    
    error_metrics.to_csv(metrics_path, index=False)

if __name__ == "__main__":
    if platform.system() == "Darwin":
        base_path = os.path.expanduser("~/Documents/Masterarbeit/Prediction_of_energy_prices/")
    else:
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'

    train_df = load_and_prepare_data(
        os.path.join(base_path, 'data/Final_data/train_df.csv'))
    test_df = load_and_prepare_data(
        os.path.join(base_path, 'data/Final_data/test_df.csv'))

    future_covariates_columns = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)',
                                 'Temperature (°C)', 'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)',
                                 'Lignite (GWh)', 'Natural_gas (GWh)', 'Other (GWh)',
                                 'Pumped_storage_generation (GWh)', 'Solar_energy (GWh)',
                                 'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
                                 'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
                                 'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Lag_1_day',
                                 'Lag_2_days', 'Lag_3_days', 'Lag_4_days', 'Lag_5_days', 'Lag_6_days',
                                 'Lag_7_days', 'Day_of_week', 'Month', 'Rolling_mean_7']

    series_train, series_test, future_covariates_train, future_covariates_for_prediction = prepare_time_series(
        train_df, test_df, future_covariates_columns)

    series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series = scale_data(
        series_train, series_test, future_covariates_train, future_covariates_for_prediction)

    optuna_epochs = 50
    optuna_trials = 250
    best_model_epochs = 50

    best_params, study = run_optuna_optimization(
        series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs)

    best_trial_number = study.best_trial.number
    best_model = train_best_model(
        best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs)

    n = len(series_test_scaled)
    forecast = best_model.predict(
        n=n, future_covariates=future_covariates_for_prediction_scaled)

    forecast = scaler_series.inverse_transform(forecast)

    tmpdir = os.getenv('TMPDIR', '/tmp')
    plot_path = os.path.join(tmpdir, 'TFT_forecast.png')
    metrics_path = os.path.join(tmpdir, 'TFT_metrics.csv')
    plot_forecast(series_test, forecast, plot_path)
    save_results(forecast, series_test_scaled, scaler_series, plot_path, metrics_path)

    final_plot_path = os.path.join(base_path, f'predictions/TFT/TFT_forecast_{optuna_epochs}.png')
    final_metrics_path = os.path.join(base_path, f'predictions/TFT/TFT_metrics_{optuna_epochs}.csv')
    final_logs_path = os.path.join(base_path, 'predictions/TFT/logs/')
    
    os.makedirs(final_logs_path, exist_ok=True)

    shutil.copy(plot_path, final_plot_path)
    shutil.copy(metrics_path, final_metrics_path)
    shutil.copytree(os.path.join(tmpdir, 'TFT_logs'), final_logs_path, dirs_exist_ok=True)
