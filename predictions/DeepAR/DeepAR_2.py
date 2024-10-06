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
from darts.models import RNNModel
from darts.metrics import mape, mae, rmse, mse
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.callbacks import TFMProgressBar
import plotly.graph_objects as go
from pytorch_lightning import loggers as pl_loggers

def check_cuda_availability():
    """
    Check and print if CUDA is available.
    """
    print(torch.cuda.is_available())

def set_random_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_early_stopping_callback():
    """
    Create and return an EarlyStopping callback.
    """
    return EarlyStopping(
        monitor='train_loss', patience=20, verbose=True
    )

def create_scalers():
    """
    Create and return scaler objects for both the time series and covariates.
    """
    scaler_series = Scaler(MaxAbsScaler())
    scaler_covariates = Scaler(MaxAbsScaler())
    return scaler_series, scaler_covariates

def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime.
    """
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_time_series(df_train, df_test, covariates_columns):
    """
    Prepare time series objects for training and testing, including future covariates.
    
    """
    # Add day of week and month as covariate
    df_train['Day_of_week'] = df_train['Date'].dt.dayofweek
    df_train['Month'] = df_train['Date'].dt.month
    df_test['Day_of_week'] = df_test['Date'].dt.dayofweek
    df_test['Month'] = df_test['Date'].dt.month

    series_train = TimeSeries.from_dataframe(
        df_train, 'Date', 'Day_ahead_price (€/MWh)').astype('float32')
    series_test = TimeSeries.from_dataframe(
        df_test, 'Date', 'Day_ahead_price (€/MWh)').astype('float32')
    future_covariates_train = TimeSeries.from_dataframe(
        df_train, 'Date', covariates_columns).astype('float32')

    max_input_chunk_length = 150  # Adjust based on hyperparameter tuning
    required_covariate_start = series_test.start_time(
    ) - pd.DateOffset(days=(max_input_chunk_length - 1))
    required_covariate_end = series_test.end_time()

    future_covariates_full = TimeSeries.from_dataframe(
        pd.concat([df_train, df_test]), 'Date', covariates_columns, fill_missing_dates=True, freq="D"
    ).astype('float32')

    future_covariates_for_prediction = future_covariates_full.slice(
        required_covariate_start, required_covariate_end)

    return series_train, series_test, future_covariates_train, future_covariates_for_prediction

def scale_data(series_train, series_test, future_covariates_train, future_covariates_for_prediction):
    """
    Scale the time series data and future covariates.
    """
    scaler_series, scaler_covariates = create_scalers()

    series_train_scaled = scaler_series.fit_transform(series_train)
    future_covariates_train_scaled = scaler_covariates.fit_transform(future_covariates_train)

    series_test_scaled = scaler_series.transform(series_test)
    future_covariates_for_prediction_scaled = scaler_covariates.transform(future_covariates_for_prediction)

    return series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series

def create_logger(trial_number=None, best_model=False):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Use TMPDIR if available, else fallback to a default temporary directory
    base_log_dir = os.getenv('TMPDIR', '/tmp')
    
    # If training the best model, create a separate directory
    if best_model:
        log_dir = f'{base_log_dir}/best_model/{timestamp}_best_model'
    else:
        log_dir = f'{base_log_dir}/{timestamp}_trial_{trial_number}'

    # Ensure the directory is created
    os.makedirs(log_dir, exist_ok=True)

    return pl_loggers.TensorBoardLogger(save_dir=log_dir, name='DeepAR', default_hp_metric=False)

def train_best_model(best_params, series_train_scaled, future_covariates_train_scaled, best_model_epochs):
    """
    Train the RNN model with the best hyperparameters found by Optuna.
    A separate TensorBoard logger is created for this training.
    """
    best_training_length = max(500, best_params['input_chunk_length'])

    # Create a new logger specifically for the best model training
    tb_logger = create_logger(best_model=True)

    dropout = best_params.get('dropout', 0.0)

    best_model = RNNModel(
        model='LSTM',
        input_chunk_length=best_params['input_chunk_length'],
        training_length=best_training_length,
        hidden_dim=best_params['hidden_dim'],
        n_rnn_layers=best_params['n_layers'],
        dropout=dropout,
        likelihood=GaussianLikelihood(),
        batch_size=best_params['batch_size'],
        n_epochs=best_model_epochs,
        optimizer_kwargs={'lr': best_params['learning_rate']},
        random_state=42,
        pl_trainer_kwargs={
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu', 
            'devices': 1,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
        }
    )

    # Train the best model
    best_model.fit(series_train_scaled,
                   future_covariates=future_covariates_train_scaled, verbose=True)

    return best_model


def objective(trial, series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs):
    """
    Optuna objective function for RNN model hyperparameter tuning.
    """
    n_layers = trial.suggest_int('n_layers', 1, 5)
    dropout = trial.suggest_float('dropout', 0.0, 0.5) if n_layers > 1 else 0.0
    input_chunk_length = trial.suggest_int('input_chunk_length', 30, 300)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

    training_length = max(500, input_chunk_length)

    # Create a new logger for each trial
    tb_logger = create_logger(trial.number)

    # RNN Model initialization
    model = RNNModel(
        model='LSTM',
        input_chunk_length=input_chunk_length,
        training_length=training_length,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_layers,
        dropout=dropout,
        likelihood=GaussianLikelihood(),
        batch_size=batch_size,
        n_epochs=optuna_epochs,
        optimizer_kwargs={'lr': learning_rate},
        random_state=42,
        save_checkpoints=False,
        model_name="rnn_model",
        force_reset=True,
        pl_trainer_kwargs={
            'accelerator': 'gpu',
            'devices': 2,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
        }
    )

    try:
        # Model fitting
        model.fit(series_train_scaled,
                  future_covariates=future_covariates_train_scaled, verbose=False)

        # Predict and calculate metrics
        n = len(series_test_scaled)
        forecast_val = model.predict(
            n=n, future_covariates=future_covariates_for_prediction_scaled)

        # Calculate error metrics
        rmse_val = rmse(series_test_scaled, forecast_val)
        mape_val = mape(series_test_scaled, forecast_val)
        mae_val = mae(series_test_scaled, forecast_val)
        mse_val = mse(series_test_scaled, forecast_val)

        # Log the error metrics in the trial object
        trial.set_user_attr('rmse', rmse_val)
        trial.set_user_attr('mape', mape_val)
        trial.set_user_attr('mae', mae_val)
        trial.set_user_attr('mse', mse_val)

    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')

    return rmse_val


def run_optuna_optimization(series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs):
    """
    Run Optuna optimization to find the best hyperparameters.
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


def plot_forecast(test_series, forecast):
    """
    Plot the actual vs predicted forecast using Plotly.
    """
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
        title='DeepAR Model',
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

    return fig


def save_results(forecast, test_series, scaler_series, fig, optuna_epochs):
    test_series = scaler_series.inverse_transform(test_series)

    # Use TMPDIR or a fallback directory
    base_path = os.getenv('TMPDIR', '/tmp')

    print('Error Metrics on Test Set:')
    print(f'  MAPE: {mape(test_series, forecast):.2f}%')
    print(f'  MAE: {mae(test_series, forecast):.2f}')
    print(f'  RMSE: {rmse(test_series, forecast):.2f}')
    print(f'  MSE: {mse(test_series, forecast):.2f}')

    # Save the forecast plot and error metrics
    forecast_plot_path = os.path.join(
        base_path, f'predictions/DeepAR/Deep_AR_forecast_epochs_{optuna_epochs}.png')
    forecast_csv_path = os.path.join(
        base_path, f'predictions/DeepAR/Deep_AR_forecast_epochs_{optuna_epochs}.csv')
    metrics_csv_path = os.path.join(
        base_path, f'predictions/DeepAR/Deep_AR_metrics_epochs_{optuna_epochs}.csv')

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(forecast_plot_path), exist_ok=True)
    
    fig.write_image(forecast_plot_path)
    error_metrics = pd.DataFrame({'MAE': [mae(test_series, forecast)], 'MAPE': [mape(test_series, forecast)],
                                  'MSE': [mse(test_series, forecast)], 'RMSE': [rmse(test_series, forecast)]})
    error_metrics.to_csv(metrics_csv_path, index=False)

    forecast.to_csv(forecast_csv_path, index=False)


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

    check_cuda_availability()

    # Detect if on Mac or Linux and adjust base path accordingly
    if platform.system() == "Darwin":  # Check if it's macOS
        base_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/'
    else:  # Assuming Linux for the cluster
        base_path = '/pfs/data5/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/'

    # Set random seed
    set_random_seed(42)

    # Create early stopping callback
    early_stop_callback = create_early_stopping_callback()

    # Load in the train and test data
    train_df = load_and_prepare_data(os.path.join(base_path, 'data/Final_data/train_df.csv'))
    test_df = load_and_prepare_data(os.path.join(base_path, 'data/Final_data/test_df.csv'))

    # Define future covariates
    future_covariates_columns = future_covariates_columns = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)',
       'Temperature (°C)', 'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)',
       'Lignite (GWh)', 'Natural_gas (GWh)', 'Other (GWh)',
       'Pumped_storage_generation (GWh)', 'Solar_energy (GWh)',
       'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
       'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
       'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Lag_1_day',
       'Lag_2_days', 'Lag_3_days', 'Lag_4_days', 'Lag_5_days', 'Lag_6_days',
       'Lag_7_days', 'Day_of_week', 'Month', 'Rolling_mean_7']

    # Prepare time series
    series_train, series_test, future_covariates_train, future_covariates_for_prediction = prepare_time_series(
        train_df, test_df, future_covariates_columns)

    # Scale data
    series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series = scale_data(
        series_train, series_test, future_covariates_train, future_covariates_for_prediction)

    # Parameters for epochs and trials
    OPTUNA_TRIALS = 500  # Define the number of Optuna trials
    OPTUNA_EPOCHS = 100  # Define the number of epochs per Optuna trial
    BEST_MODEL_EPOCHS = 100  # Define the number of epochs for the best model

    # Run Optuna optimization and get the study and best parameters
    best_params, study = run_optuna_optimization(
        series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, OPTUNA_TRIALS, OPTUNA_EPOCHS)

    # Train the best model
    best_model = train_best_model(
        best_params, series_train_scaled, future_covariates_train_scaled, BEST_MODEL_EPOCHS)

    # Use TMPDIR for model saving
    models_dir = os.getenv('TMPDIR', '/tmp') + '/models/DeepAR/'
    os.makedirs(models_dir, exist_ok=True)

    # Save best model to TMPDIR
    model_save_path = os.path.join(models_dir, f'best_deep_ar_model_epochs_{BEST_MODEL_EPOCHS}.pth')
    best_model.save(model_save_path)
    print(f"Best model saved at: {model_save_path}")

    # Make predictions
    n = len(series_test_scaled)
    forecast = best_model.predict(n=n, future_covariates=future_covariates_for_prediction_scaled)
    forecast = scaler_series.inverse_transform(forecast)

    # Inspect the error metrics from best trial in the Optuna study
    inspect_best_trial(study)

    # Plot and save results
    fig = plot_forecast(series_test, forecast)
    save_results(forecast, series_test_scaled, scaler_series, fig, OPTUNA_EPOCHS)
