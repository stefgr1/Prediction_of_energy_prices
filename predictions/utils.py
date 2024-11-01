import os
import random
import numpy as np
import torch
import pandas as pd
import platform
import shutil
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from sklearn.preprocessing import MaxAbsScaler
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import mape, mae, rmse, mse, smape
from darts.dataprocessing.transformers import Scaler
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


def create_early_stopping_callback(patience=20):
    """
    Create and return an EarlyStopping callback.
    """
    return EarlyStopping(
        monitor='train_loss', patience=patience, verbose=True
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


def prepare_time_series(df_train, df_test, covariates_columns, max_input_chunk_length):
    """
    Prepare time series objects for training and testing, including future covariates.

    """
    series_train = TimeSeries.from_dataframe(
        df_train, 'Date', 'Day_ahead_price (€/MWh)').astype('float32')
    series_test = TimeSeries.from_dataframe(
        df_test, 'Date', 'Day_ahead_price (€/MWh)').astype('float32')
    future_covariates_train = TimeSeries.from_dataframe(
        df_train, 'Date', covariates_columns).astype('float32')

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
    future_covariates_train_scaled = scaler_covariates.fit_transform(
        future_covariates_train)

    series_test_scaled = scaler_series.transform(series_test)
    future_covariates_for_prediction_scaled = scaler_covariates.transform(
        future_covariates_for_prediction)

    return series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series


def create_logger(trial_number=None, best_model=False, model_name='Model'):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Use TMPDIR if available, else fallback to a default temporary directory
    base_log_dir = os.getenv('TMPDIR', '/tmp')

    # If training the best model, create a separate directory
    if best_model:
        log_dir = f'{base_log_dir}/best_model/{timestamp}_best_model_{random.randint(0, 1000)}'
    else:
        log_dir = f'{base_log_dir}/{timestamp}_trial_{trial_number}_{random.randint(0, 1000)}'

    # Ensure the directory is created
    os.makedirs(log_dir, exist_ok=True)

    return pl_loggers.TensorBoardLogger(save_dir=log_dir, name=model_name, default_hp_metric=False)


def save_results(forecast, test_series, scaler_series, fig, optuna_epochs, output_path, lag_suffix, model_name="Model"):

    """
    Save forecast results, error metrics, and plot to files.
    """
    test_series = scaler_series.inverse_transform(test_series)

    # Print metrics for logging purposes
    print('Error Metrics on Test Set:')
    print(f'  MAE: {mae(test_series, forecast):.2f}')
    print(f'  RMSE: {rmse(test_series, forecast):.2f}')
    print(f'  MSE: {mse(test_series, forecast):.2f}')
    print(f'  MAPE: {mape(test_series, forecast):.2f}%')
    print(f'  SMAPE: {smape(test_series, forecast):.2f}%')

    # Define output paths in the temporary directory first
    forecast_plot_path = os.path.join(
        output_path, f'{model_name}_forecast_epochs_{optuna_epochs}{lag_suffix}.png')
    forecast_csv_path = os.path.join(
        output_path, f'{model_name}_forecast_epochs_{optuna_epochs}{lag_suffix}.csv')
    metrics_csv_path = os.path.join(
        output_path, f'{model_name}_metrics_epochs_{optuna_epochs}{lag_suffix}.csv')

    # Ensure the directory exists before saving
    os.makedirs(output_path, exist_ok=True)

    # Save the plot, forecast data, and metrics
    fig.write_image(forecast_plot_path)
    error_metrics = pd.DataFrame({'MAE': [mae(test_series, forecast)], 'MSE': [mse(test_series, forecast)], 
                                  'RMSE': [rmse(test_series, forecast)], 'MAPE': [mape(test_series, forecast)], 
                                  'SMAPE': [smape(test_series, forecast)]})
    error_metrics.to_csv(metrics_csv_path, index=False)

    forecast_df = pd.DataFrame({
        'Date': forecast.time_index,  # Add time index (dates)
        'Forecast': forecast.values().squeeze()  # Add forecasted values
    })
    forecast_df.to_csv(forecast_csv_path, index=False)

    # Return paths for copying or verification
    return forecast_plot_path, forecast_csv_path, metrics_csv_path



def copy_results_to_home(tmp_file_paths, home_dir_path):
    """
    Copy files and directories from TMPDIR to the home directory.
    """
    for tmp_file_path in tmp_file_paths:
        home_file_path = os.path.join(
            home_dir_path, os.path.basename(tmp_file_path))

        # Check if it is a file or a directory
        if os.path.isfile(tmp_file_path):
            shutil.copy(tmp_file_path, home_file_path)
            print(f"Copied {tmp_file_path} to {home_file_path}")
        elif os.path.isdir(tmp_file_path):
            # Use copytree for directories
            shutil.copytree(tmp_file_path, home_file_path, dirs_exist_ok=True)
            print(f"Copied directory {tmp_file_path} to {home_file_path}")
        else:
            print(f"{tmp_file_path} is neither a file nor a directory. Skipping.")


def plot_forecast(test_series, forecast, title):
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
        title=title,
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

future_covariates_columns_2 = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)',
                               'Temperature (°C)', 'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)',
                               'Lignite (GWh)', 'Natural_gas (GWh)', 'Other (GWh)',
                               'Pumped_storage_generation (GWh)', 'Solar_energy (GWh)',
                               'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
                               'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
                               'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Day_of_week', 'Month']
