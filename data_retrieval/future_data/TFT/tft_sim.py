import os
import shutil
import random
import numpy as np
import torch
import traceback
import platform
import pandas as pd
import yaml
import optuna
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
from sklearn.preprocessing import MaxAbsScaler
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mse, mae
from darts.utils.callbacks import TFMProgressBar
import plotly.graph_objects as go
from pytorch_lightning import loggers as pl_loggers

# Function to load configuration from a YAML file
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

# Set the random seed for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_random_seed(42)

# Early stopping callback
early_stop_callback = EarlyStopping(monitor='train_loss', patience=20, verbose=True)

# Create scaler object for series
scaler_series = Scaler(MaxAbsScaler())

def load_and_prepare_data(file_path):
    """Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime."""
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_time_series(df, target_column):
    """Prepare time series object for training."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    series = TimeSeries.from_dataframe(df, 'Date', target_column).astype('float32')
    return series

def scale_data(series):
    """Scale the time series data."""
    scaler_series = Scaler(MaxAbsScaler())
    series_scaled = scaler_series.fit_transform(series)
    return series_scaled, scaler_series

def create_logger():
    """Create a TensorBoard logger with a unique log folder for each run. Save logs in the temporary directory."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.getenv('TMPDIR', '/tmp'), f'TFT/logs/{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    return pl_loggers.TensorBoardLogger(save_dir=log_dir, name='TFT', default_hp_metric=False)

def train_model(best_params, series_scaled, best_model_epochs, devices):
    """Train the TFT model with the best hyperparameters found by Optuna."""
    tb_logger = create_logger()
    best_model = TFTModel(
        input_chunk_length=best_params['input_chunk_length'],
        output_chunk_length=1,
        hidden_size=best_params['hidden_dim'],
        lstm_layers=best_params['n_layers'],
        dropout=best_params.get('dropout', 0.0),
        batch_size=best_params['batch_size'],
        n_epochs=best_model_epochs,
        optimizer_kwargs={'lr': best_params['learning_rate']},
        random_state=42,
        add_relative_index=True,
        pl_trainer_kwargs={
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
        }
    )
    best_model.fit(series_scaled, verbose=True)
    return best_model

def objective(trial, series_scaled, optuna_epochs, devices):
    hidden_dim = trial.suggest_int('hidden_dim', 16, 64)
    n_layers = trial.suggest_int('n_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    input_chunk_length = trial.suggest_int('input_chunk_length', 30, 200)

    tb_logger = create_logger()
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
        add_relative_index=True,
        pl_trainer_kwargs={
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
        }
    )
    try:
        model.fit(series_scaled, verbose=False)
        forecast_train = model.predict(n=len(series_scaled))
        train_loss = rmse(series_scaled, forecast_train)
        return train_loss
    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')

def run_optuna_optimization(series_scaled, optuna_trials, optuna_epochs, devices):
    """Run Optuna optimization to find the best hyperparameters for TFT."""
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, series_scaled, optuna_epochs, devices), n_trials=optuna_trials)
    best_params = study.best_params
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')
    return best_params, study

# Function to copy files from tmpdir_path to the specified home directory
def copy_to_home_dir(src_dir, dest_dir):
    try:
        os.makedirs(dest_dir, exist_ok=True)
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"All files from {src_dir} copied to {dest_dir}")
    except Exception as e:
        print(f"Error copying files to home directory: {e}")
        traceback.print_exc()

# Update the function to plot and save the forecast as a plot and CSV file
def plot_forecast(forecast, output_path):
    """Plot the forecast and save it to an image file."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast.time_index,
        y=forecast.values().squeeze(),
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='TFT Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly'
    )
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(output_path)
    return fig

def save_forecast_to_csv(forecast, output_path):
    """Save the forecast values and dates to a CSV file."""
    forecast_df = pd.DataFrame({
        'Date': forecast.time_index,
        'Forecast': forecast.values().squeeze()
    })
    forecast_df.to_csv(output_path, index=False)
    print(f"Forecast data saved at: {output_path}")

def sanitize_filename(filename):
    """Replace or remove special characters for safe file handling."""
    return "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in filename)

# Main execution block
if __name__ == "__main__":
    config = load_config()
    if config is None:
        print("Failed to load config file.")
        exit(1)

    base_path = os.path.expanduser("~/Documents/Masterarbeit/Prediction_of_energy_prices/") \
                if platform.system() == "Darwin" \
                else os.getenv('HOME') + '/Prediction_of_energy_prices/'

    set_random_seed(42)
    df = load_and_prepare_data(os.path.join(base_path, config["data_file"]))
    series_train = prepare_time_series(df, config["target_column"])
    series_train_scaled, scaler_series = scale_data(series_train)

    best_params, study = run_optuna_optimization(series_train_scaled, config["optuna_trials"], config["optuna_epochs"], config["devices"])

    # Define the path with a subdirectory under TFT
    tmpdir_path = os.path.join(os.getenv('TMPDIR'), 'predictions/TFT/best_model/')
    os.makedirs(tmpdir_path, exist_ok=True)

    # Sanitize the target column for file naming
    sanitized_target_column = sanitize_filename(config["target_column"])

    # Save the best model in the subdirectory
    model_save_path = os.path.join(tmpdir_path, f'best_tft_model_epochs_{config["optuna_epochs"]}.pth')
    best_model = train_model(best_params, series_train_scaled, config["optuna_epochs"])
    best_model.save(model_save_path)
    print(f"Best model saved at: {model_save_path}")

    # Forecast into the future (e.g., 730 days)
    forecast = best_model.predict(n=730)
    forecast = scaler_series.inverse_transform(forecast)

    # Save the forecast plot in the subdirectory
    forecast_plot_path = os.path.join(tmpdir_path, f'tft_forecast_plot_{sanitized_target_column}.png')
    plot_forecast(forecast, forecast_plot_path)
    print(f"Forecast plot saved at: {forecast_plot_path}")

    # Save the forecast values to CSV in the subdirectory
    forecast_csv_path = os.path.join(tmpdir_path, f'tft_forecast_values_{sanitized_target_column}.csv')
    save_forecast_to_csv(forecast, forecast_csv_path)

    # Copy all files to the home directory
    home_dir_path = '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data_retrieval/future_data/TFT/best_model/'
    copy_to_home_dir(tmpdir_path, home_dir_path)
