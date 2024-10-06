import shutil
import os
import random
import numpy as np
import traceback
import torch
import platform
import pandas as pd
import optuna
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

# Set the random seed
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_random_seed(42)

# Early stopping callback
early_stop_callback = EarlyStopping(monitor='train_loss', patience=10, verbose=True)

# Create scaler object for series
scaler_series = Scaler(MaxAbsScaler())

def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime.
    """
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_time_series(df, target_column):
    """
    Prepare time series object for training.
    """
    series = TimeSeries.from_dataframe(df, 'Date', target_column).astype('float32')
    return series

def scale_data(series):
    """
    Scale the time series data.
    """
    scaler_series = Scaler(MaxAbsScaler())
    series_scaled = scaler_series.fit_transform(series)
    return series_scaled, scaler_series

def create_logger():
    """
    Create a TensorBoard logger with a unique log folder for each run.
    Save logs in the temporary directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.getenv('TMPDIR', '/tmp'), f'TFT/logs/{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    return pl_loggers.TensorBoardLogger(save_dir=log_dir, name='TFT', default_hp_metric=False)

def train_model(best_params, series_scaled, best_model_epochs):
    """
    Train the TFT model with the best hyperparameters found by Optuna.
    """
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
            'devices': 1,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
        }
    )

    # Train the model
    best_model.fit(series_scaled, verbose=True)

    return best_model

def objective(trial, series_scaled, optuna_epochs):
    hidden_dim = trial.suggest_int('hidden_dim', 16, 64)
    n_layers = trial.suggest_int('n_layers', 1, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    input_chunk_length = trial.suggest_int('input_chunk_length', 30, 200)

    # Check for NaN in data before training
    if series_scaled.pd_dataframe().isnull().values.any():
        print(f"NaN values detected in the input data during trial {trial.number}")
        return float('inf')

    # Create logger for the trial
    tb_logger = create_logger()

    # Initialize TFT Model
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
            'devices': 1,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
        }
    )

    try:
        # Train the model
        model.fit(series_scaled, verbose=False)

        # Predict the next two years (730 days)
        forecast_val = model.predict(n=730)

        # Calculate metrics on the test set (we won't have a test set, so skip this step)
        rmse_value = rmse(series_scaled[-730:], forecast_val)  # Evaluate on last 730 days

        # Check if RMSE is NaN or Inf
        if np.isnan(rmse_value) or np.isinf(rmse_value):
            print(f"NaN or Inf detected in RMSE during trial {trial.number}")
            return float('inf')

    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')

    return rmse_value


def run_optuna_optimization(series_scaled, optuna_trials, optuna_epochs):
    """
    Run Optuna optimization to find the best hyperparameters for TFT.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, series_scaled, optuna_epochs), n_trials=optuna_trials)

    best_params = study.best_params
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')

    return best_params, study

def plot_forecast(forecast, output_path):
    """
    Plot the forecast for the next two years using Plotly and save it to a file.
    """
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

    # Ensure the directory exists before saving the plot
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist

    # Save the figure to the temporary directory
    fig.write_image(output_path)

    return fig

# Main execution block
if __name__ == "__main__":

    # Detect if on Mac or Linux and adjust base path accordingly
    if platform.system() == "Darwin":  # macOS
        base_path = os.path.expanduser("~/Documents/Masterarbeit/Prediction_of_energy_prices/")
    else:  # Assuming Linux for the cluster
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'

    set_random_seed(42)

    # Load in the training data (no test data)
    df = load_and_prepare_data(os.path.join(base_path, 'data/Final_data/final_data_july.csv'))

    # Use Solar_radiation (W/m2) as the target (or another column)
    target_column = 'Solar_radiation (W/m2)'

    # Prepare and scale the time series
    series_train = prepare_time_series(df, target_column)
    series_train_scaled, scaler_series = scale_data(series_train)

    # Optuna settings
    optuna_epochs = 1  # Define the number of epochs for Optuna trials
    optuna_trials = 1  # Define the number of trials for Optuna

    # Run Optuna optimization
    best_params, study = run_optuna_optimization(series_train_scaled, optuna_trials, optuna_epochs)

    # Ensure the TMPDIR path for storing the model and results
    tmpdir_path = os.path.join(os.getenv('TMPDIR'), 'predictions/TFT/')
    os.makedirs(tmpdir_path, exist_ok=True)

    # Save the best model
    model_save_path = os.path.join(tmpdir_path, f'best_tft_model_epochs_{optuna_epochs}.pth')
    best_model = train_model(best_params, series_train_scaled, optuna_epochs)
    best_model.save(model_save_path)
    print(f"Best model saved at: {model_save_path}")

    # Make predictions
    n = len(series_train_scaled)  # Adjust depending on your test data length
    forecast = best_model.predict(n=n)

    forecast = scaler_series.inverse_transform(forecast)

    # Define output path for the forecast plot
    output_path = os.path.join(tmpdir_path, f'tft_forecast_plot_{target_column}.png')

    # Plot and save the forecast
    fig = plot_forecast(forecast, output_path)
    print(f"Forecast plot saved at: {output_path}")

    # Define the home directory where results should be copied
    home_results_dir = os.path.join(base_path, '/data_retrieval/future_data/TFT')
    os.makedirs(home_results_dir, exist_ok=True)

    # Copy the results from TMPDIR to the home directory
    try:
        if os.path.exists(tmpdir_path):
            shutil.copytree(tmpdir_path, home_results_dir, dirs_exist_ok=True)
            print(f"Results copied to {home_results_dir}")
        else:
            print(f"Directory {tmpdir_path} does not exist. Skipping copy operation.")
    except Exception as e:
        print(f"Error copying results: {e}")

    # ** Print best hyperparameters at the end **
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')


