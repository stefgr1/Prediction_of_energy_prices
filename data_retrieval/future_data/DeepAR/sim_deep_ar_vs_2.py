import os
import re
import sys
import numpy as np
import platform
import importlib
import random
import pandas as pd
import torch
import optuna
import yaml
import traceback
import tempfile
import shutil  
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.timeseries import concatenate
from pytorch_lightning.callbacks import ModelCheckpoint
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mae, rmse, mse, mape
from darts.utils.likelihood_models import GaussianLikelihood, ExponentialLikelihood
from darts.utils.callbacks import TFMProgressBar
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from darts.dataprocessing.transformers import Scaler


add_encoders = {
    'datetime_attribute': {
        'past': ['day', 'weekday', 'month'],
        'future': ['day', 'weekday', 'month']
    },
    'cyclic': {
        'past': ['day', 'weekday', 'month'],
        'future': ['day', 'weekday', 'month']
    },
    'position': {
        'future': ['relative']
    }
}


def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

# Function to create and load config file
def load_config(config_path=None):
    # Default to using config.yaml in the current script's directory
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to load and prepare data


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.reset_index(drop=True, inplace=True)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Early stopping callback function


def create_early_stopping_callback(patience):
    return EarlyStopping(
        monitor='val_loss', patience=patience, verbose=True
    )

# Logger creation function


def create_logger(tmpdir, trial_number=None, model_name='DeepAR_Model'):
    if platform.system() == 'Darwin':  # macOS
        final_base_log_dir = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data_retrieval/future_data/DeepAR/logs'
    else:  # Linux cluster
        home_dir = os.path.expanduser('~')
        final_base_log_dir = os.path.join(
            home_dir,
            'Prediction_of_energy_prices',
            'data_retrieval',
            'future_data',
            'DeepAR',
            'logs'
        )

    # Ensure the final base log directory exists
    os.makedirs(final_base_log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if trial_number is not None:
        log_dir = os.path.join(tmpdir, f'{timestamp}_trial_{trial_number}')
    else:
        log_dir = os.path.join(tmpdir, f'{timestamp}_best_model')
    os.makedirs(log_dir, exist_ok=True)

    logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir, name=model_name, default_hp_metric=False)

    # After training, move the logs to the final destination
    final_log_dir = os.path.join(final_base_log_dir, os.path.basename(log_dir))
    shutil.move(log_dir, final_log_dir)

    return logger



def main():

    # Load the config file
    config = load_config()

    target_column = config["target_column"]

    # Define tmpdir for temporary file storage
    with tempfile.TemporaryDirectory() as tmpdir:

        # Determine the platform and available GPU type
        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            # Use Apple's Metal Performance Shaders (MPS) on macOS
            accelerator = 'mps'
        elif torch.cuda.is_available():
            accelerator = 'gpu'  # Use CUDA on Linux if available
        else:
            accelerator = 'cpu'  # Fall back to CPU

        # Update the devices parameter accordingly
        devices = config["devices"] if accelerator != 'cpu' else 1

        # Detect if on Mac or Linux and adjust base path accordingly
        if platform.system() == "Darwin":  # macOS
            base_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/'
        else:  # Assuming Linux for the cluster
            base_path = os.path.join(
                os.getenv('HOME'), 'Prediction_of_energy_prices/')

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load in the train and test data
        train_df = load_and_prepare_data(os.path.join(
            base_path, 'data/Final_data/train_df.csv'))
        test_df = load_and_prepare_data(os.path.join(
            base_path, 'data/Final_data/test_df.csv'))

        # Create TimeSeries objects
        train_series = TimeSeries.from_dataframe(
            train_df, value_cols=target_column).astype('float32')
        test_series = TimeSeries.from_dataframe(
            test_df, value_cols=target_column).astype('float32')

        # Concatenate train and test data for plotting
        total_series = train_series.append(test_series)

        # Scale the data
        scaler = Scaler()
        train_transformed = scaler.fit_transform(train_series)
        test_transformed = scaler.transform(test_series)
        total_series_transformed = scaler.transform(total_series)

        # Create early stopping callback
        early_stop_callback = create_early_stopping_callback(
            patience=config["patience"])

        # Define the objective function for Optuna
        def objective(trial):
            # Hyperparameters to tune
            n_layers = trial.suggest_int('n_layers', 1, 2)
            dropout = trial.suggest_float(
                'dropout', 0.0, 0.5) if n_layers > 1 else 0.0
            input_chunk_length = trial.suggest_int(
                'input_chunk_length', 10, 100)
            hidden_dim = trial.suggest_int('hidden_dim', 50, 200)
            learning_rate = trial.suggest_float(
                'learning_rate', 1e-7, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

            if n_layers == 1:
                dropout = 0.0

            # Ensure training_length is larger than input_chunk_length but never larger than 500
            training_length = trial.suggest_int(
                'training_length', input_chunk_length + 1, 500)

            # Create a new logger for each trial
            tb_logger = create_logger(
                tmpdir, trial_number=trial.number, model_name=f'DeepAR_{target_column}')

            # Create model
            model = RNNModel(
                model='LSTM',
                input_chunk_length=input_chunk_length,
                training_length=training_length,
                hidden_dim=hidden_dim,
                n_rnn_layers=n_layers,
                dropout=dropout,
                likelihood= GaussianLikelihood(),
                batch_size=batch_size,
                n_epochs=config["optuna_epochs"],
                optimizer_kwargs={'lr': learning_rate},
                random_state=42,
                add_encoders=add_encoders,
                save_checkpoints=False,
                model_name="rnn_model",
                force_reset=True,
                pl_trainer_kwargs={
                    'accelerator': accelerator,
                    'devices': config["devices"],
                    'enable_progress_bar': True,
                    'logger': tb_logger,
                    'enable_model_summary': False,
                    'enable_checkpointing': False,  
                    'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
                    'log_every_n_steps': 20,
                }
            )
            try:
                # Fit the model
                model.fit(
                    series=train_transformed,
                    val_series=test_transformed,
                )

                # Forecast over the test period
                n = len(test_series)
                forecast = model.predict(n=n)

                forecast = scaler.inverse_transform(forecast)

                # Get rid off impossible negative values
                forecast = forecast.with_values(np.maximum(forecast.values(), 0))

                # Calculate error metrics
                rmse_val = rmse(test_series, forecast)
                mape_val = mape(test_series, forecast)
                mae_val = mae(test_series, forecast)
                mse_val = mse(test_series, forecast)

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

        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=config["n_trials"])

        # Get best hyperparameters
        best_params = study.best_params

        print('Best hyperparameters:')
        for key, value in best_params.items():
            print(f'  {key}: {value}')

        # Prepare variables for best model
        best_training_length = best_params['training_length']
        dropout = best_params.get('dropout', 0.0)
        best_trial = study.best_trial
        tb_logger = create_logger(
            tmpdir, trial_number=best_trial.number, model_name=f'DeepAR_bestmodel_{target_column}')

        # Retrain the model on the entire dataset with best hyperparameters
        best_model = RNNModel(
            model='LSTM',
            input_chunk_length=best_params['input_chunk_length'],
            training_length=best_training_length,
            hidden_dim=best_params['hidden_dim'],
            n_rnn_layers=best_params['n_layers'],
            dropout=dropout,
            likelihood=GaussianLikelihood(),
            batch_size=best_params['batch_size'],
            n_epochs=config["best_model_epochs"],
            optimizer_kwargs={'lr': best_params['learning_rate']},
            random_state=42,
            add_encoders=add_encoders,
            save_checkpoints=False,
            force_reset=True,
            model_name="deep_ar_model",
            pl_trainer_kwargs={
                'accelerator': accelerator,
                'devices': config["devices"],
                'enable_progress_bar': True,
                'enable_checkpointing': False, 
                'logger': tb_logger,
                'enable_model_summary': False,
                'log_every_n_steps': 20,
            }
        )

        # Fit the model on the entire data
        best_model.fit(
            series=total_series_transformed,
        )

        # Set forecast horizon for two years (daily data)
        forecast_horizon = 365 * 2  # Two years into the future


        # Make future prediction
        forecast = best_model.predict(
            n=forecast_horizon,
        )

        # Inverse transform the forecasted data
        forecast = scaler.inverse_transform(forecast)

        forecast = forecast.with_values(np.maximum(forecast.values(), 0))

        # Save the forecast values and dates to CSV
        forecast_df = forecast.pd_dataframe()
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns={'index': 'Date'}, inplace=True)
        forecast_csv_path = os.path.join(script_dir, 'forecast.csv')
        forecast_df.to_csv(forecast_csv_path, index=False)

        # Plot the results using Plotly
        fig = go.Figure()

        # Add the actual data
        fig.add_trace(go.Scatter(
            x=total_series.time_index,
            y=total_series.values().squeeze(),
            mode='lines',
            name='Actual Data'
        ))

        # Add the forecast data
        fig.add_trace(go.Scatter(
            x=forecast.time_index,
            y=forecast.values().squeeze(),
            mode='lines',
            name='Forecast'
        ))

        # Update layout
        fig.update_layout(
            title=f'{target_column} Forecast',
            xaxis_title='Date',
            yaxis_title=target_column
        )

        # Sanitize the target column name for filenames only
        sanitized_target_column = sanitize_filename(config["target_column"])


        # Paths for temporary storage in tmpdir
        temp_forecast_csv_path = os.path.join(
            tmpdir, f'forecast_{config["optuna_epochs"]}.csv')
        temp_plot_file_path = os.path.join(
            tmpdir, f'forecast_plot_{config["optuna_epochs"]}.png')

        # Save the forecast CSV and plot in the tmpdir
        forecast_df.to_csv(temp_forecast_csv_path, index=False)
        fig.write_image(temp_plot_file_path, width=1200, height=600)

        # Paths for final storage
        final_forecast_csv_path = os.path.join(
            script_dir, f'forecast_{sanitized_target_column}_{config["optuna_epochs"]}.csv')
        final_plot_file_path = os.path.join(
            script_dir, f'forecast_plot_{sanitized_target_column}_{config["optuna_epochs"]}.png')


        # Copy the forecast CSV and plot to the final destination
        shutil.copy2(temp_forecast_csv_path, final_forecast_csv_path)
        shutil.copy2(temp_plot_file_path, final_plot_file_path)


if __name__ == "__main__":
    main()
