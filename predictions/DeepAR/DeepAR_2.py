import os
import torch
import optuna
import traceback
import platform
import plotly.graph_objects as go
import shutil
import sys
import importlib
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape, mae, rmse, mse
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.callbacks import TFMProgressBar

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
    Train the RNN model with the best hyperparameters found by Optuna.
    """
    best_training_length = max(500, best_params['input_chunk_length'])

    # Create a new logger specifically for the best model training
    tb_logger = create_logger(best_model=True, model_name='DeepAR')

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
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'log_every_n_steps': 20,
        }
    )

    # Train the best model
    best_model.fit(series_train_scaled,
                   future_covariates=future_covariates_train_scaled, verbose=True)

    return best_model

def objective(trial, series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices, early_stop_callback):
    """
    Optuna objective function for RNN model hyperparameter tuning.
    """
    n_layers = trial.suggest_int('n_layers', 1, 2)
    dropout = trial.suggest_float('dropout', 0.0, 0.5) if n_layers > 1 else 0.0
    input_chunk_length = trial.suggest_int('input_chunk_length', 10, 100)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])

    # Ensure training_length is larger than input_chunk_length but never larger than 500
    training_length = trial.suggest_int('training_length', input_chunk_length + 1, 500)

    # Create a new logger for each trial
    tb_logger = create_logger(trial.number, model_name='DeepAR')

    # RNN Model initialization
    model = RNNModel(
        model='LSTM',
        input_chunk_length=input_chunk_length,
        training_length=training_length,  # Use the hyperparameter for training_length
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
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
            'gradient_clip_val': 0.5,  # Clip gradients if they exceed 0.5
            'log_every_n_steps': 20,
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

    return rmse_val

def run_optuna_optimization(series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_trials, optuna_epochs, devices, early_stop_callback):
    """
    Run Optuna optimization to find the best hyperparameters.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, series_train_scaled, future_covariates_train_scaled,
                                           series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices, early_stop_callback), n_trials=optuna_trials)

    best_params = study.best_params
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')

    # Return both the best parameters and the study itself
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

    check_cuda_availability()

    # Detect if on Mac or Linux and adjust base path accordingly
    if platform.system() == "Darwin":  # Check if it's macOS
        base_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/'
    else:  # Assuming Linux for the cluster
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'

    # Set random seed
    set_random_seed(42)

    # Create early stopping callback
    early_stop_callback = create_early_stopping_callback(patience = 25)

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

    MAX_INPUT_CHUNK_LENGTH = 100
    # Parameters for epochs and trials
    OPTUNA_TRIALS = 1000  # Define the number of Optuna trials
    OPTUNA_EPOCHS = 49  # Define the number of epochs per Optuna trial
    BEST_MODEL_EPOCHS = 49  # Define the number of epochs for the best model
    DEVICES = 4

    # Prepare time series
    series_train, series_test, future_covariates_train, future_covariates_for_prediction = prepare_time_series(
        train_df, test_df, future_covariates_columns, MAX_INPUT_CHUNK_LENGTH)

    # Scale data
    series_train_scaled, series_test_scaled, future_covariates_train_scaled, future_covariates_for_prediction_scaled, scaler_series = scale_data(
        series_train, series_test, future_covariates_train, future_covariates_for_prediction)

    # Run Optuna optimization and get the study and best parameters
    best_params, study = run_optuna_optimization(
        series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, OPTUNA_TRIALS, OPTUNA_EPOCHS, DEVICES, early_stop_callback)

    # Train the best model
    best_model = train_best_model(
        best_params, series_train_scaled, future_covariates_train_scaled, BEST_MODEL_EPOCHS, DEVICES)

    # Use TMPDIR for model saving
    models_dir = os.getenv('TMPDIR', '/tmp') + '/predictions/DeepAR/'
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

    # Plot and save results, and retrieve file paths
    fig = plot_forecast(series_test, forecast, title="DeepAR Model forecast")
    forecast_plot_path, forecast_csv_path, metrics_csv_path = save_results(forecast, series_test_scaled, scaler_series, fig, OPTUNA_EPOCHS)

    # Copy results from TMPDIR to the home directory
    home_results_dir = os.path.join(base_path, 'predictions/DeepAR/')
    os.makedirs(home_results_dir, exist_ok=True)

    # Copy generated files to home directory
    copy_results_to_home([forecast_plot_path, forecast_csv_path, metrics_csv_path], home_results_dir)
    print(f"Results copied to {home_results_dir}")

    # Copy the model from TMPDIR to the home directory
    home_model_dir = os.path.join(base_path, 'predictions/DeepAR/')
    os.makedirs(home_model_dir, exist_ok=True)
    shutil.copy(model_save_path, home_model_dir)
    print(f"Model copied to {home_model_dir}")

    # ** Print best hyperparameters at the end **
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')