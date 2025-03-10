import os
import torch
import optuna
import traceback
import platform
import plotly.graph_objects as go
import shutil
import sys
import importlib
import yaml
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mape, mae, rmse, mse, smape
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.callbacks import TFMProgressBar
import yaml

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
future_covariates_columns = utils.future_covariates_columns

# Function to load configuration from YAML file


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), 'config_DeepAR.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


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


def objective(trial, series_train_scaled, future_covariates_train_scaled, series_test_scaled, future_covariates_for_prediction_scaled, optuna_epochs, devices, early_stop_callback):
    """
    Optuna objective function for RNN model hyperparameter tuning.
    """
    n_layers = trial.suggest_int('n_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5) if n_layers > 1 else 0.0
    input_chunk_length = trial.suggest_int('input_chunk_length', 10, 120)
    hidden_dim = trial.suggest_int('hidden_dim', 60, 120)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Ensure training_length is larger than input_chunk_length but never larger than 500
    training_length = trial.suggest_int(
        'training_length', input_chunk_length + 1, 130)

    # Create a new logger for each trial
    tb_logger = create_logger(trial.number, model_name='DeepAR')

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
            'devices': devices,
            'enable_progress_bar': True,
            'logger': tb_logger,
            'enable_model_summary': False,
            'callbacks': [early_stop_callback, TFMProgressBar(enable_train_bar_only=True)],
            'log_every_n_steps': 10,
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
        smape_val = smape(series_test_scaled, forecast_val)

        # Log the error metrics in the trial object
        trial.set_user_attr('rmse', rmse_val)
        trial.set_user_attr('mape', mape_val)
        trial.set_user_attr('mae', mae_val)
        trial.set_user_attr('mse', mse_val)
        trial.set_user_attr('smape', smape_val)

    except Exception as e:
        print(f'Exception during model training: {e}')
        traceback.print_exc()
        return float('inf')

    return rmse_val


def inspect_best_trial(study):
    """
    Inspect and return the error metrics from the best trial in the Optuna study.
    """
    best_trial = study.best_trial
    metrics_dict = {
        "RMSE": best_trial.user_attrs.get('rmse', 'N/A'),
        "MAPE": best_trial.user_attrs.get('mape', 'N/A'),
        "MAE": best_trial.user_attrs.get('mae', 'N/A'),
        "MSE": best_trial.user_attrs.get('mse', 'N/A'),
        "sMAPE": best_trial.user_attrs.get('smape', 'N/A')
    }
    return metrics_dict


# Function to save best hyperparameters to TMPDIR and copy to home directory
def save_best_hyperparameters(best_params, optuna_trials, best_model_epochs, lag_suffix):
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    hyperparams_path = os.path.join(
        tmp_dir, f"best_hyperparameters_{optuna_trials}_{best_model_epochs}{lag_suffix}.yml")

    # Save best parameters in TMPDIR
    with open(hyperparams_path, 'w') as yaml_file:
        yaml.dump(best_params, yaml_file)
    print(f"Best hyperparameters saved at: {hyperparams_path}")

    # Define home directory for results and ensure it exists
    home_hyperparams_dir = os.path.join(base_path, 'predictions/DeepAR/')
    os.makedirs(home_hyperparams_dir, exist_ok=True)

    # Copy YAML file to home directory
    home_hyperparams_path = os.path.join(
        home_hyperparams_dir, os.path.basename(hyperparams_path))
    shutil.copy(hyperparams_path, home_hyperparams_path)
    print(f"Hyperparameters file copied to {home_hyperparams_path}")


# Main execution block
if __name__ == "__main__":

    # Load configuration parameters from the YAML file
    config = load_config()
    lag_suffix = "_with_lags" if config["lags"] else "_no_lags"

    check_cuda_availability()

    # Detect if on Mac or Linux and adjust base path accordingly
    if platform.system() == "Darwin":  # Check if it's macOS
        base_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/'
        models_dir = os.path.join(base_path, 'predictions/DeepAR/')
    else:  # Assuming Linux for the cluster
        base_path = os.getenv('HOME') + '/Prediction_of_energy_prices/'
        models_dir = os.path.join(
            os.getenv('TMPDIR', '/tmp'), 'predictions/DeepAR/')

    # Ensure model directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Set random seed
    set_random_seed(42)

    # Create early stopping callback
    early_stop_callback = create_early_stopping_callback(patience=150)

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

    # Use parameters loaded from config.yml
    MAX_INPUT_CHUNK_LENGTH = config['max_input_chunk_length']
    OPTUNA_TRIALS = config['optuna_trials']
    OPTUNA_EPOCHS = config['optuna_epochs']
    BEST_MODEL_EPOCHS = config['best_model_epochs']
    DEVICES = config['devices']

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

    # Save the best model
    model_save_path = os.path.join(
        models_dir, f'best_deep_ar_model_epochs_{BEST_MODEL_EPOCHS}_{OPTUNA_TRIALS}{lag_suffix}.pth')
    best_model.save(model_save_path)
    print(f"Best model saved at: {model_save_path}")

    # Make predictions
    n = len(series_test_scaled)
    forecast = best_model.predict(
        n=n, future_covariates=future_covariates_for_prediction_scaled)
    forecast = scaler_series.inverse_transform(forecast)

    # Inspect the error metrics from best trial in the Optuna study
    inspect_best_trial(study)

    # Plot and save results, and retrieve file paths
    fig = plot_forecast(series_test, forecast, title="DeepAR Model forecast")
    forecast_plot_path, forecast_csv_path, metrics_csv_path = save_results(
        forecast, series_test_scaled, scaler_series, fig, OPTUNA_EPOCHS, OPTUNA_TRIALS, models_dir, lag_suffix
    )

    # Save best hyperparameters and copy to home directory
    save_best_hyperparameters(
        best_params, OPTUNA_TRIALS, BEST_MODEL_EPOCHS, lag_suffix)

    if platform.system() == "Darwin":
        # If on macOS, copy files directly to the home directory
        home_results_dir = os.path.join(base_path, 'predictions/DeepAR/')
        os.makedirs(home_results_dir, exist_ok=True)

        # Copy model and results files to home directory
        shutil.copy(model_save_path, home_results_dir)
        shutil.copy(forecast_plot_path, home_results_dir)
        shutil.copy(forecast_csv_path, home_results_dir)
        shutil.copy(metrics_csv_path, home_results_dir)
        print(f"Results copied to {home_results_dir}")

    else:  # On Linux or cluster
        # Copy results and model from TMPDIR to home directory
        home_results_dir = os.path.join(base_path, 'predictions/DeepAR/')
        os.makedirs(home_results_dir, exist_ok=True)

        copy_results_to_home(
            [forecast_plot_path, forecast_csv_path, metrics_csv_path], home_results_dir)
        shutil.copy(model_save_path, home_results_dir)
        print(f"Model and results copied to {home_results_dir}")

    # Print best hyperparameters at the end
    print('Best hyperparameters:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')
