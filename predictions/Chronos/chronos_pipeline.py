import warnings
import transformers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
import plotly.graph_objs as go
import plotly.io as pio
from darts import TimeSeries
from darts.metrics import mse, mae, rmse, smape, mape

# Check for MPS availability and set device
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load and prepare the data


def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime.
    """
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    return df

# Prepare the data


def prepare_data(df):
    """
    Prepare the data by renaming columns and adding item_id.
    """
    df = df.rename(columns={
        'Date': 'timestamp',
        'Day_ahead_price (€/MWh)': 'target',
        'TTF_gas_price (€/MWh)': 'TTF_gas_price_EUR_MWh'
    })
    df['item_id'] = '1'  # Single time series ID
    return df


# Paths to data files
train_data_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Final_data/train_df_no_lags.csv'
test_data_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Final_data/test_df_no_lags.csv'
full_data_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Final_data/final_data_no_lags.csv'

# Load and preprocess data
train_df = prepare_data(load_and_prepare_data(train_data_path))
test_df = prepare_data(load_and_prepare_data(test_data_path))
df = prepare_data(load_and_prepare_data(full_data_path))

train_data = TimeSeriesDataFrame.from_data_frame(
    train_df, id_column="item_id", timestamp_column="timestamp")
test_data = TimeSeriesDataFrame.from_data_frame(
    test_df, id_column="item_id", timestamp_column="timestamp")
data = TimeSeriesDataFrame.from_data_frame(
    df, id_column="item_id", timestamp_column="timestamp")

print(train_data.columns)  # Confirm 'TTF_gas_price_EUR_MWh' is present

# Set prediction length
prediction_length = len(test_data)
SIZE = "tiny"

# Train the predictor with Chronos
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length - 1,
    target="target",
    eval_metric="rmse",
    known_covariates_names=['Solar_radiation (W/m2)', 'Wind_speed (m/s)', 'Temperature (°C)',
                            'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)', 'Lignite (GWh)',
                            'Natural_gas (GWh)', 'Other (GWh)', 'Pumped_storage_generation (GWh)',
                            'Solar_energy (GWh)', 'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
                            'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
                            'TTF_gas_price_EUR_MWh', 'Nuclear_energy (GWh)', 'Day_of_week'],
).fit(
    train_data,
    hyperparameters={
        "Chronos": [
            {
                "fine_tune": False,
                "model_path": f"bolt_{SIZE}",
                "ag_args": {"name_suffix": "ZeroShot"},
            },
            {
                "model_path": f"bolt_{SIZE}",
                "fine_tune": False,
                "covariate_regressor": "CAT",
                "target_scaler": "mean_abs",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    enable_ensemble=False,
    time_limit=10,
)

# Function to select the best model and predict


def select_best_model_and_predict(leaderboard, predictor, train_data, future_known_covariates, full_data, prediction_length):
    """
    Selects the model with the highest score_test from the leaderboard and uses it for prediction.

    Parameters:
    leaderboard (pd.DataFrame): Leaderboard containing model scores.
    predictor (TimeSeriesPredictor): Trained AutoGluon predictor.
    train_data (TimeSeriesDataFrame): Training data used for prediction.
    future_known_covariates (TimeSeriesDataFrame): Future covariates for prediction.
    full_data (TimeSeriesDataFrame): Full data for plotting.
    prediction_length (int): Prediction length for visualization.

    Returns:
    pd.DataFrame: Predictions from the best model.
    """
    best_model = leaderboard.sort_values(
        by="score_test", ascending=False).iloc[0]["model"]
    print(f"Selected best model: {best_model}")

    predictions = predictor.predict(
        train_data,
        known_covariates=future_known_covariates,
        model=best_model
    )

    predictor.plot(
        data=full_data,
        predictions=predictions,
        item_ids=full_data.item_ids[:2],
        max_history_length=prediction_length - 1,
    )

    return predictions


# Use the best model for prediction
leaderboard = predictor.leaderboard(test_data)
future_known_covariates = test_data.drop(columns=["target"])
predictions = select_best_model_and_predict(
    leaderboard=leaderboard,
    predictor=predictor,
    train_data=train_data,
    future_known_covariates=future_known_covariates,
    full_data=data,
    prediction_length=prediction_length
)

# Compute error metrics


def compute_metrics(predictions, test_data):
    """
    Compute MAPE, MAE, RMSE, MSE, and SMAPE.
    """
    test_data = test_data.reset_index()
    predictions = predictions.reset_index()

    test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
    predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])

    test_data.set_index('timestamp', inplace=True)
    predictions.set_index('timestamp', inplace=True)

    test_series = TimeSeries.from_series(test_data['target'])
    pred_series = TimeSeries.from_series(predictions['mean'])

    mape_value = mape(test_series, pred_series)
    mae_value = mae(test_series, pred_series)
    rmse_value = rmse(test_series, pred_series)
    mse_value = mse(test_series, pred_series)
    smape_value = smape(test_series, pred_series)
    return mape_value, mae_value, rmse_value, mse_value, smape_value


mape_value, mae_value, rmse_value, mse_value, smape_value = compute_metrics(
    predictions, test_data)
print(f"MAPE: {mape_value:.2f}")
print(f"MAE: {mae_value:.2f}")
print(f"RMSE: {rmse_value:.2f}")
print(f"MSE: {mse_value:.2f}")
print(f"SMAPE: {smape_value:.2f}")

# Save metrics and predictions
metrics = pd.DataFrame({
    'RMSE': [rmse_value],
    'MSE': [mse_value],
    'MAE': [mae_value],
    'MAPE': [mape_value],
    'SMAPE': [smape_value]
}).round(2)

metrics.to_csv(
    f'/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/predictions/Chronos/autogluon_metrics_{SIZE}.csv', index=False)

predictions_df = pd.DataFrame(predictions).reset_index()
predictions_df = predictions_df.drop(
    columns=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', 'item_id'])
predictions_df = predictions_df.rename(
    columns={'mean': 'Day_ahead_price (€/MWh)', 'timestamp': 'Date'})
predictions_df.set_index('Date', inplace=True)
predictions_df.to_csv(
    f'/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/predictions/Chronos/autogluon_predictions_{SIZE}.csv')

print("Script execution complete.")
