import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood


def load_and_merge_csv_files(forecasted_data_dir):
    # Get all CSV files in the forecasted_data folder
    csv_files = [f for f in os.listdir(
        forecasted_data_dir) if f.endswith('.csv')]

    # Initialize an empty DataFrame
    merged_df = pd.DataFrame()

    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(forecasted_data_dir, csv_file))

        # Merge the DataFrame on the 'Date' column
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    return merged_df


# Define directories and load files
cwd = os.getcwd()
forecasted_data_dir = os.path.join(cwd, 'future_prediction/forecasted_data')
merged_df = load_and_merge_csv_files(forecasted_data_dir)

# Load additional files
gas_prices = pd.read_csv(os.path.join(
    cwd, 'future_prediction/forecast_values_large_730_TTF_gas_price_EUR_MWh_300_32.csv'))
oil_prices = pd.read_csv(os.path.join(
    cwd, 'future_prediction/forecast_values_large_730_Oil_price_EUR_400_32_large.csv'))

# Select relevant columns and rename
gas_prices = gas_prices[['Date', 'Mean']].rename(
    columns={'Mean': 'TTF_gas_price (EUR/MWh)'})
oil_prices = oil_prices[['Date', 'Mean']].rename(
    columns={'Mean': 'Oil_price (EUR)'})

# Set Date column as index
gas_prices.set_index('Date', inplace=True)
oil_prices.set_index('Date', inplace=True)

# Merge the gas_prices and oil_prices DataFrames with the merged_df DataFrame
merged_df = pd.merge(merged_df, gas_prices, on='Date', how='outer')
merged_df.dropna(inplace=True)
merged_df = merged_df.round(2)
merged_df['Nuclear_energy (MWh)'] = 0

# Load old data and combine
df_old = pd.read_csv(
    "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Final_data/final_data_july.csv")
final_df = pd.concat([df_old, merged_df], ignore_index=True)
final_df = final_df.iloc[:, :-12]

# Prepare data for model training
data = final_df.copy()
target_column = 'Day_ahead_price (€/MWh)'
covariates = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)', 'Temperature (°C)', 'Biomass (GWh)',
              'Hard_coal (GWh)', 'Hydro (GWh)', 'Lignite (GWh)', 'Natural_gas (GWh)',
              'Other (GWh)', 'Pumped_storage_generation (GWh)', 'Solar_energy (GWh)',
              'Wind_offshore (GWh)', 'Wind_onshore (GWh)', 'Net_total_export_import (GWh)',
              'BEV_vehicles', 'Oil_price (EUR)', 'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)']

data[target_column] = data[target_column].astype(np.float32)
data[covariates] = data[covariates].astype(np.float32)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Split data for training and prediction
train_data = data[data['Date'] < '2024-07-29']
prediction_data = data[data['Date'] >= '2024-07-29']

# Create TimeSeries objects
target_series = TimeSeries.from_dataframe(
    train_data, time_col='Date', value_cols=target_column)
covariate_series = TimeSeries.from_dataframe(
    train_data, time_col='Date', value_cols=covariates)

# Scale the target and covariates
target_scaler = Scaler()
covariate_scaler = Scaler()
target_series_scaled = target_scaler.fit_transform(target_series)
covariate_series_scaled = covariate_scaler.fit_transform(covariate_series)


# Define and train the model
model = RNNModel(
    model='LSTM',
    input_chunk_length=39,
    training_length=500,
    optimizer_kwargs={'lr': 0.001},  # Reduced learning rate
    n_rnn_layers=1,
    n_epochs=3,  # Increased epochs for better convergence
    likelihood=GaussianLikelihood(sigma=sigma),  # Keep the likelihood for now
    batch_size=16,
    hidden_dim=153,
    dropout=0.1,  # Added dropout for regularization
    random_state=42,
    pl_trainer_kwargs={"accelerator": "cpu"},
)


model.fit(series=target_series_scaled,
          future_covariates=covariate_series_scaled, verbose=True)

# Prepare covariates for prediction
forecast_start_date = pd.to_datetime(prediction_data['Date'].iloc[0])
required_start_date = forecast_start_date - \
    pd.Timedelta(days=model.input_chunk_length)
forecast_end_date = pd.to_datetime(prediction_data['Date'].iloc[-1])

covariates_df = data.loc[(data['Date'] >= required_start_date) & (
    data['Date'] <= forecast_end_date), ['Date'] + covariates]
combined_covariate_series = TimeSeries.from_dataframe(
    covariates_df, time_col='Date', value_cols=covariates)
combined_covariate_series_scaled = covariate_scaler.transform(
    combined_covariate_series)

print("Target series start:", target_series.start_time())
print("Target series end:", target_series.end_time())

# Forecast and inverse scale
forecast = model.predict(
    n=730, future_covariates=combined_covariate_series_scaled)
forecast_original_scale = target_scaler.inverse_transform(forecast)

print("Forecast values:")
print(forecast_original_scale)
print("Forecast series start:", forecast_original_scale.start_time())
print("Forecast series end:", forecast_original_scale.end_time())

# Plot the results
plt.figure(figsize=(12, 6))
target_series.plot(label='Actual', color='blue')
forecast_original_scale.plot(label='Forecast', color='orange')
plt.title('Day Ahead Price Forecast')
plt.xlabel('Date')
plt.ylabel('Day Ahead Price (EUR/MWh)')
plt.legend()
plt.show()
