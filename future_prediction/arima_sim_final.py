import os
import platform
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import shutil

# Define constants
TARGET_COLUMN = "Day_ahead_price (€/MWh)"
TMPDIR = os.getenv("TMPDIR", "/tmp")
INCLUDE_LAGS = False
START_DATE = "2024-07-29"  # Start date for the new forecast
FORECAST_DAYS = 730  # Number of days to forecast
CHANGE = "constant"

# Define covariate lists
future_covariates_no_lags = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)', 'Temperature (°C)',
                             'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)', 'Lignite (GWh)',
                             'Natural_gas (GWh)', 'Other (GWh)', 'Pumped_storage_generation (GWh)',
                             'Solar_energy (GWh)', 'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
                             'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
                             'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Day_of_week', 'Month']

future_covariates_with_lags = future_covariates_no_lags + ['Lag_1_day', 'Lag_2_days', 'Lag_3_days',
                                                           'Lag_4_days', 'Lag_5_days', 'Lag_6_days',
                                                           'Lag_7_days', 'Rolling_mean_7']

# Paths based on environment
if platform.system() == 'Linux':
    file_path = f'/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/future_prediction/final_df_{CHANGE}.csv'
    model_path = '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/predictions/ARIMA/one_step_forecasts/best_sarima_model_without_lags.pkl'
    home_dir = '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/future_prediction/'
else:
    file_path = f'/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/future_prediction/final_df_{CHANGE}.csv'
    model_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/predictions/ARIMA/one_step_forecasts/best_sarima_model_without_lags.pkl'
    home_dir = './'

# Forecast function


def generate_stepwise_forecast(y, X, start_date, days, best_model):
    forecast_dates = pd.date_range(start=start_date, periods=days)
    forecasts = []

    for t in tqdm(range(days)):
        y_train = y  # Use all available historical data for each step
        X_train = X.iloc[:len(y)] if X is not None else None
        X_forecast = X.iloc[[len(y)]] if X is not None and len(
            X) > len(y) else None

        sarima_model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=best_model.order,
            seasonal_order=best_model.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="c" if best_model.with_intercept else None
        ).fit(disp=0)

        sarima_forecast = sarima_model.get_forecast(steps=1, exog=X_forecast)
        forecast_mean = sarima_forecast.predicted_mean.item()
        forecast_std = sarima_forecast.var_pred_mean.item() ** 0.5

        forecasts.append({
            "date": forecast_dates[t],
            "forecast_mean": forecast_mean,
            "forecast_std": forecast_std
        })

        # Append the forecasted value to y using pd.concat()
        y = pd.concat(
            [y, pd.Series([forecast_mean], index=[forecast_dates[t]])])

    return pd.DataFrame(forecasts)

# Main function to run forecasting


def main():
    # Load data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.index.freq = 'D'

    # Target variable and covariates
    # Use only rows with target values
    y = df.loc[df[TARGET_COLUMN].notna(), TARGET_COLUMN]
    covariate_columns = future_covariates_with_lags if INCLUDE_LAGS else future_covariates_no_lags
    available_covariate_columns = [
        col for col in covariate_columns if col in df.columns]
    X = df[available_covariate_columns] if available_covariate_columns else None

    # Load SARIMA model
    with open(model_path, 'rb') as model_file:
        best_model = pickle.load(model_file)

    # Generate forecasts
    forecasts = generate_stepwise_forecast(
        y, X, START_DATE, FORECAST_DAYS, best_model)

    # Save forecast data to TMPDIR
    forecast_path = os.path.join(TMPDIR, f"two_year_forecast_{CHANGE}.csv")
    forecasts.to_csv(forecast_path, index=False)
    print(f"Two-year forecast saved at: {forecast_path}")

    # Plot forecast with historical data using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values,
        mode='lines', name='Historical Data', line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=forecasts["date"], y=forecasts["forecast_mean"],
        mode='lines', name='Forecast', line=dict(color="orange", dash='dash')
    ))
    fig.update_layout(
        title="ARIMA Model Forecast for Next Two Years",
        xaxis_title="Date",
        yaxis_title=TARGET_COLUMN,
        template="plotly_white",
        width=1000,
        height=600
    )

    # Save plot to TMPDIR
    plot_path = os.path.join(TMPDIR, f"arima_final_forecast_{CHANGE}.png")
    fig.write_image(plot_path)
    print(f"Forecast plot saved at: {plot_path}")

    # If running on the cluster, copy the files to the home directory
    if platform.system() == 'Linux':
        shutil.copy(forecast_path, os.path.join(
            home_dir, f"two_year_forecast_{CHANGE}.csv"))
        shutil.copy(plot_path, os.path.join(
            home_dir, f"arima_final_forecast_{CHANGE}.png"))
        print("Files copied to home directory on cluster.")


if __name__ == "__main__":
    main()
