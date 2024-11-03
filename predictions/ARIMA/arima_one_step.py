import os
import shutil
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# Define the future covariates columns from your dataframe
future_covariates_no_lags = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)', 'Temperature (°C)',
                             'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)', 'Lignite (GWh)',
                             'Natural_gas (GWh)', 'Other (GWh)', 'Pumped_storage_generation (GWh)',
                             'Solar_energy (GWh)', 'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
                             'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
                             'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Day_of_week', 'Month']

future_covariates_with_lags = ['Solar_radiation (W/m2)', 'Wind_speed (m/s)',
                               'Temperature (°C)', 'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)',
                               'Lignite (GWh)', 'Natural_gas (GWh)', 'Other (GWh)',
                               'Pumped_storage_generation (GWh)', 'Solar_energy (GWh)',
                               'Wind_offshore (GWh)', 'Wind_onshore (GWh)',
                               'Net_total_export_import (GWh)', 'BEV_vehicles', 'Oil_price (EUR)',
                               'TTF_gas_price (€/MWh)', 'Nuclear_energy (GWh)', 'Lag_1_day',
                               'Lag_2_days', 'Lag_3_days', 'Lag_4_days', 'Lag_5_days', 'Lag_6_days',
                               'Lag_7_days', 'Day_of_week', 'Month', 'Rolling_mean_7']

# Global variables
TARGET_COLUMN = "Day_ahead_price (€/MWh)"
START_DATE = "2024-07-26"
END_DATE = "2024-07-28"
INCLUDE_LAGS = True  # Set to True if you want to include lags
TMPDIR = os.getenv("TMPDIR", "/tmp")
HOME_DIR = os.path.expanduser("~")


def load_and_prepare_data(file_path, include_lags):
    """
    Load and prepare the dataset for forecasting.

    Args:
        file_path (str): Path to the data file.
        include_lags (bool): Flag indicating whether to include lags.

    Returns:
        tuple: A tuple containing the target series (y) and covariates (X).
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    y = df[TARGET_COLUMN]
    covariate_columns = future_covariates_with_lags if include_lags else future_covariates_no_lags
    X = df[covariate_columns] if covariate_columns else None
    return y, X


def fit_best_sarima_model(y_train):
    """
    Fit the best SARIMA model using auto_arima on the training data.

    Args:
        y_train (pd.Series): Training data for the target variable.

    Returns:
        pmdarima.ARIMA: The fitted SARIMA model.
    """
    best_model = auto_arima(
        y=y_train,
        start_p=0, start_q=0,
        start_P=0, start_Q=0,
        m=7,
        seasonal=True,
        stepwise=True
    )
    print(best_model.summary())
    return best_model


def generate_forecasts(y, X, start_date, end_date, best_model):
    """
    Generate one-step-ahead forecasts iteratively over the forecast period.

    Args:
        y (pd.Series): Full time series data.
        X (pd.DataFrame): Covariates for the time series.
        start_date (str): Start date for forecasting.
        end_date (str): End date for forecasting.
        best_model (pmdarima.ARIMA): The fitted SARIMA model.

    Returns:
        pd.DataFrame: DataFrame containing dates, actual values, mean forecasts, and standard deviations.
    """
    forecasts = []
    for t in tqdm(range(y.index.get_loc(start_date), y.index.get_loc(end_date) + 1)):
        y_train = y.iloc[:t]
        X_train = X.iloc[:t] if X is not None else None
        X_forecast = X.iloc[[t]] if X is not None else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima_model = SARIMAX(
                endog=y_train,
                exog=X_train,
                order=best_model.order,
                seasonal_order=best_model.seasonal_order,
                trend="c" if best_model.with_intercept else None
            ).fit(disp=0)

        sarima_forecast = sarima_model.get_forecast(steps=1, exog=X_forecast)
        forecasts.append({
            "date": y.index[t],
            "actual": y.values[t],
            "mean": sarima_forecast.predicted_mean.item(),
            "std": sarima_forecast.var_pred_mean.item() ** 0.5,
        })

    return pd.DataFrame(forecasts)


def plot_forecasts(forecasts, include_lags):
    """
    Plot actual vs forecasted values with confidence intervals.

    Args:
        forecasts (pd.DataFrame): DataFrame containing dates, actual values, mean forecasts, and standard deviations.
        include_lags (bool): Flag indicating if lags were included in the model.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecasts["date"], y=forecasts["actual"],
        mode='lines', line=dict(color="#3f4751", width=1),
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=forecasts["date"], y=forecasts["mean"],
        mode='lines', line=dict(color="#ca8a04", width=1),
        name="Forecast"
    ))

    fig.add_trace(go.Scatter(
        x=forecasts["date"], y=forecasts["mean"] + forecasts["std"],
        fill=None, mode='lines', line=dict(color="#ca8a04", width=0.5), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecasts["date"], y=forecasts["mean"] - forecasts["std"],
        fill='tonexty', mode='lines', line=dict(color="#ca8a04", width=0.5),
        name="Forecast +/- 1 Std. Dev.", opacity=0.2
    ))

    fig.update_layout(
        title=f"SARIMA Forecast for {TARGET_COLUMN} {'with' if include_lags else 'without'} Lags",
        xaxis_title="Time", yaxis_title="Value",
        legend=dict(x=1.05, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white",
        width=1500, height=450
    )

    # Save the plot as PNG in TMPDIR
    plot_path = os.path.join(
        TMPDIR, f"sarima_forecast_plot_{'with' if include_lags else 'without'}_lags.png")
    fig.write_image(plot_path)
    print(f"Forecast plot saved at: {plot_path}")


def calculate_and_save_metrics(forecasts, include_lags):
    """
    Calculate error metrics and save them to a CSV file.

    Args:
        forecasts (pd.DataFrame): DataFrame with actual and predicted values.
        include_lags (bool): Flag indicating if lags were included in the model.
    """
    rmse_value = mean_squared_error(
        forecasts["actual"], forecasts["mean"], squared=False)
    mae_value = mean_absolute_error(forecasts["actual"], forecasts["mean"])
    mape_value = mean_absolute_percentage_error(
        forecasts["actual"], forecasts["mean"])
    smape_value = np.mean(
        np.abs(forecasts["actual"] - forecasts["mean"]) /
        ((np.abs(forecasts["actual"]) + np.abs(forecasts["mean"])) / 2)
    ) * 100
    mse_value = mean_squared_error(forecasts["actual"], forecasts["mean"])

    metrics_df = pd.DataFrame({
        "Metric": ["MAPE", "MAE", "RMSE", "MSE", "SMAPE"],
        "Value": [mape_value, mae_value, rmse_value, mse_value, smape_value]
    })

    # Save metrics to TMPDIR
    metrics_path = os.path.join(
        TMPDIR, f"sarima_error_metrics_{'with' if include_lags else 'without'}_lags.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Error metrics saved at: {metrics_path}")


def save_forecast_results(forecasts, include_lags):
    """
    Save forecasted values to a CSV file.

    Args:
        forecasts (pd.DataFrame): DataFrame containing dates and forecasted values.
        include_lags (bool): Flag indicating if lags were included in the model.
    """
    forecast_path = os.path.join(
        TMPDIR, f"forecasted_values_sarima_{'with' if include_lags else 'without'}_lags.csv")
    forecast_df = forecasts[['date', 'mean']].rename(
        columns={'mean': 'Forecasted_price'})
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecasted values saved at: {forecast_path}")


def copy_files_to_home_directory():
    """
    Copy all output files from TMPDIR to the home directory.
    """
    output_files = [
        os.path.join(
            TMPDIR, f"sarima_forecast_plot_{'with' if INCLUDE_LAGS else 'without'}_lags.png"),
        os.path.join(
            TMPDIR, f"sarima_error_metrics_{'with' if INCLUDE_LAGS else 'without'}_lags.csv"),
        os.path.join(
            TMPDIR, f"forecasted_values_sarima_{'with' if INCLUDE_LAGS else 'without'}_lags.csv")
    ]

    for file in output_files:
        if os.path.exists(file):
            shutil.copy(file, HOME_DIR)
            print(f"Copied {file} to {HOME_DIR}")


def main():
    # Determine file path based on the include_lags flag
    file_path = '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_no_lags.csv' if not INCLUDE_LAGS else '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv'

    # Load and prepare data
    y, X = load_and_prepare_data(file_path, INCLUDE_LAGS)

    # Fit SARIMA model
    best_sarima_model = fit_best_sarima_model(y[y.index < START_DATE])

    # Generate forecasts
    forecasts = generate_forecasts(
        y, X, START_DATE, END_DATE, best_sarima_model)

    # Plot forecasts and save to TMPDIR
    plot_forecasts(forecasts, INCLUDE_LAGS)

    # Calculate and save error metrics to TMPDIR
    calculate_and_save_metrics(forecasts, INCLUDE_LAGS)

    # Save forecasted values to TMPDIR
    save_forecast_results(forecasts, INCLUDE_LAGS)

    # Copy all files from TMPDIR to the home directory
    copy_files_to_home_directory()


if __name__ == "__main__":
    main()
