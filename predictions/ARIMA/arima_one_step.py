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
import pickle

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


def load_and_prepare_data(file_path, target_column, include_lags):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    y = df[target_column]
    covariate_columns = future_covariates_with_lags if include_lags else future_covariates_no_lags

    # Only include columns that exist in df
    available_covariate_columns = [
        col for col in covariate_columns if col in df.columns]
    print(available_covariate_columns)
    X = df[available_covariate_columns] if available_covariate_columns else None
    return y, X


def fit_best_sarima_model(y_train, tmpdir, include_lags):
    best_model = auto_arima(
        y=y_train,
        start_p=0, start_q=0,
        start_P=0, start_Q=0,
        m=7,
        seasonal=True,
        stepwise=True
    )
    print(best_model.summary())

    # Save the best model
    model_save_path = os.path.join(
        tmpdir, f"best_sarima_model_{'with' if include_lags else 'without'}_lags.pkl")
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    print(f"Best SARIMA model saved at: {model_save_path}")

    return best_model


def generate_forecasts(y, X, start_date, end_date, best_model):
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


def plot_forecasts(forecasts, tmpdir, target_column, include_lags):
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
        title=f"SARIMA Forecast for {target_column} {'with' if include_lags else 'without'} Lags",
        xaxis_title="Time", yaxis_title="Value",
        legend=dict(x=1.05, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white",
        width=1500, height=450
    )

    plot_path = os.path.join(
        tmpdir, f"sarima_forecast_plot_{'with' if include_lags else 'without'}_lags.png")
    fig.write_image(plot_path)
    print(f"Forecast plot saved at: {plot_path}")


def calculate_and_save_metrics(forecasts, tmpdir, include_lags):
    rmse_value = mean_squared_error(
        forecasts["actual"], forecasts["mean"], squared=False)
    mae_value = mean_absolute_error(forecasts["actual"], forecasts["mean"])
    mape_value = mean_absolute_percentage_error(
        forecasts["actual"], forecasts["mean"]) * 100
    smape_value = np.mean(
        np.abs(forecasts["actual"] - forecasts["mean"]) /
        ((np.abs(forecasts["actual"]) + np.abs(forecasts["mean"])) / 2)
    ) * 100
    mse_value = mean_squared_error(forecasts["actual"], forecasts["mean"])

    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MSE", "MAPE", "SMAPE"],
        "Value": [mae_value, rmse_value, mse_value, mape_value, smape_value]
    })

    metrics_path = os.path.join(
        tmpdir, f"sarima_error_metrics_{'with' if include_lags else 'without'}_lags.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Error metrics saved at: {metrics_path}")


def save_forecast_results(forecasts, tmpdir, include_lags):
    forecast_path = os.path.join(
        tmpdir, f"forecasted_values_sarima_{'with' if include_lags else 'without'}_lags.csv")
    forecast_df = forecasts[['date', 'mean']].rename(
        columns={'mean': 'Forecasted_price'})
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecasted values saved at: {forecast_path}")


def copy_files_to_results_directory(tmpdir, results_dir, include_lags):
    output_files = [
        os.path.join(
            tmpdir, f"sarima_forecast_plot_{'with' if include_lags else 'without'}_lags.png"),
        os.path.join(
            tmpdir, f"sarima_error_metrics_{'with' if include_lags else 'without'}_lags.csv"),
        os.path.join(
            tmpdir, f"forecasted_values_sarima_{'with' if include_lags else 'without'}_lags.csv"),
        os.path.join(
            tmpdir, f"best_sarima_model_{'with' if include_lags else 'without'}_lags.pkl")
    ]

    for file in output_files:
        if os.path.exists(file):
            shutil.copy(file, results_dir)
            print(f"Copied {file} to {results_dir}")


def load_best_sarima_model(tmpdir, include_lags):
    model_save_path = os.path.join(
        tmpdir, f"best_sarima_model_{'with' if include_lags else 'without'}_lags.pkl")
    with open(model_save_path, 'rb') as model_file:
        best_model = pickle.load(model_file)
    print("Best SARIMA model loaded from:", model_save_path)
    return best_model


def main():
    # Global variables
    TARGET_COLUMN = "Day_ahead_price (€/MWh)"
    START_DATE = "2022-07-01"
    END_DATE = "2024-07-28"
    INCLUDE_LAGS = True
    TMPDIR = os.getenv("TMPDIR", "/tmp")
    # Define the target directory for saved files
    RESULTS_DIR = "/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/predictions/ARIMA/one_step_forecasts"

    # Ensure RESULTS_DIR exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv' if INCLUDE_LAGS else '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_no_lags.csv'

    y, X = load_and_prepare_data(file_path, TARGET_COLUMN, INCLUDE_LAGS)

    # Check if model already exists
    model_save_path = os.path.join(
        TMPDIR, f"best_sarima_model_{'with' if INCLUDE_LAGS else 'without'}_lags.pkl")
    if os.path.exists(model_save_path):
        best_sarima_model = load_best_sarima_model(TMPDIR, INCLUDE_LAGS)
    else:
        best_sarima_model = fit_best_sarima_model(
            y[y.index < START_DATE], TMPDIR, INCLUDE_LAGS)

    forecasts = generate_forecasts(
        y, X, START_DATE, END_DATE, best_sarima_model)

    plot_forecasts(forecasts, TMPDIR, TARGET_COLUMN, INCLUDE_LAGS)

    calculate_and_save_metrics(forecasts, TMPDIR, INCLUDE_LAGS)

    save_forecast_results(forecasts, TMPDIR, INCLUDE_LAGS)

    copy_files_to_results_directory(TMPDIR, RESULTS_DIR, INCLUDE_LAGS)


if __name__ == "__main__":
    main()
