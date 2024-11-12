import warnings
import transformers
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from chronos import ChronosPipeline
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
import os
import platform

# Global configuration
TARGET_COLUMN = "Day_ahead_price (€/MWh)"
SIZE = "tiny"
# Check the platform and assign the device accordingly
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

START_DATE = "2022-07-01"
END_DATE = "2024-07-28"

def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and set 'Date' as index.
    """
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def run_sarima(data, start_date, end_date):
    """
    Perform SARIMA forecasting with hyperparameter tuning using auto_arima, and one-step ahead forecasts.
    """
    best_sarima_model = auto_arima(
        y=data[data.index < start_date],
        start_p=0, start_q=0,
        start_P=0, start_Q=0,
        m=12, seasonal=True,
    )

    sarima_forecasts = []

    for t in tqdm(range(data.index.get_loc(start_date), data.index.get_loc(end_date) + 1)):
        context = data.iloc[:t]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima_model = SARIMAX(
                endog=context,
                order=best_sarima_model.order,
                seasonal_order=best_sarima_model.seasonal_order,
                trend="c" if best_sarima_model.with_intercept else None
            ).fit(disp=0)

        sarima_forecast = sarima_model.get_forecast(steps=1)
        sarima_forecasts.append({
            "date": data.index[t],
            "actual": data.values[t],
            "mean": sarima_forecast.predicted_mean.item(),
            "std": sarima_forecast.var_pred_mean.item() ** 0.5,
        })

    return pd.DataFrame(sarima_forecasts)

def plot_forecasts(forecasts, title):
    """
    Plot the actual vs. forecasted values using Plotly.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecasts["date"],
        y=forecasts["actual"],
        mode='lines',
        line=dict(color="#3f4751", width=1),
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=forecasts["date"],
        y=forecasts["mean"],
        mode='lines',
        line=dict(color="#ca8a04", width=1),
        name="Predicted"
    ))

    fig.add_trace(go.Scatter(
        x=forecasts["date"],
        y=forecasts["mean"] + forecasts["std"],
        fill=None,
        mode='lines',
        line=dict(color="#ca8a04", width=0.5),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecasts["date"],
        y=forecasts["mean"] - forecasts["std"],
        fill='tonexty',
        mode='lines',
        line=dict(color="#ca8a04", width=0.5),
        name="Predicted +/- 1 Std. Dev.",
        opacity=0.2
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=TARGET_COLUMN,
        legend=dict(x=1.05, y=1),
        template="plotly",
        width=800,
        height=450
    )
    fig.show()

def evaluate_forecast(forecasts, model_name, output_dir):
    """
    Evaluate the forecasts using common metrics and save them to a CSV file.
    """
    rmse = np.sqrt(mean_squared_error(forecasts["actual"], forecasts["mean"]))
    mae = mean_absolute_error(forecasts["actual"], forecasts["mean"])
    metrics_df = pd.DataFrame([
        {"Metric": "RMSE", "Value": rmse},
        {"Metric": "MAE", "Value": mae}
    ]).set_index("Metric")

    # Save metrics to CSV
    metrics_file_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_file_path)

    return metrics_df

def run_chronos(data, start_date, end_date):
    """
    Run a Chronos model for time series forecasting.
    """
    chronos_model = ChronosPipeline.from_pretrained(f"amazon/chronos-t5-{SIZE}", device_map=DEVICE, torch_dtype=torch.bfloat16)

    chronos_forecasts = []

    for t in tqdm(range(data.index.get_loc(start_date), data.index.get_loc(end_date) + 1)):
        context = data.iloc[:t]
        transformers.set_seed(42)

        chronos_forecast = chronos_model.predict(
            context=torch.from_numpy(context.values),
            prediction_length=1,
            num_samples=100
        ).detach().cpu().numpy().flatten()

        chronos_forecasts.append({
            "date": data.index[t],
            "actual": data.values[t],
            "mean": np.mean(chronos_forecast),
            "std": np.std(chronos_forecast, ddof=1),
        })

    return pd.DataFrame(chronos_forecasts)

def main():
    # Load data
    df = load_and_prepare_data('../../data/Final_data/final_data_july.csv')
    # Select the needed columns
    data = df[['Date', TARGET_COLUMN]]
    # transform to a pandas series
    data = data.set_index('Date')
    data = data.squeeze()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Run SARIMA forecast
    sarima_forecasts = run_sarima(data, START_DATE, END_DATE)
    plot_forecasts(sarima_forecasts, f"SARIMA Forecast for {TARGET_COLUMN}")
    sarima_metrics = evaluate_forecast(sarima_forecasts, "sarima", script_dir)
    print(sarima_metrics)

    # Run Chronos forecast
    chronos_forecasts = run_chronos(data, START_DATE, END_DATE)
    plot_forecasts(chronos_forecasts, f"Chronos Forecast for {TARGET_COLUMN}")
    chronos_metrics = evaluate_forecast(chronos_forecasts, "chronos", script_dir)
    print(chronos_metrics)

if __name__ == "__main__":
    main()