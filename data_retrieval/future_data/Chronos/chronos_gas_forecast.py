import warnings
import transformers
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from chronos import ChronosPipeline
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import platform

# Global configuration
TARGET_COLUMN = "TTF_gas_price (€/MWh)"
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

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def run_chronos(data, prediction_length):
    """
    Run a Chronos model for time series forecasting.
    """
    chronos_model = ChronosPipeline.from_pretrained(f"amazon/chronos-t5-{SIZE}", device_map=DEVICE, torch_dtype=torch.bfloat16)

    transformers.set_seed(42)

    # Limit prediction length to 64 to avoid degradation of quality.
    prediction_length = min(prediction_length, 64)

    # Perform prediction in a single batch to avoid looping over each time step.
    chronos_forecast = chronos_model.predict(
        context=torch.from_numpy(data.values),
        prediction_length=prediction_length,
        num_samples=100,
        limit_prediction_length=False  # Disable the prediction length check
    ).detach().cpu().numpy().flatten()

    forecast_dates = pd.date_range(data.index[-1], periods=prediction_length + 1, freq='D')[1:]

    # Ensure forecast_dates and chronos_forecast are of the same length
    min_length = min(len(forecast_dates), len(chronos_forecast))
    forecast_dates = forecast_dates[:min_length]
    chronos_forecast = chronos_forecast[:min_length]

    return pd.DataFrame({'ds': forecast_dates, 'yhat': chronos_forecast})


def plot_forecasts(forecasts, title, output_path):
    """
    Plot the forecasted values using Plotly and save the figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecasts["ds"],
        y=forecasts["yhat"],
        mode='lines',
        line=dict(color="#ca8a04", width=1),
        name="Predicted"
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

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the figure
    sanitized_target_column = TARGET_COLUMN.replace(" ", "_").replace("/", "_").replace("€", "EUR")
    fig.write_image(os.path.join(output_path, f"forecast_plot_{sanitized_target_column}.png"))



def main():
    # Load data
    df = load_and_prepare_data('/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv')
    # Select the needed columns
    data = df[['Date', TARGET_COLUMN]]
    # Transform to a pandas series
    data = data.set_index('Date')
    data = data.squeeze()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Run Chronos forecast
    chronos_forecasts = run_chronos(data, prediction_length=64)  # Limit to 64 for better performance
    plot_forecasts(chronos_forecasts, f"Chronos Forecast for {TARGET_COLUMN}", script_dir)

if __name__ == "__main__":
    main()
