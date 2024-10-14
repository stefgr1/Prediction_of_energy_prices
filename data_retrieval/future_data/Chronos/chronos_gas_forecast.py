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
DEVICE = torch.device("cpu")  # Force CPU usage

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def run_chronos_recursive(data, num_steps):
    # Remove the device_map argument if it causes issues; model defaults to CPU with .to('cpu')
    chronos_model = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{SIZE}",
        torch_dtype=torch.float32  # Use float32 on CPU for compatibility
    )
    
    # Ensure model is on CPU
    chronos_model.model.to("cpu")
    
    context = data.copy().values.flatten()
    forecasts = []
    forecast_dates = []

    for i in tqdm(range(num_steps)):
        transformers.set_seed(42)
        # Convert context to float32 for CPU compatibility
        context_tensor = torch.from_numpy(context).to(torch.float32)
        
        # Generate the one-step-ahead forecast
        chronos_forecast = chronos_model.predict(
            context=context_tensor,
            prediction_length=1,
            num_samples=100
        ).detach().cpu().numpy().flatten()
        
        # Append forecast
        forecasts.append(chronos_forecast[0])

        # Update context by appending the forecasted value
        context = np.append(context, chronos_forecast[0])

        # Generate new date
        last_date = data.index[-1] if i == 0 else forecast_dates[-1]
        new_date = last_date + pd.Timedelta(days=1)
        forecast_dates.append(new_date)

    return pd.DataFrame({'ds': forecast_dates, 'yhat': forecasts})

def plot_forecasts(forecasts, title, output_path, original_data):
    """
    Plot the original and forecasted values using Plotly and save the figure.
    """
    fig = go.Figure()

    # Add original values trace
    fig.add_trace(go.Scatter(
        x=original_data.index,
        y=original_data[TARGET_COLUMN],
        mode='lines',
        line=dict(color="#1f77b4", width=1),
        name="Actual"
    ))

    # Add forecasted values trace
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
        width=1200,
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
    # Transform to a pandas dataframe indexed by 'Date'
    data = data.set_index('Date')

    # Run recursive Chronos forecast
    chronos_forecasts = run_chronos_recursive(data[TARGET_COLUMN], num_steps=730)
    plot_forecasts(chronos_forecasts, f"Chronos Forecast for {TARGET_COLUMN}", "output_path_here", data)

if __name__ == "__main__":
    main()
