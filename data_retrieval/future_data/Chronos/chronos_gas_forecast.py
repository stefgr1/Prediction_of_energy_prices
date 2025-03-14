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
TARGET_COLUMN = "Other (GWh)"
SIZE = "small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def run_chronos_recursive(data, num_steps):

    # Set the seed 
    transformers.set_seed(42)

    # Load the pretrained model
    chronos_model = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{SIZE}",
        torch_dtype=torch.float32  
    )
    
    # Ensure model is on CPU
    chronos_model.model.to("cpu")
    
    context = data.copy().values.flatten()
    forecasts = []
    forecast_dates = []

    for i in tqdm(range(num_steps)):

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
        final_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecasts})
    return final_df


def plot_forecasts(forecasts, title, original_data):
    """
    Plot the original and forecasted values using Plotly and save the figure.
    """
    # Filter original data to start from January 1, 2022
    original_data = original_data[original_data.index >= "2022-01-01"]
    
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
        height=450,
        xaxis_range=['2022-01-01', forecasts["ds"].max()]  # Ensuring x-axis starts from 2022
    )

    # Save the figure
    output_path = os.path.dirname(os.path.abspath(__file__))
    sanitized_target_column = TARGET_COLUMN.replace(" ", "_").replace("/", "_").replace("€", "EUR")
    fig.write_image(os.path.join(output_path, f"forecast_plot_{sanitized_target_column}_{SIZE}.png"))

def save_forecast_metrics(forecasts):
    """
    Calculate and save forecast metrics (MAE, MSE, RMSE) as a CSV file.
    """
    # Save the forecast values
    output_path = os.path.dirname(os.path.abspath(__file__))
    forecasts.to_csv(os.path.join(output_path, f"forecast_values_{TARGET_COLUMN}_{SIZE}.csv"), index=False)

# Usage in main
def main():
    # Load data
    df = load_and_prepare_data('/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv')
    # Select the needed columns
    data = df[['Date', TARGET_COLUMN]]
    # Transform to a pandas dataframe indexed by 'Date'
    data = data.set_index('Date')

    # Run recursive Chronos forecast
    chronos_forecasts = run_chronos_recursive(data[TARGET_COLUMN], num_steps=730)
    
    # Plot forecasts and save metrics in the same directory as the script
    plot_forecasts(chronos_forecasts, f"Chronos Forecast for {TARGET_COLUMN}", data)
    save_forecast_metrics(chronos_forecasts)

if __name__ == "__main__":
    main()
