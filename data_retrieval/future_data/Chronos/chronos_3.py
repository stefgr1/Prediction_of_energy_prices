import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from chronos import ChronosPipeline

# Global configuration
TARGET_COLUMN = "Wind_speed (m/s)"
MODEL_SIZE = "small"
CONTEXT_SIZE = 200
TOTAL_PREDICTION_LENGTH = 730
CHUNK_SIZE = 16
SMOOTHING_WINDOW = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Initialize the model
def initialize_model(size):
    return ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{size}",
        device_map=DEVICE,
        torch_dtype=torch.bfloat16  # For memory efficiency
    )

# Recursive prediction function with limited context update
def recursive_predict(pipeline, context, total_steps, step_size=32, max_context_size=100):
    forecasts = []
    for _ in range(0, total_steps, step_size):
        current_forecast = pipeline.predict(context, min(step_size, total_steps - len(forecasts)))
        current_forecast_np = current_forecast.cpu().detach().numpy()
        forecasts.append(current_forecast_np[0])
        
        forecast_flat = torch.tensor(current_forecast_np[0].flatten(), dtype=torch.bfloat16)
        context = torch.cat([context[-max_context_size:], forecast_flat], dim=0)
    
    full_forecast = np.concatenate(forecasts, axis=-1)[:total_steps]
    return full_forecast

# Smooth the predictions using a moving average
def smooth_predictions(predictions, window_size):
    smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')
    pad_width = (len(predictions) - len(smoothed)) // 2
    return np.pad(smoothed, pad_width, mode='edge')

# Plot and save the forecast
def plot_forecast(df, forecast_index, low, mean_smoothed, high, output_path):
    plt.figure(figsize=(12, 4.5))
    plt.plot(df['Date'], df[TARGET_COLUMN], color="#1f77b4", linewidth=1, label="Actual")
    plt.plot(forecast_index, mean_smoothed, color="#ca8a04", linewidth=1, label="Predicted (Smoothed Mean)")
    plt.fill_between(forecast_index, low, high, color="#ca8a04", alpha=0.3, label="80% Prediction Interval")
    plt.title(f"Forecast for {TARGET_COLUMN}")
    plt.xlabel("Date")
    plt.ylabel(TARGET_COLUMN)
    plt.xlim([pd.to_datetime("2022-01-01"), forecast_index[-1]])
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    
    plot_filename = os.path.join(output_path, f"forecast_plot_{MODEL_SIZE}_{TOTAL_PREDICTION_LENGTH}_{TARGET_COLUMN.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()
    return plot_filename

# Save forecast data to CSV
def save_forecast_to_csv(forecast_dates, low, mean_smoothed, high, output_path):
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Low_10': low,
        'Mean_Smoothed': mean_smoothed,
        'High_90': high
    })
    csv_filename = os.path.join(output_path, f"forecast_values_{MODEL_SIZE}_{TOTAL_PREDICTION_LENGTH}_{TARGET_COLUMN.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
    forecast_df.to_csv(csv_filename, index=False)
    return csv_filename

# Main function to execute forecasting process
def main():
    # Load the dataset
    df = load_and_prepare_data('/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv')
    
    # Initialize the model
    pipeline = initialize_model(MODEL_SIZE)
    
    # Prepare the context (historical data)
    context = torch.tensor(df[TARGET_COLUMN].values[-CONTEXT_SIZE:], dtype=torch.bfloat16)
    
    # Generate the full forecast recursively
    forecast = recursive_predict(pipeline, context, TOTAL_PREDICTION_LENGTH, step_size=CHUNK_SIZE, max_context_size=100)
    
    # Calculate prediction intervals and mean
    low, high = np.quantile(forecast, [0.1, 0.9], axis=0)
    mean = np.mean(forecast, axis=0)
    
    # Trim to match TOTAL_PREDICTION_LENGTH
    low, high, mean = low[:TOTAL_PREDICTION_LENGTH], high[:TOTAL_PREDICTION_LENGTH], mean[:TOTAL_PREDICTION_LENGTH]
    
    # Smooth the mean forecast
    mean_smoothed = smooth_predictions(mean, window_size=SMOOTHING_WINDOW)
    
    # Define forecast dates for plotting and saving
    forecast_index = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=TOTAL_PREDICTION_LENGTH, freq='D')
    
    # Set output path
    output_path = os.path.dirname(os.path.abspath(__file__))
    
    # Plot and save forecast
    plot_filename = plot_forecast(df, forecast_index, low, mean_smoothed, high, output_path)
    
    # Save forecast data to CSV
    csv_filename = save_forecast_to_csv(forecast_index, low, mean_smoothed, high, output_path)
    
    # Print the paths to confirm saving
    print(f"Plot saved to: {plot_filename}")
    print(f"Forecast values saved to: {csv_filename}")

if __name__ == "__main__":
    main()
