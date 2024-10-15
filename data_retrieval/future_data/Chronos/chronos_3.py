import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from chronos import ChronosPipeline

# Global configuration
TARGET_COLUMN = "Oil_price (EUR)"
SIZE="small"

# Function to load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Load the dataset
df = load_and_prepare_data('/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv')

# Create a ChronosPipeline using the pre-trained 'chronos-t5-tiny' model
pipeline = ChronosPipeline.from_pretrained(
    f"amazon/chronos-t5-{SIZE}",
    device_map="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    torch_dtype=torch.bfloat16  # For memory efficiency
)

# Prepare the context (historical data) - using a larger context window (200 data points)
context = torch.tensor(df[TARGET_COLUMN].values[-200:], dtype=torch.bfloat16)

# Set the total prediction length and a smaller chunk size for smoother predictions
total_prediction_length = 730
chunk_size = 16

# Recursive prediction function
def recursive_predict(pipeline, context, total_steps, step_size=32):
    forecasts = []
    
    for _ in range(0, total_steps, step_size):
        # Predict a chunk of the forecast
        current_forecast = pipeline.predict(context, min(step_size, total_steps - len(forecasts)))
        current_forecast_np = current_forecast.cpu().detach().numpy()
        
        # Append the results
        forecasts.append(current_forecast_np[0])
        
        # Flatten the current forecast output and convert to 1D tensor
        forecast_flat = torch.tensor(current_forecast_np[0].flatten(), dtype=torch.bfloat16)
        
        # Update the context with the new predictions
        context = torch.cat([context, forecast_flat], dim=0)
    
    # Concatenate all forecasted chunks and trim to total_steps
    full_forecast = np.concatenate(forecasts, axis=-1)[:total_steps]
    return full_forecast

# Generate the full forecast recursively
forecast = recursive_predict(pipeline, context, total_prediction_length)

# Calculate prediction intervals and mean
low, high = np.quantile(forecast, [0.1, 0.9], axis=0)
mean = np.mean(forecast, axis=0)

# Trim to match total_prediction_length
low = low[:total_prediction_length]
high = high[:total_prediction_length]
mean = mean[:total_prediction_length]

# Smoothing function for mean forecast
def smooth_predictions(predictions, window_size=5):
    return np.convolve(predictions, np.ones(window_size)/window_size, mode='same')

# Apply smoothing to the mean forecast
mean_smoothed = smooth_predictions(mean, window_size=5)

# Visualize the forecast with styling similar to Plotly
forecast_index = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=total_prediction_length, freq='D')

plt.figure(figsize=(12, 4.5))

# Plot historical data with styling
plt.plot(df['Date'], df[TARGET_COLUMN], color="#1f77b4", linewidth=1, label="Actual")

# Plot smoothed mean forecast with styling
plt.plot(forecast_index, mean_smoothed, color="#ca8a04", linewidth=1, label="Predicted (Smoothed Mean)")

# Fill between the low and high prediction intervals
plt.fill_between(forecast_index, low, high, color="#ca8a04", alpha=0.3, label="80% Prediction Interval")

# Set the title and labels
plt.title(f"Forecast for {TARGET_COLUMN}")
plt.xlabel("Date")
plt.ylabel(TARGET_COLUMN)

# Customize the x-axis to match the specified range
plt.xlim([pd.to_datetime("2022-01-01"), forecast_index[-1]])

# Customize the legend and grid
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.grid(visible=True, linestyle='--', alpha=0.7)

# Save the plot in the same folder as the running file
output_path = os.path.dirname(os.path.abspath(__file__))
plot_filename = os.path.join(output_path, f"forecast_plot_{SIZE}_{total_prediction_length}_{TARGET_COLUMN.replace(' ', '_').replace('(', '').replace(')', '')}.png")
plt.savefig(plot_filename, bbox_inches="tight")
plt.close()  # Close the plot to free memory

# Save the forecasted values as a CSV file
forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=total_prediction_length, freq='D')
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Low_10': low,
    'Mean_Smoothed': mean_smoothed,
    'High_90': high
})
csv_filename = os.path.join(output_path, f"forecast_values_{SIZE}_{total_prediction_length}_{TARGET_COLUMN.replace(' ', '_').replace('(', '').replace(')', '')}.csv")
forecast_df.to_csv(csv_filename, index=False)

# Print the paths to confirm saving
print(f"Plot saved to: {plot_filename}")
print(f"Forecast values saved to: {csv_filename}")
