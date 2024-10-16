import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from chronos import ChronosPipeline
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from sklearn.preprocessing import MinMaxScaler

# Global configuration
TARGET_COLUMN = "Wind_offshore (GWh)"
MODEL_SIZE = "large"
TOTAL_PREDICTION_LENGTH = 730
SMOOTHING_WINDOW = 5
DEVICE = "cpu"
MODEL_SAVE_DIR = "./saved_models"

# Load, scale, and prepare data
def load_and_prepare_data(file_path, TARGET_COLUMN):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Scaling the target column
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[TARGET_COLUMN] = scaler.fit_transform(df[[TARGET_COLUMN]])
    
    return df, scaler

# Replace special characters in TARGET_COLUMN to make it suitable for a file name
def sanitize_filename(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\u20ac", "EUR")

# Initialize the model, loading from saved file if available
def initialize_model(size):
    model_path = os.path.join(MODEL_SAVE_DIR, f"chronos-t5-{size}.pt")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{size}",
            device_map="cpu",  # Load to CPU initially
            torch_dtype=torch.float32
        )
        model.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.model.to(DEVICE)
        return model
    else:
        print(f"Downloading and saving model {size}...")
        model = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{size}",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        torch.save(model.model.state_dict(), model_path)
        model.model.to(DEVICE)
        return model

# Recursive prediction function with limited context update
def recursive_predict(pipeline, context, total_steps, step_size=32, max_context_size=100):
    forecasts = []
    for _ in range(0, total_steps, step_size):
        context = context.to(DEVICE)

        # Predict a chunk of the forecast
        current_forecast = pipeline.predict(
            context, min(step_size, total_steps - len(forecasts))
        ).to(DEVICE)

        current_forecast_np = current_forecast.cpu().detach().numpy()
        forecasts.append(current_forecast_np[0])

        forecast_flat = torch.tensor(
            current_forecast_np[0].flatten(), dtype=torch.float32
        ).to(DEVICE)
        context = torch.cat(
            [context[-max_context_size:], forecast_flat], dim=0
        )

    full_forecast = np.concatenate(forecasts, axis=-1)[:total_steps]
    return full_forecast

# Smooth the predictions using a moving average
def smooth_predictions(predictions, window_size):
    smoothed = np.convolve(predictions, np.ones(
        window_size)/window_size, mode='valid')
    pad_width = (len(predictions) - len(smoothed)) // 2
    return np.pad(smoothed, pad_width, mode='edge')

# Plot and save the forecast
def plot_forecast(df, forecast_index, low, mean, mean_smoothed, high, output_path, context_size, chunk_size):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df[TARGET_COLUMN], color="#1f77b4",
             linewidth=1.5, label="Actual Data", alpha=0.8)
    plt.plot(forecast_index, mean, color="darkred",
             linewidth=1, label="Predicted Mean", alpha=0.8)
    plt.fill_between(forecast_index, low, high, color="#ffbb78",
                     alpha=0.3, label="80% Prediction Interval")
    plt.title(f"Forecast for {TARGET_COLUMN} (Context Size: {context_size}, Chunk Size: {chunk_size})",
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(TARGET_COLUMN, fontsize=14)
    plt.xlim([pd.to_datetime("2022-01-01"), forecast_index[-1]])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1),
               fontsize=12, frameon=True, shadow=True)
    plt.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.6)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sanitized_target_column = sanitize_filename(TARGET_COLUMN)
    plot_filename = os.path.join(
        output_path, f"forecast_plot_{MODEL_SIZE}_{TOTAL_PREDICTION_LENGTH}_{sanitized_target_column}_{context_size}_{chunk_size}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    return plot_filename

# Save forecast data to CSV
def save_forecast_to_csv(forecast_dates, low, mean, mean_smoothed, high, output_path, context_size, chunk_size):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sanitized_target_column = sanitize_filename(TARGET_COLUMN)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Low_10': low,
        'Mean': mean,
        'Mean_Smoothed': mean_smoothed,
        'High_90': high
    })
    csv_filename = os.path.join(
        output_path, f"forecast_values_{MODEL_SIZE}_{TOTAL_PREDICTION_LENGTH}_{sanitized_target_column}_{context_size}_{chunk_size}.csv")
    forecast_df.to_csv(csv_filename, index=False)
    return csv_filename

# Main function to run forecasts with scaling
def run_forecast_thread(context_size, chunk_size, df, scaler, pipeline):
    context = torch.tensor(
        df[TARGET_COLUMN].values[-context_size:], dtype=torch.float32).to(DEVICE)

    # Generate the full forecast recursively
    forecast = recursive_predict(
        pipeline, context, TOTAL_PREDICTION_LENGTH, step_size=chunk_size, max_context_size=100)

    # Calculate prediction intervals and mean
    low, high = np.percentile(forecast, [10, 90], axis=0)
    mean = np.mean(forecast, axis=0)

    # Inverse transform the predictions
    low, high, mean = scaler.inverse_transform(low.reshape(-1, 1)).flatten(), \
                      scaler.inverse_transform(high.reshape(-1, 1)).flatten(), \
                      scaler.inverse_transform(mean.reshape(-1, 1)).flatten()

    # Smooth the mean forecast
    mean_smoothed = smooth_predictions(mean, window_size=SMOOTHING_WINDOW)

    # Define forecast dates for plotting and saving
    forecast_index = pd.date_range(
        start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=TOTAL_PREDICTION_LENGTH, freq='D')

    # Set output path
    output_path = os.path.dirname(os.path.abspath(__file__))

    # Plot and save forecast
    plot_filename = plot_forecast(
        df, forecast_index, low, mean, mean_smoothed, high, output_path, context_size, chunk_size)

    # Save forecast data to CSV, including both normal and smoothed means
    csv_filename = save_forecast_to_csv(
        forecast_index, low, mean, mean_smoothed, high, output_path, context_size, chunk_size)

    return plot_filename, csv_filename

if __name__ == "__main__":
    # Load the dataset and scale it
    if platform.system() == "Darwin":
        df, scaler = load_and_prepare_data(
            '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Final_data/final_data_july.csv', TARGET_COLUMN)
    else:
        df, scaler = load_and_prepare_data(
            '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv', TARGET_COLUMN)

    # Define combinations of context sizes and chunk sizes to test
    context_sizes = [100, 200, 300, 400, 500]
    chunk_sizes = [16, 32, 64, 128, 256]
    combinations = list(itertools.product(context_sizes, chunk_sizes))

    # Initialize the model once
    pipeline = initialize_model(MODEL_SIZE)

    # Run forecasts in parallel with scaling and the shared pipeline
    with ThreadPoolExecutor(max_workers=25) as executor:
        future_to_combination = {executor.submit(run_forecast_thread, context_size, chunk_size, df, scaler, pipeline): (context_size, chunk_size)
                                 for context_size, chunk_size in combinations}

        for future in as_completed(future_to_combination):
            context_size, chunk_size = future_to_combination[future]
            try:
                plot_filename, csv_filename = future.result()
                print(f"Completed: Context Size = {context_size}, Chunk Size = {chunk_size}")
                print(f"Plot saved to: {plot_filename}")
                print(f"CSV saved to: {csv_filename}")
            except Exception as exc:
                print(f"Exception for Context Size = {context_size}, Chunk Size = {chunk_size}: {exc}")
