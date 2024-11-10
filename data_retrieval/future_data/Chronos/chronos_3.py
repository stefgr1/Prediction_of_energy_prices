import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from chronos import ChronosPipeline
import platform

# Global configuration
TARGET_COLUMN = "TTF_gas_price (€/MWh)"
MODEL_SIZE = "large"
CONTEXT_SIZE = 300
TOTAL_PREDICTION_LENGTH = 730
CHUNK_SIZE = 16
SMOOTHING_WINDOW = 15
DEVICE = "cpu"
MODEL_SAVE_DIR = "./data_retrieval/future_data/Chronos/saved_models"

# Load and prepare data


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Replace special characters in TARGET_COLUMN to make it suitable for a file name


def sanitize_filename(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("€", "EUR")

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
        model.model.load_state_dict(torch.load(
            model_path, map_location="cpu", weights_only=True))

        # Move the model to the appropriate device (MPS or CUDA)
        model.model.to(DEVICE)
        return model
    else:
        print(f"Downloading and saving model {size}...")
        model = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{size}",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        # Save model state manually
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        torch.save(model.model.state_dict(), model_path)

        # Move the model to the appropriate device
        model.model.to(DEVICE)
        return model

# Recursive prediction function with limited context update


def recursive_predict(pipeline, context, total_steps, step_size=32, max_context_size=100):
    forecasts = []
    for _ in range(0, total_steps, step_size):
        # Ensure the context tensor is on the correct device
        context = context.to(DEVICE)

        # Predict a chunk of the forecast
        current_forecast = pipeline.predict(
            context, min(step_size, total_steps - len(forecasts)))

        # Ensure forecast tensor is on the same device as context
        current_forecast = current_forecast.to(DEVICE)

        # Move the forecast back to CPU for further processing
        current_forecast_np = current_forecast.cpu().detach().numpy()
        forecasts.append(current_forecast_np[0])

        forecast_flat = torch.tensor(
            current_forecast_np[0].flatten(), dtype=torch.float32).to(DEVICE)
        context = torch.cat(
            [context[-max_context_size:], forecast_flat], dim=0)

    full_forecast = np.concatenate(forecasts, axis=-1)[:total_steps]
    return full_forecast

# Smooth the predictions using a moving average


def smooth_predictions(predictions, window_size):
    smoothed = np.convolve(predictions, np.ones(
        window_size)/window_size, mode='valid')
    pad_width = (len(predictions) - len(smoothed)) // 2
    return np.pad(smoothed, pad_width, mode='edge')

# Plot and save the forecast


def plot_forecast(df, forecast_index, low, mean, mean_smoothed, high, output_path):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df[TARGET_COLUMN], color="#1f77b4",
             linewidth=1.5, label="Actual Data", alpha=0.8)
    plt.plot(forecast_index, mean, color="darkred",
             linewidth=1, label="Predicted Mean", alpha=0.8)
    plt.fill_between(forecast_index, low, high, color="#ffbb78",
                     alpha=0.3, label="80% Prediction Interval")
    plt.title(f"Forecast for {TARGET_COLUMN}",
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
        output_path, f"forecast_plot_{MODEL_SIZE}_{TOTAL_PREDICTION_LENGTH}_{sanitized_target_column}_{CONTEXT_SIZE}_{CHUNK_SIZE}_{SMOOTHING_WINDOW}.png")
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
    plt.close()
    return plot_filename

# Save forecast data to CSV


# Save forecast data to CSV
def save_forecast_to_csv(forecast_dates, low, mean, mean_smoothed, high, output_path):
    # Adjust the length of mean_smoothed to match other arrays if needed
    if len(mean_smoothed) != len(mean):
        pad_width = len(mean) - len(mean_smoothed)
        mean_smoothed = np.pad(mean_smoothed, (0, pad_width), mode='edge')

    # Ensure all arrays are the same length
    if not (len(forecast_dates) == len(low) == len(mean) == len(mean_smoothed) == len(high)):
        raise ValueError(
            "All arrays must be of the same length to save to CSV.")

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
        output_path, f"forecast_values_{MODEL_SIZE}_{TOTAL_PREDICTION_LENGTH}_{sanitized_target_column}_{CONTEXT_SIZE}_{CHUNK_SIZE}_{SMOOTHING_WINDOW}.csv")
    forecast_df.to_csv(csv_filename, index=False)
    return csv_filename


# Main function


def main():
    # Load the dataset
    if platform.system() == "Darwin":
        df = load_and_prepare_data(
            '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Final_data/final_data_july.csv')
    else:
        df = load_and_prepare_data(
            '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Final_data/final_data_july.csv')

    # Initialize the model
    pipeline = initialize_model(MODEL_SIZE)

    # Prepare the context (historical data)
    context = torch.tensor(
        df[TARGET_COLUMN].values[-CONTEXT_SIZE:], dtype=torch.float32).to(DEVICE)

    # Generate the full forecast recursively
    forecast = recursive_predict(
        pipeline, context, TOTAL_PREDICTION_LENGTH, step_size=CHUNK_SIZE, max_context_size=100)

    # Calculate prediction intervals and mean
    low, high = np.quantile(forecast, [0.1, 0.9], axis=0)
    mean = np.mean(forecast, axis=0)

    # Trim to match TOTAL_PREDICTION_LENGTH
    low, high, mean = low[:TOTAL_PREDICTION_LENGTH], high[:
                                                          TOTAL_PREDICTION_LENGTH], mean[:TOTAL_PREDICTION_LENGTH]

    # Smooth the mean forecast
    mean_smoothed = smooth_predictions(mean, window_size=SMOOTHING_WINDOW)

    # Define forecast dates for plotting and saving
    forecast_index = pd.date_range(
        start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=TOTAL_PREDICTION_LENGTH, freq='D')

    # Set output path
    output_path = os.path.dirname(os.path.abspath(__file__))

    # Plot and save forecast
    plot_filename = plot_forecast(
        df, forecast_index, low, mean, mean_smoothed, high, output_path)

    # Save forecast data to CSV, including both normal and smoothed means
    csv_filename = save_forecast_to_csv(
        forecast_index, low, mean, mean_smoothed, high, output_path)

    # Print the paths to confirm saving
    print(f"Plot saved to: {plot_filename}")
    print(f"Forecast values saved to: {csv_filename}")


if __name__ == "__main__":
    main()
