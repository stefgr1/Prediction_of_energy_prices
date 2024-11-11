import platform
import pandas as pd
import torch
import plotly.graph_objects as go
from darts.models import TFTModel
from darts import TimeSeries
import shutil
import os

CHANGE = "constant"

# Load the data
df = pd.read_csv(
    f'/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/future_prediction/final_df_{CHANGE}.csv')

# Define target column and covariate columns
target_column = "Day_ahead_price (€/MWh)"
covariate_columns = [
    col for col in df.columns if col not in ["Date", target_column]]

# Determine file paths based on system
if platform.system() == 'Linux':  # Assuming Linux for the cluster
    model_load_path = '/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/predictions/TFT/best_models/best_tft_model_epochs_30_300_no_lags.pth'
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    local_output_path = f'/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/future_prediction/forecast_plot_{CHANGE}.png'
else:  # Assuming MacOS or other local systems
    model_load_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/predictions/TFT/best_models/best_tft_model_epochs_30_300_no_lags.pth'
    tmp_dir = './temp'
    local_output_path = f'./forecast_plot_{CHANGE}.png'

# Ensure temp directory exists
os.makedirs(tmp_dir, exist_ok=True)

# Load the fine-tuned model on CPU
fine_tuned_model = TFTModel.load(
    model_load_path, map_location=torch.device('cuda'))

# Prepare data
series = TimeSeries.from_dataframe(
    df, time_col="Date", value_cols=target_column)
future_covariates = TimeSeries.from_dataframe(
    df, time_col="Date", value_cols=covariate_columns)

# Predict the next two years (730 days) 
n_forecast = 730  # Two years into the future
forecast = fine_tuned_model.predict(
    n=n_forecast, future_covariates=future_covariates)

# Concatenate historical data with forecast for plotting
full_series = series.append(forecast)

# Plotting with Plotly
fig = go.Figure()

# Add historical data to the plot
fig.add_trace(go.Scatter(
    x=series.time_index, y=series.values.flatten(),
    mode='lines', name='Historical Data', line=dict(color="blue")
))

# Add forecast data to the plot
fig.add_trace(go.Scatter(
    x=forecast.time_index, y=forecast.values.flatten(),
    mode='lines', name='Forecast', line=dict(color="orange", dash='dash')
))

# Customize the layout
fig.update_layout(
    title="Target Variable Forecast for Next Two Years",
    xaxis_title="Date",
    yaxis_title=target_column,
    template="plotly_white",
    width=1000,
    height=600
)

# Save plot to the temporary directory
plot_path_tmp = os.path.join(tmp_dir, f'forecast_plot_{CHANGE}.png')
fig.write_image(plot_path_tmp)

# Copy the plot from the temp directory to the desired location
if os.path.exists(plot_path_tmp):
    shutil.copy(plot_path_tmp, local_output_path)
    print(f"Plot saved to: {local_output_path}")
else:
    print("Failed to save plot.")

# Remove temporary plot file if desired
os.remove(plot_path_tmp)
