# %% [markdown]
# # Chronos

# %%
import os
import warnings
import transformers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from chronos import ChronosPipeline
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from darts import TimeSeries
import plotly.graph_objs as go
import plotly.io as pio

# %%
# Import the data


# Import the data
def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime.
    """
    df = pd.read_csv(file_path)
    df.sort_values('Date', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
    df.set_index('Date', inplace=True)
    return df


# Now use this function to load the data
df = load_and_prepare_data(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/future_prediction/final_df_constant.csv')

# Extract target column
target_column = "Day_ahead_price (€/MWh)"
data = df[target_column]  # Keep only the target column
data = data.squeeze()  # Convert to series if not already

# Set the prediction start date to the day after the last date in the data
# data.index[-1] + pd.Timedelta(days=1)
start_date = pd.Timestamp("2024-07-29")
end_date = start_date + pd.Timedelta(days=729)


# %%
train_df = load_and_prepare_data('../data/Final_data/train_df_no_lags.csv')
test_df = load_and_prepare_data('../data/Final_data/test_df_no_lags.csv')

train_df.reset_index(inplace=True)
test_df.reset_index(inplace=True)


# %% [markdown]
# ## Chronos Pipeline

# %%
# Define the size of the Chronos model
SIZE = "large"

# %%
chronos_model = ChronosPipeline.from_pretrained(
    f"amazon/chronos-t5-{SIZE}",
    device_map="mps",
    torch_dtype=torch.bfloat16,
)

# %%
# Set the prediction period (730 days)
prediction_length = 730

# Prepare a list for storing the forecasts
chronos_forecasts = []

transformers.set_seed(42)

NUM_SAMPLES = 100

# Save the start time
start_time = datetime.now()

# Loop across the future dates (730 steps into the future)
context = data.values  # Initial context from historical data
print(f"Initial context: {context}")
context = context[:-730]
print(f"Adjusted context without NAs: {context}")
# %%
for t in tqdm(range(data.index.get_loc(start_date), data.index.get_loc(end_date)+1)):

    # extract the context window
    context = data.iloc[:t]

    # Generate the one-step-ahead forecast based on the current context
    chronos_forecast = chronos_model.predict(
        context=torch.from_numpy(context.values),
        prediction_length=1,
        num_samples=NUM_SAMPLES
    ).detach().cpu().numpy().flatten()

    # Get the mean forecast and std dev
    forecast_value = np.mean(chronos_forecast)
    std_dev = np.std(chronos_forecast, ddof=1)

    # Append forecast to the list with the future date
    chronos_forecasts.append({
        "date": data.index[t],
        "mean": forecast_value,
        "std": std_dev,
    })

    # Update context with the forecast value to simulate rolling forecast
    context = np.append(context, forecast_value)
    print(f"Iteration {t+1}, Last context value: {context[-1]}")

# Cast forecasts to DataFrame
chronos_forecasts = pd.DataFrame(chronos_forecasts)

# Save the end time
end_time = datetime.now()
print(f"\nRunning time of Chronos model: {end_time - start_time}")

# %% [markdown]
# ## Display Forecast Data

# %%
chronos_forecasts.head()

# %%
chronos_forecasts.tail()

# %% [markdown]
# ## Plot the Results

# Create traces for actual and predicted values
trace_actual = go.Scatter(
    x=data.index,
    y=data.values,
    mode='lines',
    name='Actual',
    line=dict(color='#3f4751', width=1)
)

trace_predicted = go.Scatter(
    x=chronos_forecasts["date"].values,
    y=chronos_forecasts["mean"].values,
    mode='lines',
    name='Predicted',
    line=dict(color='#009ad3', width=1)
)

# Create traces for confidence intervals
trace_std_1 = go.Scatter(
    x=chronos_forecasts["date"].values,
    y=chronos_forecasts["mean"].values + chronos_forecasts["std"].values,
    mode='lines',
    name='Predicted +/- 1 Std. Dev.',
    line=dict(color='#009ad3', width=0),
    fill='tonexty',
    fillcolor='rgba(0, 154, 211, 0.2)'
)

trace_std_1_neg = go.Scatter(
    x=chronos_forecasts["date"].values,
    y=chronos_forecasts["mean"].values - chronos_forecasts["std"].values,
    mode='lines',
    line=dict(color='#009ad3', width=0),
    showlegend=False,
    fill='tonexty',
    fillcolor='rgba(0, 154, 211, 0.2)'
)

trace_std_2 = go.Scatter(
    x=chronos_forecasts["date"].values,
    y=chronos_forecasts["mean"].values + 2 * chronos_forecasts["std"].values,
    mode='lines',
    name='Predicted +/- 2 Std. Dev.',
    line=dict(color='#009ad3', width=0),
    fill='tonexty',
    fillcolor='rgba(0, 154, 211, 0.1)'
)

trace_std_2_neg = go.Scatter(
    x=chronos_forecasts["date"].values,
    y=chronos_forecasts["mean"].values - 2 * chronos_forecasts["std"].values,
    mode='lines',
    line=dict(color='#009ad3', width=0),
    showlegend=False,
    fill='tonexty',
    fillcolor='rgba(0, 154, 211, 0.1)'
)

# Create the figure with all the traces
fig = go.Figure()
fig.add_trace(trace_actual)
fig.add_trace(trace_predicted)
fig.add_trace(trace_std_2)
fig.add_trace(trace_std_2_neg)
fig.add_trace(trace_std_1)
fig.add_trace(trace_std_1_neg)

# Set layout options
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Value',
    legend=dict(
        x=0.01,       # Position legend in the top-left corner
        y=0.99,       # Position it close to the top of the plot
        xanchor='left',
        yanchor='top',
        # Optional: Add background color with some transparency
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='Black',
        borderwidth=1
    ),
    margin=dict(t=20, b=20, l=20, r=20),
    width=1200,
    height=450
)

# Show the plot
pio.show(fig)


# %%
# save the forecasted values as csv
chronos_forecasts = chronos_forecasts[['date', 'mean']]
# rename the mean column to Day_ahead_price (€/MWh)
chronos_forecasts.rename(
    columns={'mean': 'Day_ahead_price (€/MWh)'}, inplace=True)
chronos_forecasts.to_csv('chronos_forecasts_large.csv', index=False)
# %%
