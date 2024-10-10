# backtesting part:
from darts.utils.utils import SeasonalityMode, TrendMode
model_en = RNNModel.load_from_checkpoint(
    model_name="deep_ar_model", best=True)


# Perform backtesting
backtest = best_model.historical_forecasts(
    series=total_series_transformed,
    future_covariates=day_series,
    start=test_transformed.start_time(),
    forecast_horizon=forecast_horizon,
    stride=30,
    retrain=False,
    verbose=True,
    last_points_only=False,
)

# Inverse transform backtest predictions
backtest = scaler.inverse_transform(backtest)

# Plotting backtest results using Plotly
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=test_series.time_index,
    y=test_series.values().squeeze(),
    mode='lines',
    name='Actual Data'
))

# Add backtest forecast data
fig.add_trace(go.Scatter(
    x=backtest.time_index,
    y=backtest.values().squeeze(),
    mode='lines',
    name='Backtest Forecast'
))

# Update layout
fig.update_layout(
    title='Backtesting Results',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)'
)

# Save the plot as a PNG image
backtest_plot_file_path = os.path.join(script_dir, 'backtest_plot.png')
fig.write_image(backtest_plot_file_path, width=1200, height=600)
