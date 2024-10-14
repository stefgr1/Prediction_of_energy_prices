import os
import time
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from concurrent.futures import ProcessPoolExecutor
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(filename='error.log', level=logging.ERROR)

# Load and prepare data function
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')  # Ensure daily frequency
    return df

# Sanitize filenames to avoid issues with special characters in paths
def sanitize_filename(covariate):
    # Replace special characters with underscores or remove them
    return covariate.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')

# Define the parameter grid
param_dist = {  
    'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.1, 0.25, 0.5],
    'seasonality_prior_scale': [0.01, 0.05, 1, 5, 7, 10],
    'holidays_prior_scale': [0.01, 0.05, 1, 5, 7, 10],
    'seasonality_mode': ['additive', 'multiplicative'],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_dist.keys(), v)) for v in itertools.product(*param_dist.values())]

# Get the current task ID from the SLURM environment variable
slurm_task_id = int(os.getenv('SLURM_PROCID', '0'))

# Total number of tasks (should match --ntasks in Slurm script)
total_tasks = int(os.getenv('SLURM_NTASKS', '1'))

# Covariates to be used for prediction
#covariates = ['Solar_radiation (W/m2)', 'Temperature (°C)', 'Biomass (GWh)', 
#              'Hard_coal (GWh)', 'Hydro (GWh)', 'Lignite (GWh)', 
#              'Natural_gas (GWh)', 'Other (GWh)', 'Pumped_storage_generation (GWh)', 
#              'Solar_energy (GWh)', 'Net_total_export_import (GWh)']

covariates = ["Oil_price (EUR)","TTF_gas_price (€/MWh)"]

# Split covariates using np.array_split to handle uneven splits
assigned_covariates = np.array_split(covariates, total_tasks)[slurm_task_id]

# Directory to save results
save_dir = "/pfs/data5/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Future_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Function to evaluate parameters and Prophet model
def evaluate_params(params_df):
    params, df_prophet = params_df
    try:
        m = Prophet(**params).fit(df_prophet)
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
        df_p = performance_metrics(df_cv, rolling_window=0.1)
        avg_rmse = df_p['rmse'].mean()
        return params, avg_rmse
    except Exception as e:
        logging.error(f"Error with params {params}: {e}")
        return params, float('inf')

# Corporate identity settings for plots
SIZE_DEFAULT = 14
SIZE_LARGE = 16
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font weight
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

# Custom color palette to use for plots
corporate_colors = ["#2B2F42", "#8D99AE", "#EF233C"]

# Run the process for each covariate assigned to this task
for covariate in covariates:
    print(f"Task {slurm_task_id} processing covariate: {covariate}")
    
    # Load and prepare data
    df = load_and_prepare_data('../../../data/Final_data/final_data_july.csv')
    
    # Prepare data for Prophet
    df_prophet = pd.DataFrame({
        'ds': df.index, 
        'y': df[covariate]
    })

    # Start hyperparameter tuning
    start_time = time.time()

    # Use ProcessPoolExecutor and map with the new named function
    with ProcessPoolExecutor() as executor:
        # Pass both params and df_prophet to the evaluation function
        params_df_pairs = [(params, df_prophet) for params in all_params]
        results = list(tqdm(executor.map(evaluate_params, params_df_pairs), total=len(all_params)))

    end_time = time.time()
    print(f"Total tuning time for {covariate}: {end_time - start_time:.2f} seconds")
    
    # Extract the best parameters and save results
    rmses = [result[1] for result in results]
    best_params = all_params[np.argmin(rmses)]
    print(f"Best params for {covariate}: {best_params}")

    # Save tuning results using sanitized filenames
    sanitized_covariate = sanitize_filename(covariate)
    tuning_results = pd.DataFrame([result[0] for result in results])
    tuning_results['average_rmse'] = rmses
    tuning_results.to_csv(f'{save_dir}{sanitized_covariate}_tuning_results.csv', index=False)
    
    # Apply the best parameter configuration to the model
    weekly_seasonality = True if "GWh" in covariate else False  # Set weekly_seasonality based on covariate name
    
    m = Prophet(
        seasonality_mode=best_params['seasonality_mode'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale'], 
        seasonality_prior_scale=best_params['seasonality_prior_scale'], 
        yearly_seasonality=True,
        weekly_seasonality=weekly_seasonality,  
        daily_seasonality=False, 
    )
    m.add_country_holidays(country_name='DE')
    m.fit(df_prophet)
    
    # Make future predictions for 2 years
    future = m.make_future_dataframe(periods=730)
    forecast = m.predict(future)

    # Save forecast results (and potentially aggregate later)
    forecast_final = forecast[['ds', 'yhat']].copy()
    forecast_final.rename(columns={'ds': 'Date', 'yhat': covariate}, inplace=True)
    forecast_final = forecast_final.round(2)
    forecast_final.set_index('Date', inplace=True)
    forecast_final.to_csv(f'{save_dir}forecasted_{sanitized_covariate}.csv')

    # Plot the original data and forecast with improved aesthetics
    fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size

    # Plot the original data and forecast
    ax.plot(df.index, df[covariate], label="Original Data", color=corporate_colors[0], linewidth=2)
    ax.plot(forecast_final.index, forecast_final[covariate], label="Forecast", color=corporate_colors[2], linewidth=2)

    # Customize the plot
    ax.set_title(f'{covariate} - Forecast for the next two years', fontsize=SIZE_LARGE, fontweight='bold')
    ax.set_xlabel("Date", fontsize=SIZE_LARGE)
    ax.set_ylabel(covariate, fontsize=SIZE_LARGE)
    ax.legend(loc="best")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Format the date axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Use tight layout to prevent cutoffs
    plt.tight_layout()

    # Save the plot with sanitized filenames
    plot_filename = f'{save_dir}{sanitized_covariate}_forecast_plot.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')  # Use bbox_inches to include all elements
    plt.close()

    print(f"Plot saved: {plot_filename}")

print("All covariates processed.")
