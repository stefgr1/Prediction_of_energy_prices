import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import itertools
import os
import plotly.graph_objs as go
import time
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR)

# Covariates to be used for prediction
covariates = ['Solar_radiation (W/m2)', 'Temperature (°C)', 
              'Biomass (GWh)', 'Hard_coal (GWh)', 'Hydro (GWh)', 'Lignite (GWh)', 
              'Natural_gas (GWh)', 'Other (GWh)', 'Pumped_storage_generation (GWh)', 
              'Solar_energy (GWh)', 'Net_total_export_import (GWh)']

# Load and prepare data function
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('D')  # Ensure daily frequency
    return df

# Define the parameter grid
param_dist = {  
    'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
    'seasonality_prior_scale': [0.01, 0.05, 1, 5, 7, 10],
    'holidays_prior_scale': [0.01, 0.05, 1, 5, 7, 10],
    'seasonality_mode': ['additive', 'multiplicative'],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_dist.keys(), v)) for v in itertools.product(*param_dist.values())]

# Function to run Prophet with a specific parameter set
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

save_dir = "/pfs/data5/home/tu/tu_tu/tu_zxoul27/Prediction_of_energy_prices/data/Future_data/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Loop through each covariate
for covariate in covariates:
    print(f"Processing: {covariate}")
    
    # Load and prepare data
    df = load_and_prepare_data('../../data/Final_data/final_data_july.csv')
    
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
    
    # Save tuning results
    tuning_results = pd.DataFrame([result[0] for result in results])
    tuning_results['average_rmse'] = rmses
    tuning_results.to_csv(f'{save_dir}{covariate}_tuning_results.csv', index=False)
    
    # Apply the best parameter configuration to the model
    weekly_seasonality = True if "GWh" in covariate else False  # Set weekly_seasonality based on covariate name
    
    m = Prophet(
        seasonality_mode=best_params['seasonality_mode'],
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale'], 
        seasonality_prior_scale=best_params['seasonality_prior_scale'], 
        yearly_seasonality="auto",
        weekly_seasonality=weekly_seasonality,  
        daily_seasonality=False
    )
    m.add_country_holidays(country_name='DE')
    m.fit(df_prophet)
    
    # Make future predictions for 2 years
    future = m.make_future_dataframe(periods=730)
    forecast = m.predict(future)
    
    # Plot the forecast and save it
    fig = plot_plotly(m, forecast)
    fig.update_layout(
        title=f"Forecasted {covariate}",
        xaxis_title="Date",
        yaxis_title=f"{covariate}",
        template="plotly_white"
    )
    fig.write_image(f'{save_dir}{covariate}_forecast.png')  # Save plot
    
    # Save the forecasted values
    forecast_final = forecast[['ds', 'yhat']].copy()
    forecast_final.rename(columns={'ds': 'Date', 'yhat': covariate}, inplace=True)
    forecast_final = forecast_final.round(2)
    forecast_final.set_index('Date', inplace=True)
    forecast_final.to_csv(f'{save_dir}forecasted_{covariate}.csv')

print("All covariates processed.")
