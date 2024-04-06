# %%

# importing the needed libraries
import os
import sys
import numpy as np
import pandas as pd

# %%

print("Current Working Directory:", os.getcwd())

# %%

# Assuming 'data' directory is at the same level as your notebook
data_path = './EEX_stock/storage/final_data_eex/'
file_name = 'eex_stock_prices_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Function to import data


def import_data(file_path, index_col):
    data = pd.read_csv(file_path, sep=',')
    return data


# Import the data
stock_data = import_data(file_path)

# %%
# Import wind data
data_path = './Wind/storage/final_data_wind/'
file_name = 'wind_data_daily_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
wind_data = import_data(file_path)
# %%
# Import solar data
data_path = './Solar_radiation/storage/final_data_solar/'
file_name = 'solar_data_daily_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
solar_data = import_data(file_path)
# %%
# Import the weather data
data_path = './Temperature/storage/final_data_temp'
file_name = 'temp_data_daily_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
temp_data = import_data(file_path)

# %%
# Import the oil prices data
data_path = './Oil_price_brent/Brent_oil_cleaned.csv'
oil_prices = pd.read_csv(data_path)

# %%

# Merge the data by date
data = stock_data.merge(wind_data, on='Date')
data = data.merge(solar_data, on='Date')
data = data.merge(temp_data, on='Date')
data = data.merge(oil_prices, on='Date')

# %%
# Checking for missing values in the date column
# First, ensure your 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Generate a complete date range from January 1, 2006, to April 1, 2024
start_date = '2006-01-01'
end_date = '2024-04-01'
date_range = pd.date_range(start=start_date, end=end_date)

# Convert date_range to DataFrame for easy comparison
date_range_df = pd.DataFrame(date_range, columns=['Date'])

# Check if any date in the generated range is missing from your dataset
missing_dates = date_range_df[~date_range_df['Date'].isin(data['Date'])]

if missing_dates.empty:
    print("All dates from January 1, 2006, to April 1, 2024, are included in the dataset.")
else:
    print("The following dates are missing from the dataset:")
    print(missing_dates)


# %%
# rename the column "close" to oil_price and the Day Ahead Auktion just Day Ahead price
data.rename(columns={'Close': 'Oil_price',
            'Day Ahead Auktion (arithmetisch)': 'Day_ahead_price'}, inplace=True)
data.rename(columns={'AverageWindSpeed': 'Wind_speed',
            'AverageSolarRadiation': 'Solar_radiation', 'AverageTemp': 'Temperature'}, inplace=True)
# Using the data column as index
data.set_index('Date', inplace=True)
# %%
data
# %% Import the BEV_vehicle data and merge it with the existing data
data_path = './BEV_vehicles/storage/BEV_vehicles_per_day.csv'
bev_data = pd.read_csv(data_path)

# %% Now merge the data but make sure to keep the years before 2010 and insert a "0" for missing values in the DailyVehicles column
# Convert the 'Date' column in df1 to datetime if it's not already
bev_data['Date'] = pd.to_datetime(bev_data['Date'])

# Set 'Date' as the index for df1
bev_data.set_index('Date', inplace=True)

# %%
# Your df2 is already indexed by 'Date', so we can proceed to merge
# Merge df1 and df2 on their index ('Date')
result = data.join(bev_data, how='left')

# %%
# For any day in 'data' without corresponding 'bev_data', fill 'DailyVehicles' with 0
result['DailyVehicles'] = result['DailyVehicles'].fillna(0)

# remove the digit from the daily vehicles column
result['DailyVehicles'] = result['DailyVehicles'].astype(int)

# rename column to BEV_vehicles
result.rename(columns={'DailyVehicles': 'BEV_vehicles'}, inplace=True)

# %% import power_gen_2012_2024.csv
data_path = '.Energy_production/storage/final_data_energy/'
file_name = 'power_gen_2012_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
power_data = import_data(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Energy_production/storage/final_data_energy/power_gen_2012_2024_cleaned.csv', index_col="date")

# %% set date as index
power_data.set_index('date', inplace=True)

# %%
