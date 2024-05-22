# %%

# importing the needed libraries
import os
import sys
import numpy as np
import pandas as pd

# %%
print("Current Working Directory:", os.getcwd())

# %%
data_path = '../data/EEX_stock/storage/final_data_eex/'
file_name = 'eex_stock_prices_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Function to import data


def import_data(file_path, index_col=None):
    data = pd.read_csv(file_path, sep=',')
    return data


# Import the data
stock_data = import_data(file_path, 0)

# %%
# Import wind data
data_path = '../data/Wind/storage/final_data_wind/'
file_name = 'wind_data_daily_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
wind_data = import_data(file_path, 0)
# %%
# Import solar data
data_path = '../data/Solar_radiation/storage/final_data_solar/'
file_name = 'solar_data_daily_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
solar_data = import_data(file_path, 0)
# %%
# Import the weather data
data_path = '../data/Temperature/storage/final_data_temp'
file_name = 'temp_data_daily_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
temp_data = import_data(file_path, 0)

# %%
# Import the oil prices data
data_path = '../data/Oil_price_brent/Brent_oil_cleaned.csv'
oil_prices = pd.read_csv(data_path)

# %%
# Import the gas prices data
data_path = '../data/Gas_price/storage/TTF/'
file_name = 'TTF_final.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
gas_prices = import_data(file_path, 0)

# Make Unnamed column the index and call "Date"
gas_prices.set_index('Unnamed: 0', inplace=True)
gas_prices.index.names = ['Date']
# Rename the column "CLOSE" to "TTF Gas Price in €/MWh"
gas_prices.rename(columns={'CLOSE': 'TTF_gas_price (EUR/MWh)'}, inplace=True)


# %%

# Merge the data by date
data = stock_data.merge(wind_data, on='Date')
data = data.merge(solar_data, on='Date')
data = data.merge(temp_data, on='Date')
data = data.merge(oil_prices, on='Date')
data = data.merge(gas_prices, on='Date')

# %%
# Checking for missing values in the date column
# First, ensure your 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Generate a complete date range from January 1, 2006, to April 1, 2024
start_date = '2012-01-01'
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
data_path = '../data/BEV_vehicles/storage/BEV_vehicles_per_day.csv'
bev_data = pd.read_csv(data_path)

# %% Now merge the data but make sure to keep the years before 2010 and insert a "0" for missing values in the DailyVehicles column
# Convert the 'Date' column in df1 to datetime if it's not already
bev_data['Date'] = pd.to_datetime(bev_data['Date'])

# Set 'Date' as the index for df1
bev_data.set_index('Date', inplace=True)

# %%
# Merge the data
result = data.join(bev_data, how='left')

# %%
# For any day in 'data' without corresponding 'bev_data', fill 'DailyVehicles' with 0
result['DailyVehicles'] = result['DailyVehicles'].fillna(0)

# remove the digit from the daily vehicles column
result['DailyVehicles'] = result['DailyVehicles'].astype(int)

# rename column to BEV_vehicles
result.rename(columns={'DailyVehicles': 'BEV_vehicles'}, inplace=True)

## ---------------------------------------- Importing the power production data ----------------------------------------##
# %%
data_path = '.Energy_production/storage/final_data_energy/'
file_name = 'power_gen_2012_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
power_data = import_data(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/Energy_production/storage/final_data_energy/power_gen_2012_2024_cleaned.csv', index_col="date")

# %% Set date as index
power_data.set_index('date', inplace=True)


# %%# ---------------------------------------- Importing the power import/export data ----------------------------------------##
data_path = '../data/Power_import_export/storage/'
file_name = 'power_import_export_cleaned.csv'
file_path = os.path.join(data_path, file_name)
power_import_export = pd.read_csv(file_path)

# set date as index
power_import_export.set_index('date', inplace=True)

# Check for any missing values in the data
power_import_export.isnull().sum()

# change the index to datetime
power_import_export.index = pd.to_datetime(power_import_export.index)

# %% Ensure that the index is in datetime format
power_data.index = pd.to_datetime(power_data.index)
result.index = pd.to_datetime(result.index)

# %%# ---------------------------------------- Merging all data ----------------------------------------##
final_data = power_data.join(result, how='left')
final_data = final_data.join(power_import_export, how='left')
# %% Drop all rows that contain missing values
final_data.dropna(inplace=True)

# %% Reorder the data frame by alphabetical order
final_data = final_data.reindex(sorted(final_data.columns), axis=1)

# %% Drop the total column from the data
final_data.drop(columns=['Total Prod (GWh)'], inplace=True)

# %% convert BEV_vehicles to integer
final_data['BEV_vehicles'] = final_data['BEV_vehicles'].astype(int)

# %%
# move the Day_ahead_price column to the first position of all columns
if 'Day_ahead_price' in final_data.columns:
    # Create a new list of columns with 'Day_ahead_price' first
    cols = ['Day_ahead_price'] + \
        [col for col in final_data.columns if col != 'Day_ahead_price']

    # Reorder the DataFrame's columns
    final_data = final_data[cols]

# %%
final_data

# %% Save final data to a csv file
final_data.to_csv('final_data.csv')

# %%
# Search for any missing days in the data
start_date = '2012-01-01'
end_date = '2024-03-01'
date_range = pd.date_range(start=start_date, end=end_date)
date_range_df = pd.DataFrame(date_range, columns=['Date'])
missing_dates = date_range_df[~date_range_df['Date'].isin(final_data.index)]
if missing_dates.empty:
    print("All dates from January 1, 2012, to April 1, 2024, are included in the dataset.")
else:
    print("The following dates are missing from the dataset:")
    print(missing_dates)


# %%
