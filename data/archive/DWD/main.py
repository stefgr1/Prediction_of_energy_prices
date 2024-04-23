# %%
from dwd_scraper import DWDScraper
from dwd_data_utils import merge_station_data, calculate_mean_temperature_per_day
import pandas as pd
import os
from data.config_loader import load_config
from pathlib import Path

home_directory = Path(os.path.expanduser('~')) / 'Documents' / \
    'Masterarbeit' / 'Prediction_of_energy_prices'
os.chdir(home_directory)

# def main():
config = load_config()
scraper = DWDScraper()

# Example workflow
# scraper.get_all_stations("stations.csv", "2006-01-01", "2023-12-31")

# %%
stations = pd.read_csv(os.path.join(
    config['data']['dwd'], 'meta', 'stations.csv'))
relevant_ids = scraper.get_relevant_station_ids(stations)

# For demonstration, skipping actual scraping due to potential long runtime
# scraper.scrape_hist("dwd.csv", "2006-01-01", "2023-12-31", relevant_ids)
# %%
# Assuming 'dwd.csv' exists from a previous scraping operation
dwd_data = pd.read_csv(os.path.join(config['data']['dwd'], 'dwd.csv'))

# %%
# Data that represents all temperature data for all stations
merged_data = merge_station_data(dwd_data, stations)

# %%
mean_temperature_data = calculate_mean_temperature_per_day(merged_data)

# %%
# Rename the columns for better readability
mean_temperature_data.columns = ['Station_id', 'Date', 'Mean Temperature (°C)']
mean_temperature_data.groupby('Date').mean('Temperature')
# Drop station_id column
mean_temperature_data.drop('Station_id', axis=1, inplace=True)


# %% Import the stock data and merge it with the temperature data
stock_data = pd.read_csv(os.path.join(
    config['data']['eex_saving'], 'eex_stock_prices_2006_2024.csv'))
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_and_temp_data = pd.merge(
    stock_data, mean_temperature_data, on='Date', how='inner')
# Drop all rows with NaN values
stock_and_temp_data.dropna(inplace=True)


# %%
# Count nas in dwd_data
nas = dwd_data.isna().sum()
# %%
