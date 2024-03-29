# %%
from dwd_scraper import DWDScraper
from dwd_data_utils import merge_station_data, calculate_mean_temperature_per_day
import pandas as pd
import os
from config_loader import load_config
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
merged_data = merge_station_data(dwd_data, stations)

# %%
mean_temperature_data = calculate_mean_temperature_per_day(merged_data)

# Further analysis can be conducted as needed


# if __name__ == '__main__':
# main()

# %%
