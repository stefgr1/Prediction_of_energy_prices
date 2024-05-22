# pylint: skip-file
# isort: skip_file
# black: skip-file

from config_loader import load_config
from EEX_stock.data_utils import merge_data_frames, save_data_frame
from datetime import datetime
import yaml
import pandas as pd
import os
import sys

from pathlib import Path


# Add the parent directory of your `data` module to sys.path
sys.path.append(str(Path(__file__).parent))
os.chdir('/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices')


# Configuration for each data type
data_config = {
    'wind': {
        'input_key': 'wind_input',
        'output_key': 'wind_output',
        'file_prefix': 'wind_data_',
        'output_file_prefix': 'wind_data_daily_',
        'value_column': 'WindSpeed'
    },
    'solar': {
        'input_key': 'solar_input',
        'output_key': 'solar_output',
        'file_prefix': 'SOLAR_data_',
        'output_file_prefix': 'solar_data_daily_',
        'value_column': 'SolarRadiation'
    },
    'temp': {
        'input_key': 'temp_input',
        'output_key': 'temp_output',
        'file_prefix': 'temp_data_',
        'output_file_prefix': 'temp_data_daily_',
        'value_column': 'Temp'
    }
}

# Main function to load, process, and save data based on type
def process_data(data_type):
    home_directory = Path(os.path.expanduser('~')) / 'Documents' / \
        'Masterarbeit' / 'Prediction_of_energy_prices'
    os.chdir(home_directory)

    config = load_config()
    years = range(2006, 2025)

    dataframes = []
    for year in years:
        file_path = Path(config["data"][data_config[data_type]['input_key']]
                         ) / f"{data_config[data_type]['file_prefix']}{year}.csv"
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        dataframes.append(df)

    df_merged = merge_data_frames(dataframes)
    df_merged.rename(columns={
                     'Datum': 'Month', 'value': data_config[data_type]['value_column'], 'y': 'hour of the day'}, inplace=True)

    # Common processing steps for all data types
    df = df_merged.copy()
    month_mapping = {
        'Januar': 'January', 'Februar': 'February', 'März': 'March', 'April': 'April', 'Mai': 'May',
        'Juni': 'June', 'Juli': 'July', 'August': 'August', 'September': 'September',
        'Oktober': 'October', 'November': 'November', 'Dezember': 'December'
    }
    df['Month'] = df['Month'].map(month_mapping)

    total_days = len(df) // 24
    start_date = datetime(2006, 1, 1)
    date_range = pd.date_range(start=start_date, periods=total_days, freq='D')
    df['Date'] = date_range.repeat(24)

    daily_average = df.groupby('Date')[data_config[data_type]['value_column']].mean(
    ).reset_index(name=f'Average{data_config[data_type]["value_column"]}')
    daily_average = daily_average.round(2)

    output_file_name = f"{data_config[data_type]['output_file_prefix']}{years[0]}_{years[-1]}.csv"
    output_path = Path(
        config["data"][data_config[data_type]['output_key']]) / output_file_name
    save_data_frame(daily_average, output_path)
    return daily_average


# Application of the function to each data type
wind = process_data('wind')
solar = process_data('solar')
temp = process_data('temp')

wind.info()
solar.info()
temp.info()
