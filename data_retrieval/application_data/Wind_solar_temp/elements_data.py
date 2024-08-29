# %%

import pandas as pd
import os
from pathlib import Path

# %%

# Get the current working directory (where your script is running from)
current_dir = Path.cwd()

# %%
# Navigate two directories up
up_three_steps = current_dir.parents[2]

# Navigate two directories down, assuming the names 'subdir1' and 'subdir2' are the target directories
down_three_steps = up_three_steps / 'data' / 'Application_data' / 'Elements'

# Print the final path to confirm
print("Final path:", down_three_steps)


# %%

# Dictionary mapping month numbers to names in German
month_map = {
    1: 'Januar', 2: 'Februar', 3: 'März', 4: 'April',
    5: 'Mai', 6: 'Juni', 7: 'Juli', 8: 'August',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Dezember'
}


def load_and_process_data(directory, data_type, start_month, end_month, year):
    """
    Generic function to load and process data files from a specified directory and time range.
    Adaptable for solar, wind, and temperature data.

    Parameters:
    - directory: Directory containing the data files.
    - data_type: Type of data ('Solar', 'Wind', or 'Temp').
    - start_month: Starting month as integer (e.g., 3 for March).
    - end_month: Ending month as integer (e.g., 6 for June).
    - year: Year for which the data is applicable.

    Returns:
    - Dictionary of DataFrames, with keys formatted as '<data_type>_<Month>_<Year>'.
    """
    data_frames = {}
    for month in range(start_month, end_month + 1):
        month_name = month_map[month]
        file_name = f'{data_type}_{month_name}_{year}.csv'
        file_path = os.path.join(directory, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=1)
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            key = f'{data_type}_{month_name}_{year}'
            data_frames[key] = df
        else:
            print(f"File not found: {file_path}")
    return data_frames


def process_data(df, date_column_name, value_column_name, year_prefix='2024'):
    """
    Generic function to process data to calculate daily averages of measurements.

    Parameters:
    - df: DataFrame containing the data.
    - date_column_name: Name of the column containing date information.
    - value_column_name: Name of the column containing measurement values.
    - year_prefix: Prefix string that valid datetime data should start with.

    Returns:
    - Series containing the daily average values.
    """
    df = df[df[date_column_name].astype(str).str.startswith(year_prefix)]
    try:
        df[date_column_name] = pd.to_datetime(
            df[date_column_name], errors='coerce', utc=True)
    except Exception as e:
        print(f"Datetime conversion failed: {e}")
        return None

    df[date_column_name] = df[date_column_name].dt.tz_convert(None)

    if df[date_column_name].isna().any():
        print("Warning: Some dates failed to convert and are NaT.")
        return None

    if df[date_column_name].dtype == 'datetime64[ns]':
        daily_average = df.groupby(df[date_column_name].dt.date)[
            value_column_name].mean()
        return daily_average.round(2)
    else:
        print("Column not in datetime format. Please check the data.")
        return None


def aggregate_data(data_dict, date_column, value_column):
    """
    Aggregates processed dataframes into a single DataFrame.

    Parameters:
    - data_dict: Dictionary of DataFrames.
    - date_column: Name of the date column in the DataFrames.
    - value_column: Name of the value column in the DataFrames.

    Returns:
    - DataFrame containing aggregated daily averages of all data.
    """
    daily_averages = []
    for key, df in data_dict.items():
        daily_avg = process_data(df, date_column, value_column)
        if daily_avg is not None:
            daily_avg = daily_avg.reset_index()
            # Extracting month name from the key
            daily_avg['Month'] = key.split('_')[1]
            daily_averages.append(daily_avg)

    if daily_averages:
        combined_daily_avg = pd.concat(
            daily_averages, axis=0, ignore_index=True)
        combined_daily_avg.drop(columns=['Month'], inplace=True)
        combined_daily_avg.set_index('date', inplace=True)
        return combined_daily_avg
    else:
        return pd.DataFrame()


def combine_data(solar_data, wind_data, temp_data):
    """
    Combines solar, wind, and temperature data into a single DataFrame.

    Parameters:
    - solar_data: DataFrame containing daily averages of solar radiation.
    - wind_data: DataFrame containing daily averages of wind speed.
    - temp_data: DataFrame containing daily averages of temperature.

    Returns:
    - DataFrame containing combined data with solar radiation, wind speed, and temperature.
    """
    # Ensure all data frames are aligned on the same date index
    if solar_data.empty or wind_data.empty or temp_data.empty:
        print("Warning: One or more data frames are empty. Check data loading process.")
        # Return an empty DataFrame if any of the input DataFrames are empty.
        return pd.DataFrame()

    resulting_df = pd.concat([solar_data, wind_data, temp_data], axis=1)

    if resulting_df.empty:
        print("Warning: Combined data is empty. Check individual data sources.")
        return resulting_df

    # Check if resulting_df has the expected number of columns before setting column names
    if resulting_df.shape[1] == 3:
        resulting_df.columns = [
            'Solar Radiation (W/m²)', 'Wind Speed (m/s)', 'Temperature (°C)']
    else:
        print(
            f"Unexpected number of columns in the resulting DataFrame: {resulting_df.shape[1]}")

    return resulting_df


# Assuming you have already loaded and processed the data as shown in previous examples
directory = down_three_steps
solar_data = load_and_process_data(directory, 'Solar', 3, 7, 2024)
wind_data = load_and_process_data(directory, 'Wind', 3, 7, 2024)
temp_data = load_and_process_data(directory, 'Temp', 3, 7, 2024)

aggregated_solar_data = aggregate_data(
    solar_data, 'date', 'Globalstrahlung (W/m²)')
aggregated_wind_data = aggregate_data(
    wind_data, 'date', 'Geschwindigkeit (m/s)')
aggregated_temp_data = aggregate_data(temp_data, 'date', 'Lufttemperatur (°C)')

# Combine the data into one DataFrame
combined_data = combine_data(
    aggregated_solar_data, aggregated_wind_data, aggregated_temp_data)

# Print combined data
print(combined_data)

# Save the combined data to a CSV file
csv_file_path = down_three_steps / 'Elements_March_July.csv'
combined_data.to_csv(csv_file_path)
print("Data saved successfully to './data/Application_data/Elements/Elements_March_July.csv'.")
