# %%

import pandas as pd
from pathlib import Path


def load_and_clean_data(year, base_path):
    file_path = Path(base_path) / f"day_ahead_prices_{year}.csv"
    df = pd.read_csv(file_path, header=1)

    # Renaming and cleaning data
    df = df.rename(columns={'Datum (MEZ)': 'Date'})
    df['Day Ahead Auktion (DE-LU)'] = pd.to_numeric(
        df['Day Ahead Auktion (DE-LU)'], errors='coerce')

    # Drop NaN values that result from conversion issues (or handle them differently if needed)
    df.dropna(subset=['Day Ahead Auktion (DE-LU)'], inplace=True)

    return df


# Define your directory path
# Get the current working directory (where your script is running from)
current_dir = Path.cwd()

# Navigate two directories up
up_three_steps = current_dir.parents[2]

# Navigate two directories down, assuming the names 'subdir1' and 'subdir2' are the target directories
down_three_steps = up_three_steps / 'data' / 'Application_data' / 'Day_Ahead'

# Print the final path to confirm
print("Final path:", down_three_steps)
df_2024 = load_and_clean_data(2024, down_three_steps)
# Delete value for 2023
df_2024 = df_2024.iloc[1:]

# Now process the data


def process_data(df, date_column_name, value_column_name):
    df[date_column_name] = pd.to_datetime(
        df[date_column_name], errors='coerce', utc=True)
    df[date_column_name] = df[date_column_name].dt.tz_convert(None)

    if df[date_column_name].isna().any():
        print("Warning: Some dates failed to convert and are NaT.")
        return None

    daily_average = df.groupby(df[date_column_name].dt.date)[
        value_column_name].mean().round(2)

    # Rename the column to "Day_Ahead_price"
    daily_average.name = 'Day_Ahead_Price (€/MWh)'

    return daily_average


daily_averages_2024 = process_data(
    df_2024, 'Date', 'Day Ahead Auktion (DE-LU)')
print(daily_averages_2024)

# Save the data as csv file
daily_averages_2024.to_csv(
    down_three_steps / 'day_ahead_prices_Jan_July.csv', header=True)


# %%
