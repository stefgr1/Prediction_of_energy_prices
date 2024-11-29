from pathlib import Path
import pandas as pd


def load_and_clean_data(year, base_path):
    """
    Load and clean the day-ahead prices data for a given year.

    Parameters:
        year (int): The year of the data to load.
        base_path (Path): The base directory where the data is stored.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    file_path = Path(base_path) / f"day_ahead_prices_{year}.csv"
    try:
        df = pd.read_csv(file_path, header=1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    # Rename and clean the data
    df = df.rename(columns={'Datum (MEZ)': 'Date'})
    df['Day Ahead Auktion (DE-LU)'] = pd.to_numeric(
        df['Day Ahead Auktion (DE-LU)'], errors='coerce'
    )

    # Drop rows with NaN values in the target column
    df.dropna(subset=['Day Ahead Auktion (DE-LU)'], inplace=True)

    return df


def process_data(df, date_column_name, value_column_name):
    """
    Process the data to compute daily average values.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        date_column_name (str): The name of the column with date information.
        value_column_name (str): The name of the column with value information.

    Returns:
        pd.Series: A series with daily average prices.
    """
    # Convert dates to datetime, removing time zone information
    df[date_column_name] = pd.to_datetime(
        df[date_column_name], errors='coerce', utc=True)
    df[date_column_name] = df[date_column_name].dt.tz_convert(None)

    if df[date_column_name].isna().any():
        print("Warning: Some dates failed to convert and are NaT.")
        return None

    # Calculate daily averages and rename the column
    daily_average = df.groupby(df[date_column_name].dt.date)[
        value_column_name].mean().round(2)
    daily_average.name = 'Day_Ahead_Price (€/MWh)'

    return daily_average


def save_to_csv(data, output_path):
    try:
        data.to_csv(output_path, header=True)
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


# Define the base directory structure
current_dir = Path.cwd()
data_path = current_dir.parents[2] / 'data' / 'Application_data' / 'Day_Ahead'

print("Data path:", data_path)

# Load and clean data for 2024
df_2024 = load_and_clean_data(2024, data_path)
if df_2024 is not None:
    # Exclude the first row (for 2023)
    df_2024 = df_2024.iloc[1:]

    # Process the data to compute daily averages
    daily_averages_2024 = process_data(
        df_2024, 'Date', 'Day Ahead Auktion (DE-LU)')
    if daily_averages_2024 is not None:
        # Define output file path
        output_file = data_path / 'day_ahead_prices_Jan_July.csv'

        # Save the processed data to a CSV file
        save_to_csv(daily_averages_2024, output_file)
