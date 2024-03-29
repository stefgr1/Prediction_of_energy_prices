import pandas as pd
from pathlib import Path


def load_and_clean_data(year, base_path):
    """Load, clean, and return a DataFrame for a given year."""
    file_path = Path(
        base_path) / f"energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_{year}.csv"
    df_year = pd.read_csv(file_path).dropna()
    df_year = df_year.rename(
        columns={'Tag': 'Date', 'Day Ahead Auktion (volumengewichtet)': 'Price'})
    df_year['Date'] = pd.to_datetime(df_year['Date'], format='%d.%m.%Y')
    return df_year


def merge_data_frames(data_frames):
    """Merge a list of DataFrames into a single DataFrame."""
    return pd.concat(data_frames)


def save_data_frame(df, output_path):
    """Save the DataFrame to the specified output path."""
    df.to_csv(output_path, index=False)
    print(f"Data successfully merged, cleaned, and saved as {output_path}")
