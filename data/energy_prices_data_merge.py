import pandas as pd
from pathlib import Path


def load_and_clean_data(year, base_path):
    """Load, clean, and return a DataFrame for a given year."""
    file_path = base_path / \
        f"energy_prices_each_year/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_{year}.csv"
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


def main():
    base_path = Path(
        "/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/")
    years = [2019, 2020, 2021, 2022, 2023]

    # Use a list comprehension to load, clean, and store DataFrames
    dataframes = [load_and_clean_data(year, base_path) for year in years]

    # Merge the DataFrames into one
    df_merged = merge_data_frames(dataframes)

    # Construct the output file path
    output_file_name = f"energy_prices_{years[0]}_{years[-1]}.csv"
    output_path = base_path / f"energy_prices_all_years/{output_file_name}"

    # Save the cleaned and merged DataFrame
    save_data_frame(df_merged, output_path)

    # Directly display the head of the DataFrame
    return df_merged.head()


if __name__ == "__main__":
    df_head = main()
    print(df_head)
