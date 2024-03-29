import os
import yaml
from pathlib import Path
from data_utils import load_and_clean_data, merge_data_frames, save_data_frame


def main():
    # Set the working directory to the project folder
    home_directory = Path(os.path.expanduser('~')) / 'Documents' / \
        'Masterarbeit' / 'Prediction_of_energy_prices'
    os.chdir(home_directory)

    # Load configuration
    config_path = home_directory / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    years = range(2006, 2025)
    dataframes = [load_and_clean_data(
        year, config["data"]["eex_input"]) for year in years]
    df_merged = merge_data_frames(dataframes)

    output_file_name = f"eex_stock_prices_{years[0]}_{years[-1]}.csv"
    output_path = Path(config["data"]["eex_saving"]) / output_file_name
    save_data_frame(df_merged, output_path)

    print(df_merged.head())


if __name__ == "__main__":
    main()
