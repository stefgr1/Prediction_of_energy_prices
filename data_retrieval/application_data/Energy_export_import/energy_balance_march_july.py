from pathlib import Path
import pandas as pd


def load_data(file_name, base_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_name (str): Name of the CSV file to load.
        base_path (Path): Base directory where the file is located.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = base_path / file_name
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def preprocess_data(data):
    """
    Preprocess the data by renaming and dropping columns, setting the index, and adjusting column names.

    Parameters:
        data (pd.DataFrame): DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Convert 'date_id' to datetime and rename it to 'date'
    data['date_id'] = pd.to_datetime(data['date_id'])
    data.rename(columns={'date_id': 'date'}, inplace=True)

    # Drop unnecessary columns
    columns_to_drop = [
        'date_id.1', 'Österreich', 'Schweiz', 'Tschechien', 'Dänemark',
        'Frankreich', 'Luxemburg', 'Niederlande', 'Polen', 'Strompreis',
        'Schweden', 'Norwegen', 'Belgien'
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    # Set 'date' as the index
    data.set_index('date', inplace=True)

    # Rename the 'Net Total' column to 'net_total_export_import (GWh)'
    data.rename(
        columns={'Net Total': 'net_total_export_import (GWh)'}, inplace=True)

    return data


def save_data(data, output_file_path):
    try:
        data.to_csv(output_file_path, index=True)
        print(f"Data successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def main():
    """
    Function to load, preprocess, and save electricity import/export data.
    """
    # Define the base directory structure
    current_dir = Path.cwd()
    data_path = current_dir.parents[2] / \
        'data' / 'Application_data' / 'Import_export'

    print("Data path:", data_path)

    # Load the data
    file_name = 'electricity_balance_agora_march_july.csv'
    data = load_data(file_name, data_path)

    if data is not None:
        # Preprocess the data
        processed_data = preprocess_data(data)

        # Save the processed data to a new file
        output_file_name = 'power_import_export_march_july.csv'
        output_file_path = data_path / output_file_name
        save_data(processed_data, output_file_path)


if __name__ == "__main__":
    main()
