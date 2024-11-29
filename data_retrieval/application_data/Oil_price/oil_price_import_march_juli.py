import pandas as pd
import requests
from pathlib import Path
import matplotlib.pyplot as plt


def get_exchange_rate(base_currency, target_currency):
    """
    Fetches the exchange rate from the base currency to the target currency using an API.

    Parameters:
        base_currency (str): The base currency (e.g., 'USD').
        target_currency (str): The target currency (e.g., 'EUR').

    Returns:
        float: The exchange rate.
    """
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        return data['rates'][target_currency]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching exchange rates: {e}")
        return None


def load_and_process_oil_data(file_path):
    """
    Loads and processes oil price data from a CSV file.

    Parameters:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with cleaned and converted data.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Select and clean the required columns
    df = df[['Date', 'Price']]
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y", errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.rename(columns={"Price": "Oil price (USD)"}, inplace=True)

    # Sort by date and set as index
    df.sort_values(by="Date", inplace=True)
    df.set_index("Date", inplace=True)

    return df


def process_and_save_oil_data(df, exchange_rate, output_path):
    """
    Converts oil prices to a different currency, interpolates missing values, 
    and saves the processed data to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing oil prices in USD.
        exchange_rate (float): Conversion rate from USD to EUR.
        output_path (Path): Path to save the cleaned data.
    """
    # Convert prices to EUR
    df['Oil_price (EUR)'] = df['Oil price (USD)'] * exchange_rate

    # Drop the USD column
    df.drop(columns='Oil price (USD)', inplace=True)

    # Create a full date range and reindex
    date_range = pd.date_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(date_range)

    # Fill missing values with linear interpolation and round to 2 decimals
    df['Oil_price (EUR)'] = df['Oil_price (EUR)'].interpolate(method='linear')
    df['Oil_price (EUR)'] = df['Oil_price (EUR)'].round(2)

    # Save the cleaned data
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


def plot_oil_price_trend(df):
    """
    Plots the trend of oil prices over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing oil price data.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Oil_price (EUR)'],
             label='Oil Price (EUR)', color='blue')
    plt.title('Trend of the Oil Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price in EUR')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    """
    Main function to load, process, and save oil price data.
    """
    # Define the file paths
    home_directory = Path.home() / 'Documents' / 'Masterarbeit' / \
        'Prediction_of_energy_prices'
    file_path = home_directory / 'data' / 'Application_data' / \
        'Oil_price' / 'Oil_historic_data_march_july.csv'
    output_path = home_directory / 'data' / 'Application_data' / \
        'Oil_price' / 'Brent_oil_cleaned.csv'

    # Load and process the oil data
    oil_data = load_and_process_oil_data(file_path)

    if oil_data is not None:
        # Fetch the exchange rate
        usd_to_eur_rate = get_exchange_rate('USD', 'EUR')
        if usd_to_eur_rate is not None:
            # Process and save the data
            process_and_save_oil_data(oil_data, usd_to_eur_rate, output_path)

            # Plot the trend
            plot_oil_price_trend(oil_data)
        else:
            print("Exchange rate could not be retrieved. Exiting.")
    else:
        print("Oil data could not be loaded. Exiting.")


if __name__ == "__main__":
    main()
