import requests
import pandas as pd
from datetime import datetime

# API base URL
BASE_URL = "https://api.energy-charts.info/price?bzn=DE-LU&start=2023-01-01T00%3A00%2B01%3A00&end=2023-01-02T23%3A45%2B01%3A00"

# Function to fetch prices for a given year


def fetch_prices_for_year(bidding_zone, start_date, end_date):
    endpoint = f"{BASE_URL}"
    params = {
        "bzn": bidding_zone,
        "start": start_date.isoformat(),
        "end": end_date.isoformat()
    }
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data: Status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Main function to iterate over years and fetch data


def download_price_data(bidding_zone='DE-LU'):
    """Download energy price data for the given bidding zone and save to CSV """
    all_data = []
    for year in range(2020, 2023):
        print(f"Fetching data for {year}...")
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        yearly_data = fetch_prices_for_year(bidding_zone, start_date, end_date)
        if yearly_data:
            all_data.extend(yearly_data)
        else:
            print(f"No data fetched for {year}.")

    # Convert data to DataFrame and save to CSV if any data was fetched
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("energy_prices_2005_2023.csv", index=False)
        print("Data successfully downloaded and saved.")
    else:
        print("No data was downloaded.")


# Uncomment the following line to run the script with the actual base URL and correct parameters
download_price_data()
