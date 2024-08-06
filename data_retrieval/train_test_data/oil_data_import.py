import pandas as pd
import requests
from pathlib import Path
import matplotlib.pyplot as plt

# Function to get exchange rates


def get_exchange_rate(base, target):
    url = f"https://api.exchangerate-api.com/v4/latest/{base}"
    response = requests.get(url)
    data = response.json()
    return data['rates'][target]


# Set the path to the file
home_directory = Path.home() / 'Documents' / 'Masterarbeit' / \
    'Prediction_of_energy_prices'
file_path = home_directory / 'data' / \
    'Oil_price_brent' / 'storage' / 'Brent_oil.csv'

# Load and process the CSV file
df = pd.read_csv(file_path)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df = df[['Date', 'Close']]

# Convert date and ensure numeric data
df['Date'] = pd.to_datetime(df['Date'], format="%d.%m.%Y")
df['Close'] = pd.to_numeric(df['Close'].str.replace(',', '.'), errors='coerce')

# Sorting and setting the index
df.sort_values(by="Date", inplace=True)
df.set_index("Date", inplace=True)
df.rename(columns={"Close": "Oil price (USD)"}, inplace=True)

# Fetch the exchange rate
usd_to_eur_rate = get_exchange_rate('USD', 'EUR')

# Convert USD prices to EUR
df['Oil_price (EUR)'] = df['Oil price (USD)'] * usd_to_eur_rate

# Drop the USD column
df.drop(columns='Oil price (USD)', inplace=True)

# Create a full date range from min to max date
date_range = pd.date_range(start=df.index.min(), end=df.index.max())
df = df.reindex(date_range)

# Fill missing values with linear interpolation
df['Oil_price (EUR)'] = df['Oil_price (EUR)'].interpolate(method='linear')
df['Oil_price (EUR)'] = df['Oil_price (EUR)'].round(2)

# Plotting
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(df.index, df['Oil_price (EUR)'],
         label='Oil Price (EUR)', color='blue')  # Plot the oil price
plt.title('Trend of the Oil Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price in EUR')
plt.grid(True)  # Enable the grid for better readability
plt.legend()
plt.show()

# Save the cleaned data
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)
df.to_csv(home_directory / 'data' / 'Oil_price_brent' /
          'Brent_oil_cleaned.csv', index=False)
