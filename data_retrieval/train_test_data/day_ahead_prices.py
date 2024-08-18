# %%
# Load in all files called
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# List to store each year's data as a DataFrame
data_list = []

# Load in the data for each year and append it to the list
for year in range(2006, 2025):
    file_path = f'./../../data/EEX_stock/storage/prices_per_year/energy-charts_Tägliche_Börsenstrompreise_in_Deutschland_{year}.csv'
    try:
        yearly_data = pd.read_csv(file_path)
        yearly_data['Year'] = year  # Add a column to keep track of the year
        data_list.append(yearly_data)
    except FileNotFoundError:
        print(f"File for {year} not found. Skipping this year.")

# Concatenate all the yearly DataFrames into a single DataFrame
data = pd.concat(data_list, ignore_index=True)

# Display the first few rows of the combined DataFrame
print(data.head())

# %%
# Drop the first row
data = data.drop(0)
# Rename "Tag" to "Date"
data = data.rename(columns={"Tag": "Date"})
# rename "Day Ahead Auktion (arithmetisch)" to "Day_ahead_price (€/MWh)"
data = data.rename(
    columns={"Day Ahead Auktion (arithmetisch)": "Day_ahead_price (€/MWh)"})
# Drop the "Year" column
data = data.drop(columns='Year')
# Drop all NA values
data = data.dropna()

# Convert the "Date" column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
# Set the "Date" column as the index
data.set_index('Date', inplace=True)
# Display the first few rows of the data
data.head()

# %%
# Import the missing data
missing_data = pd.read_csv(
    './../../data/EEX_stock/storage/prices_per_year/gap_data_2015.csv')
# %%
# Keep only the columns "date_id" and "Strompreis"
missing_data = missing_data[['date_id', 'Strompreis']]

# Rename the "Strompreis" column to "Day_ahead_price (€/MWh)"
missing_data = missing_data.rename(
    columns={'Strompreis': 'Day_ahead_price (€/MWh)'})

# %%
# Create a copy of the DataFrame to avoid SettingWithCopyWarning
missing_data = missing_data.copy()

# Convert the 'date_id' column to datetime format
missing_data['date_id'] = pd.to_datetime(missing_data['date_id'])

# Rename the 'date_id' column to 'Date'
missing_data = missing_data.rename(columns={'date_id': 'Date'})

# Set the 'Date' column as the index
missing_data.set_index('Date', inplace=True)

# remove 2014-12-31, 2015-01-05	, and 2015-01-06 from the missing data
missing_data = missing_data.drop(['2014-12-31', '2015-01-05', '2015-01-06'])

# %%
# Display the first few rows to confirm the conversion
missing_data


# %%
# Merge the missing data with the main data
data = pd.concat([data, missing_data])
# %%
data

# Search for the dates first oif january 2015 to the fifth of january 2015
data.loc['2014-12-31':'2015-01-07']

# %%
# Search for duplicate dates in the data
duplicate_dates = data[data.index.duplicated()]
duplicate_dates

# %%
# Sort the data by the index
data = data.sort_index()

# %%
# Search for the dates with missing values
missing_dates = data[data['Day_ahead_price (€/MWh)'].isnull()]
# %%
missing_dates
# %%
# Round all the values in the 'Day_ahead_price (€/MWh)' column to 2 decimal places
# Convert the 'Day_ahead_price (€/MWh)' column to numeric, coercing any errors to NaN
data['Day_ahead_price (€/MWh)'] = pd.to_numeric(
    data['Day_ahead_price (€/MWh)'], errors='coerce')

# Rounding all the values in the 'Day_ahead_price (€/MWh)' column to 2 decimal places
data['Day_ahead_price (€/MWh)'] = data['Day_ahead_price (€/MWh)'].round(2)

# Display the first few rows to confirm the rounding
print(data.head())

# %%
# Save the data to a CSV file
data.to_csv('./../../data/EEX_stock/storage/final_data_eex/day_ahead_prices.csv')
# %%
