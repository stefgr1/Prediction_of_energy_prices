# Processing the eex data
# %%
import os
import sys
import numpy as np
import pandas as pd


# %%
def import_eex_data(file_name):
    data = pd.read_csv(file_name, sep=';')
    return data


# Create the path to the data
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    current_path, 'data/EEX_stock/storage/final_data_eex/')
file_name = 'eex_stock_prices_2006_2024.csv'
file_path = os.path.join(data_path, file_name)

# Import the data
data = import_eex_data(file_path)

# %%
data[['Date', 'Day Ahead Auktion']
     ] = data['Date,Day Ahead Auktion (arithmetisch)'].str.split(',', expand=True)
data = data.drop(columns=['Date,Day Ahead Auktion (arithmetisch)'])

# %%
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)
# %%
# Search for any missing values in the data
missing_values = data.isnull().sum()
missing_values
# %%
data.tail()
# %%
