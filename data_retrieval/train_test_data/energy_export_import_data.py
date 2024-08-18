# %%
import pandas as pd
import numpy as np
import os

# %%
data_path = './../../data/Power_import_export/storage/'
file_name = 'power_import_export.csv'
file_path = os.path.join(data_path, file_name)
data = pd.read_csv(file_path)
# %% Change the date_id column just to normal date column
data['date_id'] = pd.to_datetime(data['date_id'])

# Rename date_id column to date
data.rename(columns={'date_id': 'date'}, inplace=True)

# %% drop all columns except for the Net Total column and the date column
data.drop(columns=['date_id.1', 'Austria', 'Switzerland', 'Czech Republic',
                   'Denmark', 'France', 'Luxembourg', 'Netherlands', 'Poland',
                   'Power price', 'Sweden', 'Norway', 'Belgium'], inplace=True)
# %%
# set date as index
data.set_index('date', inplace=True)

# %% rename the Net Total column to net_total (GWh)
data.rename(
    columns={'Net Total': 'net_total_export_import (GWh)'}, inplace=True)

# %% save the data to a new csv file with different name
# Save the DataFrame to a new CSV file
new_file_name = 'power_import_export_cleaned.csv'
new_file_path = os.path.join(data_path, new_file_name)
# Set index=False if you don't want to include the index in the CSV file
data.to_csv(new_file_path, index=True)

# %%
