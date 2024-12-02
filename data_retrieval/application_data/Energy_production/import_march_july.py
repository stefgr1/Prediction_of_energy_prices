# %%
import pandas as pd
import numpy as np
import os

# %%
path = os.path.join(os.getcwd(
), '..', '..', '..', 'data/Application_data/Energy_production/', 'energy_production_march_july.csv')

# Use the absolute path for clarity and reliability
absolute_path = os.path.abspath(path)
data_new = pd.read_csv(absolute_path)

# %%
# Convert the 'date_id' columns to datetime
data_new['date_id'] = pd.to_datetime(data_new['date_id'])

# Drop the duplicate 'date_id.1' column
data_new.drop(columns=['date_id.1'], inplace=True)

# Set 'date_id' as the index of the DataFrame
data_new.set_index('date_id', inplace=True)

# rename date_id to date
data_new.index.names = ['date']

# %%
# Drop the second column
data_new.drop(data_new.columns[1], axis=1, inplace=True)

# Drop the last four columns
data_new.drop(data_new.columns[-4:], axis=1, inplace=True)

# %%
# Drop these columns: Conventional, Total electricity demand, Total grid emissions
data_new.drop(columns=['Konventionell', 'Gesamtstromverbrauch',
              'Absolute Emissionen'], inplace=True)

# %% Add the ending (GWh) to all columns
data_new.columns = [col + ' (GWh)' for col in data_new.columns]
# %% Count number of missing values in the data
data_new.isnull().sum()

# %% Create a new column for the sum of all the columns
data_new['Total Prod (GWh)'] = data_new.sum(axis=1)

# %% Compute the total production per year for data_new
data_new['Total Prod (GWh)'].resample('Y').sum()

# %% save data as a csv file
data_new.to_csv(f'{absolute_path[:-4]}_cleaned.csv')
