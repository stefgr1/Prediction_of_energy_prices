# %%
import pandas as pd
import numpy as np
import os

# %%
# Navigate one directory up from the current working directory and then into the 'data' directory
path = os.path.join(os.getcwd(
), '..', 'data/Energy_production/storage/Per_year/', 'power_gen_2012_2024.csv')

# Use the absolute path for clarity and reliability
absolute_path = os.path.abspath(path)

# Now 'absolute_path' can be used to access your file
data_new = pd.read_csv(absolute_path)

# %%
# Convert the 'date_id' columns to datetime
data_new['date_id'] = pd.to_datetime(data_new['date_id'])

# Drop the duplicate 'date_id.1' column if it's identical to 'date_id'
# Ensure this is correct for your dataset before dropping
data_new.drop(columns=['date_id.1'], inplace=True)

# Set 'date_id' as the index of the DataFrame
data_new.set_index('date_id', inplace=True)

# rename date_id to date
data_new.index.names = ['date']

# %%
# Drop the second column (index 1 as pandas is 0-based index)
data_new.drop(data_new.columns[1], axis=1, inplace=True)

# Drop the last four columns
data_new.drop(data_new.columns[-4:], axis=1, inplace=True)

# %%
# Drop these columns: Conventional	Total electricity demand	Total grid emissions
data_new.drop(columns=['Conventional', 'Total electricity demand',
              'Total grid emissions'], inplace=True)

# %% Add the ending (GWh) to all columns
data_new.columns = [col + ' (GWh)' for col in data_new.columns]
# %% COunt number of missing values in the data
data_new.isnull().sum()

# %% Create a new column for the sum of all the columns
data_new['Total Prod (GWh)'] = data_new.sum(axis=1)

# %% Compute the total production per year for data_new
data_new['Total Prod (GWh)'].resample('Y').sum()

# %% save the data as a csv file
data_new.to_csv(f'{absolute_path[:-4]}_cleaned.csv')
