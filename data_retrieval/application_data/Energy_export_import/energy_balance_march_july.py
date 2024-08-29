# %%
import pandas as pd
from pathlib import Path
import os


# %%
# Load the data
# Get the current working directory (where your script is running from)
current_dir = Path.cwd()

# %%
# Navigate two directories up
up_three_steps = current_dir.parents[2]

# Navigate two directories down, assuming the names 'subdir1' and 'subdir2' are the target directories
down_three_steps = up_three_steps / 'data' / 'Application_data' / 'Import_export'

# Print the final path to confirm
print("Final path:", down_three_steps)
# %%
# Load the data
data = pd.read_csv(down_three_steps /
                   'electricity_balance_agora_march_july.csv')

# %%

# %% Change the date_id column just to normal date column
data['date_id'] = pd.to_datetime(data['date_id'])

# Rename date_id column to date
data.rename(columns={'date_id': 'date'}, inplace=True)

# %% drop all columns except for the Net Total column and the date column
data.drop(columns=['date_id.1', 'Österreich', 'Schweiz', 'Tschechien',
                   'Dänemark', 'Frankreich', 'Luxemburg', 'Niederlande', 'Polen',
                   'Strompreis', 'Schweden', 'Norwegen', 'Belgien'], inplace=True)
# %%
# set date as index
data.set_index('date', inplace=True)

# %% rename the Net Total column to net_total (GWh)
data.rename(
    columns={'Net Total': 'net_total_export_import (GWh)'}, inplace=True)

# %% save the data to a new csv file with different name
# Save the DataFrame to a new CSV file
new_file_name = 'power_import_export_march_july.csv'

# %% save file under new name in the same directory
new_file_path = down_three_steps / new_file_name

# Set index=False if you don't want to include the index in the CSV file
data.to_csv(new_file_path, index=True)
# %%
