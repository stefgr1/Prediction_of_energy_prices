# %%
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from calendar import monthrange

# %%
# Load the excel file from the same folder
df = pd.read_excel(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/BEV_vehicles/storage/BEV_vehicles_per_month.xlsx')

# %%
# Change the second column to the month column and use as index
df = df.rename(columns={'Unnamed: 0': 'Month'})
df = df.set_index('Month')

# %%
# Dropping the last two rows
df = df.drop(df.index[-2:])
# change the column headers to strings
df.columns = df.columns.astype(str)
# %%
df
# %%
# First, let's drop the year 2009
df.drop(columns='2009', inplace=True)

# Reshape the DataFrame to a long format
df_long = df.stack().reset_index()
df_long.columns = ['Month', 'Year', 'Vehicles']

# Convert 'Month' from German to English if needed, and then to datetime
month_translation = {
    'Januar': 'January', 'Februar': 'February', 'März': 'March', 'April': 'April',
    'Mai': 'May', 'Juni': 'June', 'Juli': 'July', 'August': 'August',
    'September': 'September', 'Oktober': 'October', 'November': 'November', 'Dezember': 'December'
}
df.index = df.index.map(lambda x: month_translation[x])

# Flatten DataFrame and create a proper datetime index
df_melted = df.melt(ignore_index=False, var_name='Year',
                    value_name='Vehicles').reset_index()
df_melted['Date'] = pd.to_datetime(
    df_melted['Month'] + ' ' + df_melted['Year'])

# Create an empty DataFrame for the final result
final_df = pd.DataFrame()

# Iterate over each row and distribute the vehicles more evenly
for index, row in df_melted.iterrows():
    year, month = row['Date'].year, row['Date'].month
    total_days = monthrange(year, month)[1]
    total_vehicles = row['Vehicles']
    base_daily_vehicles = total_vehicles // total_days
    remainder = total_vehicles % total_days

    # Check if the remainder can be evenly distributed
    if remainder > 0 and ((base_daily_vehicles + 1) * total_days - total_vehicles) >= total_days / 2:
        # If so, distribute the remainder across all days
        daily_vehicles = np.full(total_days, base_daily_vehicles + 1)
        extra_vehicles_to_remove = (
            base_daily_vehicles + 1) * total_days - total_vehicles
        days_to_decrement = np.random.choice(
            range(total_days), size=int(extra_vehicles_to_remove), replace=False)
        for day in days_to_decrement:
            daily_vehicles[day] -= 1
    else:
        # Otherwise, proceed with the original approach
        daily_vehicles = np.full(total_days, base_daily_vehicles)
        daily_vehicles[-1] += remainder  # Adjust the last day to round up

    # Create a date range for the month
    dates = pd.date_range(start=f"{year}-{month}-01", periods=total_days)
    temp_df = pd.DataFrame({'Date': dates, 'DailyVehicles': daily_vehicles})
    final_df = pd.concat([final_df, temp_df])


# Reset the index of the final DataFrame
final_df.reset_index(drop=True, inplace=True)

# %% Drop the data after the 331 of march 2024
# final_df = final_df[final_df['Date'] <= '2024-03-31']

# Changing type of DailyVehicles to integer
final_df['DailyVehicles'] = final_df['DailyVehicles'].astype(int)

# Count the number of daily vehicles
final_df['DailyVehicles'].sum()


# %%
# Save resulting data from as csv file in the same folder
final_df.to_csv(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/BEV_vehicles/storage/BEV_vehicles_per_day.csv', index=False)

# %%
