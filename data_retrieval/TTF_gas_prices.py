# %%
import pandas as pd
import holidays
import os
import matplotlib.pyplot as plt
# %%
# Load data


def import_data(file_path, index_col=None):
    """Import data from CSV with optional parsing of dates."""
    data = pd.DataFrame()
    if index_col is not None:
        data = pd.read_csv(file_path, sep=',', parse_dates=[index_col])
    else:
        data = pd.read_csv(file_path, sep=',')
    return data


# %%
# File path configuration
data_path = '../data/Gas_price/storage/TTF/'
file_name = 'TTF.csv'
file_path = os.path.join(data_path, file_name)

# Data processing
data = import_data(file_path)
print(data.head())
# %%
# COunt number of NAs
print(data.isna().sum())
# SHow the missing data
print(data[data['CLOSE'].isna()])

# %%


def analyze_missing_dates(date_range, data_frame):
    """Analyze missing dates for weekends and holidays."""
    if 'Date' in data_frame.columns:
        dates = data_frame['Date']
    else:
        dates = data_frame.index

    missing_dates = date_range[~date_range.isin(dates)]
    missing_dates_df = pd.DataFrame(missing_dates, columns=['Missing Dates'])
    missing_dates_df['Is Weekend'] = missing_dates_df['Missing Dates'].apply(
        lambda x: x.weekday() in [5, 6])
    de_holidays = holidays.Germany(years=range(2012, 2025))
    missing_dates_df['Is Holiday'] = missing_dates_df['Missing Dates'].apply(
        lambda x: x in de_holidays)
    return missing_dates_df


# %%
# Look for any missing dates
date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max())
missing_dates_df = analyze_missing_dates(date_range, data)
print(missing_dates_df)
# %%
# Show the dates that are not weekends or holidays
missing_dates_df[~(missing_dates_df['Is Weekend'] |
                   missing_dates_df['Is Holiday'])]
# %%
# Fill missing data with average


def fill_missing_data_with_average(data, date_col='Date', value_col='CLOSE'):
    """Fill missing data by averaging the nearest available values for missing days and existing NaNs."""
    # Ensure the Date column is in datetime format
    data[date_col] = pd.to_datetime(data[date_col])

    # Set 'Date' as the index
    data.set_index(date_col, inplace=True)

    # Generate a full range from min to max date found in the data
    full_range = pd.date_range(start=data.index.min(), end=data.index.max())

    # Reindex the data to include missing dates - fills with NaN for missing entries
    data = data.reindex(full_range)

    # Identify all dates that need to be filled (both missing dates and those with NaN values)
    dates_to_fill = data[data[value_col].isnull()].index

    # Fill in NaN values by averaging adjacent days
    for date in dates_to_fill:
        before = data.loc[:date].dropna()
        after = data.loc[date:].dropna()

        prev_value = before[value_col].last_valid_index()
        next_value = after[value_col].first_valid_index()

        if pd.notnull(prev_value) and pd.notnull(next_value):
            average_value = (
                data.at[prev_value, value_col] + data.at[next_value, value_col]) / 2
            data.at[date, value_col] = average_value
        elif pd.notnull(prev_value):  # If there's only a previous value available
            data.at[date, value_col] = data.at[prev_value, value_col]
        elif pd.notnull(next_value):  # If there's only a next value available
            data.at[date, value_col] = data.at[next_value, value_col]

    # Reset the index to move 'Date' back to a column
    # data.reset_index(inplace=True)
    return data


# %% Appyling the function to the data
filled_data = fill_missing_data_with_average(data)
filled_data

# %%
# Count number of NAs
print(filled_data.isna().sum())
print(filled_data[filled_data['CLOSE'].isna()])

# %%
# Plotting
plt.figure(figsize=(12, 6))  # Set the figure size for better visibility
# Plot the CLOSE prices
plt.plot(filled_data.index,
         filled_data['CLOSE'], label='CLOSE Price', color='blue')
plt.title('CLOSE Prices Over Time')  # Title of the plot
plt.xlabel('Date')  # Label for the X-axis
plt.ylabel('CLOSE Price')  # Label for the Y-axis
plt.grid(True)  # Turn on the grid
plt.legend()  # Show legend
plt.show()  # Display the plot


# %%
# Save resulting filled_data as csv file called TTT_final.csv
final_file_name = 'TTF_final.csv'
final_file_path = os.path.join(data_path, final_file_name)
filled_data.to_csv(final_file_path)
# %%
