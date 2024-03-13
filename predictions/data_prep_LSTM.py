import sys
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from linear_regression import load_and_prepare_data

sys.path.append(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/predictions')


def normalize_dataframe(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns of a pandas DataFrame using MinMaxScaler.
    Args:
        dataframe: The pandas DataFrame to normalize.
        columns: A list of column names to normalize.
    Returns:
        A pandas DataFrame with the specified columns normalized.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe[columns] = scaler.fit_transform(dataframe[columns])

    return dataframe

# function to get the min and max values of the original data from the scaler


def get_min_max(dataframe: pd.DataFrame, columns: list):
    original_min = dataframe[columns].min()
    original_max = dataframe[columns].max()
    return original_min, original_max


def series_to_supervised(data: pd.DataFrame, n_in: int = 1, n_out: int = 1, dropnan: bool = True) -> pd.DataFrame:
    """Convert time series data to a supervised learning format.
    Args:
        data: Time series data in a pandas DataFrame.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean indicating whether to drop rows with NaN values.
    Returns:
        A pandas DataFrame suitable for supervised learning.
    """
    n_vars = data.shape[1]
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_data_for_model(file_path: str, date_split: str, n_days: int = 1, n_features: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and split data into training and test sets for model training.
    Args:
        file_path: Path to the CSV file containing the data.
        date_split: Date to split the data into training and testing.
    Returns:
        A tuple containing the training and test pandas DataFrames.
    """
    # Load and prepare data
    train_df, test_df = load_and_prepare_data(file_path, date_split)

    # Normalize the Price data
    train_df = normalize_dataframe(train_df, ['Price'])
    test_df = normalize_dataframe(test_df, ['Price'])

    # Convert the Series to a DataFrame if it's not already
    reframed_train = series_to_supervised(
        pd.DataFrame(train_df['Price']), n_days, n_features)
    reframed_test = series_to_supervised(
        pd.DataFrame(test_df['Price']), n_days, n_features)

    return reframed_train, reframed_test


def reshape_for_lstm(X: pd.DataFrame, n_days: int = 1, n_features: int = 1) -> np.ndarray:
    """Reshape input data into the 3D format required by LSTM networks.
    Args:
        X: Input data in pandas DataFrame format.
        n_days: Number of days per sample.
        n_features: Number of features per sample.
    Returns:
        Input data reshaped into 3D array [samples, timesteps, features] for LSTM.
    """
    return X.values.reshape((X.shape[0], n_days, n_features))
