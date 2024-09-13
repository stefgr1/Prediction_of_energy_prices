# Data Import and Preparation
import pandas as pd
import numpy as np


# Load the data


def load_and_prepare_data(file_path):
    """
    Load data, convert dates, and ensure chronological order.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
