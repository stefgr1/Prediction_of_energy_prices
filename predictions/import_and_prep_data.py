# Data Import and Preparation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data


def load_and_prepare_data(file_path, start_test_date_str):
    """
    Load data, convert dates, and ensure chronological order.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    start_test_date = pd.to_datetime(start_test_date_str)
    train_df = df.loc[df['Date'] < start_test_date]
    test_df = df.loc[df['Date'] >= start_test_date]

    return train_df, test_df
