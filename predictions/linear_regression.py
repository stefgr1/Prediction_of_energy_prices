import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from pathlib import Path


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


def feature_engineering(df):
    """
    Add time-based features to the dataframe.
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Further features like day of week could be added here

    return df


def perform_regression(train_df, test_df):
    """
    Train and evaluate a linear regression model with polynomial features.
    """
    # Feature engineering for both train and test sets
    train_df_fe = feature_engineering(train_df.copy())
    test_df_fe = feature_engineering(test_df.copy())

    # Selecting features and target
    X_train = train_df_fe[['Year', 'Month', 'Day']]
    y_train = train_df_fe['Price']
    X_test = test_df_fe[['Year', 'Month', 'Day']]
    y_test = test_df_fe['Price']

    # Polynomial Features
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred, test_df_fe


def plot_results(file_path, train_df, test_df, y_pred):
    """
    Plot training, testing data, and predictions with advanced visualization.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(train_df['Date'], train_df['Price'],
             label='Training Data', color='blue', linewidth=2)
    plt.plot(test_df['Date'], test_df['Price'],
             label='Testing Data', color='red', linewidth=2)
    plt.scatter(test_df['Date'], y_pred,
                label='Predictions', color='green', s=10)

    plt.title('Energy Prices Prediction with Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Price (EUR)')
    plt.legend()
    plt.grid(True)

    plt.savefig(Path(file_path) / 'energy_prices_regression_plot.png')
    print(f"Plot successfully saved to {file_path}")
    plt.show()


def main():
    """
    Main execution function.
    """
    file_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices__all_years/energy_prices_2019_2023.csv'
    train_df, test_df = load_and_prepare_data(file_path, '2023-01-01')

    mse, y_pred, test_df_for_plot = perform_regression(train_df, test_df)
    print(f'Mean Squared Error: {mse:.4f}')

    plot_results(Path(file_path).parent, train_df, test_df_for_plot, y_pred)


if __name__ == '__main__':
    main()
