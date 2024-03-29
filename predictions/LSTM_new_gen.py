import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import math
import torch.nn as nn
import torch.optim as optim

# Initialize a new WandB run
wandb.init(project="LSTM_energy_prices", entity="skyfano1")

# Data Loading and Preprocessing


def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def scale_data(data_frame, column_name):
    """
    Scale data within the specified column of a DataFrame using MinMaxScaler.
    Returns the scaler and the scaled DataFrame.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_frame.loc[:, column_name] = scaler.fit_transform(
        data_frame[column_name].values.reshape(-1, 1))
    return scaler, data_frame


def split_sequence_data(data, lookback):
    """
    Create sequences from the data set for training and testing LSTM models.
    """
    data_raw = data.to_numpy()
    sequences = [data_raw[i: i + lookback]
                 for i in range(len(data_raw) - lookback)]
    sequences = np.array(sequences)

    test_set_size = int(np.round(0.2 * sequences.shape[0]))
    train_set_size = sequences.shape[0] - test_set_size

    x_train = sequences[:train_set_size, :-1]
    y_train = sequences[:train_set_size, -1]
    x_test = sequences[train_set_size:, :-1]
    y_test = sequences[train_set_size:, -1]

    return x_train, y_train, x_test, y_test

# LSTM Model Definition


class EnergyPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Training and Evaluation Functions


def train_model(model, num_epochs, x_train, y_train, criterion, optimiser):
    """
    Train the LSTM model.
    """
    hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print(f"Epoch {t}: Loss: {loss.item()}")
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return hist


def evaluate_model(model, x_test, y_test, scaler):
    """
    Make predictions and evaluate the model using RMSE.
    """
    y_test_pred = model(x_test)
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print(f'Test Score: {testScore:.2f} RMSE')
    return y_test, y_test_pred

# Plotting Functions


def plot_predictions(y_train, y_train_pred, title='Energy Price Prediction'):
    """
    Plot the actual vs. predicted energy prices.
    """
    plt.figure(figsize=(15, 8))
    plt.plot(y_train, label="Actual Price", color='royalblue', linewidth=2)
    plt.plot(y_train_pred, label="Predicted Price (LSTM)",
             color='tomato', linewidth=2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Days', fontsize=14)
    plt.ylabel('Energy Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_loss(hist, title='Training Loss Over Epochs'):
    """
    Plot the training loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(hist, color='royalblue', linewidth=2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.show()


# Main execution flow
if __name__ == "__main__":
    # File path for the dataset
    file_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_all_years/energy_prices_2019_2023.csv'

    # Load and prepare data
    data = load_and_prepare_data(file_path)

   # Scale the 'Price' column and split the data into training and testing sets
    scaler, scaled_data = scale_data(data[['Price']], 'Price')
    lookback = 20  # Define the lookback period for sequence creation
    x_train, y_train, x_test, y_test = split_sequence_data(
        scaled_data, lookback)

    # Convert the data into PyTorch tensors
    x_train_tensor, y_train_tensor = torch.tensor(
        x_train).float(), torch.tensor(y_train).float()
    x_test_tensor, y_test_tensor = torch.tensor(
        x_test).float(), torch.tensor(y_test).float()

    # Model initialization
    input_dim = 1  # Input dimension: 1 feature (price)
    hidden_dim = 32  # Number of hidden layers
    num_layers = 2  # Number of LSTM layers
    output_dim = 1  # Output dimension: 1 feature (predicted price)
    num_epochs = 100  # Number of epochs for training

    model = EnergyPriceLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                            num_layers=num_layers, output_dim=output_dim)
    criterion = torch.nn.MSELoss(reduction='mean')  # Loss function
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)  # Optimizer

    # Train the model
    print("Training started...")
    start_time = time.time()
    hist = train_model(model, num_epochs, x_train_tensor,
                       y_train_tensor, criterion, optimiser)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.3f} seconds.")

    # Evaluate the model
    y_train, y_train_pred = evaluate_model(
        model, x_train_tensor, y_train_tensor, scaler)
    y_test, y_test_pred = evaluate_model(
        model, x_test_tensor, y_test_tensor, scaler)

    # Plot the training results
    plot_predictions(y_train, y_train_pred,
                     title='Energy Price Prediction on Training Set')
    plot_predictions(y_test, y_test_pred,
                     title='Energy Price Prediction on Test Set')
    plot_loss(hist)
