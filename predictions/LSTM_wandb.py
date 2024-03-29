import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pyhopper
import wandb
from pyhopper.callbacks.wandb import WandbCallback

# Initialize a new WandB run
wandb.init(project="LSTM_energy_prices", entity="skyfano1", reinit=True)

# Data loading, preprocessing, and utility functions


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def scale_data(data_frame, column_name):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_frame.loc[:, column_name] = scaler.fit_transform(
        data_frame[column_name].values.reshape(-1, 1))
    return scaler, data_frame


def split_sequence_data(data, lookback):
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


def train_and_evaluate(params):
    # Safely update WandB configuration with hyperparameters
    wandb.config.update(params, allow_val_change=True)

    file_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_all_years/energy_prices_2019_2023.csv'
    data = load_and_prepare_data(file_path)
    scaler, scaled_data = scale_data(data[['Price']], 'Price')
    lookback = 20
    x_train, y_train, x_test, y_test = split_sequence_data(
        scaled_data, lookback)

    # Convert to PyTorch tensors and adjust dimensions as necessary
    x_train_tensor = torch.tensor(x_train).float()
    y_train_tensor = torch.tensor(y_train).float().unsqueeze(-1)
    x_test_tensor = torch.tensor(x_test).float()
    y_test_tensor = torch.tensor(y_test).float().unsqueeze(-1)

    # Initialize the model with hyperparameters from PyHopper
    model = EnergyPriceLSTM(input_dim=1, hidden_dim=params["hidden_dim"],
                            num_layers=params["num_layers"], output_dim=1).cuda()

    # Define loss function and optimizer with hyperparameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Training loop
    for epoch in range(params["num_epochs"]):
        model.train()
        optimizer.zero_grad()

        y_train_pred = model(x_train_tensor.cuda())
        loss = criterion(y_train_pred, y_train_tensor.cuda())

        loss.backward()
        optimizer.step()

        wandb.log({"epoch": epoch, "training_loss": loss.item()})

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_tensor.cuda()).cpu().numpy()
        y_test = y_test_tensor.cpu().numpy()

        y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
        wandb.log({"Test RMSE": test_rmse})

    return test_rmse


# Define the search space and run the optimization
search = pyhopper.Search({
    "lr": pyhopper.float(0.001, 0.1, log=True),
    "hidden_dim": pyhopper.int(20, 100),
    "num_layers": pyhopper.int(1, 4),
    "num_epochs": pyhopper.int(50, 100),
})

if __name__ == "__main__":
    best_params = search.run(
        train_and_evaluate, direction="min", runtime="1h", callbacks=[
            WandbCallback(project="LSTM_energy_prices",
                          name="Experiment 1")
        ])
    print("Best Hyperparameters:", best_params)
