# %%
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Preprocessing the data

# Load the data
file_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/final_data.csv'
final_data = pd.read_csv(file_path)

# Set the date as index
final_data['date'] = pd.to_datetime(final_data['date'])
final_data.set_index('date', inplace=True)

# Separate features and target
X = final_data.drop('Day_ahead_price', axis=1).values
y = final_data['Day_ahead_price'].values.reshape(-1, 1)  # Reshape for scaling

# Normalize features
scaler_x = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_x.fit_transform(X)

# Normalize target
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)

# %% Create sequences

# Create sequences of time steps


def create_sequences(input_data, output_data, time_steps):
    xs, ys = [], []
    for i in range(len(input_data) - time_steps):
        xs.append(input_data[i:(i + time_steps)])
        ys.append(output_data[i + time_steps])
    return np.array(xs), np.array(ys)


time_steps = 90
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

# Converting to PyTorch tensors
X_train_seq = torch.FloatTensor(X_train_seq)
y_train_seq = torch.FloatTensor(y_train_seq)
X_test_seq = torch.FloatTensor(X_test_seq)
y_test_seq = torch.FloatTensor(y_test_seq)

# %% Define the LSTM model


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTMModel(
    input_size=X_train_seq.shape[2], hidden_layer_size=100, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% Train the model
epochs = 4

for i in range(epochs):
    for seq, labels in zip(X_train_seq, y_train_seq):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} final loss: {single_loss.item():10.10f}')

# %% Make predictions

# Detach the tensor from the current graph since it's only needed for inference
with torch.no_grad():
    model.eval()
    predictions = []
    for seq in X_test_seq:
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        predictions.append(model(seq).item())

# Inverse transform to original scale
predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test_seq.numpy().reshape(-1, 1))

# %% Print the predictions
print(predictions)

# %% Evaluate the model
# Calculate RMSE in the original scale
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
print(f"Root Mean Squared Error: {rmse}")

# %%
# Plot predictions against the actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test_original, label='True Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title('Energy Prices Prediction with LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# %%
