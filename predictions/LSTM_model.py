# %%
# Creating the LSTM
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import sys
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_prep_LSTM import prepare_data_for_model, reshape_for_lstm, get_min_max, normalize_dataframe
from linear_regression import load_and_prepare_data
sys.path.append(
    '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/predictions')

# Example function call, commented out to prevent execution
file_path = '/Users/skyfano/Documents/Masterarbeit/Prediction_of_energy_prices/data/storage/energy_prices_all_years/energy_prices_2019_2023.csv'
train_data, test_data = prepare_data_for_model(file_path, '2023-01-01')

# Splitting the data into X and Y
X_train, Y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, Y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Reshaping X for LSTM
X_train_reshaped = reshape_for_lstm(X_train, 1, X_train.shape[1])
X_test_reshaped = reshape_for_lstm(X_test, 1, X_test.shape[1])

# Shapes
X_train_shape, Y_train_shape = X_train_reshaped.shape, Y_train.shape
X_test_shape, Y_test_shape = X_test_reshaped.shape, Y_test.shape

print(X_train_shape, Y_train_shape, X_test_shape, Y_test_shape)

############## LSTM Model ##############

# Step 1: Create the model


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
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


# Step 2: Instantiate the model with hyperparameters
model = LSTMModel(input_size=1, hidden_layer_size=100, output_size=1)

# Step 3: Prepare the model for training
# Assuming X_train_reshaped is from your previous code
X_train_tensor = torch.Tensor(X_train_reshaped)
# Reshape Y_train to match output dimensions
Y_train_tensor = torch.Tensor(Y_train.values).view(-1, 1)

# Step 4: Training the model
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10  # This is arbitrary and can be adjusted

for i in range(epochs):
    for seq, labels in zip(X_train_tensor, Y_train_tensor):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 2 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Step 5: Make predictions

# Convert test set to PyTorch tensors
# Assuming X_test_reshaped is prepared
X_test_tensor = torch.Tensor(X_test_reshaped)
# Assuming Y_test is prepared
Y_test_tensor = torch.Tensor(Y_test.values).view(-1, 1)

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    # Assuming your model can handle the batch size, you might need to adjust this for your model
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                         torch.zeros(1, 1, model.hidden_layer_size))

    predictions = []
    for seq in X_test_tensor:
        y_pred = model(seq)
        predictions.append(y_pred.numpy())

predictions = np.array(predictions).flatten()

# Calculate performance metrics
test_mse = mean_squared_error(Y_test_tensor, predictions)
test_rmse = math.sqrt(test_mse)

print(f'Test MSE: {test_mse}')
print(f'Test RMSE: {test_rmse}')

# %%
# Example: Inspect the first few predictions
print(predictions[:10])

# %%
# Load and prepare data
train_df, test_df = load_and_prepare_data(file_path, '2023-01-01')

# Assuming get_min_max returns Series or DataFrame with a single value
original_min_series, original_max_series = get_min_max(train_df, ['Price'])

# Convert to scalars
original_min = original_min_series.iloc[0]
original_max = original_max_series.iloc[0]


def inverse_transform(normalized_values, original_min, original_max):
    return normalized_values * (original_max - original_min) + original_min


# Example usage (assuming you have these values)
# Replace with your actual min and max values
# Now these should be scalars
actual_prices = inverse_transform(
    Y_test_tensor.numpy().flatten(), original_min, original_max)
predicted_prices = inverse_transform(predictions, original_min, original_max)


days = range(len(actual_prices))

# Adjust this to your actual start date
start_date = pd.to_datetime('2023-01-01')
end_date = start_date + pd.Timedelta(days=len(actual_prices) - 1)
date_range = pd.date_range(start_date, end_date)

plt.figure(figsize=(10, 6))

# Use date_range if you have specific dates, else use days
plt.plot(date_range, actual_prices, label='Actual Prices')
plt.plot(date_range, predicted_prices, label='Predicted Prices', alpha=0.7)

plt.title('Energy Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# Adjust interval for better readability
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.legend()
plt.show()

# %%
