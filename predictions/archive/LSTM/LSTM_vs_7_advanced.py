# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
# Import MAE if you need it
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %%


def select_device(requested_device=None):
    """
    Selects the most appropriate device available for computations.
    If a device is specified, it checks if that device is available and uses it.
    If no device is specified, it defaults to CUDA if available, then MPS, and finally CPU.

    Parameters:
    - requested_device (str, optional): The requested device as a string (e.g., 'cuda', 'mps', 'cpu').

    Returns:
    - torch.device: The selected PyTorch device.
    - str: Description of the selected device.
    """
    if requested_device:
        # User has requested a specific device
        if requested_device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda'), "NVIDIA GPU"
        elif requested_device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps'), "Apple Silicon GPU"
        elif requested_device == 'cpu':
            return torch.device('cpu'), "CPU"
        else:
            raise ValueError(
                f"Requested device '{requested_device}' is not available or not recognized.")

    # Default selection logic if no device is specified
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA GPU"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU"
    else:
        return torch.device("cpu"), "CPU"


# Select the best available device
device, device_name = select_device()
print(f"Using {device_name} for computation.")

# %%

# Import the data


def load_and_prepare_data(file_path):
    """
    Load energy prices data from a CSV file, ensure chronological order, and convert 'Date' to datetime.
    """
    df = pd.read_csv(file_path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df = pd.DataFrame(df)
    return df


df = load_and_prepare_data('../../data/Final_data/final_data.csv')

# %%
# Feature Engineering (add your features here)
df['previous_day_price'] = df['Day_ahead_price'].shift(1)
df.fillna(method='bfill', inplace=True)  # Handle any NaNs by backfilling

# Normalize the data
scaler = MinMaxScaler()
# Reset index
df.reset_index(inplace=True)
df_scaled = scaler.fit_transform(df.drop(columns=['date']))
df_scaled = pd.DataFrame(df_scaled, columns=df.columns.drop('date'))

# %%
# Creating sequences for the LSTM


def create_sequences(df, target_column, sequence_length):
    sequences = []
    data_size = len(df)
    for i in range(data_size - sequence_length):
        sequence = df[i:i + sequence_length]
        label = df.iloc[i + sequence_length][target_column]
        sequences.append((sequence, label))
    return sequences


sequence_length = 14
data_sequences = create_sequences(
    df_scaled, 'Day_ahead_price', sequence_length)


def train_test_split_sequential(data, test_size=0.2):
    """
    Split the sequence data into training and testing datasets sequentially.

    Parameters:
    - data: List of tuples where each tuple contains (sequence, label).
    - test_size: Fraction of the dataset to be used as test data.

    Returns:
    - train_data: Training data containing sequences and labels.
    - test_data: Testing data containing sequences and labels.
    """
    split_idx = int(len(data) * (1 - test_size))  # Calculate split index
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


# Split data
train_sequences, test_sequences = train_test_split_sequential(
    data_sequences, test_size=0.2)

# %%


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # Ensure this attribute is properly set

        # Initialize the LSTM layer with specified dropout and make it bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_prob, bidirectional=True)

        # Fully connected layer that outputs our predicted value
        # Note: the output size is doubled because the LSTM is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        # *2 for bidirectionality
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_dim).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the classifier
        # Only use the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model with the right parameters
model = LSTMModel(input_dim=len(df.columns) - 1,  # input dimension
                  hidden_dim=50,                  # number of neurons in the LSTM layer
                  num_layers=2,                   # number of LSTM layers
                  output_dim=1,                   # output dimension
                  dropout_prob=0.2                # dropout rate
                  ).to(device)


# Data Loaders Preparation


def sequences_to_tensor(sequences, device):
    # This unpacks sequences into two lists of numpy arrays
    X, y = zip(*sequences)

    # Convert lists of arrays into a single 2D numpy array for X
    X = np.array([np.array(xi) for xi in X], dtype=np.float32)

    # Convert y into a 1D numpy array
    # Reshape y to be 2D as expected by torch
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    # Convert numpy arrays to tensors and move them to the specified device
    X_tensor = torch.tensor(X).to(device)
    y_tensor = torch.tensor(y).to(device)

    return X_tensor, y_tensor


X_train, y_train = sequences_to_tensor(train_sequences, device)
X_test, y_test = sequences_to_tensor(test_sequences, device)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# %%
# Training the Model
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
# Reduces the learning rate every 25 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)


def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')


train_model(100)

# %%
# Evaluation
# Fit the scaler on the 'Day_ahead_price' column only
price_scaler = MinMaxScaler()
prices_scaled = price_scaler.fit_transform(df[['Day_ahead_price']])


# Adjust evaluate_model function to correctly apply inverse transform
def evaluate_model(model, test_loader, price_scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            predictions.extend(output.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    # Correctly inverse transform predictions and actuals
    predictions = price_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))
    actuals = price_scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    return predictions.flatten(), actuals.flatten()


predictions, actuals = evaluate_model(model, test_loader, price_scaler)


# Compute metrics
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(actuals, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.title('Day-Ahead Energy Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price (EUR)')
plt.legend()
plt.show()
# %%


# %%
