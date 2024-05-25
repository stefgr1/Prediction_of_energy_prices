import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load and preprocess data


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    return df

# Selecting the computation device


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Preparing data
df = load_and_prepare_data('./data/Final_data/final_data.csv')
X = df.drop(columns=['Day_ahead_price']).values
y = df['Day_ahead_price'].values.reshape(-1, 1)

# Scaling features
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

# Converting to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Reshaping data for LSTM input
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

# Model initialization
device = select_device()


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Ensure the model input size is correctly set
input_size = 18  # Number of features in your dataset
model = LSTM(input_size=input_size, hidden_size=200,
             num_layers=2, num_classes=1).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        test_outputs = model(X_test.to(device))
        test_loss = criterion(test_outputs, y_test.to(device))
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Predicting and Inverting scaling
model.eval()
predicted = model(X_test.to(device))
predicted = scaler_y.inverse_transform(predicted.cpu().detach().numpy())

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(y_test.cpu(), predicted))
print(f"RMSE: {rmse:.4f}")
