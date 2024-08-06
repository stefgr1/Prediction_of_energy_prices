
# %%
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare the data


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # Feature Engineering
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['lag1'] = df['Day_ahead_price'].shift(1)  # Lag feature
    df.dropna(inplace=True)  # Handle any NaNs introduced by lag features
    return df


# Load data
df = load_and_prepare_data('../../data/Final_data/final_data.csv')
df.reset_index(inplace=True)

# Split the data into train and test sets
X = df.drop(['Day_ahead_price', 'date'], axis=1)
y = df['Day_ahead_price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# %%
# Define the objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
        'objective': 'reg:squarederror'
    }

    model = XGBRegressor(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# %%


# Create a study object and perform the optimization
study = optuna.create_study(direction='minimize')
# Adjust n_trials or timeout as per computation resource
study.optimize(objective, n_trials=100, timeout=3600)

# Output the best parameters
print("Best parameters:", study.best_params)

# %%

# Fit a model with the best parameters
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# Make predictions and calculate RMSE
predictions = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Final RMSE: {final_rmse}")

# %%


# Combine training and test indices to sort the entire dataset for plotting
full_data = pd.concat([X_train, X_test])
full_data['Day_ahead_price'] = pd.concat(
    [y_train, y_test])  # Combine corresponding target values

# Sort by original data index or date if available
full_data.sort_index(inplace=True)

# Prepare plot data
train_indices = y_train.index
test_indices = y_test.index

# Creating the plot
plt.figure(figsize=(14, 7))
plt.plot(full_data.index, df['Day_ahead_price'], label='Actual', color='blue')
plt.scatter(test_indices, predictions, color='red',
            label='Predicted', alpha=0.6)  # Scatter for visibility
plt.title('Trend of Training, Test Data and Predictions')
plt.xlabel('Date')
plt.ylabel('Day Ahead Price')
plt.legend()
plt.grid(True)
plt.show()

# %%


# %%
