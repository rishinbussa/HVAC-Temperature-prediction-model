import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load the data
hvac_data = pd.read_csv('hvac_data_cleaned.csv')

# Clean the data

# 1. Remove unnecessary columns (e.g., 'created_at' and 'entry_id' are usually not needed for forecasting)
if 'created_at' in hvac_data.columns:
    hvac_data.drop(['created_at', 'entry_id'], axis=1, inplace=True)

# 2. Handle outliers by capping them to a specified threshold
threshold = 1.5  # Example threshold, can be adjusted
for column in hvac_data.columns:
    hvac_data[column] = hvac_data[column].clip(lower=hvac_data[column].quantile(0.05),
                                               upper=hvac_data[column].quantile(0.95))

# 3. Fill missing values using forward fill
hvac_data.fillna(method='ffill', inplace=True)

# Function to check stationarity and differencing if necessary
def check_stationarity_and_difference(series):
    series = series.dropna()  # Drop missing values
    if len(series) < 10:  # Ensure there are enough data points
        return series, True
    result = adfuller(series)
    if result[1] > 0.05:
        return np.diff(series), False
    return series, True

# Function to fit ARIMA model
def fit_arima(series):
    model = ARIMA(series, order=(7, 1, 0))
    model_fit = model.fit()
    return model_fit

# Function to calculate forecast metrics
def calculate_metrics(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    return mae, mse, rmse, r2

# Initialize a dictionary to store the results
forecast_results = {}

# Create subplots for ACF of residuals
fig_acf, axes_acf = plt.subplots(4, 2, figsize=(15, 20))
axes_acf = axes_acf.flatten()

# Loop through each field and perform the operations
for idx, field in enumerate(hvac_data.columns):
    print(f"\nProcessing {field}")

    # Check stationarity and differencing
    series = hvac_data[field]
    series_diff, is_stationary = check_stationarity_and_difference(series)
    if not is_stationary:
        series = series_diff

    # Ensure the series has sufficient data points after differencing
    if len(series) < 10:
        print(f"Not enough data points for {field} after differencing.")
        continue

    # Fit ARIMA model
    model_fit = fit_arima(series)

    # Make forecast
    forecast = model_fit.forecast(steps=5)
    forecast_results[field] = forecast

    # Calculate metrics
    mae, mse, rmse, r2 = calculate_metrics(series[-5:], forecast)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2:.4f}")  # Format R-squared to 4 decimal places

    # Plot ACF of residuals
    pd.plotting.autocorrelation_plot(pd.Series(model_fit.resid), ax=axes_acf[idx])
    axes_acf[idx].set_title(f"ACF of Residuals for {field}")

plt.tight_layout()
plt.show()

# Plot the results
plt.figure(figsize=(12, 8))
for i, field in enumerate(hvac_data.columns, 1):
    if field in forecast_results:
        plt.subplot(4, 2, i)
        plt.plot(hvac_data[field][-100:], label='Actual')
        plt.plot(pd.Series(np.concatenate([hvac_data[field][-5:].values, forecast_results[field]]),
                           index=np.arange(len(hvac_data[field]) - 5, len(hvac_data[field]) + 5)), label='Forecast')
        plt.title(field)
        plt.legend()
plt.tight_layout()
plt.show()

print(forecast_results)
