import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import requests
import time
import io

warnings.filterwarnings("ignore")


# Function to fetch data from ThingSpeak
def fetch_data_from_thingspeak(channel_id, read_api_key, results=8000):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_api_key}&results={results}"
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text))
    data['created_at'] = pd.to_datetime(data['created_at'])
    return data


# Function to clean the data
def clean_data(data):
    if 'created_at' in data.columns:
        data.drop(['entry_id'], axis=1, inplace=True)

    for col in data.columns:
        data[col] = preprocess_field(data[col])

    data.fillna(method='ffill', inplace=True)
    return data


# Function to preprocess a field
def preprocess_field(series):
    series = series.replace(r'[^0-9.-]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    series.fillna(method='ffill', inplace=True)
    return series


# Function to check stationarity and differencing if necessary
def check_stationarity_and_difference(series):
    series = series.dropna()
    if len(series) < 10:
        return pd.Series(series), True
    result = adfuller(series)
    if result[1] > 0.05:
        return pd.Series(np.diff(series)), False
    return pd.Series(series), True


# Function to fit ARIMA model
def fit_arima(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


# Function to forecast HVAC data
def forecast_hvac_data(hvac_data):
    forecast_results = {}
    orders = (7, 1, 0)

    for field in hvac_data.columns:
        if field == 'created_at':
            continue

        print(f"\nProcessing {field}")
        series = hvac_data[field]

        series_diff, is_stationary = check_stationarity_and_difference(series)
        if not is_stationary:
            series = series_diff

        if len(series) < 10:
            print(f"Not enough data points for {field} after differencing.")
            continue

        series = series.dropna()

        try:
            model_fit = fit_arima(series, orders)
            forecast = model_fit.forecast(steps=10)
            forecast_results[field] = forecast
        except Exception as e:
            print(f"An error occurred while processing {field}: {e}")

    return forecast_results


# Function to train the power consumption model
def train_power_consumption_model(hvac_data, power_data):
    hvac_data = hvac_data.drop('created_at', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(hvac_data, power_data, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"R-squared: {r2_score(y_test, y_pred)}")

    return model


# Function to predict power consumption
def predict_power_consumption(forecasted_hvac_data, power_model):
    forecast_df = pd.DataFrame(forecasted_hvac_data).T
    power_consumption_forecast = power_model.predict(forecast_df)
    return power_consumption_forecast


# ThingSpeak channel details
CHANNEL_ID = '2590984'
READ_API_KEY = 'BOPRV0GTWVROPYPR'

# Assume power_data is available
power_data = np.random.rand(8000) * 100  # Dummy power consumption data


# Train power consumption model
def initialize_models():
    hvac_data = fetch_data_from_thingspeak(CHANNEL_ID, READ_API_KEY)
    hvac_data = clean_data(hvac_data)

    # Ensure the length of power_data matches hvac_data
    if len(hvac_data) < len(power_data):
        power_data_trimmed = power_data[:len(hvac_data)]
    else:
        power_data_trimmed = np.append(power_data, np.zeros(len(hvac_data) - len(power_data)))

    power_model = train_power_consumption_model(hvac_data, power_data_trimmed)
    return power_model


power_model = initialize_models()

# Main loop to continuously fetch, process data, and predict
while True:
    try:
        hvac_data = fetch_data_from_thingspeak(CHANNEL_ID, READ_API_KEY)
        hvac_data = clean_data(hvac_data)
        forecasted_hvac_data = forecast_hvac_data(hvac_data)
        power_consumption_forecast = predict_power_consumption(forecasted_hvac_data, power_model)

        print("Forecasted Power Consumption:", power_consumption_forecast)

        # Sleep for a specified interval before fetching new data (e.g., 15 minutes)
        time.sleep(900)  # 900 seconds = 15 minutes
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(60)  # Wait for 1 minute before retrying
