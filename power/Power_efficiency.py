import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("setpoint19.csv")

# Select features and target
features = ["Evap_Out", "Comp_Out", "Cond_Out", "Evap_In"]
X = df[features]
y = df["COP"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)


rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)

print("Random Forest MSE:", rf_mse)
print("Gradient Boosting MSE:", gb_mse)

rf_best_predictions = rf_model.predict(X_test)
gb_best_predictions = gb_model.predict(X_test)

rf_best_mse = mean_squared_error(y_test, rf_best_predictions)
gb_best_mse = mean_squared_error(y_test, gb_best_predictions)

print("Best Random Forest MSE:", rf_best_mse)
print("Best Gradient Boosting MSE:", gb_best_mse)

final_predictions = np.mean([rf_best_predictions, gb_best_predictions], axis=0)
final_mse = mean_squared_error(y_test, final_predictions)

print("Final Ensemble MSE:", final_mse)
print(rf_best_predictions)
