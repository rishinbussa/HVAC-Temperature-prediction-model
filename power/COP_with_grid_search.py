import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("setpoint18.csv")

# Select features and target
features = ["Evap_Out", "Comp_Out", "Cond_Out", "Evap_In"]
X = df[features]
y = df["COP"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predict COP
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate models
rf_mse = mean_squared_error(y_test, rf_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)

print("Random Forest MSE:", rf_mse)
print("Gradient Boosting MSE:", gb_mse)

# Hyperparameter tuning with Grid Search
# Parameters for tuning Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
)
rf_grid_search.fit(X_train, y_train)

print("Best Random Forest Params:", rf_grid_search.best_params_)
print("Best Random Forest Score:", rf_grid_search.best_score_)

# Parameters for tuning Gradient Boosting
gb_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 4, 5],
}

gb_grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gb_param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
)
gb_grid_search.fit(X_train, y_train)

print("Best Gradient Boosting Params:", gb_grid_search.best_params_)
print("Best Gradient Boosting Score:", gb_grid_search.best_score_)

# Train models with best parameters
rf_best_model = rf_grid_search.best_estimator_
gb_best_model = gb_grid_search.best_estimator_

rf_best_model.fit(X_train, y_train)
gb_best_model.fit(X_train, y_train)

# Predict COP with best models
rf_best_predictions = rf_best_model.predict(X_test)
gb_best_predictions = gb_best_model.predict(X_test)

# Evaluate best models
rf_best_mse = mean_squared_error(y_test, rf_best_predictions)
gb_best_mse = mean_squared_error(y_test, gb_best_predictions)

print("Best Random Forest MSE:", rf_best_mse)
print("Best Gradient Boosting MSE:", gb_best_mse)

# Ensemble averaging
final_predictions = np.mean([rf_best_predictions, gb_best_predictions], axis=0)
final_mse = mean_squared_error(y_test, final_predictions)

print("Final Ensemble MSE:", final_mse)
